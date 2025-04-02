from ..utils import Rewarder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
import numpy as np
from ..data_managers import TrainingDataset
import hashlib
from typing import List, Any
import wandb

class TorchRewarder(nn.Module, Rewarder):
    """
    A rewarder that uses a torch model. 

    The only method that needs to be implemented is the forward method.

    Note: using this class assume that the outputs are torch tensors (dtype=torch.float32) saved as pt files, with torch.save.
    """
    def __init__(self, config: dict, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", wandb_params: dict = None):
        """
        Initialize the TorchRewarder.
    
        Args:
            config: Dictionary containing configuration parameters. All are optional.
                lr (default 0.001): Learning rate
                epochs (default 100): Number of epochs to train
                batch_size (default 16): Batch size
                val_split (default 0.2): Validation split
                early_stopping_patience (default 10): Early stopping patience
                loss (default "cross_entropy"): Loss function to use. Can be "margin" or "cross_entropy".
            model_path: Path to save or load the model
            device: Device to run the model on. Defaults to "cuda" if available, otherwise "cpu".
            wandb_params: Dictionary containing wandb parameters. Defaults to None.
        """
        nn.Module.__init__(self)
        self.config = config or {}
        self.device = device
        self.optimizer = None
        self.model_path = model_path
        self.wandb_params = wandb_params
        
        def cross_entropy_loss(scores1, scores2, y):
            p = torch.exp(scores2) / (torch.exp(scores1) + torch.exp(scores2)) # as y = 0 means "left win", p = P("right wins")
            return -torch.sum(y * torch.log(p) + (1 - y) * torch.log(1 - p))
        
        self.loss = {
            "margin": lambda scores1, scores2, y: F.margin_ranking_loss(scores1, scores2, y*2-1, margin=0.1), # convert to {-1, 1}
            "cross_entropy": cross_entropy_loss
        }[self.config.get('loss', 'cross_entropy')]

    #-------- Methods to implement --------#
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor, assumed to be torch tensors already on the correct device. Should be a batch of data.
            
        Returns:
            Output tensor, shape (batch_size)
        """
        raise NotImplementedError("Forward method must be implemented by child classes")
    
    def preprocess(self, data: List[Any]) -> torch.Tensor:
        """
        Preprocess the data.

        Args:
            data: List of data to preprocess.
            
        Returns:
            Preprocessed data ready for forward pass (B, *)
        """
        return data
    
    #-------- End of methods to implement --------#

    def _setup_optimizer(self):
        """Set up the optimizer using the config parameters"""
        lr = self.config.get('lr', 0.001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def rank(self, data):
        """
        Rank the data based on the model's predictions.
        
        Args:
            data: Data to rank; shape (batch_size, *)
            
        Returns:
            List of float rewards for each input
        """
        data = self.preprocess(data)

        # Make the data a torch tensor
        if isinstance(data, torch.Tensor):
            data = data.to(self.device)
        elif isinstance(data, np.ndarray):
            data = torch.FloatTensor(data).to(self.device)
        elif isinstance(data, list):
            data = torch.stack(data).to(self.device)
        else:
            raise ValueError("Data must be a numpy array or a list")

        super().train(False)
        with torch.no_grad():
            scores = self(data)
            
        # Convert to numpy and normalize to [0, 1] if more than one sample
        scores_np = scores.cpu().numpy()
        if len(scores_np) > 1:
            scores_np = (scores_np - np.min(scores_np)) / (np.max(scores_np) - np.min(scores_np) + 1e-10)
        
        return scores_np.tolist()
    
    def _get_batches(self, dataset: TrainingDataset, indices, batch_size):
        """
        Get batches from the dataset using specified indices.
        
        Args:
            dataset: The dataset to sample from.
            indices: The indices to use for sampling.
            batch_size: Size of each batch.
            
        Yields:
            Batches of data.
        """
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_data = [dataset[j] for j in batch_indices]
            
            # Unpack the batch data
            path1s, path2s, winners = zip(*batch_data)
            
            # Load and preprocess the data using caching
            data1 = torch.stack([self._load_or_preprocess(path) for path in path1s])
            data2 = torch.stack([self._load_or_preprocess(path) for path in path2s])
            
            if not isinstance(data1[0], torch.Tensor):
                raise ValueError("The outputs are not saved as torch tensors. Please save the outputs as torch tensors using torch.save. Note: numpy arrays saved with torch.save will still be saved as numpy arrays.")

            yield data1, data2, torch.tensor(winners, dtype=torch.float32, device=self.device)
    
    def _evaluate_dataset(self, dataset_batches):
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset_batches: Generator yielding batches of data
            
        Returns:
            Average loss and accuracy over the dataset
        """
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        
        super().train(False)
        with torch.no_grad():
            for batch_data1, batch_data2, batch_winners in dataset_batches:
                # Get predictions
                scores1 = self(batch_data1)
                scores2 = self(batch_data2)
                
                # Compute loss
                loss = self.loss(scores1, scores2, batch_winners)
                total_loss += loss.item()
                
                # Compute accuracy
                predictions = (scores1 < scores2).int()
                correct += (predictions == batch_winners).sum().item()
                total += len(batch_winners)
                
                batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def _create_folds(self, dataset_size: int, indices: list, fold_ratio: float = None) -> tuple:
        """
        Create training and validation folds based on the fold ratio.
        
        Args:
            dataset_size: Total number of samples in the dataset
            indices: List of indices to split into folds
            fold_ratio: Ratio of data to use for validation in each fold
            
        Returns:
            tuple: (fold_train_indices, fold_val_indices, num_folds)
        """
        if fold_ratio is None or fold_ratio <= 0:
            # Original behavior with single validation split
            val_split = self.config.get('val_split', 0.2)
            random.shuffle(indices)
            val_size = int(val_split * dataset_size)
            return [indices[val_size:]], [indices[:val_size]], 1
        
        # Cross validation setup based on fold_ratio
        num_folds = max(2, round(1 / fold_ratio))  # Limit between 2 and 10 folds
        fold_size = int(dataset_size * fold_ratio)
        fold_train_indices = []
        fold_val_indices = []
        
        # Create folds
        for fold in range(num_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < num_folds - 1 else dataset_size
            
            val_indices = indices[val_start:val_end]
            train_indices = indices[:val_start] + indices[val_end:]
            
            fold_train_indices.append(train_indices)
            fold_val_indices.append(val_indices)
        
        return fold_train_indices, fold_val_indices, num_folds

    def _train_single_batch(self, batch_data1, batch_data2, batch_winners):
        """
        Train on a single batch of data.
        
        Args:
            batch_data1: First batch of data
            batch_data2: Second batch of data
            batch_winners: Batch of winner labels
            
        Returns:
            tuple: (loss, correct_predictions, total_samples)
        """        
        # Get predictions
        scores1 = self(batch_data1)
        scores2 = self(batch_data2)
        
        # Compute loss
        loss = self.loss(scores1, scores2, batch_winners)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Calculate metrics
        predictions = (scores1 < scores2).int()
        correct = (predictions == batch_winners).sum().item()
        
        return loss.item(), correct, len(batch_winners)

    def _train_single_epoch(self, train_indices: list, dataset: TrainingDataset, batch_size: int):
        """
        Train for a single epoch.
        
        Args:
            train_indices: List of training indices
            dataset: Training dataset
            batch_size: Batch size
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        
        # Shuffle training indices for each epoch
        random.shuffle(train_indices)
        
        # Get batches for training
        for batch_data1, batch_data2, batch_winners in self._get_batches(dataset, train_indices, batch_size):
            loss, batch_correct, batch_total = self._train_single_batch(batch_data1, batch_data2, batch_winners)
            
            total_loss += loss
            correct += batch_correct
            total += batch_total
            batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy

    def _log_progress(self, fold: int, num_folds: int, epoch: int, epochs: int,
                     train_loss: float, train_accuracy: float,
                     val_loss: float, val_accuracy: float):
        """
        Log training progress to console and wandb if configured.
        
        Args:
            fold: Current fold number
            num_folds: Total number of folds
            epoch: Current epoch number
            epochs: Total number of epochs
            train_loss: Training loss
            train_accuracy: Training accuracy
            val_loss: Validation loss
            val_accuracy: Validation accuracy
        """
        if self.wandb_run:
            log_data = {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }
            if num_folds > 1:
                log_data.update({
                    "fold": fold + 1,
                    f"fold_{fold + 1}_train_loss": train_loss,
                    f"fold_{fold + 1}_val_loss": val_loss
                })
            self.wandb_run.log(log_data)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            fold_info = f"Fold {fold + 1}/{num_folds} - " if num_folds > 1 else ""
            print(f"{fold_info}Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    def train(self, dataset: TrainingDataset):
        """
        Train the model on the dataset of pairs using cross validation if specified.
        
        Args:
            dataset: TrainingDataset instance containing pairs of simulations with labels
            
        The config can contain:
            fold_ratio (float, default None): Ratio of data to use for validation in each fold.
                If None or 0, uses val_split for a single validation set.
                Otherwise, creates multiple folds where each fold uses this ratio of data for validation.
                For example, 0.2 means each fold uses 20% of data for validation, resulting in 5 folds.
            val_split (float, default 0.2): Ratio of data to use for validation when fold_ratio is None.
            early_stopping_patience (int, default 3): Number of epochs to wait for improvement before stopping.
            Other parameters remain the same as before.
        """
        if not hasattr(self, 'optimizer'):
            raise ValueError("Optimizer not set up. Call __init__ method at initialization of the child class.")
        
        self._setup_optimizer()

        if self.wandb_params:
            self.wandb_run = wandb.init(**self.wandb_params)

        # Training parameters
        batch_size = self.config.get('batch_size', 16)
        epochs = self.config.get('epochs', 100)
        fold_ratio = self.config.get('fold_ratio', None)
        early_stopping_patience = self.config.get('early_stopping_patience', 3)
        
        # Create folds
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        fold_train_indices, fold_val_indices, num_folds = self._create_folds(dataset_size, indices, fold_ratio)
        
        # Track best model across all folds
        best_overall_val_loss = float('inf')
        best_overall_model_state = None

        print(f"Training on {dataset_size} pairs, for {dataset.simulations_number} simulations, with {num_folds} validation folds.")
        
        # Training loop for each fold
        for fold in range(num_folds):
            train_indices = fold_train_indices[fold]
            val_indices = fold_val_indices[fold]
            
            print(f"Fold {fold + 1}: training on {len(train_indices)} samples, validating on {len(val_indices)} samples")
            
            # Initial fold state
            best_val_loss = float('inf')
            best_model_state = None
            patience_counter = 0
            
            # Training loop for current fold
            for epoch in range(epochs):
                # Train for one epoch
                super().train()
                train_loss, train_accuracy = self._train_single_epoch(train_indices, dataset, batch_size)
                
                # Validation phase
                val_loss, val_accuracy = self._evaluate_dataset(
                    self._get_batches(dataset, val_indices, batch_size)
                )
                
                # Log progress
                self._log_progress(fold, num_folds, epoch, epochs,
                                 train_loss, train_accuracy,
                                 val_loss, val_accuracy)
                
                # Early stopping and model checkpointing
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Update best overall model if this fold performed better
            if best_val_loss < best_overall_val_loss:
                best_overall_val_loss = best_val_loss
                best_overall_model_state = best_model_state
        
        # Load best model across all folds
        if best_overall_model_state is not None:
            self.load_state_dict(best_overall_model_state)
            print(f"Loaded best model with validation loss: {best_overall_val_loss:.4f}")
        if self.wandb_run:
            self.wandb_run.finish()
    
    def save(self):
        """
        Save the model to disk using the model_path attribute.
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load(self):
        """
        Load the model from disk using the model_path attribute.
        
        Returns:
            The loaded rewarder
        """
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Update config
            self.config = checkpoint.get('config', self.config)
            
            # Load model state
            self.load_state_dict(checkpoint['model_state_dict'])
            super().train(False)
            
            # Re-setup optimizer with possibly updated config
            self._setup_optimizer()
            
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"No model found at {self.model_path}, using initialized model")
            
        return self

    def _get_preprocessed_path(self, data_path: str) -> str:
        """
        Get the path where the preprocessed version of a file should be stored.
        
        Args:
            data_path: Original data file path
            
        Returns:
            Path where preprocessed version should be stored
        """
        # Create temp directory next to model path if it doesn't exist
        model_dir = os.path.dirname(self.model_path)
        preprocessed_dir = os.path.join(model_dir, "preprocessed_temp")
        os.makedirs(preprocessed_dir, exist_ok=True)
        
        # Create unique filename based on original path and last modified time
        file_stat = os.stat(data_path)
        unique_id = f"{data_path}_{file_stat.st_mtime}"
        filename_hash = hashlib.md5(unique_id.encode()).hexdigest()
        
        return os.path.join(preprocessed_dir, f"{filename_hash}.pt")
    
    def _load_or_preprocess(self, data_path: str) -> torch.Tensor:
        """
        Load preprocessed data if it exists, otherwise preprocess and save.
        
        Args:
            data_path: Path to the original data file
            
        Returns:
            Preprocessed data as torch tensor
        """
        preprocessed_path = self._get_preprocessed_path(data_path)
        
        # If preprocessed file exists and is newer than original, load it
        if os.path.exists(preprocessed_path):
            if os.path.getmtime(preprocessed_path) >= os.path.getmtime(data_path):
                return torch.load(preprocessed_path, map_location=self.device)
        
        # Otherwise preprocess and save
        data = torch.load(data_path, map_location=self.device)
        preprocessed_data = self.preprocess([data])[0]
        torch.save(preprocessed_data, preprocessed_path)
        return preprocessed_data
