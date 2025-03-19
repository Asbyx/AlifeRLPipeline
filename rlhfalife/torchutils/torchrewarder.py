from ..utils import Rewarder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
import numpy as np
from ..data_managers import TrainingDataset

class TorchRewarder(nn.Module, Rewarder):
    """
    A rewarder that uses a torch model. 

    The only method that needs to be implemented is the forward method.

    Note: using this class assume that the outputs are torch tensors (dtype=torch.float32) saved as pt files, with torch.save.
    """
    def __init__(self, config: dict, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the TorchRewarder.
    
        Args:
            config: Dictionary containing configuration parameters. All are optional.
                lr (default 0.001): Learning rate
                epochs (default 100): Number of epochs to train
                batch_size (default 16): Batch size
                val_split (default 0.2): Validation split
                early_stopping_patience (default 10): Early stopping patience
            model_path: Path to save or load the model
            device: Device to run the model on. Defaults to "cuda" if available, otherwise "cpu".
        """
        nn.Module.__init__(self)
        self.config = config or {}
        self.device = device
        self.optimizer = None
        self.model_path = model_path

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
            
            # Load the actual data from the paths
            data1 = [torch.load(path, map_location=self.device) for path in path1s]
            data2 = [torch.load(path, map_location=self.device) for path in path2s]
            
            if not isinstance(data1[0], torch.Tensor):
                raise ValueError("The outputs are not saved as torch tensors. Please save the outputs as torch tensors using torch.save. Note: numpy arrays saved with torch.save will still be saved as numpy arrays.")

            yield data1, data2, winners
    
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
                # Stack tensors in each batch
                data1_tensor = torch.stack(batch_data1)
                data2_tensor = torch.stack(batch_data2)
                
                # Prepare target for margin ranking loss
                y = torch.tensor([-1 if w == 1 else 1 for w in batch_winners], device=self.device)
                
                # Get predictions
                scores1 = self(data1_tensor)
                scores2 = self(data2_tensor)
                
                # Compute loss
                loss = F.margin_ranking_loss(scores1, scores2, y, margin=0.1)
                total_loss += loss.item()
                
                # Compute accuracy
                predictions = (scores1 > scores2).int() * 2 - 1  # Convert to -1 or 1
                correct += (predictions == y).sum().item()
                total += len(y)
                
                batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def train(self, dataset):
        """
        Train the model on the dataset of pairs.
        
        Args:
            dataset: TrainingDataset instance containing pairs of simulations with labels
        """
        # check that the __init__ method has been called
        if not hasattr(self, 'optimizer'):
            raise ValueError("Optimizer not set up. Call __init__ method at initialization of the child class.")
        
        self._setup_optimizer()

        # Training parameters
        batch_size = self.config.get('batch_size', 16)
        epochs = self.config.get('epochs', 100)
        val_split = self.config.get('val_split', 0.2)
        early_stopping_patience = self.config.get('early_stopping_patience', 10)
        
        # Split dataset into training and validation sets
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        random.shuffle(indices)
        
        val_size = int(val_split * dataset_size)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        print(f"Training on {len(train_indices)} samples, validating on {len(val_indices)} samples")
        
        # Initial model state
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # Training loop
        super().train()
        for epoch in range(epochs):
            total_train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_batch_count = 0
            
            # Shuffle training indices for each epoch
            random.shuffle(train_indices)
            
            # Get batches for training
            for batch_data1, batch_data2, batch_winners in self._get_batches(dataset, train_indices, batch_size):
                # Stack tensors in each batch
                data1_tensor = torch.stack(batch_data1)
                data2_tensor = torch.stack(batch_data2)
                
                # Prepare target for margin ranking loss
                y = torch.tensor([-1 if w == 1 else 1 for w in batch_winners], device=self.device)
                
                # Get predictions
                scores1 = self(data1_tensor)
                scores2 = self(data2_tensor)
                
                # Compute loss
                loss = F.margin_ranking_loss(scores1, scores2, y, margin=0.1)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                total_train_loss += loss.item()
                predictions = (scores1 > scores2).int() * 2 - 1
                train_correct += (predictions == y).sum().item()
                train_total += len(y)
                train_batch_count += 1
            
            # Calculate training metrics
            avg_train_loss = total_train_loss / train_batch_count if train_batch_count > 0 else 0
            train_accuracy = train_correct / train_total if train_total > 0 else 0
            
            # Validation phase
            val_loss, val_accuracy = self._evaluate_dataset(
                self._get_batches(dataset, val_indices, batch_size)
            )
            
            # Report progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
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
        
        # Load best model if available
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            print(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    
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
