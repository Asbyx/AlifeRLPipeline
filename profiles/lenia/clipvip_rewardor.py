import torch
import torch.nn as nn
import pandas as pd
import pickle as pk
import os
from src.utils import Rewardor
from .clipvip_reward import CLIPVIPReward

class LeniaCLIPVIPRewardor(Rewardor):
    """
    A Rewardor implementation that uses a simplified CLIPVIP model to rank Lenia simulations.
    """
    def __init__(self, num_frames=12, minihead=True, device='cpu'):
        """
        Initialize the CLIPVIP-based rewardor.
        
        Args:
            num_frames: Number of frames to process
            minihead: Whether to use minihead architecture
            device: Device to run the model on
        """
        super().__init__()
        self.model = CLIPVIPReward(
            num_frames=num_frames,
            minihead=minihead,
            device=device
        )
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def rank(self, data):
        """
        Rank the data using the CLIPVIP model.
        
        Args:
            data: Tensor of shape (B, T, C, H, W) containing video frames
            
        Returns:
            Tensor of shape (B,) containing scores for each video
        """
        self.model.eval()
        with torch.no_grad():
            scores = self.model(data.to(self.device))
        return scores.cpu()

    def train(self, pairs_path):
        """
        Train the rewardor using paired comparisons.
        
        Args:
            pairs_path: Path to CSV file containing paired comparisons
        """
        pairs_df = pd.read_csv(pairs_path)
        # Filter out rows where winner is null
        pairs_df = pairs_df[pairs_df['winner'].notna()]
        
        # Convert pairs to training data
        for _, row in pairs_df.iterrows():
            param1_path = os.path.join('out/lenia/outputs', f"{row['param1']}.pkl")
            param2_path = os.path.join('out/lenia/outputs', f"{row['param2']}.pkl")
            
            # Load the outputs
            with open(param1_path, 'rb') as f:
                video1 = pk.load(f)
            with open(param2_path, 'rb') as f:
                video2 = pk.load(f)
                
            # Convert to tensors and ensure correct shape (B,T,C,H,W)
            video1 = torch.from_numpy(video1).float()
            video2 = torch.from_numpy(video2).float()
            
            # Ensure videos are in the correct format
            if video1.shape[1] == 3:  # If in format (B,C,T,H,W)
                video1 = video1.permute(0,2,1,3,4)  # Convert to (B,T,C,H,W)
            if video2.shape[1] == 3:  # If in format (B,C,T,H,W)
                video2 = video2.permute(0,2,1,3,4)  # Convert to (B,T,C,H,W)
                
            video1 = video1.to(self.device)
            video2 = video2.to(self.device)
            
            # Create labels (1 if param1 won, 0 if param2 won)
            label = torch.tensor(1.0 if row['winner'] == row['param1'] else 0.0).to(self.device)
            
            # Train step
            self.model.train()
            self.optimizer.zero_grad()
            
            score1 = self.model(video1)
            score2 = self.model(video2)
            
            # Compute loss using binary cross entropy
            pred = torch.sigmoid(score1 - score2)
            loss = self.criterion(pred, label)
            
            loss.backward()
            self.optimizer.step()

    def save(self, path):
        """
        Save the model state to the specified path.
        
        Args:
            path: Path to save the model state
        """
        self.model.save_weights(path)

    def load(self, path):
        """
        Load the model state from the specified path.
        
        Args:
            path: Path to load the model state from
        """
        self.model.load_weights(path) 