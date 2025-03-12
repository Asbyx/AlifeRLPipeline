import src.utils
import torch
import torch.nn as nn
import pandas as pd
import os
import torchvision.models as models

class Lenia_Rewarder(src.utils.Rewarder):
    """
    A Rewarder implementation that uses a simplified CLIPVIP model to rank Lenia simulations.
    """
    def __init__(self, num_frames=12, minihead=True, device='cpu'):
        """
        Initialize the CLIPVIP-based rewarder.
        
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
        self.num_frames = num_frames

    def set_simulation(self, simulation):
        self.simulation = simulation
    
    def rank(self, data):
        """
        Rank the data using the CLIPVIP model.
        
        Args:
            data: List of tensors, each of shape (T, C, H, W) containing video frames
            
        Returns:
            Tensor of shape (B,) containing scores for each video
        """
        torch.cuda.empty_cache()
        self.model.eval()
        with torch.no_grad(), torch.autocast(device_type=self.device):
            # Add batch dimension to each video
            videos = [video.unsqueeze(0) for video in data]

            # Ensure videos are in the correct format
            for i in range(len(videos)):
                if videos[i].shape[1] == 3:  # If in format (C,T,H,W)
                    videos[i] = videos[i].permute(0,2,1,3,4)  # Convert to (B,T,C,H,W)

            # Ensure videos are the correct number of frames
            videos = [video[:, :self.num_frames] for video in videos]
                
            # Process videos in batches to reduce GPU memory usage
            batch_size = 4  # Adjust based on available GPU memory
            scores = []
            
            for i in range(0, len(videos), batch_size):
                batch = videos[i:i+batch_size]
                # Move batch to device and get scores
                batch = [video.to(self.device) for video in batch]
                batch_scores = [self.model(video) for video in batch]
                scores.extend(batch_scores)
                
                # Clear GPU memory after processing batch
                for video in batch:
                    video.cpu()
                torch.cuda.empty_cache()

        return scores

    def train(self, pairs_path, out_path):
        """
        Train the rewarder using paired comparisons.
        
        Args:
            pairs_path: Path to CSV file containing paired comparisons
        """
        pairs_df = pd.read_csv(pairs_path, dtype='str')
        # Filter out rows where winner is null
        pairs_df = pairs_df[pairs_df['winner'].notna()]
        
        print(f"Starting training with {len(pairs_df)} comparison pairs")
        
        # clear cache
        torch.cuda.empty_cache()

        self.model.train()
        total_loss = 0
        correct_predictions = 0
        
        # Convert pairs to training data
        for idx, row in pairs_df.iterrows():
            param1_path = os.path.join(out_path["outputs"], f"{row['param1']}.pkl")
            param2_path = os.path.join(out_path["outputs"], f"{row['param2']}.pkl")
            
            # Load the outputs
            video1 = self.simulation.load_outputs([param1_path])[0]
            video2 = self.simulation.load_outputs([param2_path])[0]

            # Add batch dimension
            video1 = video1.unsqueeze(0)
            video2 = video2.unsqueeze(0)

            # Ensure videos are in the correct format
            if video1.shape[1] == 3:  # If in format (C,T,H,W)
                video1 = video1.permute(0,2,1,3,4)  # Convert to (B,T,C,H,W)
            if video2.shape[1] == 3:  # If in format (C,T,H,W)
                video2 = video2.permute(0,2,1,3,4)  # Convert to (B,T,C,H,W)

            # Ensure videos are the correct number of frames
            video1 = video1[:, :self.num_frames]
            video2 = video2[:, :self.num_frames]
                
            video1 = video1.to(self.device)
            video2 = video2.to(self.device)
            
            # Create labels (1 if param1 won, 0 if param2 won)
            label = torch.tensor([1.0 if row['winner'] == row['param1'] else 0.0]).to(self.device)
            
            # Train step
            self.optimizer.zero_grad()
            score1 = self.model(video1)
            score2 = self.model(video2)
            
            # Compute loss using binary cross entropy
            pred = torch.sigmoid(score1 - score2)
            loss = self.criterion(pred, label)
            
            # Track metrics
            total_loss += loss.item()
            predicted_winner = 1 if pred.item() > 0.5 else 0
            actual_winner = 1 if row['winner'] == row['param1'] else 0
            correct_predictions += (predicted_winner == actual_winner)
            
            loss.backward()
            self.optimizer.step()
            
            # Print progress
            avg_loss = total_loss / (idx + 1)
            accuracy = correct_predictions / (idx + 1) * 100
            if (idx + 1) % 5 == 0:
                print(f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f})")
                print(f"Prediction: {pred.item():.4f} (threshold: 0.5)")
                print(f"Current accuracy: {accuracy:.2f}%")
                print(f"Completed {idx + 1}/{len(pairs_df)} pairs")
                print()
            
        # Print final statistics
        print(f"Final average loss: {total_loss / len(pairs_df):.4f}")
        print(f"Final accuracy: {(correct_predictions / len(pairs_df)) * 100:.2f}%")

    def save(self, path):
        """
        Save the model state to the specified path.
        
        Args:
            path: Path to save the model state
        """
        self.model.save_weights(path+"/rewarder.pth")

    def load(self, path):
        """
        Load the model state from the specified path.
        
        Args:
            path: Path to load the model state from
        """
        self.model.load_weights(path+"/rewarder.pth") 

class MiniBlock(nn.Module):
    """A simple transformer-style block for processing sequences"""
    def __init__(self, embed_dim, num_tokens, device='cpu'):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=2, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.to(device)

    def forward(self, x):
        x = x + self.attention(x, x, x)[0]
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x

class CLIPVIPReward(nn.Module):
    """
    A simplified version of CLIPVIP reward model that uses a ResNet backbone.
    Expected video shape: (B, T, C, H, W)
    """
    def __init__(self, num_frames=12, minihead=False, device='cpu'):
        super().__init__()
        
        # Use ResNet18 as backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
        self.backbone_dim = 512  # ResNet18's final feature dimension
        
        if minihead:
            self.head = MiniBlock(self.backbone_dim, num_frames, device)
        else:
            self.head = nn.Identity()
            
        self.final = nn.Linear(self.backbone_dim, 1)
        self.to(device)
        
    def forward(self, x):
        """
        x: (B, T, C, H, W) tensor of video frames
        Returns: (B,) tensor of scores
        """
        B, T, C, H, W = x.shape
        
        # Process each frame through the backbone
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)  # (B*T, backbone_dim, 1, 1)
        features = features.view(B, T, self.backbone_dim)  # (B, T, backbone_dim)
        
        # Process sequence
        features = self.head(features)  # (B, T, backbone_dim)
        
        # Take the last token's features
        features = features[:, -1]  # (B, backbone_dim)
        
        # Final scoring
        scores = self.final(features)  # (B, 1)
        return scores.squeeze(1)  # (B,)
    
    def save_weights(self, path):
        """Save model weights"""
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        """Load model weights"""
        self.load_state_dict(torch.load(path))