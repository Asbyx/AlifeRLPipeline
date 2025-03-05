import torch
import torch.nn as nn
import torchvision.models as models

class MiniBlock(nn.Module):
    """A simple transformer-style block for processing sequences"""
    def __init__(self, embed_dim, num_tokens, device='cpu'):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
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
        
        # Use ResNet50 as backbone
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
        self.backbone_dim = 2048  # ResNet50's final feature dimension
        
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