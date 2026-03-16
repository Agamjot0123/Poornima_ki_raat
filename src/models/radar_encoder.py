"""
Radar Encoder — ResNet-50 based 2D encoder for delay-Doppler images.

Uses a pre-trained ResNet-50 backbone with modified first conv layer
for single-channel input. Outputs a 256-dim feature vector.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class RadarEncoder(nn.Module):
    """ResNet-50 encoder for 2D delay-Doppler radar images.
    
    Architecture:
        Input: (B, 1, 224, 224) single-channel radar image
        → Modified ResNet-50 (pretrained, 1-channel input)
        → Global Average Pooling
        → FC projection to encoder_dim
        → Output: (B, encoder_dim)
    """
    
    def __init__(self, encoder_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.encoder_dim = encoder_dim
        
        # Load pre-trained ResNet-50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Modify first conv layer: 3 channels → 1 channel
        # Average the pretrained weights across the 3 input channels
        original_conv = resnet.conv1
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        if pretrained:
            # Initialise with mean of RGB channel weights
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Copy remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Projection head: 2048 (ResNet) + 2 (coords) → encoder_dim
        self.projection = nn.Sequential(
            nn.Linear(2050, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, encoder_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 224, 224) radar image tensor
            coords: (B, 2) tensor of [psi, delta] physical coordinates
        
        Returns:
            (B, encoder_dim) feature vector
        """
        # ResNet backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling: (B, 2048, 7, 7) → (B, 2048)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Concatenate physical coordinates
        x_fused = torch.cat([x, coords], dim=1)  # (B, 2050)
        
        # Project to encoder_dim
        out = self.projection(x_fused)
        
        return out
