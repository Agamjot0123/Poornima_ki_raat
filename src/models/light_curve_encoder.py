"""
Light Curve Encoder — 1D CNN + BiLSTM for temporal light-curve features.

Extracts local flux-variation features with 1D convolutions,
then captures rotational-phase dependencies with a BiLSTM.
Output: 256-dim feature vector.
"""

import torch
import torch.nn as nn


class LightCurveEncoder(nn.Module):
    """1D CNN + BiLSTM encoder for optical light curves.
    
    Architecture:
        Input: (B, 1, L) where L = light_curve_length (default 512)
        → 3-layer 1D CNN (Conv1d → BatchNorm → ReLU → MaxPool)
        → BiLSTM
        → Output: (B, encoder_dim)
    """
    
    def __init__(self, input_length: int = 512, encoder_dim: int = 256):
        super().__init__()
        self.encoder_dim = encoder_dim
        
        # 1D CNN feature extractor
        self.cnn = nn.Sequential(
            # Block 1: 2 (flux + phase) → 32 channels
            nn.Conv1d(2, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # L → L/2
            
            # Block 2: 32 → 64 channels
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # L/2 → L/4
            
            # Block 3: 64 → 128 channels
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # L/4 → L/8
        )
        
        # BiLSTM for temporal dependencies
        # Input: (B, L/8, 128) — we transpose for LSTM
        lstm_hidden = encoder_dim // 2  # BiLSTM doubles the hidden size
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
    
    def forward(self, x: torch.Tensor, phase_angles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, L) light curve flux tensor
            phase_angles: (B, 1, L) solar phase angle tensor
        
        Returns:
            (B, encoder_dim) feature vector
        """
        # Concatenate flux and phase angles along the channel dimension
        # Shape: (B, 2, L)
        fused_input = torch.cat([x, phase_angles], dim=1)
        
        # CNN: (B, 2, L) → (B, 128, L/8)
        features = self.cnn(fused_input)
        
        # Transpose for LSTM: (B, 128, L/8) → (B, L/8, 128)
        features = features.permute(0, 2, 1)
        
        # BiLSTM: (B, L/8, 128) → (B, L/8, encoder_dim)
        lstm_out, (h_n, _) = self.lstm(features)
        
        # Use the final hidden states from both directions
        # h_n shape: (2*num_layers, B, lstm_hidden)
        # Take last layer forward and backward
        h_forward = h_n[-2]  # (B, lstm_hidden)
        h_backward = h_n[-1]  # (B, lstm_hidden)
        combined = torch.cat([h_forward, h_backward], dim=1)  # (B, encoder_dim)
        
        # Project
        out = self.projection(combined)
        
        return out
