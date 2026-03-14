"""
SPHARM Coefficient Predictor — FC layers to predict spherical harmonic coefficients.

Takes the fused feature vector and predicts {a_lm, b_lm} coefficients
up to a specified degree L.
"""

import torch
import torch.nn as nn


class SPHARMPredictor(nn.Module):
    """Fully-connected network to predict spherical harmonic coefficients.
    
    Architecture:
        Input: (B, fused_dim) fused features
        → FC layers with residual connections
        → Output: (B, num_coefficients) SPHARM coefficients
    
    For degree L=25: num_coefficients = 2 * (L+1)^2 = 2 * 676 = 1352
    """
    
    def __init__(self, fused_dim: int = 512, max_degree: int = 25,
                 hidden_dim: int = 1024):
        super().__init__()
        self.max_degree = max_degree
        self.num_coefficients = 2 * (max_degree + 1) ** 2  # a_lm and b_lm
        
        # Main prediction network
        self.network = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            
            nn.Linear(hidden_dim, self.num_coefficients),
        )
        
        # Residual projection (fused_dim → num_coefficients)
        self.residual_proj = nn.Linear(fused_dim, self.num_coefficients)
        
        # Scale factor for output (SPHARM coefficients are typically small)
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialise weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_features: (B, fused_dim) from MultimodalFusion
        
        Returns:
            (B, num_coefficients) predicted SPHARM coefficients
        """
        # Main prediction + residual
        main = self.network(fused_features)
        residual = self.residual_proj(fused_features)
        
        coefficients = (main + residual) * self.output_scale
        
        return coefficients
    
    def get_coefficients_by_degree(self, coefficients: torch.Tensor) -> dict:
        """Split flat coefficient vector into per-degree components.
        
        Useful for regularisation and analysis.
        
        Args:
            coefficients: (B, num_coefficients) or (num_coefficients,)
        
        Returns:
            Dict mapping degree l to tensor of coefficients at that degree
        """
        if coefficients.dim() == 1:
            coefficients = coefficients.unsqueeze(0)
        
        half = self.num_coefficients // 2
        a_coeffs = coefficients[:, :half]
        b_coeffs = coefficients[:, half:]
        
        result = {}
        idx = 0
        for l in range(self.max_degree + 1):
            num_m = l + 1  # m goes from 0 to l
            result[l] = {
                'a': a_coeffs[:, idx:idx + num_m],
                'b': b_coeffs[:, idx:idx + num_m],
            }
            idx += num_m
        
        return result
