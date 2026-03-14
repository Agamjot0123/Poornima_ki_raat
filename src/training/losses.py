"""
Loss Functions — Composite losses for SPHARM coefficient prediction.

Includes Chamfer loss (on decoded meshes), MSE on coefficients,
and SPHARM regularisation to penalise high-frequency noise.
"""

import torch
import torch.nn as nn
import numpy as np


class ChamferLoss(nn.Module):
    """Chamfer Distance loss between two point sets.
    
    Computes bidirectional nearest-neighbour L2 distance.
    Differentiable and suitable for training.
    """
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, N, 3) predicted point cloud
            target: (B, M, 3) target point cloud
        
        Returns:
            Scalar loss
        """
        # Pairwise distances: (B, N, M)
        diff = pred.unsqueeze(2) - target.unsqueeze(1)  # (B, N, M, 3)
        dist = torch.sum(diff ** 2, dim=-1)  # (B, N, M)
        
        # Forward: for each predicted point, find nearest target
        min_pred_to_target = torch.min(dist, dim=2)[0]  # (B, N)
        
        # Backward: for each target point, find nearest predicted
        min_target_to_pred = torch.min(dist, dim=1)[0]  # (B, M)
        
        # Mean bidirectional distance
        loss = min_pred_to_target.mean() + min_target_to_pred.mean()
        
        return loss


class SPHARMRegularisationLoss(nn.Module):
    """Regularisation loss penalising high-frequency SPHARM coefficients.
    
    Encourages smooth surfaces by applying increasing weight to
    higher-degree coefficients: weight_l = l^2
    """
    
    def __init__(self, max_degree: int = 25):
        super().__init__()
        self.max_degree = max_degree
        
        # The model outputs 2*(L+1)^2 coefficients total
        # First half = a_lm, second half = b_lm
        # We assign a degree-based weight to each coefficient position
        num_coeffs_per_half = (max_degree + 1) ** 2
        
        weights = []
        # Map flat index back to degree: index i corresponds to degree floor(sqrt(i))
        for i in range(num_coeffs_per_half):
            l = int(np.sqrt(i))
            weights.append(float(l ** 2))
        
        # Duplicate for a_lm and b_lm
        weights = np.array(weights + weights, dtype=np.float32)
        # Normalise
        weights = weights / (weights.max() + 1e-10)
        
        self.register_buffer('weights', torch.tensor(weights))
    
    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coefficients: (B, num_coefficients) predicted coefficients
        
        Returns:
            Scalar regularisation loss
        """
        # Weighted L2 on coefficients
        weighted = coefficients ** 2 * self.weights.unsqueeze(0)
        return weighted.mean()


class CoefficientMSELoss(nn.Module):
    """MSE loss directly on SPHARM coefficients."""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, num_coefficients) predicted
            target: (B, num_coefficients) ground truth
        
        Returns:
            Scalar MSE loss
        """
        return nn.functional.mse_loss(pred, target)


class CompositeLoss(nn.Module):
    """Composite loss combining multiple objectives.
    
    L = w_mse * L_mse + w_chamfer * L_chamfer + w_reg * L_spharm_reg
    
    By default operates on coefficient space (MSE + regularisation).
    Chamfer loss can be enabled but requires mesh decoding during training.
    """
    
    def __init__(self, 
                 mse_weight: float = 1.0,
                 chamfer_weight: float = 0.0,
                 spharm_reg_weight: float = 0.01,
                 max_degree: int = 25):
        super().__init__()
        
        self.mse_weight = mse_weight
        self.chamfer_weight = chamfer_weight
        self.spharm_reg_weight = spharm_reg_weight
        
        self.mse_loss = CoefficientMSELoss()
        self.spharm_reg = SPHARMRegularisationLoss(max_degree)
        
        if chamfer_weight > 0:
            self.chamfer_loss = ChamferLoss()
    
    def forward(self, pred_coefficients: torch.Tensor, 
                target_coefficients: torch.Tensor,
                pred_points: torch.Tensor = None,
                target_points: torch.Tensor = None) -> dict:
        """
        Args:
            pred_coefficients: (B, num_coefficients) predicted
            target_coefficients: (B, num_coefficients) ground truth
            pred_points: Optional (B, N, 3) predicted point cloud for Chamfer
            target_points: Optional (B, M, 3) target point cloud for Chamfer
        
        Returns:
            Dict with 'total', 'mse', 'chamfer', 'regularisation' losses
        """
        losses = {}
        
        # MSE on coefficients
        mse = self.mse_loss(pred_coefficients, target_coefficients)
        losses['mse'] = mse
        
        # SPHARM regularisation
        reg = self.spharm_reg(pred_coefficients)
        losses['regularisation'] = reg
        
        # Total
        total = self.mse_weight * mse + self.spharm_reg_weight * reg
        
        # Optional Chamfer loss
        if self.chamfer_weight > 0 and pred_points is not None and target_points is not None:
            chamfer = self.chamfer_loss(pred_points, target_points)
            losses['chamfer'] = chamfer
            total += self.chamfer_weight * chamfer
        
        losses['total'] = total
        
        return losses
