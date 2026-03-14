"""
AsteroMesh — Full model combining all components.

Dual-stream multimodal fusion network for autonomous asteroid
3-D shape reconstruction from light curves and radar images.
"""

import torch
import torch.nn as nn
from typing import Optional

from .light_curve_encoder import LightCurveEncoder
from .radar_encoder import RadarEncoder
from .fusion import MultimodalFusion
from .spharm_predictor import SPHARMPredictor
from .mesh_decoder import MeshDecoder


class AsteroMesh(nn.Module):
    """Complete AsteroMesh network.
    
    Architecture:
        (light_curve, radar_image)
        → LightCurveEncoder → 256-dim
        → RadarEncoder → 256-dim
        → MultimodalFusion → 512-dim
        → SPHARMPredictor → 1352 coefficients
        
    Separate decode step (non-differentiable):
        coefficients → MeshDecoder → watertight mesh
    """
    
    def __init__(self, 
                 light_curve_length: int = 512,
                 radar_image_size: int = 224,
                 encoder_dim: int = 256,
                 fused_dim: int = 512,
                 max_degree: int = 25,
                 mesh_resolution: int = 100,
                 pretrained_backbone: bool = True):
        super().__init__()
        
        self.max_degree = max_degree
        self.num_coefficients = 2 * (max_degree + 1) ** 2
        
        # Encoders
        self.light_curve_encoder = LightCurveEncoder(
            input_length=light_curve_length,
            encoder_dim=encoder_dim,
        )
        self.radar_encoder = RadarEncoder(
            encoder_dim=encoder_dim,
            pretrained=pretrained_backbone,
        )
        
        # Fusion
        self.fusion = MultimodalFusion(
            encoder_dim=encoder_dim,
            fused_dim=fused_dim,
        )
        
        # SPHARM prediction
        self.spharm_predictor = SPHARMPredictor(
            fused_dim=fused_dim,
            max_degree=max_degree,
        )
        
        # Mesh decoder (non-differentiable, for inference only)
        self.mesh_decoder = MeshDecoder(
            max_degree=max_degree,
            resolution=mesh_resolution,
        )
    
    def forward(self, light_curve: torch.Tensor, 
                radar_image: torch.Tensor) -> torch.Tensor:
        """Forward pass — predict SPHARM coefficients.
        
        Args:
            light_curve: (B, 1, L) light curve tensor
            radar_image: (B, 1, H, W) radar image tensor
        
        Returns:
            (B, num_coefficients) predicted SPHARM coefficients
        """
        # Encode each modality
        lc_features = self.light_curve_encoder(light_curve)   # (B, 256)
        radar_features = self.radar_encoder(radar_image)       # (B, 256)
        
        # Fuse modalities
        fused = self.fusion(lc_features, radar_features)       # (B, 512)
        
        # Predict SPHARM coefficients
        coefficients = self.spharm_predictor(fused)            # (B, 1352)
        
        return coefficients
    
    def reconstruct(self, light_curve: torch.Tensor,
                    radar_image: torch.Tensor,
                    scale: float = 1.0) -> list:
        """Full reconstruction pipeline — predict and decode to meshes.
        
        Args:
            light_curve: (B, 1, L) light curve tensor
            radar_image: (B, 1, H, W) radar image tensor
            scale: Scale factor for output meshes
        
        Returns:
            List of trimesh.Trimesh objects
        """
        self.eval()
        with torch.no_grad():
            coefficients = self.forward(light_curve, radar_image)
        
        meshes = []
        coeffs_np = coefficients.cpu().numpy()
        for i in range(len(coeffs_np)):
            mesh = self.mesh_decoder.to_mesh(coeffs_np[i], scale=scale)
            meshes.append(mesh)
        
        return meshes
    
    def reconstruct_single(self, light_curve: Optional[torch.Tensor] = None,
                           radar_image: Optional[torch.Tensor] = None,
                           scale: float = 1.0):
        """Reconstruct from a single observation (handles missing modalities).
        
        Args:
            light_curve: (1, 1, L) or None
            radar_image: (1, 1, H, W) or None
            scale: Scale factor
        
        Returns:
            trimesh.Trimesh object
        """
        device = next(self.parameters()).device
        
        if light_curve is None:
            light_curve = torch.zeros(1, 1, 512, device=device)
        if radar_image is None:
            radar_image = torch.zeros(1, 1, 224, 224, device=device)
        
        meshes = self.reconstruct(light_curve, radar_image, scale)
        return meshes[0]
    
    def count_parameters(self) -> dict:
        """Count trainable parameters per component."""
        components = {
            'light_curve_encoder': self.light_curve_encoder,
            'radar_encoder': self.radar_encoder,
            'fusion': self.fusion,
            'spharm_predictor': self.spharm_predictor,
        }
        
        counts = {}
        total = 0
        for name, module in components.items():
            count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = count
            total += count
        counts['total'] = total
        
        return counts
