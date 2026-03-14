"""
Multimodal Fusion — Attention-based fusion of light-curve and radar features.

Cross-modal attention learns which modality to weight for different surface
characteristics. Includes a gating mechanism for single-modality degradation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Cross-modal attention module.
    
    Each modality attends to the other to learn complementary features.
    """
    
    def __init__(self, dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, dim) — the modality being enriched
            key_value: (B, dim) — the modality providing context
        
        Returns:
            (B, dim) — attended features
        """
        B = query.shape[0]
        
        # Add sequence dimension: (B, dim) → (B, 1, dim)
        query = query.unsqueeze(1)
        key_value = key_value.unsqueeze(1)
        
        # Project Q, K, V
        Q = self.q_proj(query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_value).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention: (B, heads, 1, head_dim) × (B, heads, head_dim, 1) → (B, heads, 1, 1)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = (attn @ V).transpose(1, 2).contiguous().view(B, 1, -1)
        out = self.out_proj(out).squeeze(1)  # (B, dim)
        
        return out


class ModalityGate(nn.Module):
    """Gating mechanism for graceful single-modality degradation.
    
    Learns to detect when a modality is missing (all zeros) and
    adjusts the fusion accordingly.
    """
    
    def __init__(self, dim: int = 256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-sample gate value in [0, 1]."""
        return self.gate(x)  # (B, 1)


class MultimodalFusion(nn.Module):
    """Attention-based multimodal fusion module.
    
    Architecture:
        Two encoder features → Cross-attention (bidirectional)
        → Gated combination → Concatenation → MLP → fused_dim output
    
    Handles missing modalities via learned gating.
    """
    
    def __init__(self, encoder_dim: int = 256, fused_dim: int = 512, 
                 num_heads: int = 4):
        super().__init__()
        
        # Cross-attention: light curve attends to radar (and vice versa)
        self.lc_attends_radar = CrossAttention(encoder_dim, num_heads)
        self.radar_attends_lc = CrossAttention(encoder_dim, num_heads)
        
        # Layer norms for residual connections
        self.ln_lc = nn.LayerNorm(encoder_dim)
        self.ln_radar = nn.LayerNorm(encoder_dim)
        
        # Modality gates
        self.lc_gate = ModalityGate(encoder_dim)
        self.radar_gate = ModalityGate(encoder_dim)
        
        # Fusion MLP: 2×encoder_dim → fused_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(encoder_dim * 2, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, lc_features: torch.Tensor, 
                radar_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lc_features: (B, encoder_dim) from LightCurveEncoder
            radar_features: (B, encoder_dim) from RadarEncoder
        
        Returns:
            (B, fused_dim) fused feature vector
        """
        # Cross-attention with residual connections
        lc_attended = self.ln_lc(lc_features + self.lc_attends_radar(lc_features, radar_features))
        radar_attended = self.ln_radar(radar_features + self.radar_attends_lc(radar_features, lc_features))
        
        # Gated combination (handles missing modalities)
        lc_gate = self.lc_gate(lc_features)      # (B, 1)
        radar_gate = self.radar_gate(radar_features)  # (B, 1)
        
        lc_gated = lc_attended * lc_gate
        radar_gated = radar_attended * radar_gate
        
        # Concatenate and fuse
        combined = torch.cat([lc_gated, radar_gated], dim=1)  # (B, 2*encoder_dim)
        fused = self.fusion_mlp(combined)  # (B, fused_dim)
        
        return fused
