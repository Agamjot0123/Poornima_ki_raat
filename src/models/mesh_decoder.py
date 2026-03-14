"""
Mesh Decoder — Reconstruct watertight 3D mesh from SPHARM coefficients.

Evaluates the spherical harmonic expansion r(θ,φ) on a uniform spherical grid,
then triangulates to produce a genus-0 watertight mesh.
"""

import numpy as np
import torch
import trimesh
from scipy.special import sph_harm_y
from typing import Optional


class MeshDecoder:
    """Decode SPHARM coefficients into a watertight triangular mesh.
    
    r(θ,φ) = Σ_{l=0}^{L} Σ_{m=0}^{l} [a_lm cos(mφ) + b_lm sin(mφ)] P_lm(cosθ)
    
    The mesh is guaranteed to be watertight and genus-0 by construction,
    since we evaluate on a closed spherical grid.
    """
    
    def __init__(self, max_degree: int = 25, resolution: int = 100):
        """
        Args:
            max_degree: Maximum spherical harmonic degree L
            resolution: Grid resolution (number of θ and φ samples)
        """
        self.max_degree = max_degree
        self.resolution = resolution
        self.num_coefficients = 2 * (max_degree + 1) ** 2
        
        # Precompute spherical grid
        self.theta_grid, self.phi_grid = self._create_grid(resolution)
        
        # Precompute basis functions on the grid
        self.basis_cos, self.basis_sin = self._precompute_basis()
        
        # Precompute triangulation
        self.faces = self._create_triangulation(resolution)
    
    def _create_grid(self, resolution: int):
        """Create a uniform spherical grid."""
        # Avoid poles exactly at 0 and π to prevent degenerate triangles
        theta = np.linspace(0.01, np.pi - 0.01, resolution)
        phi = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
        return theta_grid.flatten(), phi_grid.flatten()
    
    def _precompute_basis(self):
        """Precompute spherical harmonic basis functions on the grid."""
        num_points = len(self.theta_grid)
        num_coeffs = (self.max_degree + 1) ** 2
        
        basis_cos = np.zeros((num_points, num_coeffs), dtype=np.float64)
        basis_sin = np.zeros((num_points, num_coeffs), dtype=np.float64)
        
        idx = 0
        for l in range(self.max_degree + 1):
            for m in range(l + 1):
                # sph_harm_y(l, m, theta, phi) — new scipy API
                Y = sph_harm_y(l, m, self.theta_grid, self.phi_grid)
                
                if m == 0:
                    basis_cos[:, idx] = Y.real
                    basis_sin[:, idx] = 0.0
                else:
                    basis_cos[:, idx] = Y.real * np.sqrt(2)
                    basis_sin[:, idx] = -Y.imag * np.sqrt(2)
                idx += 1
        
        return basis_cos, basis_sin
    
    def _create_triangulation(self, resolution: int) -> np.ndarray:
        """Create triangulation for the spherical grid.
        
        Connects adjacent grid points to form triangles, ensuring
        a closed surface (genus-0 topology).
        """
        faces = []
        for i in range(resolution - 1):
            for j in range(resolution):
                # Current and next indices (with wrapping in φ direction)
                i0 = i * resolution + j
                i1 = i * resolution + (j + 1) % resolution
                i2 = (i + 1) * resolution + j
                i3 = (i + 1) * resolution + (j + 1) % resolution
                
                # Two triangles per quad
                faces.append([i0, i1, i2])
                faces.append([i1, i3, i2])
        
        return np.array(faces, dtype=np.int32)
    
    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        """Decode SPHARM coefficients to vertex positions.
        
        Args:
            coefficients: 1D array of length num_coefficients
                         First half: a_lm (cosine), second half: b_lm (sine)
        
        Returns:
            vertices: (N, 3) array of vertex positions
        """
        half = len(coefficients) // 2
        a_coeffs = coefficients[:half]
        b_coeffs = coefficients[half:]
        
        # Reconstruct radial distances
        r = self.basis_cos @ a_coeffs + self.basis_sin @ b_coeffs
        
        # Ensure positive radii (physical constraint)
        r = np.abs(r) + 0.01
        
        # Convert spherical to Cartesian
        x = r * np.sin(self.theta_grid) * np.cos(self.phi_grid)
        y = r * np.sin(self.theta_grid) * np.sin(self.phi_grid)
        z = r * np.cos(self.theta_grid)
        
        vertices = np.stack([x, y, z], axis=1).astype(np.float64)
        
        return vertices
    
    def to_mesh(self, coefficients, scale: float = 1.0) -> trimesh.Trimesh:
        """Convert SPHARM coefficients to a trimesh object.
        
        Args:
            coefficients: Tensor or ndarray of SPHARM coefficients
            scale: Scale factor for the mesh
        
        Returns:
            trimesh.Trimesh — watertight mesh
        """
        if isinstance(coefficients, torch.Tensor):
            coefficients = coefficients.detach().cpu().numpy()
        
        vertices = self.decode(coefficients)
        vertices *= scale
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces, process=True)
        
        return mesh
    
    def to_obj(self, coefficients, path: str, scale: float = 1.0) -> None:
        """Export SPHARM coefficients directly to .obj file.
        
        Args:
            coefficients: Tensor or ndarray of SPHARM coefficients
            path: Output file path
            scale: Scale factor
        """
        mesh = self.to_mesh(coefficients, scale)
        mesh.export(path, file_type='obj')
    
    def batch_decode(self, coefficients_batch: torch.Tensor) -> list:
        """Decode a batch of coefficient vectors to meshes.
        
        Args:
            coefficients_batch: (B, num_coefficients) tensor
        
        Returns:
            List of trimesh.Trimesh objects
        """
        coeffs_np = coefficients_batch.detach().cpu().numpy()
        meshes = []
        for i in range(len(coeffs_np)):
            mesh = self.to_mesh(coeffs_np[i])
            meshes.append(mesh)
        return meshes
