"""
Photometric forward model for asteroid light-curve simulation.

Given a 3D shape (as SPHARM coefficients or mesh) and viewing geometry,
this module computes the predicted flux at each rotational phase.

Scattering Laws:
    - Lambert: I = cos(i) * A / pi
    - Lommel-Seeliger: I = 2 * A * cos(i) / (cos(i) + cos(e))
    - Hapke (simplified): I = w/(4*pi) * cos(i)/(cos(i)+cos(e)) * [1 + B(g)] * P(g)

This module uses PyTorch for differentiability, allowing gradient-based optimisation
of the SPHARM shape coefficients against observed light curves.
"""

import numpy as np
import torch
from typing import Tuple, Optional


def compute_facet_geometry(vertices: torch.Tensor,
                          faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute face normals, areas, and centroids from mesh vertices and faces.

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face vertex indices (int)

    Returns:
        normals: (F, 3) outward-facing unit normals
        areas: (F,) face areas
        centroids: (F, 3) face centroids
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Cross product for normal and area
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = torch.cross(edge1, edge2, dim=1)

    area_2x = torch.norm(cross, dim=1, keepdim=True)
    # Avoid division by zero for degenerate faces
    normals = cross / (area_2x + 1e-12)
    areas = area_2x.squeeze(-1) * 0.5

    # Centroids
    centroids = (v0 + v1 + v2) / 3.0

    return normals, areas, centroids


def lommel_seeliger_flux(normals: torch.Tensor,
                        areas: torch.Tensor,
                        observer_dir: torch.Tensor,
                        sun_dir: torch.Tensor) -> torch.Tensor:
    """
    Compute total scattered flux using the Lommel-Seeliger scattering law.

    I_facet = cos(i) / (cos(i) + cos(e)) * A_facet
    where i = incidence angle, e = emission angle

    This law is well-suited for low-albedo asteroids (dark, rough surfaces).

    Args:
        normals: (F, 3) face normals
        areas: (F,) face areas
        observer_dir: (3,) unit vector toward observer
        sun_dir: (3,) unit vector toward Sun

    Returns:
        total_flux: scalar, total scattered flux
    """
    # Cosines of incidence and emission angles
    cos_i = torch.sum(normals * sun_dir.unsqueeze(0), dim=1)      # (F,)
    cos_e = torch.sum(normals * observer_dir.unsqueeze(0), dim=1)  # (F,)

    # Only facets that are both illuminated and visible contribute
    # Use soft thresholding for differentiability (sigmoid instead of hard cutoff)
    visibility = torch.sigmoid(cos_i * 50.0) * torch.sigmoid(cos_e * 50.0)

    # Lommel-Seeliger scattering
    cos_i_safe = torch.clamp(cos_i, min=1e-6)
    cos_e_safe = torch.clamp(cos_e, min=1e-6)
    scattering = cos_i_safe / (cos_i_safe + cos_e_safe)

    # Total flux
    flux = torch.sum(scattering * areas * visibility)
    return flux


def lambert_flux(normals: torch.Tensor,
                 areas: torch.Tensor,
                 observer_dir: torch.Tensor,
                 sun_dir: torch.Tensor) -> torch.Tensor:
    """
    Compute total scattered flux using Lambert's cosine law.

    I_facet = cos(i) * cos(e) * A_facet

    Args:
        normals: (F, 3) face normals
        areas: (F,) face areas
        observer_dir: (3,) unit vector toward observer
        sun_dir: (3,) unit vector toward Sun

    Returns:
        total_flux: scalar
    """
    cos_i = torch.sum(normals * sun_dir.unsqueeze(0), dim=1)
    cos_e = torch.sum(normals * observer_dir.unsqueeze(0), dim=1)

    visibility = torch.sigmoid(cos_i * 50.0) * torch.sigmoid(cos_e * 50.0)

    flux = torch.sum(cos_i * cos_e * areas * visibility)
    return flux


class DifferentiablePhotometricModel:
    """
    Differentiable photometric model for light-curve simulation.

    Given SPHARM coefficients (as torch parameters), computes the predicted
    light curve as a function of rotational phase. All operations use PyTorch
    for automatic differentiation.
    """

    def __init__(self, n_theta: int = 48, n_phi: int = 96,
                 scattering_law: str = 'lommel_seeliger',
                 device: str = 'cpu'):
        """
        Args:
            n_theta: number of colatitude samples for mesh generation
            n_phi: number of longitude samples for mesh generation
            scattering_law: 'lommel_seeliger' or 'lambert'
            device: 'cpu' or 'cuda'
        """
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.device = device

        if scattering_law == 'lommel_seeliger':
            self.scatter_fn = lommel_seeliger_flux
        elif scattering_law == 'lambert':
            self.scatter_fn = lambert_flux
        else:
            raise ValueError(f"Unknown scattering law: {scattering_law}")

        # Precompute the grid and face connectivity (constant)
        self._setup_grid()

    def _setup_grid(self):
        """Precompute the (theta, phi) sampling grid and face indices."""
        # Colatitude and longitude grids
        theta_1d = np.linspace(0.01, np.pi - 0.01, self.n_theta)
        phi_1d = np.linspace(0, 2 * np.pi, self.n_phi, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta_1d, phi_1d, indexing='ij')

        self.theta_flat = torch.tensor(theta_grid.ravel(), dtype=torch.float64, device=self.device)
        self.phi_flat = torch.tensor(phi_grid.ravel(), dtype=torch.float64, device=self.device)

        # Precompute sin/cos for spherical->cartesian conversion
        self.sin_theta = torch.sin(self.theta_flat)
        self.cos_theta = torch.cos(self.theta_flat)
        self.sin_phi = torch.sin(self.phi_flat)
        self.cos_phi = torch.cos(self.phi_flat)

        # Face connectivity (same as spharm_to_mesh)
        faces = []
        for i in range(self.n_theta - 1):
            for j in range(self.n_phi):
                j_next = (j + 1) % self.n_phi
                v00 = i * self.n_phi + j
                v01 = i * self.n_phi + j_next
                v10 = (i + 1) * self.n_phi + j
                v11 = (i + 1) * self.n_phi + j_next
                faces.append([v00, v10, v01])
                faces.append([v01, v10, v11])

        self.faces = torch.tensor(faces, dtype=torch.long, device=self.device)

    def radii_to_vertices(self, r: torch.Tensor) -> torch.Tensor:
        """
        Convert radii at grid points to 3D Cartesian vertices.

        Args:
            r: (n_theta * n_phi,) surface radii

        Returns:
            vertices: (n_theta * n_phi, 3)
        """
        r_safe = torch.clamp(r, min=0.01)
        x = r_safe * self.sin_theta * self.cos_phi
        y = r_safe * self.sin_theta * self.sin_phi
        z = r_safe * self.cos_theta
        return torch.stack([x, y, z], dim=1)

    def compute_lightcurve(self, r: torch.Tensor,
                           phases: torch.Tensor,
                           observer_dir_ecl: torch.Tensor,
                           sun_dir_ecl: torch.Tensor,
                           spin_lambda: float,
                           spin_beta: float) -> torch.Tensor:
        """
        Compute the predicted light curve for given surface radii and viewing geometry.

        Args:
            r: (M,) surface radii at grid points
            phases: (N,) rotational phases in radians
            observer_dir_ecl: (3,) observer direction in ecliptic frame
            sun_dir_ecl: (3,) Sun direction in ecliptic frame
            spin_lambda: ecliptic longitude of spin axis (radians)
            spin_beta: ecliptic latitude of spin axis (radians)

        Returns:
            fluxes: (N,) predicted flux at each phase
        """
        vertices = self.radii_to_vertices(r)
        normals, areas, _ = compute_facet_geometry(vertices, self.faces)

        n_phases = len(phases)
        fluxes = torch.zeros(n_phases, dtype=torch.float64, device=self.device)

        # Precompute spin-axis rotation matrix (ecliptic -> spin-aligned)
        cos_lam = torch.cos(torch.tensor(spin_lambda, dtype=torch.float64))
        sin_lam = torch.sin(torch.tensor(spin_lambda, dtype=torch.float64))
        cos_bet = torch.cos(torch.tensor(spin_beta, dtype=torch.float64))
        sin_bet = torch.sin(torch.tensor(spin_beta, dtype=torch.float64))

        R_pole = torch.tensor([
            [-sin_lam, cos_lam, 0],
            [-cos_lam * sin_bet, -sin_lam * sin_bet, cos_bet],
            [cos_lam * cos_bet, sin_lam * cos_bet, sin_bet]
        ], dtype=torch.float64, device=self.device)

        for i in range(n_phases):
            phase = phases[i]
            cos_p = torch.cos(phase)
            sin_p = torch.sin(phase)

            R_phase = torch.tensor([
                [cos_p, -sin_p, 0],
                [sin_p, cos_p, 0],
                [0, 0, 1]
            ], dtype=torch.float64, device=self.device)

            R = R_phase @ R_pole

            # Transform observer and sun to body frame
            obs_body = R @ observer_dir_ecl
            sun_body = R @ sun_dir_ecl

            # Compute flux at this phase
            fluxes[i] = self.scatter_fn(normals, areas, obs_body, sun_body)

        # Normalize to unit mean
        mean_flux = torch.mean(fluxes)
        if mean_flux > 1e-10:
            fluxes = fluxes / mean_flux

        return fluxes


def photometric_loss(predicted: torch.Tensor, observed: torch.Tensor,
                     errors: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute the photometric chi-squared loss between predicted and observed light curves.

    Args:
        predicted: (N,) predicted normalised fluxes
        observed: (N,) observed normalised fluxes
        errors: (N,) flux uncertainties, or None for uniform weighting

    Returns:
        loss: scalar chi-squared loss
    """
    residuals = predicted - observed

    if errors is not None:
        # Weighted chi-squared
        weights = 1.0 / (errors ** 2 + 1e-10)
        loss = torch.sum(weights * residuals ** 2) / torch.sum(weights)
    else:
        # Simple MSE
        loss = torch.mean(residuals ** 2)

    return loss


if __name__ == "__main__":
    # Quick test: compute flux on a sphere
    print("=== Photometric Model Test ===")

    model = DifferentiablePhotometricModel(n_theta=24, n_phi=48, scattering_law='lommel_seeliger')

    # Uniform sphere radius
    n_pts = model.n_theta * model.n_phi
    r = torch.ones(n_pts, dtype=torch.float64) * 1.0

    phases = torch.linspace(0, 2 * np.pi, 50, dtype=torch.float64)
    observer = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    sun = torch.tensor([1.0, 0.1, 0.0], dtype=torch.float64)
    sun = sun / torch.norm(sun)

    fluxes = model.compute_lightcurve(
        r, phases, observer, sun,
        spin_lambda=0.0, spin_beta=np.deg2rad(60.0)
    )

    print(f"Sphere light curve: min={fluxes.min():.4f}, max={fluxes.max():.4f}, "
          f"mean={fluxes.mean():.4f}")
    print(f"Expected: nearly flat (sphere looks same from all angles)")
    variation = (fluxes.max() - fluxes.min()) / fluxes.mean()
    print(f"Variation: {variation:.4f} (should be small for a sphere)")
