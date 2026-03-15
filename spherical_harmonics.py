"""
Spherical Harmonics (SPHARM) surface representation for asteroid shape reconstruction.

The asteroid surface is parameterised as:
    r(theta, phi) = Sum_{l=0}^{N} Sum_{m=0}^{l}
        [a_lm * cos(m*phi) + b_lm * sin(m*phi)] * P_lm(cos(theta))

where:
    - (theta, phi) are colatitude and longitude on the unit sphere
    - P_lm are associated Legendre polynomials
    - {a_lm, b_lm} are the shape coefficients to be optimised
"""

import numpy as np
import torch
from scipy.special import sph_harm_y
import trimesh


def count_coefficients(max_order: int) -> int:
    """
    Count total number of SPHARM coefficients for a given maximum order N.
    Total = (N+1)^2
    For each l from 0..N, there are (2l+1) terms.
    We use real-valued SPHARM: for each (l, m), one cosine coeff and one sine coeff,
    except m=0 only has cosine.
    Total real coefficients = (N+1)^2
    """
    return (max_order + 1) ** 2


def init_coefficients_sphere(max_order: int, radius: float = 1.0) -> np.ndarray:
    """
    Initialize SPHARM coefficients for a sphere of given radius.
    Only the (l=0, m=0) coefficient is non-zero.
    
    The Y_0^0 = 1 / (2*sqrt(pi)), so to get radius R:
        a_00 = R * 2 * sqrt(pi)

    Returns:
        coeffs: 1D array of length (N+1)^2 representing [a_00, a_10, a_11, b_11, a_20, ...]
    """
    n_coeffs = count_coefficients(max_order)
    coeffs = np.zeros(n_coeffs, dtype=np.float64)
    # a_00 coefficient: sphere of given radius
    # Real Y_0^0 = 1/(2*sqrt(pi)), so r = a_00 * Y_0^0 => a_00 = R / Y_0^0 = R * 2*sqrt(pi)
    coeffs[0] = radius * 2.0 * np.sqrt(np.pi)
    return coeffs


def init_coefficients_ellipsoid(max_order: int, a: float, b: float, c: float) -> np.ndarray:
    """
    Initialize SPHARM coefficients for an approximate ellipsoid.
    Uses l=0 for mean radius and l=2 terms for ellipticity.

    Args:
        a, b, c: semi-axes of the ellipsoid (x, y, z)
    """
    mean_r = (a + b + c) / 3.0
    coeffs = init_coefficients_sphere(max_order, radius=mean_r)

    if max_order >= 2:
        # Add l=2, m=0 for polar flattening (c vs a,b)
        # Y_2^0 = sqrt(5/(16*pi)) * (3*cos^2(theta) - 1)
        # This stretches/compresses along z-axis
        idx_20 = _coeff_index(2, 0, is_sin=False)
        coeffs[idx_20] = (c - mean_r) * np.sqrt(16.0 * np.pi / 5.0) * 0.5

        # Add l=2, m=2 for equatorial ellipticity (a vs b)
        idx_22 = _coeff_index(2, 2, is_sin=False)
        coeffs[idx_22] = (a - b) * np.sqrt(16.0 * np.pi / 15.0) * 0.25

    return coeffs


def _coeff_index(l: int, m: int, is_sin: bool = False) -> int:
    """
    Map (l, m, is_sin) to a linear index in the coefficient array.
    
    Layout for each l:
        m=0: 1 coeff (a_l0) — no sine term for m=0
        m=1..l: 2 coeffs each (a_lm, b_lm)
    
    Index for degree l starts at l^2.
    Within degree l:
        m=0 -> offset 0  (a_l0)
        m>0 -> offset 2*m - 1 (a_lm), 2*m (b_lm)
    """
    base = l * l
    if m == 0:
        return base
    offset = 2 * m - 1 if not is_sin else 2 * m
    return base + offset


def evaluate_surface(coeffs: np.ndarray, max_order: int,
                     theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Evaluate the SPHARM surface radius at given (theta, phi) directions.

    Args:
        coeffs: 1D coefficient array of length (N+1)^2
        max_order: maximum SPHARM order N
        theta: colatitude angles, shape (M,)
        phi: longitude angles, shape (M,)

    Returns:
        r: surface radii at each direction, shape (M,)
    """
    r = np.zeros_like(theta, dtype=np.float64)

    for l in range(max_order + 1):
        for m in range(l + 1):
            # Compute real spherical harmonic basis
            # scipy sph_harm returns complex Y_l^m; we need real form
            if m == 0:
                # Y_l^0 is already real
                Y = np.real(sph_harm_y(l, 0, theta, phi))
                idx = _coeff_index(l, 0, is_sin=False)
                r += coeffs[idx] * Y
            else:
                # Real SPHARM:
                #   Y_lm^c = sqrt(2) * Re(Y_l^m) = sqrt(2) * |Y_l^m| * cos(m*phi) * P_l^m(cos theta)
                #   Y_lm^s = sqrt(2) * Im(Y_l^m) = sqrt(2) * |Y_l^m| * sin(m*phi) * P_l^m(cos theta)
                # But scipy sph_harm already includes the phase factor, so:
                Y_complex = sph_harm_y(l, m, theta, phi)
                Y_cos = np.sqrt(2.0) * np.real(Y_complex)  # cosine part
                Y_sin = -np.sqrt(2.0) * np.imag(Y_complex)  # sine part (note sign convention)

                idx_a = _coeff_index(l, m, is_sin=False)
                idx_b = _coeff_index(l, m, is_sin=True)
                r += coeffs[idx_a] * Y_cos + coeffs[idx_b] * Y_sin

    return r


def evaluate_surface_torch(coeffs: torch.Tensor, max_order: int,
                           Y_basis: torch.Tensor) -> torch.Tensor:
    """
    Evaluate SPHARM surface using precomputed basis (for GPU optimization).

    Args:
        coeffs: shape ((N+1)^2,) — learnable parameters
        Y_basis: shape (M, (N+1)^2) — precomputed basis values at sample points

    Returns:
        r: shape (M,) — surface radii
    """
    return Y_basis @ coeffs


def precompute_basis(max_order: int, theta: np.ndarray,
                     phi: np.ndarray) -> np.ndarray:
    """
    Precompute the real SPHARM basis matrix for given sample directions.

    Args:
        max_order: maximum SPHARM order N
        theta: colatitude angles, shape (M,)
        phi: longitude angles, shape (M,)

    Returns:
        Y_basis: shape (M, (N+1)^2) basis matrix
    """
    n_coeffs = count_coefficients(max_order)
    M = len(theta)
    Y_basis = np.zeros((M, n_coeffs), dtype=np.float64)

    for l in range(max_order + 1):
        for m in range(l + 1):
            if m == 0:
                Y = np.real(sph_harm_y(l, 0, theta, phi))
                idx = _coeff_index(l, 0, is_sin=False)
                Y_basis[:, idx] = Y
            else:
                Y_complex = sph_harm_y(l, m, theta, phi)
                Y_cos = np.sqrt(2.0) * np.real(Y_complex)
                Y_sin = -np.sqrt(2.0) * np.imag(Y_complex)

                idx_a = _coeff_index(l, m, is_sin=False)
                idx_b = _coeff_index(l, m, is_sin=True)
                Y_basis[:, idx_a] = Y_cos
                Y_basis[:, idx_b] = Y_sin

    return Y_basis


def generate_sampling_grid(n_theta: int = 64, n_phi: int = 128):
    """
    Generate a regular (theta, phi) sampling grid on the unit sphere.

    Returns:
        theta: shape (n_theta * n_phi,)
        phi: shape (n_theta * n_phi,)
        theta_grid: shape (n_theta, n_phi) for mesh generation
        phi_grid: shape (n_theta, n_phi) for mesh generation
    """
    # Avoid exact poles (theta=0, pi) for numerical stability
    theta_1d = np.linspace(0.01, np.pi - 0.01, n_theta)
    phi_1d = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta_1d, phi_1d, indexing='ij')

    return (theta_grid.ravel(), phi_grid.ravel(),
            theta_grid, phi_grid)


def spharm_to_mesh(coeffs: np.ndarray, max_order: int,
                   n_theta: int = 64, n_phi: int = 128) -> trimesh.Trimesh:
    """
    Convert SPHARM coefficients to a triangulated mesh.

    Args:
        coeffs: 1D coefficient array
        max_order: integer SPHARM order
        n_theta, n_phi: resolution of the sampling grid

    Returns:
        mesh: trimesh.Trimesh object (watertight, genus-0)
    """
    theta_flat, phi_flat, theta_grid, phi_grid = generate_sampling_grid(n_theta, n_phi)

    # Evaluate radii
    r = evaluate_surface(coeffs, max_order, theta_flat, phi_flat)
    r_grid = r.reshape(n_theta, n_phi)

    # Ensure positive radii (physical constraint)
    r_grid = np.maximum(r_grid, 0.01)

    # Convert spherical to Cartesian
    x = r_grid * np.sin(theta_grid) * np.cos(phi_grid)
    y = r_grid * np.sin(theta_grid) * np.sin(phi_grid)
    z = r_grid * np.cos(theta_grid)

    # Build grid vertices
    grid_verts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)

    # Add pole vertices for watertight closure
    # North pole (theta=0): evaluate radius at exact pole
    r_north = evaluate_surface(coeffs, max_order,
                                np.array([1e-6]), np.array([0.0]))[0]
    r_north = max(r_north, 0.01)
    north_pole = np.array([[0.0, 0.0, r_north]])

    # South pole (theta=pi): evaluate radius at exact pole
    r_south = evaluate_surface(coeffs, max_order,
                                np.array([np.pi - 1e-6]), np.array([0.0]))[0]
    r_south = max(r_south, 0.01)
    south_pole = np.array([[0.0, 0.0, -r_south]])

    # Combine: grid vertices, then north pole, then south pole
    vertices = np.vstack([grid_verts, north_pole, south_pole])
    n_grid = len(grid_verts)
    north_idx = n_grid
    south_idx = n_grid + 1

    # Build faces for the grid body
    faces = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            j_next = (j + 1) % n_phi
            v00 = i * n_phi + j
            v01 = i * n_phi + j_next
            v10 = (i + 1) * n_phi + j
            v11 = (i + 1) * n_phi + j_next

            faces.append([v00, v10, v01])
            faces.append([v01, v10, v11])

    # North pole fan: connect pole to first row of grid
    for j in range(n_phi):
        j_next = (j + 1) % n_phi
        faces.append([north_idx, j, j_next])

    # South pole fan: connect pole to last row of grid
    last_row_start = (n_theta - 1) * n_phi
    for j in range(n_phi):
        j_next = (j + 1) % n_phi
        faces.append([south_idx, last_row_start + j_next, last_row_start + j])

    faces = np.array(faces, dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Fix normals to point outward
    mesh.fix_normals()

    return mesh


def spharm_to_point_cloud(coeffs: np.ndarray, max_order: int,
                          n_points: int = 10000) -> np.ndarray:
    """
    Convert SPHARM coefficients to a point cloud via uniform sphere sampling.

    Returns:
        points: shape (n_points, 3)
    """
    # Fibonacci sphere sampling for uniform distribution
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)  # colatitude (theta convention)
    theta_samp = 2 * np.pi * indices / ((1 + np.sqrt(5)) / 2)  # longitude (phi convention)

    # Note: our convention is theta=colatitude, phi=longitude
    # Fibonacci gives phi_fib = colatitude, theta_samp = longitude
    r = evaluate_surface(coeffs, max_order, phi, theta_samp % (2 * np.pi))
    r = np.maximum(r, 0.01)

    x = r * np.sin(phi) * np.cos(theta_samp)
    y = r * np.sin(phi) * np.sin(theta_samp)
    z = r * np.cos(phi)

    return np.stack([x, y, z], axis=-1)


if __name__ == "__main__":
    # Quick sanity check: create a sphere and verify
    print("=== SPHARM Sanity Check ===")
    N = 4
    coeffs = init_coefficients_sphere(N, radius=1.0)
    print(f"Sphere coefficients (order {N}): {len(coeffs)} coefficients")
    print(f"  a_00 = {coeffs[0]:.4f} (expected: {2*np.sqrt(np.pi):.4f})")

    # Evaluate at some points
    theta_test = np.array([0.5, 1.0, 1.5, 2.0])
    phi_test = np.array([0.0, 1.0, 2.0, 3.0])
    r_test = evaluate_surface(coeffs, N, theta_test, phi_test)
    print(f"  Radii at test points: {r_test}")
    print(f"  Expected: all ~1.0")

    # Generate mesh
    mesh = spharm_to_mesh(coeffs, N, n_theta=32, n_phi=64)
    print(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"  Watertight: {mesh.is_watertight}")
    mesh.export("test_sphere.obj")
    print("  Exported test_sphere.obj")

    # Test ellipsoid
    coeffs_ell = init_coefficients_ellipsoid(N, a=2.0, b=1.5, c=1.0)
    mesh_ell = spharm_to_mesh(coeffs_ell, N, n_theta=32, n_phi=64)
    print(f"\nEllipsoid mesh: {len(mesh_ell.vertices)} vertices, {len(mesh_ell.faces)} faces")
    print(f"  Watertight: {mesh_ell.is_watertight}")
    mesh_ell.export("test_ellipsoid.obj")
    print("  Exported test_ellipsoid.obj")
