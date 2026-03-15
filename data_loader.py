"""
Data loading utilities for the asteroid shape reconstruction pipeline.

Handles:
    - DAMIT database files (spin.txt, shape.txt, lc.json)
    - Ground-truth OBJ/PLY meshes (Bennu, Eros)
    - Coordinate frame transformations (ecliptic <-> body frame, radar frame)
"""

import os
import json
import numpy as np
import trimesh
from typing import Dict, List, Tuple, Optional


# =============================================================================
# Data Structures
# =============================================================================

class SpinState:
    """Asteroid spin-axis parameters from DAMIT spin.txt."""

    def __init__(self, lambda_ecl: float, beta_ecl: float,
                 period: float, epoch: float, phi0: float):
        """
        Args:
            lambda_ecl: ecliptic longitude of spin axis (degrees)
            beta_ecl: ecliptic latitude of spin axis (degrees)
            period: sidereal rotation period (hours)
            epoch: reference epoch (JD)
            phi0: initial rotational phase at epoch (degrees)
        """
        self.lambda_ecl = lambda_ecl  # degrees
        self.beta_ecl = beta_ecl      # degrees
        self.period = period           # hours
        self.epoch = epoch             # Julian Date
        self.phi0 = phi0              # degrees

    def rotational_phase(self, time_jd: float) -> float:
        """
        Compute rotational phase at a given Julian Date.

        Returns:
            phase in radians [0, 2*pi)
        """
        dt_hours = (time_jd - self.epoch) * 24.0  # JD to hours
        phase = np.deg2rad(self.phi0) + 2.0 * np.pi * dt_hours / self.period
        return phase % (2.0 * np.pi)

    def rotational_phases(self, times_jd: np.ndarray) -> np.ndarray:
        """Compute rotational phases for an array of times."""
        dt_hours = (times_jd - self.epoch) * 24.0
        phases = np.deg2rad(self.phi0) + 2.0 * np.pi * dt_hours / self.period
        return phases % (2.0 * np.pi)

    def spin_axis_ecliptic(self) -> np.ndarray:
        """
        Get the spin axis unit vector in ecliptic coordinates.

        Returns:
            (3,) unit vector
        """
        lam = np.deg2rad(self.lambda_ecl)
        bet = np.deg2rad(self.beta_ecl)
        return np.array([
            np.cos(bet) * np.cos(lam),
            np.cos(bet) * np.sin(lam),
            np.sin(bet)
        ])


class LightCurve:
    """A single photometric light-curve observation."""

    def __init__(self, times: np.ndarray, fluxes: np.ndarray,
                 errors: Optional[np.ndarray] = None,
                 metadata: Optional[Dict] = None):
        """
        Args:
            times: observation times (JD or phase), shape (N,)
            fluxes: relative flux values, shape (N,)
            errors: flux uncertainties, shape (N,) or None
            metadata: optional dict with observer geometry, filter, etc.
        """
        self.times = np.asarray(times, dtype=np.float64)
        self.fluxes = np.asarray(fluxes, dtype=np.float64)
        self.errors = np.asarray(errors, dtype=np.float64) if errors is not None else None
        self.metadata = metadata or {}

    def normalize(self):
        """Normalize fluxes to unit mean."""
        mean_flux = np.mean(self.fluxes)
        if mean_flux > 0:
            self.fluxes = self.fluxes / mean_flux
            if self.errors is not None:
                self.errors = self.errors / mean_flux

    @property
    def n_points(self) -> int:
        return len(self.times)


class AsteroidData:
    """Container for all data related to one asteroid target."""

    def __init__(self, name: str):
        self.name = name
        self.spin: Optional[SpinState] = None
        self.light_curves: List[LightCurve] = []
        self.ground_truth_mesh: Optional[trimesh.Trimesh] = None
        self.damit_shape: Optional[trimesh.Trimesh] = None


# =============================================================================
# DAMIT File Parsers
# =============================================================================

def parse_damit_spin(filepath: str) -> SpinState:
    """
    Parse a DAMIT spin.txt file.

    Expected format (one line):
        lambda  beta  period  epoch  phi0
    All angles in degrees, period in hours, epoch in JD.
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not lines:
        raise ValueError(f"Empty spin file: {filepath}")

    parts = lines[0].split()
    if len(parts) < 5:
        raise ValueError(f"spin.txt must contain at least 5 values, got {len(parts)}")

    return SpinState(
        lambda_ecl=float(parts[0]),
        beta_ecl=float(parts[1]),
        period=float(parts[2]),
        epoch=float(parts[3]),
        phi0=float(parts[4])
    )


def parse_damit_shape(filepath: str) -> trimesh.Trimesh:
    """
    Parse a DAMIT shape.txt file (vertex + facet format).

    Expected format:
        n_vertices  n_facets
        x1  y1  z1
        x2  y2  z2
        ...
        v1  v2  v3  (1-indexed face vertex indices)
        ...
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    # First line: counts
    n_verts, n_faces = map(int, lines[0].split()[:2])

    # Read vertices
    vertices = np.zeros((n_verts, 3), dtype=np.float64)
    for i in range(n_verts):
        parts = lines[1 + i].split()
        vertices[i] = [float(parts[0]), float(parts[1]), float(parts[2])]

    # Read faces (convert from 1-indexed to 0-indexed)
    faces = np.zeros((n_faces, 3), dtype=np.int32)
    for i in range(n_faces):
        parts = lines[1 + n_verts + i].split()
        faces[i] = [int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()
    return mesh


def parse_damit_lightcurves(filepath: str) -> List[LightCurve]:
    """
    Parse DAMIT light-curve data from a JSON file.

    Expected format:
    {
        "lightcurves": [
            {
                "times": [...],
                "fluxes": [...],
                "errors": [...],   (optional)
                "metadata": {...}  (optional)
            },
            ...
        ]
    }

    Also supports simple CSV-like text files with columns: time  flux  [error]
    """
    if filepath.endswith('.json'):
        return _parse_lc_json(filepath)
    else:
        return _parse_lc_text(filepath)


def _parse_lc_json(filepath: str) -> List[LightCurve]:
    """Parse light curves from JSON format."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    curves = []
    lc_list = data.get('lightcurves', data.get('lcs', [data]))

    if isinstance(lc_list, dict):
        lc_list = [lc_list]

    for lc_data in lc_list:
        times = np.array(lc_data['times'], dtype=np.float64)
        fluxes = np.array(lc_data['fluxes'], dtype=np.float64)
        errors = np.array(lc_data['errors'], dtype=np.float64) if 'errors' in lc_data else None
        metadata = lc_data.get('metadata', {})

        lc = LightCurve(times=times, fluxes=fluxes, errors=errors, metadata=metadata)
        curves.append(lc)

    return curves


def _parse_lc_text(filepath: str) -> List[LightCurve]:
    """Parse light curves from whitespace-delimited text file."""
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    times, fluxes, errors = [], [], []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            times.append(float(parts[0]))
            fluxes.append(float(parts[1]))
            if len(parts) >= 3:
                errors.append(float(parts[2]))

    lc = LightCurve(
        times=np.array(times),
        fluxes=np.array(fluxes),
        errors=np.array(errors) if errors else None
    )
    return [lc]


# =============================================================================
# Ground-Truth Mesh Loaders
# =============================================================================

def load_mesh(filepath: str) -> trimesh.Trimesh:
    """
    Load a mesh from OBJ, PLY, or STL file.

    Returns:
        trimesh.Trimesh with normals fixed
    """
    mesh = trimesh.load(filepath, force='mesh')
    mesh.fix_normals()
    return mesh


def normalize_mesh(mesh: trimesh.Trimesh, target_scale: float = 1.0) -> trimesh.Trimesh:
    """
    Center and scale a mesh for comparison.

    Args:
        mesh: input mesh
        target_scale: desired half-extent (max extent in any axis)

    Returns:
        Copy of mesh, centred at origin, scaled to target_scale
    """
    mesh_copy = mesh.copy()

    # Center at centroid
    centroid = mesh_copy.centroid
    mesh_copy.vertices -= centroid

    # Scale to target size
    extent = np.max(np.abs(mesh_copy.vertices))
    if extent > 0:
        mesh_copy.vertices *= target_scale / extent

    return mesh_copy


# =============================================================================
# Coordinate Frame Transforms
# =============================================================================

def ecliptic_to_body_rotation(spin: SpinState, phase: float) -> np.ndarray:
    """
    Build rotation matrix from ecliptic to asteroid body frame.

    The body frame has z-axis along the spin axis.

    Args:
        spin: SpinState with ecliptic pole orientation
        phase: rotational phase in radians

    Returns:
        R: (3, 3) rotation matrix
    """
    lam = np.deg2rad(spin.lambda_ecl)
    bet = np.deg2rad(spin.beta_ecl)

    # Rotation to align z with spin axis
    # First rotate about z by lambda, then about new y by (pi/2 - beta)
    cos_lam, sin_lam = np.cos(lam), np.sin(lam)
    cos_bet, sin_bet = np.cos(bet), np.sin(bet)

    # Rotation matrix: ecliptic -> spin-aligned frame
    R_pole = np.array([
        [-sin_lam, cos_lam, 0],
        [-cos_lam * sin_bet, -sin_lam * sin_bet, cos_bet],
        [cos_lam * cos_bet, sin_lam * cos_bet, sin_bet]
    ])

    # Rotation about z-axis by phase
    cos_p, sin_p = np.cos(phase), np.sin(phase)
    R_phase = np.array([
        [cos_p, -sin_p, 0],
        [sin_p, cos_p, 0],
        [0, 0, 1]
    ])

    return R_phase @ R_pole


def body_to_radar_frame(x_body: np.ndarray, psi: float, delta: float) -> np.ndarray:
    """
    Transform coordinates from asteroid body frame to radar frame.

    As defined in the problem statement:
        x_r = (x*cos(psi) - y*sin(psi)) * cos(delta) + z*sin(delta)
        y_r = x*sin(psi) + y*cos(psi)
        z_r = -(x*cos(psi) - y*sin(psi)) * sin(delta) + z*cos(delta)

    Args:
        x_body: (N, 3) coordinates in body frame
        psi: rotational phase (radians)
        delta: subradar latitude (radians)

    Returns:
        x_radar: (N, 3) coordinates in radar frame
    """
    cos_p, sin_p = np.cos(psi), np.sin(psi)
    cos_d, sin_d = np.cos(delta), np.sin(delta)

    x, y, z = x_body[:, 0], x_body[:, 1], x_body[:, 2]

    xp = x * cos_p - y * sin_p  # rotated x

    x_r = xp * cos_d + z * sin_d
    y_r = x * sin_p + y * cos_p
    z_r = -xp * sin_d + z * cos_d

    return np.stack([x_r, y_r, z_r], axis=-1)


# =============================================================================
# Synthetic Data Generation (for testing)
# =============================================================================

def generate_synthetic_lightcurve(mesh: trimesh.Trimesh,
                                  spin: SpinState,
                                  n_phases: int = 100,
                                  observer_dir_ecl: np.ndarray = None,
                                  sun_dir_ecl: np.ndarray = None) -> LightCurve:
    """
    Generate a synthetic light curve from a known mesh and spin state.
    Uses Lommel-Seeliger scattering law for realism.

    Args:
        mesh: asteroid shape mesh
        spin: spin parameters
        n_phases: number of rotational phases to sample
        observer_dir_ecl: observer direction in ecliptic frame (default: +x)
        sun_dir_ecl: Sun direction in ecliptic frame (default: +x, i.e. zero phase angle)

    Returns:
        LightCurve with phases as times and synthetic fluxes
    """
    if observer_dir_ecl is None:
        observer_dir_ecl = np.array([1.0, 0.0, 0.0])
    if sun_dir_ecl is None:
        sun_dir_ecl = np.array([1.0, 0.1, 0.0])
        sun_dir_ecl /= np.linalg.norm(sun_dir_ecl)

    phases = np.linspace(0, 2 * np.pi, n_phases, endpoint=False)
    fluxes = np.zeros(n_phases)

    # Get mesh data
    face_normals = mesh.face_normals
    face_areas = mesh.area_faces
    face_centroids = mesh.triangles_center

    for i, phase in enumerate(phases):
        # Build rotation matrix for this phase
        R = ecliptic_to_body_rotation(spin, phase)

        # Transform observer and sun directions to body frame
        observer_body = R @ observer_dir_ecl
        sun_body = R @ sun_dir_ecl

        # Compute cosines
        cos_emission = np.dot(face_normals, observer_body)  # cos(e)
        cos_incidence = np.dot(face_normals, sun_body)       # cos(i)

        # Visibility: both cos_i > 0 (illuminated) and cos_e > 0 (visible)
        visible = (cos_emission > 0) & (cos_incidence > 0)

        # Lommel-Seeliger scattering: I = cos(i) / (cos(i) + cos(e))
        mu = cos_emission[visible]
        mu0 = cos_incidence[visible]
        scattering = mu0 / (mu + mu0)  # Lommel-Seeliger

        fluxes[i] = np.sum(scattering * face_areas[visible])

    # Normalize
    lc = LightCurve(times=phases, fluxes=fluxes)
    lc.normalize()
    return lc


# =============================================================================
# High-level data loading
# =============================================================================

def load_asteroid_data(data_dir: str, asteroid_name: str) -> AsteroidData:
    """
    Load all available data for one asteroid from a data directory.

    Expected directory structure:
        data_dir/
            spin.txt
            shape.txt (optional, DAMIT-derived)
            lc.json or lc.txt
            ground_truth.obj (optional)

    Returns:
        AsteroidData container
    """
    data = AsteroidData(name=asteroid_name)

    # Spin state
    spin_path = os.path.join(data_dir, 'spin.txt')
    if os.path.exists(spin_path):
        data.spin = parse_damit_spin(spin_path)
        print(f"  Loaded spin state: lambda={data.spin.lambda_ecl}, beta={data.spin.beta_ecl}, "
              f"P={data.spin.period}h")

    # Light curves
    for lc_file in ['lc.json', 'lc.txt', 'lightcurve.json', 'lightcurve.txt']:
        lc_path = os.path.join(data_dir, lc_file)
        if os.path.exists(lc_path):
            data.light_curves = parse_damit_lightcurves(lc_path)
            print(f"  Loaded {len(data.light_curves)} light curve(s) from {lc_file}")
            break

    # DAMIT shape
    shape_path = os.path.join(data_dir, 'shape.txt')
    if os.path.exists(shape_path):
        data.damit_shape = parse_damit_shape(shape_path)
        print(f"  Loaded DAMIT shape: {len(data.damit_shape.vertices)} vertices, "
              f"{len(data.damit_shape.faces)} faces")

    # Ground-truth mesh
    for gt_file in ['ground_truth.obj', 'ground_truth.ply', 'ground_truth.stl',
                    f'{asteroid_name}.obj', f'{asteroid_name}.ply']:
        gt_path = os.path.join(data_dir, gt_file)
        if os.path.exists(gt_path):
            data.ground_truth_mesh = load_mesh(gt_path)
            print(f"  Loaded ground truth: {len(data.ground_truth_mesh.vertices)} vertices")
            break

    return data


if __name__ == "__main__":
    # Quick test with synthetic data
    print("=== Data Loader Test ===")

    # Create a synthetic spin state
    spin = SpinState(lambda_ecl=0.0, beta_ecl=60.0, period=4.297, epoch=2451545.0, phi0=0.0)
    print(f"Spin axis (ecliptic): {spin.spin_axis_ecliptic()}")
    print(f"Phase at epoch: {spin.rotational_phase(spin.epoch):.4f} rad")
    print(f"Phase 1 hour later: {spin.rotational_phase(spin.epoch + 1/24):.4f} rad")

    # Test coordinate transforms
    test_pts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    radar_pts = body_to_radar_frame(test_pts, psi=0.5, delta=0.3)
    print(f"\nBody -> Radar transform test:")
    print(f"  Input: {test_pts[0]} -> Radar: {radar_pts[0]}")
    print("  (verify against PS equation manually)")
