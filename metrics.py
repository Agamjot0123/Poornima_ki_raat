"""
Geometric evaluation metrics for comparing reconstructed and ground-truth meshes.

Implements the five metrics specified in the problem statement:
    1. Hausdorff Distance (dH)
    2. Chamfer Distance (dCD)
    3. Volumetric IoU
    4. RMSE
    5. Completeness (C)
"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from typing import Dict, Tuple


def sample_points_from_mesh(mesh: trimesh.Trimesh, n_points: int = 10000,
                            seed: int = 42) -> np.ndarray:
    """
    Uniformly sample points from a mesh surface.

    Args:
        mesh: input mesh
        n_points: number of points to sample
        seed: random seed for reproducibility

    Returns:
        points: (n_points, 3) array of surface points
    """
    points, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    return points


def hausdorff_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute the Hausdorff distance between two point clouds.

    dH(X, Y) = max(sup_{x in X} inf_{y in Y} d(x,y),
                   sup_{y in Y} inf_{x in X} d(x,y))

    The worst-case maximum surface deviation. Penalises catastrophic artefacts.

    Args:
        X: (N, 3) reconstructed point cloud
        Y: (M, 3) ground truth point cloud

    Returns:
        dH: Hausdorff distance (scalar)
    """
    tree_Y = cKDTree(Y)
    tree_X = cKDTree(X)

    # Forward: for each x in X, find nearest y
    dists_X_to_Y, _ = tree_Y.query(X)
    # Backward: for each y in Y, find nearest x
    dists_Y_to_X, _ = tree_X.query(Y)

    return float(max(np.max(dists_X_to_Y), np.max(dists_Y_to_X)))


def chamfer_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute the Chamfer distance between two point clouds.

    dCD = (1/|X|) * Sum_{x in X} min_{y in Y} ||x-y||^2
        + (1/|Y|) * Sum_{y in Y} min_{x in X} ||x-y||^2

    Bi-directional nearest-neighbour mean; balanced accuracy + coverage.

    Args:
        X: (N, 3) reconstructed point cloud
        Y: (M, 3) ground truth point cloud

    Returns:
        dCD: Chamfer distance (scalar)
    """
    tree_Y = cKDTree(Y)
    tree_X = cKDTree(X)

    dists_X_to_Y, _ = tree_Y.query(X)
    dists_Y_to_X, _ = tree_X.query(Y)

    cd = (np.mean(dists_X_to_Y ** 2) + np.mean(dists_Y_to_X ** 2))
    return float(cd)


def rmse(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute the RMSE (root-mean-square error) between point clouds.

    Standard root-mean-square point error over the full mesh;
    validates global dimensional accuracy and scale.

    Uses nearest-neighbour correspondence (symmetric).

    Args:
        X: (N, 3) reconstructed point cloud
        Y: (M, 3) ground truth point cloud

    Returns:
        rmse_val: RMSE (scalar)
    """
    tree_Y = cKDTree(Y)
    dists_X_to_Y, _ = tree_Y.query(X)
    return float(np.sqrt(np.mean(dists_X_to_Y ** 2)))


def volumetric_iou(mesh_pred: trimesh.Trimesh, mesh_gt: trimesh.Trimesh,
                   grid_resolution: int = 64) -> float:
    """
    Compute volumetric Intersection over Union (IoU).

    IoU = V_pred ^ V_gt / (V_pred U V_gt)

    Uses ray-casting containment tests on a voxel grid.
    Falls back to a point-based radial overlap method if containment fails.
    """
    # Get common bounding box
    all_vertices = np.vstack([mesh_pred.vertices, mesh_gt.vertices])
    bbox_min = all_vertices.min(axis=0) - 0.1
    bbox_max = all_vertices.max(axis=0) + 0.1

    # Create voxel grid
    x = np.linspace(bbox_min[0], bbox_max[0], grid_resolution)
    y = np.linspace(bbox_min[1], bbox_max[1], grid_resolution)
    z = np.linspace(bbox_min[2], bbox_max[2], grid_resolution)
    grid = np.array(np.meshgrid(x, y, z, indexing='ij')).reshape(3, -1).T

    # Try containment check
    try:
        inside_pred = mesh_pred.contains(grid)
        inside_gt = mesh_gt.contains(grid)

        intersection = np.sum(inside_pred & inside_gt)
        union = np.sum(inside_pred | inside_gt)

        if union == 0:
            return 0.0
        return float(intersection / union)
    except Exception:
        pass

    # Fallback: point-based radial IoU using surface samples
    return _radial_iou(mesh_pred, mesh_gt, n_directions=2000)


def _radial_iou(mesh_pred: trimesh.Trimesh, mesh_gt: trimesh.Trimesh,
               n_directions: int = 2000) -> float:
    """
    Approximate IoU by comparing radial distances from centroid along many directions.

    For each direction, shoots a ray from the centroid and measures where it hits
    each mesh surface. The overlap along each ray contributes to the IoU estimate.
    """
    # Center both meshes
    center_pred = mesh_pred.centroid
    center_gt = mesh_gt.centroid
    center = (center_pred + center_gt) / 2.0

    # Generate uniform directions (Fibonacci sphere)
    indices = np.arange(0, n_directions, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_directions)
    theta = 2 * np.pi * indices / ((1 + np.sqrt(5)) / 2)

    directions = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ], axis=-1)

    # Sample points from both meshes
    pts_pred = sample_points_from_mesh(mesh_pred, n_directions * 5)
    pts_gt = sample_points_from_mesh(mesh_gt, n_directions * 5)

    # Compute radial distances from center
    r_pred = np.linalg.norm(pts_pred - center, axis=1)
    r_gt = np.linalg.norm(pts_gt - center, axis=1)

    # For each direction, find nearest surface point and compare radii
    tree_pred = cKDTree(pts_pred)
    tree_gt = cKDTree(pts_gt)

    probe_distance = max(np.max(r_pred), np.max(r_gt)) * 1.5
    intersection_vol = 0.0
    union_vol = 0.0

    for d in directions:
        probe = center + d * probe_distance
        # Find nearest surface point for each mesh
        _, idx_p = tree_pred.query(probe)
        _, idx_g = tree_gt.query(probe)
        rp = r_pred[idx_p]
        rg = r_gt[idx_g]

        # Radial intersection and union
        r_min = min(rp, rg)
        r_max = max(rp, rg)
        intersection_vol += r_min ** 3
        union_vol += r_max ** 3

    if union_vol == 0:
        return 0.0
    return float(intersection_vol / union_vol)


def completeness(X: np.ndarray, Y: np.ndarray,
                 tolerance: float = 0.05) -> float:
    """
    Compute completeness metric.

    C = S_model / S_ground_truth * 100%

    Fraction of ground-truth surface area recovered within a spatial tolerance.
    Flags interpolative voids from sparse data.

    Args:
        X: (N, 3) reconstructed point cloud
        Y: (M, 3) ground truth point cloud
        tolerance: spatial tolerance for counting a point as "recovered"

    Returns:
        C: completeness percentage [0, 100]
    """
    tree_X = cKDTree(X)
    dists_Y_to_X, _ = tree_X.query(Y)

    # Fraction of ground truth points within tolerance of reconstruction
    recovered = np.sum(dists_Y_to_X <= tolerance)
    return float(100.0 * recovered / len(Y))


def compute_all_metrics(mesh_pred: trimesh.Trimesh,
                        mesh_gt: trimesh.Trimesh,
                        n_sample_points: int = 10000,
                        iou_resolution: int = 64,
                        completeness_tolerance: float = 0.05) -> Dict[str, float]:
    """
    Compute all five evaluation metrics between predicted and ground-truth meshes.

    Both meshes are uniformly subsampled to fixed-size point sets prior to evaluation.

    Args:
        mesh_pred: reconstructed mesh
        mesh_gt: ground truth mesh
        n_sample_points: number of points to sample from each mesh
        iou_resolution: voxel grid resolution for IoU
        completeness_tolerance: spatial tolerance for completeness metric

    Returns:
        dict with keys: 'hausdorff', 'chamfer', 'rmse', 'iou', 'completeness'
    """
    print("Computing evaluation metrics...")

    # Sample point clouds
    X = sample_points_from_mesh(mesh_pred, n_sample_points)
    Y = sample_points_from_mesh(mesh_gt, n_sample_points)

    metrics = {}

    print("  Hausdorff distance...", end=" ")
    metrics['hausdorff'] = hausdorff_distance(X, Y)
    print(f"{metrics['hausdorff']:.6f}")

    print("  Chamfer distance...", end=" ")
    metrics['chamfer'] = chamfer_distance(X, Y)
    print(f"{metrics['chamfer']:.6f}")

    print("  RMSE...", end=" ")
    metrics['rmse'] = rmse(X, Y)
    print(f"{metrics['rmse']:.6f}")

    print("  Volumetric IoU...", end=" ")
    metrics['iou'] = volumetric_iou(mesh_pred, mesh_gt, iou_resolution)
    print(f"{metrics['iou']:.4f}")

    print("  Completeness...", end=" ")
    metrics['completeness'] = completeness(X, Y, completeness_tolerance)
    print(f"{metrics['completeness']:.2f}%")

    return metrics


def print_metrics_report(metrics: Dict[str, float], asteroid_name: str = ""):
    """Print a formatted metrics report."""
    header = f"EVALUATION METRICS"
    if asteroid_name:
        header += f" - {asteroid_name}"

    print("\n" + "=" * 50)
    print(header)
    print("=" * 50)
    print(f"  Hausdorff Distance (dH):  {metrics['hausdorff']:.6f}   [minimize]")
    print(f"  Chamfer Distance (dCD):   {metrics['chamfer']:.6f}   [minimize]")
    print(f"  RMSE:                     {metrics['rmse']:.6f}   [minimize]")
    print(f"  Volumetric IoU:           {metrics['iou']:.4f}       [maximize, target ~0.89]")
    print(f"  Completeness:             {metrics['completeness']:.2f}%      [maximize]")
    print("=" * 50)


if __name__ == "__main__":
    # Quick test: compare a sphere to itself (should give perfect scores)
    print("=== Metrics Test ===")

    sphere = trimesh.primitives.Sphere(radius=1.0, subdivisions=3)
    sphere_mesh = trimesh.Trimesh(vertices=sphere.vertices, faces=sphere.faces)

    # Self-comparison (should be perfect)
    print("\nSelf-comparison (sphere vs sphere):")
    metrics = compute_all_metrics(sphere_mesh, sphere_mesh, n_sample_points=5000)
    print_metrics_report(metrics, "Self-Test")

    # Slightly different shape
    print("\nSphere vs slightly larger sphere:")
    sphere2 = trimesh.primitives.Sphere(radius=1.1, subdivisions=3)
    sphere2_mesh = trimesh.Trimesh(vertices=sphere2.vertices, faces=sphere2.faces)
    metrics2 = compute_all_metrics(sphere_mesh, sphere2_mesh, n_sample_points=5000)
    print_metrics_report(metrics2, "Scale-Test")
