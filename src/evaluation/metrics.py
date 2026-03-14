"""
Evaluation Metrics — All 5 quantitative geometric metrics.

Hausdorff Distance, Chamfer Distance, RMSE, Volumetric IoU, and Completeness.
Operates on point clouds (numpy arrays) sampled from meshes.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Optional


def _build_kdtree(points: np.ndarray) -> cKDTree:
    """Build a KD-tree for efficient nearest-neighbour queries."""
    return cKDTree(points)


def hausdorff_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute the Hausdorff distance between two point clouds.
    
    dH(X,Y) = max(sup_x min_y d(x,y), sup_y min_x d(x,y))
    
    Worst-case maximum surface deviation.
    
    Args:
        X: (N, 3) predicted point cloud
        Y: (M, 3) ground-truth point cloud
    
    Returns:
        Hausdorff distance (scalar)
    """
    tree_X = _build_kdtree(X)
    tree_Y = _build_kdtree(Y)
    
    # Forward: max over X of min distance to Y
    dist_X_to_Y, _ = tree_Y.query(X)
    forward = np.max(dist_X_to_Y)
    
    # Backward: max over Y of min distance to X
    dist_Y_to_X, _ = tree_X.query(Y)
    backward = np.max(dist_Y_to_X)
    
    return max(forward, backward)


def chamfer_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute the Chamfer distance between two point clouds.
    
    dCD = (1/|X|) Σ_x min_y ||x-y||² + (1/|Y|) Σ_y min_x ||x-y||²
    
    Bidirectional nearest-neighbour mean; balanced accuracy + coverage.
    
    Args:
        X: (N, 3) predicted point cloud
        Y: (M, 3) ground-truth point cloud
    
    Returns:
        Chamfer distance (scalar)
    """
    tree_X = _build_kdtree(X)
    tree_Y = _build_kdtree(Y)
    
    # Forward: mean over X of squared min distance to Y
    dist_X_to_Y, _ = tree_Y.query(X)
    forward = np.mean(dist_X_to_Y ** 2)
    
    # Backward: mean over Y of squared min distance to X
    dist_Y_to_X, _ = tree_X.query(Y)
    backward = np.mean(dist_Y_to_X ** 2)
    
    return forward + backward


def rmse(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute RMSE between two point clouds.
    
    Root-mean-square of nearest-neighbour distances from X to Y.
    Validates global dimensional accuracy and scale.
    
    Args:
        X: (N, 3) predicted point cloud
        Y: (M, 3) ground-truth point cloud
    
    Returns:
        RMSE (scalar)
    """
    tree_Y = _build_kdtree(Y)
    distances, _ = tree_Y.query(X)
    return np.sqrt(np.mean(distances ** 2))


def volumetric_iou(pred_points: np.ndarray, gt_points: np.ndarray,
                   resolution: int = 64) -> float:
    """Compute volumetric IoU by voxelising both point clouds.
    
    IoU = V_pred ∩ V_gt / V_pred ∪ V_gt
    
    Validates physical volume enclosure; critical for non-convex features.
    
    Args:
        pred_points: (N, 3) predicted point cloud
        gt_points: (M, 3) ground-truth point cloud
        resolution: Voxel grid resolution
    
    Returns:
        IoU in [0, 1]
    """
    # Find bounding box encompassing both
    all_points = np.vstack([pred_points, gt_points])
    mins = all_points.min(axis=0) - 0.1
    maxs = all_points.max(axis=0) + 0.1
    
    # Voxelise
    def voxelise(points, mins, maxs, resolution):
        # Normalise to [0, resolution-1]
        normalised = (points - mins) / (maxs - mins) * (resolution - 1)
        indices = np.floor(normalised).astype(int)
        indices = np.clip(indices, 0, resolution - 1)
        
        # Create occupancy grid
        grid = np.zeros((resolution, resolution, resolution), dtype=bool)
        grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
        
        return grid
    
    pred_grid = voxelise(pred_points, mins, maxs, resolution)
    gt_grid = voxelise(gt_points, mins, maxs, resolution)
    
    intersection = np.logical_and(pred_grid, gt_grid).sum()
    union = np.logical_or(pred_grid, gt_grid).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def completeness(X: np.ndarray, Y: np.ndarray, 
                 tolerance: float = 0.05) -> float:
    """Compute completeness: fraction of ground-truth surface recovered.
    
    C = |{y ∈ Y : min_x d(x,y) < tolerance}| / |Y| × 100%
    
    Flags interpolative voids from sparse data.
    
    Args:
        X: (N, 3) predicted (model) point cloud
        Y: (M, 3) ground-truth point cloud
        tolerance: Spatial tolerance for "recovered" points
    
    Returns:
        Completeness percentage in [0, 100]
    """
    tree_X = _build_kdtree(X)
    distances, _ = tree_X.query(Y)
    
    recovered = np.sum(distances < tolerance)
    return float(recovered) / len(Y) * 100.0


def compute_all_metrics(pred_points: np.ndarray, gt_points: np.ndarray,
                        tolerance: float = 0.05,
                        iou_resolution: int = 64) -> dict:
    """Compute all 5 evaluation metrics.
    
    Args:
        pred_points: (N, 3) predicted point cloud
        gt_points: (M, 3) ground-truth point cloud
        tolerance: Completeness spatial tolerance
        iou_resolution: Voxel grid resolution for IoU
    
    Returns:
        Dict with all metric values
    """
    return {
        'hausdorff_distance': hausdorff_distance(pred_points, gt_points),
        'chamfer_distance': chamfer_distance(pred_points, gt_points),
        'rmse': rmse(pred_points, gt_points),
        'volumetric_iou': volumetric_iou(pred_points, gt_points, iou_resolution),
        'completeness': completeness(pred_points, gt_points, tolerance),
    }


def sample_points_from_mesh(mesh, num_points: int = 10000) -> np.ndarray:
    """Uniformly sample points from a mesh surface.
    
    Args:
        mesh: trimesh.Trimesh object
        num_points: Number of points to sample
    
    Returns:
        (num_points, 3) point cloud
    """
    points, _ = mesh.sample(num_points, return_index=True)
    return np.array(points)


def evaluate_meshes(pred_mesh, gt_mesh, num_points: int = 10000,
                   tolerance: float = 0.05) -> dict:
    """Evaluate predicted mesh against ground truth.
    
    Args:
        pred_mesh: Predicted trimesh.Trimesh
        gt_mesh: Ground-truth trimesh.Trimesh
        num_points: Points to sample from each mesh
        tolerance: Completeness tolerance
    
    Returns:
        Dict with all metrics
    """
    pred_points = sample_points_from_mesh(pred_mesh, num_points)
    gt_points = sample_points_from_mesh(gt_mesh, num_points)
    
    return compute_all_metrics(pred_points, gt_points, tolerance)


def main():
    """CLI for evaluating two meshes."""
    import argparse
    import trimesh
    
    parser = argparse.ArgumentParser(description='Evaluate mesh reconstruction')
    parser.add_argument('--pred', type=str, required=True, help='Predicted mesh path')
    parser.add_argument('--gt', type=str, required=True, help='Ground-truth mesh path')
    parser.add_argument('--num-points', type=int, default=10000)
    parser.add_argument('--tolerance', type=float, default=0.05)
    args = parser.parse_args()
    
    pred_mesh = trimesh.load(args.pred, force='mesh')
    gt_mesh = trimesh.load(args.gt, force='mesh')
    
    metrics = evaluate_meshes(pred_mesh, gt_mesh, args.num_points, args.tolerance)
    
    print("\n=== Evaluation Results ===")
    print(f"  Hausdorff Distance: {metrics['hausdorff_distance']:.6f}  (↓)")
    print(f"  Chamfer Distance:   {metrics['chamfer_distance']:.6f}  (↓)")
    print(f"  RMSE:               {metrics['rmse']:.6f}  (↓)")
    print(f"  Volumetric IoU:     {metrics['volumetric_iou']:.4f}  (↑)")
    print(f"  Completeness:       {metrics['completeness']:.2f}%  (↑)")


if __name__ == '__main__':
    main()
