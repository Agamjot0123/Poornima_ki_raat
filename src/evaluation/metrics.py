"""
Evaluation Metrics — All 5 quantitative geometric metrics.

Hausdorff Distance, Chamfer Distance, RMSE, Volumetric IoU, and Completeness.
Operates on point clouds (numpy arrays) sampled from meshes.
Uses SciPy `cKDTree` for high-speed spatial queries, bypassing Open3D entirely.
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


def volumetric_iou(pred_mesh, gt_mesh, pitch: float = 0.05) -> float:
    """Compute volumetric IoU by voxelising both meshes topologically.
    
    IoU = V_pred ∩ V_gt / V_pred ∪ V_gt
    
    Validates physical volume enclosure; critical for non-convex features.
    Uses trimesh voxelization fill to create solid voxel representations.
    
    Args:
        pred_mesh: trimesh.Trimesh predicted model
        gt_mesh: trimesh.Trimesh ground-truth model
        pitch: Voxel side length size
    
    Returns:
        IoU in [0, 1]
    """
    try:
        # Create solid filled voxel grids
        pred_vox = pred_mesh.voxelized(pitch).fill()
        gt_vox = gt_mesh.voxelized(pitch).fill()
        
        # Get centers of filled voxels
        pred_pts = pred_vox.points
        gt_pts = gt_vox.points
        
        if len(pred_pts) == 0 or len(gt_pts) == 0:
            return 0.0
            
        # Check intersection via cross-containment
        # How many predicted volume centers exist inside the ground truth volume?
        pred_in_gt = gt_vox.is_filled(pred_pts)
        intersect_vol = pred_in_gt.sum()
        
        # Union = Vol(A) + Vol(B) - Intersection
        union_vol = len(pred_pts) + len(gt_pts) - intersect_vol
        
        if union_vol == 0:
            return 0.0
            
        return float(intersect_vol) / float(union_vol)
    except Exception as e:
        # Failsafe if meshes are radically broken or unfilled
        return 0.0


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


def compute_all_metrics(pred_mesh, gt_mesh,
                        num_points: int = 10000,
                        tolerance: float = 0.05,
                        iou_pitch: float = 0.05) -> dict:
    """Compute all 5 evaluation metrics directly from meshes.
    
    Args:
        pred_mesh: Predicted trimesh.Trimesh
        gt_mesh: Ground-truth trimesh.Trimesh
        num_points: Points to sample for surface metrics
        tolerance: Completeness spatial tolerance
        iou_pitch: Voxel pitch for volume calculation
    
    Returns:
        Dict with all metric values
    """
    # Sample points for surface-level metrics (dH, dCD, RMSE, Completeness)
    pred_points = sample_points_from_mesh(pred_mesh, num_points)
    gt_points = sample_points_from_mesh(gt_mesh, num_points)
    
    return {
        'hausdorff_distance': hausdorff_distance(pred_points, gt_points),
        'chamfer_distance': chamfer_distance(pred_points, gt_points),
        'rmse': rmse(pred_points, gt_points),
        'volumetric_iou': volumetric_iou(pred_mesh, gt_mesh, iou_pitch),
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
    # Scale meshes to unit bounding box for consistent metric calculation
    scale = np.max(gt_mesh.extents)
    if scale > 0:
        pred_mesh = pred_mesh.copy()
        gt_mesh = gt_mesh.copy()
        pred_mesh.apply_scale(1.0 / scale)
        gt_mesh.apply_scale(1.0 / scale)
        
    # Scale tolerance by the unit bound for fair completeness mapping
    # Determine the pitch for voxelization based on the new unit scale
    iou_pitch = 0.05 # 5% of the bounding box size
    
    return compute_all_metrics(pred_mesh, gt_mesh, num_points, tolerance, iou_pitch)


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
