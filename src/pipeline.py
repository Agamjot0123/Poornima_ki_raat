"""
Pipeline — End-to-end inference for asteroid shape reconstruction.

CLI entry point: takes config + input data → outputs watertight .obj mesh.
Fully automated, no manual parameter tuning.
"""

import torch
import yaml
import argparse
from pathlib import Path
from typing import Optional

from .models.asteromesh import AsteroMesh
from .models.mesh_decoder import MeshDecoder
from .data.light_curve_loader import load_light_curve
from .data.radar_loader import load_radar_image
from .evaluation.metrics import evaluate_meshes, compute_all_metrics, sample_points_from_mesh


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(config: dict, checkpoint_path: str, 
               device: torch.device) -> AsteroMesh:
    """Load trained AsteroMesh model from checkpoint."""
    model_cfg = config.get('model', {})
    
    model = AsteroMesh(
        light_curve_length=model_cfg.get('light_curve_length', 512),
        radar_image_size=model_cfg.get('radar_image_size', 224),
        encoder_dim=model_cfg.get('encoder_dim', 256),
        fused_dim=model_cfg.get('fused_dim', 512),
        max_degree=model_cfg.get('spharm_degree', 25),
        mesh_resolution=model_cfg.get('mesh_resolution', 100),
        pretrained_backbone=False,  # Not needed for inference
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    return model


def run_inference(model: AsteroMesh,
                  light_curve_path: Optional[str] = None,
                  radar_image_path: Optional[str] = None,
                  period: Optional[float] = None,
                  output_path: str = 'output.obj',
                  scale: float = 1.0,
                  device: torch.device = torch.device('cpu')):
    """Run single-sample inference.
    
    Args:
        model: Trained AsteroMesh model
        light_curve_path: Path to light curve file (or None)
        radar_image_path: Path to radar image file (or None)
        period: Rotational period for phase-folding
        output_path: Output mesh file path
        scale: Scale factor for the mesh
        device: Compute device
    
    Returns:
        trimesh.Trimesh — reconstructed mesh
    """
    # Load inputs
    if light_curve_path:
        lc = load_light_curve(light_curve_path, period=period).unsqueeze(0).to(device)
        print(f"Loaded light curve: {light_curve_path}")
    else:
        lc = torch.zeros(1, 1, 512, device=device)
        print("No light curve provided — using zeros")
    
    if radar_image_path:
        radar = load_radar_image(radar_image_path).unsqueeze(0).to(device)
        print(f"Loaded radar image: {radar_image_path}")
    else:
        radar = torch.zeros(1, 1, 224, 224, device=device)
        print("No radar image provided — using zeros")
    
    # Reconstruct
    print("Reconstructing 3-D shape...")
    mesh = model.reconstruct_single(lc, radar, scale=scale)
    
    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_type = output_path.suffix.lstrip('.')
    if file_type not in ('obj', 'stl', 'ply'):
        file_type = 'obj'
    
    mesh.export(str(output_path), file_type=file_type)
    
    print(f"\n=== Reconstruction Complete ===")
    print(f"  Output: {output_path}")
    print(f"  Vertices: {len(mesh.vertices):,}")
    print(f"  Faces: {len(mesh.faces):,}")
    print(f"  Watertight: {mesh.is_watertight}")
    print(f"  Volume: {mesh.volume:.6f}")
    
    return mesh


def run_evaluation(pred_mesh, gt_mesh_path: str, 
                   num_points: int = 10000) -> dict:
    """Evaluate reconstruction against ground truth.
    
    Args:
        pred_mesh: Predicted trimesh.Trimesh
        gt_mesh_path: Path to ground-truth mesh
        num_points: Points to sample for evaluation
    
    Returns:
        Dict with all metrics
    """
    import trimesh
    gt_mesh = trimesh.load(gt_mesh_path, force='mesh')
    
    metrics = evaluate_meshes(pred_mesh, gt_mesh, num_points)
    
    print(f"\n=== Evaluation vs Ground Truth ===")
    print(f"  Hausdorff Distance: {metrics['hausdorff_distance']:.6f}  (↓)")
    print(f"  Chamfer Distance:   {metrics['chamfer_distance']:.6f}  (↓)")
    print(f"  RMSE:               {metrics['rmse']:.6f}  (↓)")
    print(f"  Volumetric IoU:     {metrics['volumetric_iou']:.4f}  (↑)")
    print(f"  Completeness:       {metrics['completeness']:.2f}%  (↑)")
    
    return metrics


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='AsteroMesh — Autonomous Asteroid Shape Reconstruction'
    )
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Config file path')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Model checkpoint path')
    parser.add_argument('--light-curve', type=str, default=None,
                       help='Input light curve file')
    parser.add_argument('--radar', type=str, default=None,
                       help='Input radar image file')
    parser.add_argument('--period', type=float, default=None,
                       help='Rotational period (hours)')
    parser.add_argument('--output', type=str, default='outputs/meshes/reconstruction.obj',
                       help='Output mesh path')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Mesh scale factor')
    parser.add_argument('--gt', type=str, default=None,
                       help='Ground-truth mesh for evaluation')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    device = torch.device(args.device if args.device else 
                         ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Load model
    checkpoint = args.checkpoint or config.get('inference', {}).get(
        'model_checkpoint', 'checkpoints/best_model.pth'
    )
    model = load_model(config, checkpoint, device)
    
    # Run inference
    mesh = run_inference(
        model,
        light_curve_path=args.light_curve,
        radar_image_path=args.radar,
        period=args.period,
        output_path=args.output,
        scale=args.scale,
        device=device,
    )
    
    # Optional evaluation
    if args.gt:
        run_evaluation(mesh, args.gt)


if __name__ == '__main__':
    main()
