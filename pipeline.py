"""
End-to-end pipeline for autonomous 3D asteroid shape reconstruction.

Usage:
    # Run with synthetic test data (no downloads needed):
    python pipeline.py --mode synthetic

    # Run with real DAMIT data:
    python pipeline.py --mode damit --data_dir data/bennu

    # Run with ground-truth comparison:
    python pipeline.py --mode evaluate --data_dir data/bennu --gt_mesh data/bennu/ground_truth.obj
"""

import os
import sys
import json
import argparse
import numpy as np
import trimesh
from typing import Optional

from spherical_harmonics import (
    init_coefficients_sphere, init_coefficients_ellipsoid,
    spharm_to_mesh, count_coefficients
)
from data_loader import (
    SpinState, LightCurve, AsteroidData,
    load_asteroid_data, generate_synthetic_lightcurve, normalize_mesh
)
from shape_optimizer import ShapeOptimizer
from photometric_model import DifferentiablePhotometricModel
from metrics import compute_all_metrics, print_metrics_report
from visualize import (
    plot_mesh_comparison, plot_lightcurve_fit,
    plot_loss_history, plot_metrics_bar, plot_multi_view
)


def run_synthetic_test(output_dir: str = "output/synthetic",
                       config: Optional[dict] = None):
    """
    Full pipeline test using a synthetic asteroid shape.

    This creates a known ellipsoidal shape, generates a synthetic light curve,
    then attempts to reconstruct the shape from scratch.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("SYNTHETIC TEST: Reconstruct ellipsoid from light curve")
    print("=" * 60)

    # 1. Create ground-truth shape (elongated asteroid-like ellipsoid)
    print("\n[1/6] Creating ground-truth shape...")
    gt_axes = (1.5, 1.0, 0.8)  # semi-axes
    gt_order = 8
    gt_coeffs = init_coefficients_ellipsoid(gt_order, *gt_axes)
    gt_mesh = spharm_to_mesh(gt_coeffs, gt_order, n_theta=64, n_phi=128)
    gt_mesh.export(os.path.join(output_dir, "ground_truth.obj"))
    print(f"  Shape: {gt_axes} ellipsoid")
    print(f"  Mesh: {len(gt_mesh.vertices)} verts, {len(gt_mesh.faces)} faces")
    print(f"  Watertight: {gt_mesh.is_watertight}")

    # 2. Create spin state
    print("\n[2/6] Setting up spin state...")
    spin = SpinState(
        lambda_ecl=45.0,   # spin axis: ecliptic longitude
        beta_ecl=30.0,     # spin axis: ecliptic latitude
        period=6.0,        # rotation period (hours)
        epoch=2451545.0,   # J2000.0
        phi0=0.0           # initial phase
    )
    print(f"  lambda={spin.lambda_ecl}, beta={spin.beta_ecl}, P={spin.period}h")

    # 3. Generate synthetic light curve
    print("\n[3/6] Generating synthetic light curve...")
    observer_dir = np.array([1.0, 0.0, 0.0])
    sun_dir = np.array([1.0, 0.1, 0.0])
    sun_dir /= np.linalg.norm(sun_dir)

    lc = generate_synthetic_lightcurve(
        gt_mesh, spin, n_phases=100,
        observer_dir_ecl=observer_dir,
        sun_dir_ecl=sun_dir
    )
    print(f"  Points: {lc.n_points}")
    print(f"  Flux range: [{lc.fluxes.min():.4f}, {lc.fluxes.max():.4f}]")

    # 4. Run optimization
    print("\n[4/6] Running shape optimization...")
    opt_config = config or {
        'orders': [4, 8],
        'n_iterations': 300,
        'lr': 0.02,
        'lambda_smooth': 0.05,
        'lambda_volume': 0.01,
        'lambda_positive': 1.0,
        'n_theta': 48,
        'n_phi': 96,
        'device': 'cpu',
    }

    optimizer = ShapeOptimizer(config=opt_config)
    result_coeffs = optimizer.optimize_coarse_to_fine(
        light_curves=[lc],
        spin=spin,
        initial_shape='sphere',
        initial_radius=1.0,
        observer_dir=observer_dir,
        sun_dir=sun_dir
    )

    # 5. Generate result mesh
    print("\n[5/6] Generating result mesh...")
    final_order = opt_config['orders'][-1]
    result_mesh = spharm_to_mesh(result_coeffs, final_order, n_theta=64, n_phi=128)
    result_path = os.path.join(output_dir, "reconstructed.obj")
    result_mesh.export(result_path)
    print(f"  Mesh: {len(result_mesh.vertices)} verts, {len(result_mesh.faces)} faces")
    print(f"  Watertight: {result_mesh.is_watertight}")
    print(f"  Exported: {result_path}")

    # 6. Evaluate and visualize
    print("\n[6/6] Computing metrics and generating visualizations...")

    # Normalize both meshes for fair comparison
    gt_norm = normalize_mesh(gt_mesh)
    pred_norm = normalize_mesh(result_mesh)

    metrics = compute_all_metrics(pred_norm, gt_norm, n_sample_points=10000)
    print_metrics_report(metrics, "Synthetic Ellipsoid")

    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Visualizations
    plot_mesh_comparison(result_mesh, gt_mesh, title="Synthetic Test: Reconstructed vs Ground Truth",
                         save_path=os.path.join(output_dir, "shape_comparison.png"))
    plot_multi_view(result_mesh, title="Reconstructed Shape",
                    save_path=os.path.join(output_dir, "multi_view.png"))
    plot_loss_history(optimizer.history, title="Optimization History",
                      save_path=os.path.join(output_dir, "loss_history.png"))

    # Generate predicted light curve for comparison
    pred_lc = generate_synthetic_lightcurve(
        result_mesh, spin, n_phases=200,
        observer_dir_ecl=observer_dir, sun_dir_ecl=sun_dir
    )
    plot_lightcurve_fit(lc.times, lc.fluxes, pred_lc.times, pred_lc.fluxes,
                        title="Light Curve Fit",
                        save_path=os.path.join(output_dir, "lightcurve_fit.png"))

    plot_metrics_bar(metrics, title="Evaluation Metrics",
                     save_path=os.path.join(output_dir, "metrics_bar.png"))

    print(f"\nAll outputs saved to: {output_dir}/")
    return metrics


def run_damit_reconstruction(data_dir: str, output_dir: str = "output/damit",
                              gt_mesh_path: Optional[str] = None,
                              config: Optional[dict] = None):
    """
    Run reconstruction on real DAMIT data.

    Args:
        data_dir: path to directory with spin.txt, lc.json/lc.txt, and optionally shape.txt
        output_dir: where to save outputs
        gt_mesh_path: optional path to ground-truth mesh for evaluation
        config: optimizer configuration dict
    """
    os.makedirs(output_dir, exist_ok=True)
    asteroid_name = os.path.basename(data_dir)

    print("\n" + "=" * 60)
    print(f"DAMIT RECONSTRUCTION: {asteroid_name}")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data...")
    data = load_asteroid_data(data_dir, asteroid_name)

    if data.spin is None:
        print("ERROR: No spin.txt found. Cannot proceed without spin state.")
        sys.exit(1)

    if not data.light_curves:
        print("WARNING: No light curves found. Using DAMIT shape for synthetic LC.")
        if data.damit_shape is not None:
            observer_dir = np.array([1.0, 0.0, 0.0])
            sun_dir = np.array([1.0, 0.1, 0.0])
            sun_dir /= np.linalg.norm(sun_dir)
            data.light_curves = [generate_synthetic_lightcurve(
                data.damit_shape, data.spin, n_phases=100,
                observer_dir_ecl=observer_dir, sun_dir_ecl=sun_dir
            )]
        else:
            print("ERROR: No light curves or DAMIT shape found.")
            sys.exit(1)

    # Load ground truth if provided
    gt_mesh = None
    if gt_mesh_path and os.path.exists(gt_mesh_path):
        gt_mesh = trimesh.load(gt_mesh_path, force='mesh')
        print(f"  Ground truth: {len(gt_mesh.vertices)} vertices")

    # 2. Normalize light curves
    print("\n[2/5] Preparing light curves...")
    for i, lc in enumerate(data.light_curves):
        lc.normalize()
        print(f"  LC {i}: {lc.n_points} points, flux range [{lc.fluxes.min():.3f}, {lc.fluxes.max():.3f}]")

    # 3. Optimize
    print("\n[3/5] Running shape optimization...")
    opt_config = config or {
        'orders': [4, 8, 12],
        'n_iterations': 500,
        'lr': 0.015,
        'lambda_smooth': 0.1,
        'lambda_volume': 0.01,
        'lambda_positive': 1.0,
        'n_theta': 48,
        'n_phi': 96,
        'device': 'cpu',
    }

    observer_dir = np.array([1.0, 0.0, 0.0])
    sun_dir = np.array([1.0, 0.1, 0.0])
    sun_dir /= np.linalg.norm(sun_dir)

    optimizer = ShapeOptimizer(config=opt_config)
    result_coeffs = optimizer.optimize_coarse_to_fine(
        light_curves=data.light_curves,
        spin=data.spin,
        initial_shape='sphere',
        initial_radius=1.0,
        observer_dir=observer_dir,
        sun_dir=sun_dir
    )

    # 4. Export result
    print("\n[4/5] Generating result mesh...")
    final_order = opt_config['orders'][-1]
    result_mesh = spharm_to_mesh(result_coeffs, final_order, n_theta=64, n_phi=128)
    result_path = os.path.join(output_dir, f"{asteroid_name}_reconstructed.obj")
    result_mesh.export(result_path)
    print(f"  Exported: {result_path}")

    # Also export .stl and .ply
    result_mesh.export(os.path.join(output_dir, f"{asteroid_name}_reconstructed.stl"))
    result_mesh.export(os.path.join(output_dir, f"{asteroid_name}_reconstructed.ply"))

    # 5. Evaluate and visualize
    print("\n[5/5] Generating outputs...")

    plot_mesh_comparison(result_mesh, gt_mesh,
                         title=f"{asteroid_name} Reconstruction",
                         save_path=os.path.join(output_dir, "shape_comparison.png"))
    plot_multi_view(result_mesh, title=f"{asteroid_name} Multi-View",
                    save_path=os.path.join(output_dir, "multi_view.png"))
    plot_loss_history(optimizer.history,
                      save_path=os.path.join(output_dir, "loss_history.png"))

    # Metrics (if ground truth available)
    if gt_mesh is not None:
        gt_norm = normalize_mesh(gt_mesh)
        pred_norm = normalize_mesh(result_mesh)
        metrics = compute_all_metrics(pred_norm, gt_norm)
        print_metrics_report(metrics, asteroid_name)
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        plot_metrics_bar(metrics, save_path=os.path.join(output_dir, "metrics_bar.png"))

    print(f"\nAll outputs saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous 3D Asteroid Shape Reconstruction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --mode synthetic
  python pipeline.py --mode damit --data_dir data/bennu
  python pipeline.py --mode damit --data_dir data/bennu --gt_mesh data/bennu/ground_truth.obj
        """
    )

    parser.add_argument('--mode', choices=['synthetic', 'damit'],
                        default='synthetic',
                        help='Run mode: synthetic test or DAMIT data')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to asteroid data directory (for DAMIT mode)')
    parser.add_argument('--gt_mesh', type=str, default=None,
                        help='Path to ground-truth mesh for evaluation')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: output/<mode>)')
    parser.add_argument('--orders', type=int, nargs='+', default=[4, 8],
                        help='SPHARM orders for coarse-to-fine (e.g. 4 8 12)')
    parser.add_argument('--iterations', type=int, default=300,
                        help='Iterations per SPHARM order')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda')

    args = parser.parse_args()

    config = {
        'orders': args.orders,
        'n_iterations': args.iterations,
        'lr': args.lr,
        'lambda_smooth': 0.05,
        'lambda_volume': 0.01,
        'lambda_positive': 1.0,
        'n_theta': 48,
        'n_phi': 96,
        'device': args.device,
    }

    if args.mode == 'synthetic':
        out = args.output_dir or "output/synthetic"
        run_synthetic_test(output_dir=out, config=config)

    elif args.mode == 'damit':
        if args.data_dir is None:
            print("ERROR: --data_dir required for DAMIT mode")
            sys.exit(1)
        out = args.output_dir or f"output/{os.path.basename(args.data_dir)}"
        run_damit_reconstruction(
            data_dir=args.data_dir,
            output_dir=out,
            gt_mesh_path=args.gt_mesh,
            config=config
        )


if __name__ == "__main__":
    main()
