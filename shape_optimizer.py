"""
Shape optimizer: coarse-to-fine SPHARM fitting via gradient descent.

This module optimises the SPHARM coefficients to minimize the mismatch
between predicted and observed light curves, subject to smoothness
and volume regularisation constraints.

Strategy:
    1. Start at low SPHARM order (N=4, 25 coefficients) for coarse shape
    2. Optimise with Adam until convergence
    3. Upsample to higher order (N=8, then N=12, etc.)
    4. Repeat — the coarse solution provides a good initial guess for fine detail
"""

import numpy as np
import torch
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm

from spherical_harmonics import (
    count_coefficients, init_coefficients_sphere, init_coefficients_ellipsoid,
    precompute_basis, evaluate_surface_torch, spharm_to_mesh
)
from photometric_model import (
    DifferentiablePhotometricModel, photometric_loss
)
from data_loader import LightCurve, SpinState


class ShapeOptimizer:
    """
    Coarse-to-fine SPHARM shape optimizer.

    Optimises SPHARM coefficients to fit observed light curves using
    gradient descent with PyTorch autograd.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: optional configuration dict with keys:
                - orders: list of SPHARM orders for coarse-to-fine [4, 8, 12]
                - lr: learning rate (default 0.01)
                - n_iterations: iterations per order (default 500)
                - lambda_smooth: smoothness regularisation weight
                - lambda_volume: volume regularisation weight
                - lambda_positive: positive-radius penalty weight
                - n_theta: mesh resolution theta
                - n_phi: mesh resolution phi
                - device: 'cpu' or 'cuda'
        """
        self.config = config or {}
        self.orders = self.config.get('orders', [4, 8, 12])
        self.lr = self.config.get('lr', 0.01)
        self.n_iterations = self.config.get('n_iterations', 500)
        self.lambda_smooth = self.config.get('lambda_smooth', 0.1)
        self.lambda_volume = self.config.get('lambda_volume', 0.01)
        self.lambda_positive = self.config.get('lambda_positive', 1.0)
        self.n_theta = self.config.get('n_theta', 48)
        self.n_phi = self.config.get('n_phi', 96)
        self.device = self.config.get('device', 'cpu')

        # Track optimization history
        self.history: Dict[str, List[float]] = {
            'total_loss': [],
            'photo_loss': [],
            'smooth_loss': [],
            'volume_loss': [],
        }

    def smoothness_regularization(self, coeffs: torch.Tensor,
                                  max_order: int) -> torch.Tensor:
        """
        Penalise high-frequency SPHARM coefficients to encourage smooth surfaces.

        Higher-order coefficients (larger l) are penalised more heavily:
            R_smooth = Sum_l Sum_m l^2 * (a_lm^2 + b_lm^2)

        This is analogous to a Laplacian smoothing penalty on the sphere.
        """
        penalty = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        idx = 0
        for l in range(max_order + 1):
            weight = float(l * l)  # Quadratic penalty growth with order
            for m in range(l + 1):
                if m == 0:
                    penalty = penalty + weight * coeffs[idx] ** 2
                    idx += 1
                else:
                    penalty = penalty + weight * (coeffs[idx] ** 2 + coeffs[idx + 1] ** 2)
                    idx += 2
        return penalty

    def volume_regularization(self, r: torch.Tensor,
                              target_volume: float) -> torch.Tensor:
        """
        Soft constraint on total volume to prevent collapse or explosion.

        Approximate volume from the radii:
            V ≈ (4/3 * pi / N) * Sum_i r_i^3
        """
        n_pts = len(r)
        approx_volume = (4.0 / 3.0 * np.pi / n_pts) * torch.sum(r ** 3)
        return (approx_volume - target_volume) ** 2

    def positive_radius_penalty(self, r: torch.Tensor) -> torch.Tensor:
        """
        Penalise negative radii (physically impossible).

        Uses a one-sided quadratic penalty: only activates for r < 0.
        """
        negative_r = torch.clamp(-r, min=0)
        return torch.mean(negative_r ** 2)

    def optimize_single_order(self, max_order: int,
                              init_coeffs: np.ndarray,
                              light_curves: List[LightCurve],
                              spin: SpinState,
                              observer_dir: np.ndarray = None,
                              sun_dir: np.ndarray = None,
                              n_iterations: Optional[int] = None,
                              lr: Optional[float] = None) -> np.ndarray:
        """
        Optimize SPHARM coefficients at a single order.

        Args:
            max_order: current SPHARM order
            init_coeffs: initial coefficient array, length (max_order+1)^2
            light_curves: list of observed light curves
            spin: asteroid spin state
            observer_dir: observer direction in ecliptic frame (default: +x)
            sun_dir: Sun direction in ecliptic frame
            n_iterations: override for number of iterations
            lr: override for learning rate

        Returns:
            optimized coefficients as numpy array
        """
        n_iter = n_iterations or self.n_iterations
        learning_rate = lr or self.lr

        if observer_dir is None:
            observer_dir = np.array([1.0, 0.0, 0.0])
        if sun_dir is None:
            sun_dir = np.array([1.0, 0.1, 0.0])
            sun_dir /= np.linalg.norm(sun_dir)

        n_coeffs = count_coefficients(max_order)

        # Ensure init_coeffs has the right size
        if len(init_coeffs) < n_coeffs:
            # Pad with zeros for new higher-order coefficients
            padded = np.zeros(n_coeffs, dtype=np.float64)
            padded[:len(init_coeffs)] = init_coeffs
            init_coeffs = padded
        elif len(init_coeffs) > n_coeffs:
            init_coeffs = init_coeffs[:n_coeffs]

        # Setup photometric model
        photo_model = DifferentiablePhotometricModel(
            n_theta=self.n_theta, n_phi=self.n_phi,
            scattering_law='lommel_seeliger', device=self.device
        )

        # Precompute SPHARM basis at grid points
        theta_np = photo_model.theta_flat.numpy()
        phi_np = photo_model.phi_flat.numpy()
        Y_basis_np = precompute_basis(max_order, theta_np, phi_np)
        Y_basis = torch.tensor(Y_basis_np, dtype=torch.float64, device=self.device)

        # Torch tensors for viewing geometry
        obs_torch = torch.tensor(observer_dir, dtype=torch.float64, device=self.device)
        sun_torch = torch.tensor(sun_dir, dtype=torch.float64, device=self.device)
        spin_lambda_rad = np.deg2rad(spin.lambda_ecl)
        spin_beta_rad = np.deg2rad(spin.beta_ecl)

        # Learnable coefficients
        coeffs = torch.tensor(init_coeffs, dtype=torch.float64,
                              device=self.device, requires_grad=True)

        # Target volume (from initial shape)
        with torch.no_grad():
            r_init = evaluate_surface_torch(coeffs, max_order, Y_basis)
            n_pts = len(r_init)
            target_vol = float((4.0 / 3.0 * np.pi / n_pts) * torch.sum(r_init ** 3))

        # Optimizer
        optimizer = torch.optim.Adam([coeffs], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter)

        # Prepare observed light curves as torch tensors
        obs_lcs = []
        for lc in light_curves:
            lc_copy = LightCurve(times=lc.times.copy(), fluxes=lc.fluxes.copy(),
                                 errors=lc.errors.copy() if lc.errors is not None else None)
            lc_copy.normalize()
            phases_torch = torch.tensor(lc_copy.times, dtype=torch.float64, device=self.device)
            fluxes_torch = torch.tensor(lc_copy.fluxes, dtype=torch.float64, device=self.device)
            errors_torch = (torch.tensor(lc_copy.errors, dtype=torch.float64, device=self.device)
                            if lc_copy.errors is not None else None)
            obs_lcs.append((phases_torch, fluxes_torch, errors_torch))

        # Optimization loop
        best_loss = float('inf')
        best_coeffs = init_coeffs.copy()

        pbar = tqdm(range(n_iter), desc=f"Order {max_order} ({n_coeffs} coeffs)")
        for iteration in pbar:
            optimizer.zero_grad()

            # Compute surface radii from current coefficients
            r = evaluate_surface_torch(coeffs, max_order, Y_basis)

            # Photometric loss (sum over all light curves)
            total_photo_loss = torch.tensor(0.0, dtype=torch.float64, device=self.device)
            for phases, obs_fluxes, obs_errors in obs_lcs:
                pred_fluxes = photo_model.compute_lightcurve(
                    r, phases, obs_torch, sun_torch,
                    spin_lambda=spin_lambda_rad, spin_beta=spin_beta_rad
                )
                total_photo_loss = total_photo_loss + photometric_loss(
                    pred_fluxes, obs_fluxes, obs_errors
                )

            # Regularisation
            smooth_loss = self.lambda_smooth * self.smoothness_regularization(coeffs, max_order)
            vol_loss = self.lambda_volume * self.volume_regularization(r, target_vol)
            pos_loss = self.lambda_positive * self.positive_radius_penalty(r)

            # Total loss
            total_loss = total_photo_loss + smooth_loss + vol_loss + pos_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # Track history
            loss_val = total_loss.item()
            self.history['total_loss'].append(loss_val)
            self.history['photo_loss'].append(total_photo_loss.item())
            self.history['smooth_loss'].append(smooth_loss.item())
            self.history['volume_loss'].append(vol_loss.item())

            # Track best
            if loss_val < best_loss:
                best_loss = loss_val
                best_coeffs = coeffs.detach().cpu().numpy().copy()

            if iteration % 50 == 0:
                pbar.set_postfix({
                    'loss': f'{loss_val:.6f}',
                    'photo': f'{total_photo_loss.item():.6f}',
                    'best': f'{best_loss:.6f}'
                })

        return best_coeffs

    def optimize_coarse_to_fine(self, light_curves: List[LightCurve],
                                spin: SpinState,
                                initial_shape: str = 'sphere',
                                initial_radius: float = 1.0,
                                initial_axes: Optional[Tuple[float, float, float]] = None,
                                observer_dir: np.ndarray = None,
                                sun_dir: np.ndarray = None) -> np.ndarray:
        """
        Run the full coarse-to-fine optimization pipeline.

        Args:
            light_curves: list of observed light curves
            spin: asteroid spin state
            initial_shape: 'sphere' or 'ellipsoid'
            initial_radius: radius for sphere initialization
            initial_axes: (a, b, c) for ellipsoid initialization
            observer_dir: observer direction in ecliptic frame
            sun_dir: Sun direction in ecliptic frame

        Returns:
            optimized coefficients at the highest order
        """
        print("\n" + "=" * 60)
        print("COARSE-TO-FINE SPHARM OPTIMIZATION")
        print("=" * 60)

        # Initialize coefficients at the lowest order
        min_order = self.orders[0]
        if initial_shape == 'ellipsoid' and initial_axes is not None:
            coeffs = init_coefficients_ellipsoid(min_order, *initial_axes)
            print(f"Initial shape: ellipsoid ({initial_axes})")
        else:
            coeffs = init_coefficients_sphere(min_order, radius=initial_radius)
            print(f"Initial shape: sphere (r={initial_radius})")

        # Iterate through orders
        for order in self.orders:
            n_coeffs = count_coefficients(order)
            print(f"\n--- Optimizing at order N={order} ({n_coeffs} coefficients) ---")

            # Learning rate decreases with order (fine details need smaller steps)
            lr = self.lr / (1 + 0.5 * (order - min_order))

            coeffs = self.optimize_single_order(
                max_order=order,
                init_coeffs=coeffs,
                light_curves=light_curves,
                spin=spin,
                observer_dir=observer_dir,
                sun_dir=sun_dir,
                lr=lr
            )

            # Generate mesh at this stage for monitoring
            mesh = spharm_to_mesh(coeffs, order, n_theta=self.n_theta, n_phi=self.n_phi)
            print(f"  Mesh: {len(mesh.vertices)} vertices, watertight={mesh.is_watertight}")

        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print(f"Final order: N={self.orders[-1]}, coefficients: {len(coeffs)}")
        print("=" * 60)

        return coeffs

    def get_final_mesh(self, coeffs: np.ndarray, max_order: int,
                       n_theta: int = 64, n_phi: int = 128):
        """
        Generate the final high-resolution mesh from optimized coefficients.

        Returns:
            trimesh.Trimesh object
        """
        return spharm_to_mesh(coeffs, max_order, n_theta=n_theta, n_phi=n_phi)


if __name__ == "__main__":
    # Quick test: optimize a synthetic case
    print("=== Shape Optimizer Test ===")
    print("Creating synthetic light curve from an ellipsoid...")

    # Create a known ellipsoidal shape
    from data_loader import generate_synthetic_lightcurve
    from spherical_harmonics import spharm_to_mesh, init_coefficients_ellipsoid

    # Ground truth: elongated asteroid
    gt_coeffs = init_coefficients_ellipsoid(8, a=1.5, b=1.0, c=0.8)
    gt_mesh = spharm_to_mesh(gt_coeffs, 8, n_theta=48, n_phi=96)
    print(f"Ground-truth mesh: {len(gt_mesh.vertices)} verts, watertight={gt_mesh.is_watertight}")

    # Create synthetic spin and light curve
    spin = SpinState(lambda_ecl=45.0, beta_ecl=30.0, period=6.0, epoch=2451545.0, phi0=0.0)
    synthetic_lc = generate_synthetic_lightcurve(gt_mesh, spin, n_phases=100)
    print(f"Synthetic light curve: {synthetic_lc.n_points} points")

    # Optimize starting from a sphere
    optimizer = ShapeOptimizer(config={
        'orders': [4, 8],
        'n_iterations': 200,
        'lr': 0.02,
        'lambda_smooth': 0.05,
        'lambda_volume': 0.01,
        'n_theta': 32,
        'n_phi': 64,
    })

    result_coeffs = optimizer.optimize_coarse_to_fine(
        light_curves=[synthetic_lc],
        spin=spin,
        initial_shape='sphere',
        initial_radius=1.0
    )

    result_mesh = optimizer.get_final_mesh(result_coeffs, max_order=8, n_theta=48, n_phi=96)
    result_mesh.export("test_optimized.obj")
    print(f"\nResult mesh: {len(result_mesh.vertices)} verts, watertight={result_mesh.is_watertight}")
    print("Exported test_optimized.obj")
