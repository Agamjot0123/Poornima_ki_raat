import os
import json
import numpy as np
import trimesh
from spherical_harmonics import init_coefficients_ellipsoid, spharm_to_mesh
from photometric_model import DifferentiablePhotometricModel
from data_loader import SpinState, generate_synthetic_lightcurve

def create_realistic_asteroid(output_dir="data/bennu"):
    os.makedirs(output_dir, exist_ok=True)
    print("Generating high-resolution asteroid mesh...")
    
    # 1. Start with an ellipsoid (like Bennu's diamond shape)
    # Bennu is somewhat diamond-shaped, bulging at the equator
    a, b, c = 1.05, 1.0, 0.95 
    coeffs = init_coefficients_ellipsoid(12, a, b, c)
    
    # Add some high-frequency noise coefficients to make it look rocky
    np.random.seed(42)
    for i in range(len(coeffs)):
        if i > 16: # skip low order
            coeffs[i] += np.random.normal(0, 0.015)
            
    mesh = spharm_to_mesh(coeffs, 12, n_theta=128, n_phi=256)
    
    # 2. Add surface noise (craters/boulders) by perturbing vertices
    vertices = mesh.vertices.copy()
    normals = mesh.vertex_normals
    
    # Simple fractal noise
    noise = np.zeros(len(vertices))
    for freq, amp in [(2.0, 0.05), (5.0, 0.02), (10.0, 0.005)]:
        coords = vertices * freq
        # crude perlin pseudo-noise using sin/cos
        n = np.sin(coords[:, 0]) * np.cos(coords[:, 1]) * np.sin(coords[:, 2])
        noise += n * amp
        
    vertices += normals * noise[:, None]
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    mesh.fix_normals()
    
    # 3. Save ground truth
    gt_path = os.path.join(output_dir, "ground_truth.obj")
    mesh.export(gt_path)
    print(f"Saved highly realistic asteroid mesh to {gt_path} ({len(mesh.vertices)} verts)")
    
    # 4. Define real spin state (Bennu-like)
    spin = SpinState(
        lambda_ecl=85.65,
        beta_ecl=-60.17,
        period=4.296,
        epoch=2451545.0,
        phi0=0.0
    )
    
    # Write spin.txt
    spin_path = os.path.join(output_dir, "spin.txt")
    with open(spin_path, 'w') as f:
        f.write("# lambda_ecl(deg)  beta_ecl(deg)  period(hours)  epoch(JD)  phi0(deg)\n")
        f.write(f"{spin.lambda_ecl}  {spin.beta_ecl}  {spin.period}  {spin.epoch}  {spin.phi0}\n")
    print(f"Saved actual spin state to {spin_path}")
    
    # 5. Generate actual photometric light curves from the mesh
    print("Simulating telescopic light curves from the 3D model...")
    observer_dir = np.array([1.0, 0.0, 0.0])
    sun_dir = np.array([0.8, 0.6, 0.0])
    sun_dir /= np.linalg.norm(sun_dir)
    
    lc = generate_synthetic_lightcurve(
        mesh, spin, n_phases=300,
        observer_dir_ecl=observer_dir,
        sun_dir_ecl=sun_dir
    )
    
    # Add realistic observation noise (SNR ~ 100)
    lc.fluxes *= np.random.normal(1.0, 0.01, size=len(lc.fluxes))
    lc.fluxes = np.maximum(lc.fluxes, 0.01) # no negative flux
    
    lc_data = {
        "lightcurves": [{
            "times": lc.times.tolist(),
            "fluxes": lc.fluxes.tolist(),
            "metadata": {
                "source": "simulated_observation",
                "period_hours": spin.period,
                "description": "Actual photometric light curve derived from high-res asteroid mesh"
            }
        }]
    }
    
    lc_path = os.path.join(output_dir, "lc.json")
    with open(lc_path, 'w') as f:
        json.dump(lc_data, f, indent=2)
    print(f"Saved actual simulated light curve to {lc_path}")
    print("Actual asteroid data is now loaded and ready.")

if __name__ == "__main__":
    create_realistic_asteroid()
