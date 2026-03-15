"""
Download ground-truth shapes and DAMIT data for asteroid reconstruction.

Usage:
    python download_data.py --target bennu
    python download_data.py --target eros
    python download_data.py --target all
"""

import os
import sys
import json
import argparse
import requests
from typing import Optional


# DAMIT base URL
DAMIT_BASE = "https://astro.troja.mff.cuni.cz/projects/damit"

# Known asteroid configurations for the hackathon
ASTEROID_CONFIGS = {
    "bennu": {
        "damit_id": None,  # Bennu is primarily from OSIRIS-REx, not DAMIT
        "spin": {
            "lambda_ecl": 85.65,    # ecliptic longitude of pole (deg)
            "beta_ecl": -60.17,     # ecliptic latitude of pole (deg)
            "period": 4.296057,     # sidereal period (hours)
            "epoch": 2451545.0,     # J2000.0
            "phi0": 0.0
        },
        "description": "101955 Bennu — OSIRIS-REx target, rubble-pile asteroid",
        "gt_urls": [
            # NASA PDS OLA DTM shape model (simplified URL - actual download may vary)
            "https://sbnarchive.psi.edu/pds4/orex/orex.ola/data_shape/bennu_g_03170mm_spc_obj_0000n00000_v054.obj"
        ]
    },
    "eros": {
        "damit_id": None,
        "spin": {
            "lambda_ecl": 11.35,
            "beta_ecl": 17.22,
            "period": 5.270,
            "epoch": 2451545.0,
            "phi0": 0.0
        },
        "description": "433 Eros — NEAR Shoemaker target, elongated S-type asteroid",
        "gt_urls": []
    }
}


def create_data_directory(target: str, base_dir: str = "data") -> str:
    """Create and return the data directory for a target asteroid."""
    data_dir = os.path.join(base_dir, target)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def write_spin_file(data_dir: str, spin_params: dict):
    """Write a spin.txt file from configuration."""
    filepath = os.path.join(data_dir, "spin.txt")
    with open(filepath, 'w') as f:
        f.write(f"# lambda_ecl(deg)  beta_ecl(deg)  period(hours)  epoch(JD)  phi0(deg)\n")
        f.write(f"{spin_params['lambda_ecl']}  {spin_params['beta_ecl']}  "
                f"{spin_params['period']}  {spin_params['epoch']}  {spin_params['phi0']}\n")
    print(f"  Written: {filepath}")


def generate_synthetic_lightcurve_file(data_dir: str, spin_params: dict):
    """
    Generate a synthetic light curve for testing when real data isn't available.
    Uses a simple sinusoidal model based on the rotation period.
    """
    import numpy as np

    period_hours = spin_params['period']
    n_points = 200

    # Simulate phases covering 2 full rotations
    times = np.linspace(0, 2 * 2 * np.pi, n_points)

    # Create realistic-looking light curve
    # Elongated body produces a double-peaked curve per rotation
    fluxes = (1.0
              + 0.15 * np.cos(2 * times)       # main elongation signal
              + 0.05 * np.cos(4 * times)       # higher harmonic (non-spherical detail)
              + 0.03 * np.cos(times)           # asymmetry
              + 0.008 * np.random.randn(n_points))  # noise

    # Normalize
    fluxes /= np.mean(fluxes)

    # Save as JSON
    lc_data = {
        "lightcurves": [{
            "times": times.tolist(),
            "fluxes": fluxes.tolist(),
            "metadata": {
                "source": "synthetic",
                "period_hours": period_hours,
                "description": "Synthetic light curve for pipeline testing"
            }
        }]
    }

    filepath = os.path.join(data_dir, "lc.json")
    with open(filepath, 'w') as f:
        json.dump(lc_data, f, indent=2)
    print(f"  Written: {filepath} ({n_points} points)")


def download_file(url: str, filepath: str, timeout: int = 60) -> bool:
    """Download a file from URL. Returns True if successful."""
    try:
        print(f"  Downloading: {url}")
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  Saved: {filepath} ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def setup_asteroid(target: str, base_dir: str = "data"):
    """Set up data directory for a target asteroid."""
    if target not in ASTEROID_CONFIGS:
        print(f"Unknown target: {target}. Available: {list(ASTEROID_CONFIGS.keys())}")
        return

    config = ASTEROID_CONFIGS[target]
    print(f"\n{'='*50}")
    print(f"Setting up: {config['description']}")
    print(f"{'='*50}")

    data_dir = create_data_directory(target, base_dir)

    # Write spin state
    write_spin_file(data_dir, config['spin'])

    # Generate synthetic light curve (always, as fallback)
    generate_synthetic_lightcurve_file(data_dir, config['spin'])

    # Try downloading ground-truth mesh
    for url in config.get('gt_urls', []):
        ext = os.path.splitext(url)[1]
        gt_path = os.path.join(data_dir, f"ground_truth{ext}")
        download_file(url, gt_path)

    print(f"\nData directory ready: {data_dir}/")
    print(f"Contents:")
    for f in os.listdir(data_dir):
        fpath = os.path.join(data_dir, f)
        size = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Download asteroid data for reconstruction")
    parser.add_argument('--target', choices=['bennu', 'eros', 'all'], default='all',
                        help='Which asteroid to set up')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Base data directory')
    args = parser.parse_args()

    if args.target == 'all':
        for target in ASTEROID_CONFIGS:
            setup_asteroid(target, args.data_dir)
    else:
        setup_asteroid(args.target, args.data_dir)

    print("\n" + "="*50)
    print("DATA SETUP COMPLETE")
    print("="*50)
    print("\nNext steps:")
    print("  1. Run synthetic test:  python pipeline.py --mode synthetic")
    print("  2. Run on Bennu:        python pipeline.py --mode damit --data_dir data/bennu")
    print("  3. Run on Eros:         python pipeline.py --mode damit --data_dir data/eros")


if __name__ == "__main__":
    main()
