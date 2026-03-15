# Autonomous 3D Asteroid Shape Reconstruction

**Orion Astrathon — ASTRAX'26 | 48-Hour Hackathon**

Reconstructs 3D asteroid shapes from optical light curves using Spherical Harmonic (SPHARM) parameterisation with genus-0 topology constraints and gradient-based optimization.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run synthetic test (no data download needed)
python pipeline.py --mode synthetic

# 3. Set up real asteroid data
python download_data.py --target all

# 4. Run on Bennu
python pipeline.py --mode damit --data_dir data/bennu --orders 4 8 12

# 5. Run on Eros
python pipeline.py --mode damit --data_dir data/eros
```

## How It Works

1. **Spherical Harmonic Surface**: Asteroid shape is represented as r(θ,φ) — a radial function on the unit sphere, expanded in spherical harmonics
2. **Photometric Forward Model**: Simulates light curves from a given shape using Lommel-Seeliger scattering 
3. **Gradient Optimization**: Minimises mismatch between predicted and observed light curves via Adam optimizer (PyTorch autograd)
4. **Coarse-to-Fine**: Starts with low-order harmonics (ellipsoid), progressively adds detail

## Project Structure

```
├── pipeline.py              # Main entry point — end-to-end orchestration
├── spherical_harmonics.py   # SPHARM basis, mesh generation, topology constraint
├── photometric_model.py     # Differentiable scattering model (Lommel-Seeliger)
├── shape_optimizer.py       # Coarse-to-fine gradient descent optimizer
├── data_loader.py           # DAMIT parsers, coordinate transforms
├── metrics.py               # Hausdorff, Chamfer, IoU, RMSE, Completeness
├── visualize.py             # 3D mesh plots, light curve fits, loss history
├── download_data.py         # Data downloader for Bennu/Eros
├── requirements.txt         # Python dependencies
└── output/                  # Generated meshes, plots, metrics
```

## Outputs

- **3D Mesh**: `.obj`, `.stl`, `.ply` files importable into SBMT / MeshLab
- **Metrics Report**: `metrics.json` with all 5 evaluation metrics
- **Visualizations**: Shape comparison, multi-view, light curve fit, loss history

## Dependencies

- Python 3.9+
- PyTorch (CPU or CUDA)
- NumPy, SciPy, trimesh, Open3D, matplotlib

## Evaluation Metrics

| Metric | Formula | Goal |
|--------|---------|------|
| Hausdorff Distance | max(sup-inf d(x,y)) | Minimize |
| Chamfer Distance | Mean bidirectional NN distance² | Minimize |
| RMSE | √(mean NN distance²) | Minimize |
| Volumetric IoU | V∩/V∪ | Maximize (≈0.89) |
| Completeness | % of GT surface recovered | Maximize |

## CLI Options

```
python pipeline.py --help

--mode        synthetic | damit
--data_dir    Path to asteroid data directory
--gt_mesh     Path to ground-truth mesh for evaluation
--orders      SPHARM orders (e.g. 4 8 12)
--iterations  Iterations per order (default: 300)
--lr          Learning rate (default: 0.02)
--device      cpu | cuda
```