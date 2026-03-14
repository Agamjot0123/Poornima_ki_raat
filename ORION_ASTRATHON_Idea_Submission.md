# ORION ASTRATHON – ASTRAX'26
## Idea Submission: Autonomous 3-D Asteroid Surface Shape Reconstruction

**Team:** Agastya
**Problem Statement:** Autonomous 3-D Asteroid Surface Shape Reconstruction from Multimodal Partial Observations
**Date:** 14 March 2026

---

## 1. Problem Statement Chosen

We have selected the **Autonomous 3-D Asteroid Surface Shape Reconstruction** challenge. The task requires building a fully automated pipeline that ingests sparse, multimodal observational data — optical light curves and delay-Doppler radar images — and produces a high-fidelity, watertight 3-D mesh of an asteroid's surface, benchmarked against spacecraft-derived ground-truth models (OSIRIS-REx Bennu and NEAR Shoemaker Eros).

---

## 2. Problem Understanding

### The Core Challenge
Only ~300 of 1,000,000+ discovered asteroids have rigorous 3-D shape models. Current methods (e.g., SHAPE suite) require **months of expert intervention** and iterative parameter tuning — an approach that fundamentally cannot scale to the growing catalogue of Near-Earth Objects (NEOs).

### Why Shape Matters
Accurate asteroid geometry is critical for:
- **Planetary defence** — deflection mission planning requires precise mass/shape knowledge
- **Yarkovsky/YORP modelling** — orbital drift and spin-state evolution depend on surface geometry
- **Physical characterisation** — volume, bulk density, and albedo all require known geometry

### The Data Problem
Two complementary but individually incomplete modalities exist:

| Modality | Strength | Blind Spot |
|---|---|---|
| **Delay-Doppler Radar** | Sub-10m resolution cross-sections | North-South ambiguity collapses 3-D → 2-D |
| **Optical Light Curves** | Constrains spin axis + period precisely | Cannot see concave features (craters, lobes) |

**Key insight:** Neither modality alone is sufficient. Radar resolves fine surface detail but loses a dimension; light curves capture global shape constraints but only the convex hull. A multimodal fusion approach can compensate for both blind spots simultaneously.

### What We Must Solve
1. **Automation** — Zero human-in-the-loop after initial configuration
2. **Non-convex recovery** — Craters, boulder fields, bifurcated lobes
3. **Watertight output** — Closed, hole-free mesh exportable as `.obj`/`.stl`
4. **Physical correctness** — Radar-to-body frame coordinate transforms must be sound

---

## 3. Implementation Roadmap

### Phase 1: Data Pipeline & Preprocessing (Hours 0–8)
- Download and parse Bennu/Eros ground-truth meshes from NASA PDS
- Fetch light-curve data from DAMIT database and OSIRIS-REx ground-based observations
- Acquire delay-Doppler radar matrices from JPL Asteroid Radar Research
- Implement preprocessing: light-curve normalisation, phase-folding, radar image standardisation
- Build synthetic training data generator: render simulated observations from known 3-D models

### Phase 2: Model Architecture & Training (Hours 8–24)
- Implement dual-stream MFFnet-style encoder:
  - **Stream A:** 1D CNN + BiLSTM for temporal light-curve features
  - **Stream B:** ResNet-50 backbone for 2D delay-Doppler radar images
- Build attention-based multimodal fusion module
- Implement SPHARM coefficient predictor (FC layers → a_lm, b_lm up to degree N=25)
- Spherical harmonic mesh decoder with genus-0 topology constraint
- Training with composite loss: L_chamfer + λ₁·L_hausdorff + λ₂·L_volume_IoU

### Phase 3: Training, Inference & Optimisation (Hours 24–36)
- Train on synthetic dataset (augmented with noise, partial observations)
- Fine-tune on real Bennu/Eros observational data
- Optimise SPHARM degree for accuracy vs. computational cost trade-off
- Implement automated hyperparameter selection (no manual tuning)

### Phase 4: Evaluation & Presentation (Hours 36–48)
- Compute all 5 required metrics against ground truth (Hausdorff, Chamfer, RMSE, IoU, Completeness)
- Generate side-by-side 3D renderings (reconstruction vs. spacecraft ground truth)
- Export final meshes in .obj format (SBMT-compatible)
- Write metric report (Jupyter notebook with visualisations)
- Prepare final presentation

---

## 4. Tech Stack

| Component | Technology | Justification |
|---|---|---|
| **Deep Learning Framework** | PyTorch | Best ecosystem for custom architectures, dynamic computation graphs |
| **Backbone CNN** | ResNet-50 (torchvision) | Pre-trained on ImageNet, proven feature extractor for 2D data |
| **Temporal Encoder** | 1D CNN + BiLSTM | Captures both local and long-range temporal patterns in light curves |
| **3-D Mesh Operations** | Open3D + trimesh | Industry-standard mesh I/O, Poisson reconstruction, metric computation |
| **Spherical Harmonics** | SHTools (pyshtools) | Robust SPHARM coefficient computation and mesh generation |
| **Ephemeris & Coordinates** | SpiceyPy | NASA-standard SPICE kernels for coordinate frame transforms |
| **Asteroid Data Access** | sbpy + astroquery | Programmatic access to PDS, DAMIT, and JPL databases |
| **Visualisation** | PyVista + Matplotlib | Interactive 3D rendering and publication-quality metric plots |
| **Compute** | CUDA / Google Colab Pro | GPU-accelerated training (T4/A100) |
| **Version Control** | Git + GitHub | Repository management, collaboration, submission |
| **Report** | Jupyter Notebook | Reproducible metric computation with inline visualisations |

---

## 5. Architecture

### System Architecture: AsteroNeRF — Dual-Stream Multimodal Fusion Network

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                  │
│                                                                     │
│   ┌─────────────────────┐       ┌─────────────────────────┐        │
│   │  Optical Light       │       │  Delay-Doppler Radar     │       │
│   │  Curves (1D)         │       │  Images (2D)             │       │
│   │  [flux vs phase]     │       │  [range × velocity]      │       │
│   └────────┬────────────┘       └──────────┬──────────────┘        │
│            │                                │                       │
│            ▼                                ▼                       │
│   ┌─────────────────────┐       ┌─────────────────────────┐        │
│   │  1D CNN + BiLSTM     │       │  ResNet-50 Encoder       │       │
│   │  Temporal Encoder    │       │  (pretrained backbone)   │       │
│   │  → Feature Vec (256) │       │  → Feature Vec (256)     │       │
│   └────────┬────────────┘       └──────────┬──────────────┘        │
│            │                                │                       │
│            └────────────┬───────────────────┘                       │
│                         ▼                                           │
│            ┌─────────────────────────┐                              │
│            │  Attention-Based Fusion  │                              │
│            │  Cross-modal attention   │                              │
│            │  → Fused Vec (512)       │                              │
│            └────────────┬────────────┘                              │
│                         ▼                                           │
│            ┌─────────────────────────┐                              │
│            │  SPHARM Coefficient      │                              │
│            │  Predictor (FC layers)   │                              │
│            │  → {a_lm, b_lm}         │                              │
│            │    l=0..25, m=0..l       │                              │
│            └────────────┬────────────┘                              │
│                         ▼                                           │
│            ┌─────────────────────────┐                              │
│            │  Spherical Harmonic      │                              │
│            │  Mesh Decoder            │                              │
│            │  r(θ,φ) = Σ (a_lm cos   │                              │
│            │  + b_lm sin) P_lm        │                              │
│            │  Genus-0 guaranteed      │                              │
│            └────────────┬────────────┘                              │
│                         ▼                                           │
│            ┌─────────────────────────┐                              │
│            │  Watertight 3-D Mesh     │                              │
│            │  Output (.obj / .stl)    │                              │
│            └────────────┬────────────┘                              │
│                         ▼                                           │
│            ┌─────────────────────────┐                              │
│            │  VALIDATION PIPELINE     │                              │
│            │  vs Ground Truth         │                              │
│            │  (Bennu / Eros)          │                              │
│            │                          │                              │
│            │  • Hausdorff Distance    │                              │
│            │  • Chamfer Distance      │                              │
│            │  • RMSE                  │                              │
│            │  • Volumetric IoU        │                              │
│            │  • Completeness C        │                              │
│            └──────────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

### Training Strategy
- **Synthetic data generation:** Render simulated light curves and radar images from known 3-D asteroid models (DAMIT database) to create a large training corpus
- **Composite loss function:** L = L_chamfer + λ₁·L_hausdorff + λ₂·(1 - IoU) + λ₃·L_SPHARM_regularisation
- **Transfer learning:** Pre-trained ResNet-50 backbone, fine-tuned on asteroid radar images
- **Data augmentation:** Noise injection, phase shifts, partial observation masking

### Key Design Decisions
1. **SPHARM representation** over direct vertex prediction — guarantees watertight, genus-0 topology by construction
2. **Attention-based fusion** over simple concatenation — learns which modality to trust for different surface regions
3. **Dual-stream architecture** — each modality gets a specialised encoder before fusion, preserving domain-specific features

---

## 6. GitHub Repository

**Repository:** [https://github.com/Agastya18/Orion](https://github.com/Agastya18/Orion)

### Initial Repository Structure
```
Orion/
├── README.md                    # Project overview & setup instructions
├── requirements.txt             # Python dependencies
├── config/
│   └── default.yaml             # Default pipeline configuration
├── src/
│   ├── data/
│   │   ├── light_curve_loader.py    # DAMIT light-curve parser
│   │   ├── radar_loader.py          # Delay-Doppler matrix loader
│   │   └── synthetic_generator.py   # Training data renderer
│   ├── models/
│   │   ├── light_curve_encoder.py   # 1D CNN + BiLSTM stream
│   │   ├── radar_encoder.py         # ResNet-50 stream
│   │   ├── fusion.py                # Attention-based fusion module
│   │   ├── spharm_predictor.py      # SPHARM coefficient heads
│   │   └── mesh_decoder.py          # Spherical harmonic → mesh
│   ├── training/
│   │   ├── trainer.py               # Training loop
│   │   └── losses.py                # Composite loss functions
│   ├── evaluation/
│   │   ├── metrics.py               # Hausdorff, Chamfer, IoU, RMSE, C
│   │   └── visualise.py             # Side-by-side renderings
│   └── pipeline.py                  # End-to-end inference pipeline
├── notebooks/
│   └── metric_report.ipynb          # Quantitative evaluation report
├── outputs/
│   └── meshes/                      # Reconstructed .obj files
└── data/
    ├── ground_truth/                # Bennu & Eros reference models
    └── observations/                # Light curves & radar data
```

---

*Submitted for ORION ASTRATHON – ASTRAX'26 | 14 March 2026*
