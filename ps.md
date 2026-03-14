# ORION ASTRATHON - ASTRAX'26
## Autonomous 3-D Asteroid Surface Shape Reconstruction from Multimodal Partial Observations

**Duration:** 48 Hours
**Domain:** Planetary Science · Machine Learning · Astrodynamics
**Difficulty:** Advanced

---

## 1. Background

Asteroids are primordial remnants of the early solar system whose precise three-dimensional morphologies are foundational to planetary defence, spacecraft navigation, and the modelling of non-gravitational forces. The **Yarkovsky effect** — the gradual orbital drift caused by anisotropic thermal emission — and the **YORP effect** — the torque-driven spin evolution arising from asymmetric photon recoil — both depend critically on accurate shape models. Constraining an asteroid's volume, bulk density, and albedo is likewise impossible without knowing its geometry.

Despite the discovery of over one million minor planets, only a few hundred possess rigorously constrained 3-D shape models. Two primary observational modalities feed ground-based shape reconstruction pipelines:

- **Delay-Doppler radar imaging** — resolves an asteroid's cross-section in range and radial-velocity dimensions, yielding sub-10-m spatial resolution but suffering from an inherent North–South ambiguity that collapses 3-D structure onto a 2-D plane.
- **Optical light-curve photometry** — records the rotationally modulated flux of scattered sunlight, tightly constraining the spin-axis orientation and sidereal period, but inherently blind to concave features hidden within the convex hull.

Current state-of-practice tools (e.g., the SHAPE suite) ingest these data through non-linear χ² minimisation over a spherical-harmonic surface parameterisation. While physically rigorous, this approach requires **months of expert intervention** and is prone to local-optima traps. The scientific community urgently needs autonomous, data-driven alternatives capable of processing the flood of newly discovered Near-Earth Objects (NEOs).

---

## 2. Problem Statement

**Core Objective:** Design and implement an autonomous software pipeline that ingests sparse, multimodal observational data — optical light curves and/or delay-Doppler radar images — and outputs a high-fidelity, watertight 3-D mesh or volumetric representation of an asteroid's surface, validated against spacecraft-derived ground-truth models.

### 2.1 Technical Requirements

1. **Multimodal ingestion** — The pipeline must process at least one of:
   - (a) time-series optical photometry (flux vs. rotational phase), or
   - (b) 2-D delay-Doppler radar matrices.
   - Bonus credit is awarded for fusing both.

2. **3-D output** — The reconstruction must produce an explicit geometric representation (triangular mesh `.obj`/`.stl`, point cloud, or voxel grid) that can be exported and visualised.

3. **Automation** — The pipeline must run end-to-end without manual parameter tuning after initial configuration. Human-in-the-loop iterative adjustments are not acceptable.

4. **Non-convex awareness** — The algorithm must attempt to recover concave features (craters, boulder fields, bifurcated lobes) that are inaccessible to purely convex inversion methods.

5. **Physical coordinate correctness** — Coordinate transformations from the radar frame to the asteroid body frame must follow the standard rotation equations where ψ is the rotational phase and δ is the subradar latitude.

### 2.2 Recommended Approaches

Participants are strongly encouraged to move beyond legacy optimisation and explore:

- **Neural Radiance Fields (NeRF)** adapted to ingest delay-Doppler coordinate mappings in place of optical perspective projections, learning implicit volumetric density directly from radar echoes.
- **Point-cloud deep networks** that exploit deviations between photometric light curves and hypothetical convex hulls to predict concave geometry at millisecond inversion speeds.
- **Dual-stream CNN encoders** (e.g., MFFnet-style architectures with ResNet backbones) for multimodal feature fusion, extracting SPHARM coefficients from both radar and optical channels before decoding to a 3-D mesh.
- **Spherical parameterisation** with genus-0 topology constraints to guarantee a closed, hole-free surface even from severely sparse input data.

---

## 3. Expected Deliverables

| Deliverable | Description |
|---|---|
| **Source Code** | Well-structured repository (GitHub/GitLab) with a comprehensive `README.md` covering installation, dependencies, and step-by-step reproduction instructions. All inference scripts must run without modification on the provided test datasets. |
| **3-D Shape Output** | Reconstructed mesh or point cloud exported in a standard format (`.obj`, `.ply`, or `.stl`) for at least one target asteroid (Bennu or Eros). Output must be importable into the Small Body Mapping Tool (SBMT). |
| **Metric Report** | A concise PDF or Jupyter notebook documenting quantitative performance on the ground-truth datasets: Hausdorff Distance, Chamfer Distance, IoU, RMSE, and Completeness C. |

---

## 4. Datasets & Resources

All data is open-source and freely accessible. The primary repository is the **NASA Planetary Data System (PDS) Small Bodies Node**.

### 4.1 Ground-Truth Shape Models (Validation Targets)

| Mission | Target | Instrument / Product | PDS Identifier |
|---|---|---|---|
| OSIRIS-REx | (101955) Bennu | OLA LIDAR — Digital Terrain Models (DTMs) via Poisson reconstruction | `urn:nasa:pds:orex.ola` |
| OSIRIS-REx | (101955) Bennu | Ground-based light curves (sparse optical input) | `orex.gbo.astbennu.lightcurvesimages::1.0` |
| NEAR Shoemaker | (433) Eros | MSI optical + NLR laser rangefinder; Peter Thomas shape model (elongated S-type) | `urn:nasa:pds:near_rss_derive` |

### 4.2 Ground-Based Observational Inputs

- **DAMIT** (`damit.cuni.cz`) — Database of Asteroid Models from Inversion Techniques. Provides triangular-facet polyhedra, spin-state parameters, and light-curve histories in machine-readable formats:
  - `spin.txt` — ecliptic longitude λ, latitude β, period P, epoch t0, phase ϕ0
  - `shape.txt` — vertex coordinates & facet indices
  - `lc.json` — photometric time-series

- **JPL Asteroid Radar Research** (`echo.jpl.nasa.gov`) — Goldstone and Arecibo delay-Doppler 2-D matrices for NEOs and comets.

### 4.3 Recommended Libraries

| Library | Purpose |
|---|---|
| `SpiceyPy` | Python/NAIF SPICE wrapper for ephemeris, coordinate-frame transforms, and observer-geometry modelling |
| `sbpy` | Astropy-affiliated package for light-curve analysis, thermal models, and small-body database access |
| `Open3D / trimesh / PyMeshLab` | Mesh manipulation, Poisson reconstruction, and metric computation |
| `PyTorch / JAX` | Recommended frameworks for neural inversion architectures |

---

## 5. Evaluation Metrics

### 5.1 Quantitative Geometric Metrics

| Metric | Definition & Purpose | Target |
|---|---|---|
| **Hausdorff Distance** | Worst-case maximum surface deviation; penalises catastrophic artefacts | Minimise |
| **Chamfer Distance** | Bi-directional nearest-neighbour mean; balanced accuracy + coverage | Minimise |
| **RMSE** | Root-mean-square point error over the full mesh; validates global dimensional accuracy and scale | Minimise |
| **Volumetric IoU** | Validates physical volume enclosure; critical for non-convex craters and voids. State-of-the-art NN methods achieve IoU ≈ 0.89 | Maximise |
| **Completeness C** | Fraction of ground-truth surface area recovered within a spatial tolerance; flags interpolative voids from sparse data | Maximise |

### 5.2 Judging Rubric (1–5 Scale)

| Criterion | Scoring Guidance |
|---|---|
| **Impact** | Does the tool demonstrably accelerate NEO characterisation pipelines from months to minutes? Can outputs be integrated into SBMT? |
| **Creativity** | Novelty of the inversion approach. NeRF adaptations for radar data, MFFnet fusion channels, generative networks governed by physical scattering laws, and SNR-improving signal processing all score highly. |
| **Validity** | Quantitative benchmarking against OSIRIS-REx Bennu and NEAR Eros ground truth using the five metrics above. Code must run without failures; coordinate transforms must be physically sound. |
| **Relevance** | Tool must perform actual 3-D inversion from partial observations. |
| **Presentation** | Side-by-side rendering of algorithmic output vs. spacecraft ground truth, clean error-metric visualisations, and a coherent narrative explaining planetary-defence significance. |