ORION ASTRATHON - ASTRAX'26 Autonomous 3-D Asteroid Surface Shape
Reconstruction from Multimodal Partial Observations Duration:48 Hours
Domain:Planetary Science·Machine Learning·Astrodynamics
Difficulty:Advanced 1. Background Asteroids are primordial remnants of
the early solar system whose precise three-dimensional mor- phologies
are foundational to planetary defence, spacecraft navigation, and the
modelling of non- gravitational forces. TheYarkovsky effect---the
gradual orbital drift caused by anisotropic thermal emission---and
theYORP effect---the torque-driven spin evolution arising from
asymmetric photon recoil---both depend critically on accurate shape
models. Constraining an asteroid's volume, bulk density, and albedo is
likewise impossible without knowing its geometry. Despite the discovery
of over one million minor planets, only a few hundred possess rigorously
constrained 3-D shape models. Two primary observational modalities feed
ground-based shape reconstruction pipelines: • Delay-Doppler radar
imaging--- resolves an asteroid's cross-section in range ( τ = 2R/c) and
radial-velocity (ν = −2 λ dR dt ) dimensions, yielding sub-10-m spatial
resolution but suffering from an inherentNorth--South ambiguitythat
collapses 3-D structure onto a 2-D plane. • Optical light-curve
photometry--- records the rotationally modulated flux of scattered
sunlight, tightly constraining the spin-axis orientation and sidereal
period, but inherently blind to concave features hidden within the
convex hull. Current state-of-practice tools (e.g., theSHAPEsuite)
ingest these data through non-linear χ2 minimisation over a
spherical-harmonic surface parameterisation: r(θ, ϕ) = NX l=0 lX m=1 
alm cos(mϕ) +b lm sin(mϕ)  Plm(cosθ) While physically rigorous, this
approach requires months of expert intervention and is prone to
local-optima traps. The scientific community urgently needs autonomous,
data-driven alternatives capable of processing the flood of newly
discovered Near-Earth Objects (NEOs). 2. Problem Statement Core
Objective:Design and implement an autonomous software pipeline that
ingests sparse, multimodal observational data---optical light curves
and/or delay-Doppler radar images---and outputs a high-fidelity,
watertight 3-D mesh or volumetric representation of an asteroid's
surface, validated against spacecraft-derived ground-truth models. 2.1.
Technical Requirements 1. Multimodal ingestion.The pipeline must process
at least one of: (a) time-series optical photometry (flux vs. rotational
phase), or (b) 2-D delay-Doppler radar matrices. Bonus credit is awarded
for fusing both.

ORION ASTRATHON - ASTRAX'26Asteroid Shape Reconstruction 2. 3-D
output.The reconstruction must produce an explicit geometric
representation (triangular mesh.obj/.stl, point cloud, or voxel grid)
that can be exported and visualised. 3. Automation.The pipeline must run
end-to-end without manual parameter tuning after initial configuration.
Human-in-the-loop iterative adjustments are not acceptable. 4.
Non-convex awareness.The algorithm must attempt to recover concave
features (craters, boulder fields, bifurcated lobes) that are
inaccessible to purely convex inversion methods. 5. Physical coordinate
correctness.Coordinate transformations from the radar frame ( xr, yr,
zr) to the asteroid body frame (x, y, z) must follow the standard
rotation: xr = (xcosψ−ysinψ) cosδ+zsinδ, y r =xsinψ+ycosψ, z r =−zcosδ
whereψis the rotational phase andδis the subradar latitude. 2.2.
Recommended Approaches Participants are strongly encouraged to move
beyond legacy optimisation and explore: • Neural Radiance Fields
(NeRF)adapted to ingest delay-Doppler coordinate mappings in place of
optical perspective projections, learning implicit volumetric density
F(x, θ, ϕ) → (c, σ) directly from radar echoes. • Point-cloud deep
networksthat exploit deviations between photometric light curves and
hypothetical convex hulls to predict concave geometry at millisecond
inversion speeds. • Dual-stream CNN encoders(e.g., MFFnet-style
architectures with ResNet backbones) for multimodal feature fusion,
extracting SPHARM coefficients from both radar and optical channels
before decoding to a 3-D mesh. • Spherical parameterisation with genus-0
topology constraintsto guarantee a closed, hole-free surface even from
severely sparse input data. 3. Expected Deliverables Deliverable
Description Source Code Well-structured repository (GitHub/GitLab) with
a comprehensive README.md covering installation, dependencies, and
step-by-step reproduction instructions. All inference scripts must run
without modification on the provided test datasets. 3-D Shape Out- put
Reconstructed mesh or point cloud exported in a standard format ( .obj,
.ply, or .stl) for at least one target asteroid (Bennu or Eros). Output
must be importable into the Small Body Mapping Tool (SBMT). Metric
Report A concise PDF or Jupyter notebook documenting quantitative
performance on the ground-truth datasets: Hausdorff Distance, Chamfer
Distance, IoU, RMSE, and CompletenessC. 4. Datasets & Resources All data
is open-source and freely accessible. The primary repository is the NASA
Planetary Data System (PDS) Small Bodies Node. 2

ORION ASTRATHON - ASTRAX'26Asteroid Shape Reconstruction 4.1.
Ground-Truth Shape Models (Validation Targets) MissionTarget Instrument
/ Product PDS Identifier OSIRIS-REx (101955) Bennu OLA LIDAR --- Digital
Terrain Models (DTMs) via Poisson reconstruction; post-TAG modell
00050mm spc obj ...vp54.obj urn:nasa:pds:orex.ola OSIRIS-REx (101955)
Bennu Ground-based light curves (sparse optical in- put) orex.gbo.ast-
bennu.lightcurves- images::1.0 NEAR Shoe- maker (433) Eros MSI optical +
NLR laser rangefinder; Peter Thomas shape model (elongated S-type)
urn:nasa:pds:near rss derived 4.2. Ground-Based Observational Inputs •
DAMIT(damit.cuni.cz) --- Database of Asteroid Models from Inversion
Techniques. Provides triangular-facet polyhedra, spin-state parameters,
and light-curve histories in machine-readable formats: spin.txt
(ecliptic longitude λ, latitude β, period P, epoch t0, phase ϕ0),
shape.txt (vertex coordinates & facet indices),lc.json(photometric
time-series). • JPL Asteroid Radar Research(echo.jpl.nasa.gov) ---
Goldstone and Arecibo delay-Doppler 2-D matrices for NEOs and comets.
4.3. Recommended Libraries • SpiceyPy--- Python/NAIF SPICE wrapper for
ephemeris, coordinate-frame transforms, and observer-geometry modelling.
• sbpy--- Astropy-affiliated package for light-curve analysis, thermal
models, and small-body database access. • Open3D / trimesh /
PyMeshLab--- mesh manipulation, Poisson reconstruction, and metric
computation. •PyTorch / JAX--- recommended frameworks for neural
inversion architectures. 5. Evaluation Metrics 5.1. Quantitative
Geometric Metrics Both meshes are uniformly subsampled to fixed-size
point sets prior to evaluation. Let X (recon- structed) andY(ground
truth) denote the resulting point clouds. 3

ORION ASTRATHON - ASTRAX'26Asteroid Shape Reconstruction
MetricDefinition & Purpose Target Hausdorff Dis- tanced H dH(X, Y) = max
supx∈X infy∈Y d(x, y),supy∈Y infx∈X d(x, y)

Worst-case maximum surface deviation; penalises catastrophic artefacts.
Minimise Chamfer Dis- tanced CD dCD = 1 \|X\| P x∈X miny∈Y ∥x−y∥ 2 2 + 1
\|Y\| P y∈Y minx∈X ∥x−y∥ 2 2 Bi-directional nearest-neighbour mean;
balanced accuracy + coverage. Minimise RMSE Standard root-mean-square
point error over the full mesh; validates global dimensional accuracy
and scale. Minimise Volumetric IoU IoU = Vpred∩Vgt Vpred∪Vgt Validates
physical volume enclosure; critical for non-convex craters and voids.
State-of-the-art NN methods achieve IoU ≈0.89. Maximise Completeness C
C= Smodel Sgr tr ×100% Fraction of ground-truth surface area recovered
within a spatial tolerance; flags interpolative voids from sparse data.
Maximise 5.2. Judging Rubric (1--5 Scale) Criterion Scoring Guidance
Impact Does the tool demonstrably accelerate NEO characterisation
pipelines from months to minutes? Can outputs be integrated into SBMT?
Creativity Novelty of the inversion approach. NeRF adaptations for radar
data, MFFnet fusion channels, generative networks governed by physical
scattering laws, and SNR-improving signal processing all score highly.
Validity Quantitative benchmarking against OSIRIS-REx Bennu and NEAR
Eros ground truth using the five metrics above. Code must run without
failures; coordinate transforms must be physically sound. Relevance Tool
must perform actual 3-D inversion from partial observations.
Presentation Side-by-side rendering of algorithmic output vs. spacecraft
ground truth, clean error-metric visualisations, and a coherent
narrative explaining planetary-defence significance. 4
