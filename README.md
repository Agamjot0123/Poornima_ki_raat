# AsteroMesh — Autonomous 3-D Asteroid Shape Reconstruction

Dual-stream multimodal fusion network that reconstructs watertight 3-D asteroid meshes from optical light curves and delay-Doppler radar images.

## Architecture

```
Light Curves → 1D CNN + BiLSTM → ┐
                                  ├→ Attention Fusion → SPHARM Predictor → Mesh Decoder → .obj
Radar Images → ResNet-50 ────────┘
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python -m src.training.trainer --config config/default.yaml
```

### Inference
```bash
python -m src.pipeline --config config/default.yaml --input data/observations/ --output outputs/meshes/
```

### Evaluation
```bash
python -m src.evaluation.metrics --pred outputs/meshes/bennu_pred.obj --gt data/ground_truth/bennu.obj
```

## Project Structure
```
src/
├── data/                  # Data loading & synthetic generation
├── models/                # Neural network architecture
├── training/              # Training loop & losses
├── evaluation/            # Metrics & visualisation
└── pipeline.py            # End-to-end inference
```

## Team
**Poornima ki raat** — Agastya, Agam

## References
- NASA PDS Small Bodies Node
- DAMIT Database (damit.cuni.cz)
- JPL Asteroid Radar Research
