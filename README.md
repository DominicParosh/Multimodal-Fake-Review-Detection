# Multimodal Fake Review Detector

A robust multimodal framework for detecting AI-generated fake reviews by jointly analyzing
review text and associated images using transformer-based text encoders and CLIP-style vision models.

## Project Structure

```
multimodal-fake-review-detector/
├── configs/
│   └── config.yaml              # All hyperparameters and settings
├── data/
│   ├── dataset.py               # Dataset classes (real, synthetic, hybrid)
│   ├── generators.py            # Multi-generator fake review synthesis
│   ├── augmentation.py          # Data augmentation & adversarial examples
│   └── preprocessing.py         # Text/image preprocessing pipelines
├── models/
│   ├── text_encoder.py          # Transformer-based text encoder
│   ├── image_encoder.py         # CLIP-style vision encoder
│   ├── fusion.py                # Novel multimodal fusion modules
│   ├── classifier.py            # Main classifier with confidence scoring
│   ├── explainability.py        # Explainability / attribution module
│   └── lightweight.py           # Lightweight model for real-time deployment
├── training/
│   ├── trainer.py               # Main training loop
│   ├── losses.py                # Custom loss functions (contrastive, adversarial)
│   ├── adversarial_trainer.py   # GAN-style arms race simulation
│   └── callbacks.py             # Training callbacks and schedulers
├── evaluation/
│   ├── metrics.py               # Evaluation metrics & confidence calibration
│   ├── benchmark.py             # Standardized benchmark protocol
│   ├── robustness.py            # Temporal & adversarial robustness tests
│   └── ablation.py              # Modality contribution analysis
├── toolkit/
│   ├── detector.py              # High-level API for end users
│   ├── api_server.py            # FastAPI server for real-time detection
│   └── cli.py                   # Command-line interface
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Evaluation entry point
│   └── generate_dataset.py      # Dataset generation script
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Generate hybrid dataset
python scripts/generate_dataset.py --config configs/config.yaml

# Train model
python scripts/train.py --config configs/config.yaml

# Evaluate
python scripts/evaluate.py --config configs/config.yaml --checkpoint best_model.pt

# Run API server
python -m toolkit.api_server --port 8000
```

## Usage (Python API)

```python
from toolkit.detector import FakeReviewDetector

detector = FakeReviewDetector.from_pretrained("checkpoints/best_model.pt")
result = detector.predict(
    text="This product is amazing! Best purchase ever!",
    image_path="review_image.jpg"
)
print(result)
# {'prediction': 'fake', 'confidence': 0.92, 'explanation': {...}}
```