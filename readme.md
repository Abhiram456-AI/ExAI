

# ExAI – Soil Microbial Risk Assessment System

ExAI is an AI-driven soil health assessment system that analyzes microbial plate images to estimate the likelihood of harmful microbial activity using morphology-based feature extraction and calibrated risk scoring.

The system is designed to provide both technical insights and practical recommendations through dual reporting modes.

---

## Overview

ExAI performs the following core tasks:

- Extracts microbial morphology features from plate images
- Computes calibrated risk scores (0–1 scale)
- Classifies soil condition as Harmful or Non-Harmful
- Generates detailed analytical reports
- Provides simplified farmer-friendly recommendations
- Includes a complete validation and performance evaluation pipeline

The design emphasizes explainability, robustness, and structured risk modeling rather than black-box prediction.

---

## Key Features

- Morphology-based microbial dominance detection
- Feature extraction including density, clustering, coverage, solidity, and circularity
- Risk scoring with threshold optimization
- Dual reporting modes (Scientific / Farmer)
- Full validation pipeline with ROC, PR, calibration, and confusion matrix outputs
- Publication-quality visualizations (PNG + PDF export)

---

## Project Structure

ExAI/
│
├── backend/
│   ├── models/
│   │   ├── microbial_features.py
│   │   ├── risk_engine.py
│   │
│   ├── inference/
│   │   └── pipeline.py
│   │
│   └── validation/
│       ├── validation_engine.py
│       ├── performance_metrics.py
│       ├── validation_results.csv
│       ├── validation_summary.csv
│       └── metrics_summary_ieee.csv
│
├── dataset/
│   ├── Tomato-LB/
│   ├── Tomato-NA/
│   ├── Cabbage-LB/
│   ├── Cabbage-NA/
│   ├── Brinjal-LB/
│   └── Brinjal-NA/
│
└── README.md

---

## Dataset

- Approximately 400 soil microbial plate images
- Multi-crop dataset (Tomato, Cabbage, Brinjal)
- Harmful (LB) and Non-Harmful (NA) labeling
- Cross-crop evaluation supported

Ground truth is derived from dataset folder labeling (LB = Harmful, NA = Non-Harmful).

---

## Installation

```bash
git clone https://github.com/<your-username>/ExAI.git
cd ExAI

python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# OR
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

---

## Running Inference

```bash
python backend/inference/pipeline.py
```

Interactive mode allows you to:
- Select crop type
- Specify treatment condition
- Provide image path
- Choose report mode (scientific or farmer)

---

## Running Validation

### Full Dataset Validation

```bash
python backend/validation/validation_engine.py
```

### Generate Performance Metrics & Visualizations

```bash
python backend/validation/performance_metrics.py
```

Outputs include:

- ROC curve
- Precision–Recall curve
- Calibration curve
- Risk score density plots
- Normalized confusion matrix
- Metrics summary CSV

---

## Design Principles

- Structured morphology-based feature extraction
- Calibrated probabilistic risk scoring
- Transparent threshold selection
- Clear separation between analysis and reporting layers
- Reproducible validation workflow

---

## Future Enhancements

- Extended dataset validation
- API deployment layer
- Feature ablation analysis
- Explainability visualization tools

---

If you use or extend this project, please ensure proper attribution.