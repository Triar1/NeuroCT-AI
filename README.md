# NeuroCT AI — 48h MedTech Hackathon 🧠

**Experimental brain CT image classification demo** built during the EvoScan MedTech hackathon (November 2026). **Research/education only — NOT for clinical use.**

## Overview

Streamlit application to explore six-class brain CT slice classification using a weighted ensemble of **two ResNet18 models and one EfficientNet-B0**. It displays per-model predictions, ensemble scores and an illustrative **Grad-CAM** map from one ResNet18 (not lesion localization). The application processes **PNG/JPG slices**, not complete DICOM CT examinations or raw CT projections. It is **not** a CT reconstruction system.

Classes: `EDH`, `Healthy`, `IPH`, `IVH`, `SAH`, `SDH`.

## Experimental results (not clinically validated)

| Metric | Reported result |
|---|---:|
| Six-class accuracy | 52.2% |
| Balanced accuracy | 54.4% |
| Binary hemorrhage sensitivity | 91.6% |
| Binary hemorrhage specificity | 80.4% |

**Critical data-integrity limitation:** the original audit found exact duplicates affecting approximately **34.7% of test images**, conflicting class-folder labels, and no established patient-level independence. These results may be inflated and must **not** be interpreted as reliable clinical performance. The trained weights and dataset are **not included** in this repository. See [technical report](docs/technical_report.md).

## Run locally

Requires Python, a compatible PyTorch/torchvision installation, and the **three original checkpoints**. The original checkpoints are **not included** in this archive.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run neuroct_studio_v09.py
```

Before starting inference, place the following checkpoints in `models/`:

```text
models/neuroct_resnet18.pt
models/neuroct_resnet18_v04.pt
models/neuroct_efficientnet_b0.pt
```

The interface can open without them, but **predictions cannot run**. Checkpoints must use the same class order and saved `state_dict` structure as the original project. See [models/README.md](models/README.md).

## Dataset

Original source: [Kaggle — brain-hemorrhage-multiclass](https://www.kaggle.com/datasets/yogeshkn/brain-hemorrhage-multiclass) (47,014 PNG images across train/validation/test). **No images are redistributed here.** Verify dataset permissions and medical-data provenance before reuse. Expected local folder layout for scripts and optional example picker:

```text
data/raw/{train,validation,test}/{EDH Class,Healthy Class,IPH Class,IVH Class,SAH Class,SDH Class}/*.png
```

The original split is **not suitable for a trustworthy clinical benchmark**; rebuild an independent patient-level split and verify labels before further experiments.

## Repository structure

```text
NeuroCT-AI/
├── neuroct_studio_v09.py      # Streamlit demo
├── audit_leakage_v07.py       # Duplicate/label audit
├── scripts/                   # Historical training and evaluation scripts
├── models/README.md           # Missing checkpoint requirements
├── docs/technical_report.md   # Full technical dossier and limitations
├── requirements.txt
└── .gitignore
```

Training scripts are preserved from historical iterations and are **not** an automated reproducible end-to-end pipeline. Some may require being run from the repository root and may expect additional locally generated files.

## Scope and safety

This project is a hackathon proof of concept, **not a medical device**. It must not be used for diagnosis, triage, clinical decision-making or patient care. Grad-CAM is not an anatomical lesion mask. Predictions are not calibrated clinical probabilities. No clinical validation, external validation or regulatory certification has been performed.

## License and attribution

No software license is granted in this initial publication pending confirmation of contributor rights and third-party licensing. Please review attribution and choose an appropriate license with all contributors before inviting reuse.
