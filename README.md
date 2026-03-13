# 3DMM-k-Same

**Privacy-Preserving Facial Data Publishing via 3D Morphable Masking and k-Same De-identification**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This project implements a 3D face model-based anonymization system. It combines improved k-same algorithm with 3DDFA_V2 to protect personal privacy while preserving facial expressions.

---

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Experimental Results](#experimental-results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## ✨ Features

### Three Anonymization Methods

1. **Improved Anonymization**
   - Shape parameter perturbation based on 3DDFA_V2
   - Global noise and component-level local noise
   - Expression preservation mechanism
   - Adjustable rendering blending

2. **Original 3DDFA**
   - 3D modeling and rendering only
   - Serves as upper bound baseline for visual quality
   - No anonymization applied

3. **K-Same Algorithm**
   - k-anonymity approach by Sweeney et al.
   - FaceNet feature-based clustering
   - Pixel-level face averaging
   - Pure image processing independent of 3D modeling

### Evaluation System

- **Privacy Protection**: FaceNet recognition rate and similarity
- **Utility Preservation**: FER expression recognition rate and similarity
- **Image Quality**: PSNR and SSIM
- **Performance**: Processing speed and PRF metrics

---

## 📁 Project Structure

```
3DMM-k-Same/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Dependencies
│
├── Anonymization Modules
│   ├── shape_anonymization.py         # Original k-same
│   ├── shape_anonymization_improved.py # Improved method
│   ├── ablation_experiment.py         # Ablation study
│   └── baseline_mask_face_adaptive.py # Adaptive mask baseline
│
├── Evaluation Modules
│   ├── face_recognition_eval.py       # Face recognition evaluation
│   ├── emotion_recognition_eval.py    # Emotion recognition evaluation
│   ├── run_full_evaluation.py         # Full evaluation pipeline
│   └── compare_baseline_vs_improved.py # Comparison tool
│
└── 3DDFA Dependencies
    ├── TDDFA.py                       # 3DDFA main class
    ├── TDDFA_ONNX.py                  # ONNX version
    ├── FaceBoxes/                     # Face detection
    ├── bfm/                           # Basel Face Model
    ├── models/                        # Network architectures
    ├── utils/                         # Utility functions
    ├── Sim3DR/                        # 3D rendering
    ├── configs/                       # Model configurations
    └── weights/                       # Pretrained weights
```

---

## 🚀 Quick Start

### 1. Requirements

- Python 3.8+ (3.8-3.11 recommended)
- CUDA (optional, for GPU acceleration)
- 2GB+ available disk space

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/hisirlab/3DMM-k-Same.git

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; import cv2; import facenet_pytorch; import fer; print('All dependencies installed successfully')"
```

---

## 📖 Usage

### Method 1: Improved Anonymization

```bash
python shape_anonymization_improved.py \
    --dataset-path dataset/Celeb-A \
    --output-dir results/improved \
    --strategy noise \
    --noise-level 0.20 \
    --component-noise 0.08 \
    --exp-blend 0.9 \
    --render-alpha 0.8
```

**Parameters:**
- `--noise-level`: Global noise strength (0.15-0.25 recommended)
- `--component-noise`: Component-level noise (0.05-0.10 recommended)
- `--exp-blend`: Expression preservation weight (0.8-1.0 recommended)
- `--render-alpha`: Rendering blending ratio (0.6-0.8 for privacy-quality balance)

### Method 2: Ablation Study

```bash
python ablation_experiment.py \
    --dataset-name Celeb-A \
    --dataset-root dataset \
    --out-root results/ablation \
    --k-value 5 \
    --noise 0.20 \
    --component-noise 0.08 \
    --render-alpha-a 0.8
```

This generates comparison results for three methods:
- Complete evaluation metrics (recognition rate, expression preservation, image quality)
- Processing speed comparison
- Detailed results in JSON format

### Method 3: Full Evaluation Pipeline

```bash
python run_full_evaluation.py \
    --original dataset/Celeb-A \
    --anonymized results/improved/anonymized_faces \
    --face-model facenet \
    --emotion-model fer
```

### Method 4: Baseline Comparison

```bash
python compare_baseline_vs_improved.py \
    --dataset-path dataset/Celeb-A \
    --output-dir results/comparison \
    --use-adaptive-baseline
```

This compares the improved method against baseline approaches.

---

## 📊 Evaluation Metrics

### Privacy Protection (Lower is Better)

- **Identity Recognition Rate**: Rate at which anonymized faces are re-identified
- **Average Similarity**: Cosine similarity between original and anonymized face features

### Utility Preservation (Higher is Better)

- **Expression Preservation Rate**: Rate at which expressions are correctly recognized
- **Expression Similarity**: Similarity between original and anonymized expression features

### Image Quality (Higher is Better)

- **PSNR** (Peak Signal-to-Noise Ratio): Image quality metric
- **SSIM** (Structural Similarity Index): Perceptual quality metric

### Performance

- **Processing Speed**: Time per image (seconds)
- **Precision/Recall/F1**: Classification performance metrics


## 🔧 Advanced Configuration

### Parameter Tuning

Edit `configs/mb1_120x120.yml` to adjust 3DDFA model parameters:

```yaml
batch_size: 32
img_size: 120
checkpoint_fp: weights/mb1_120x120.pth
# ... other parameters
```

### Extending Evaluation Models

Add new face recognition models in `face_recognition_eval.py`:

```python
class FaceRecognitionEvaluator:
    def __init__(self, model_type='facenet'):
        if model_type == 'your_model':
            # Load custom model
            pass
```

**Related Work:**

- 3DDFA_V2: Guo et al. "Towards Fast, Accurate and Stable 3D Dense Face Alignment" (ECCV 2020)
- k-anonymity: Sweeney, L. "k-anonymity: A model for protecting privacy" (2002)

---

## 🙏 Acknowledgements

This project is built upon:

- [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) - 3D face alignment framework
- [FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch) - Face recognition model
- [FER](https://github.com/justinshenk/fer) - Facial expression recognition library

Thanks to all contributors!

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

**Note**: Please comply with the usage agreements of related datasets (Celeb-A, FFHQ, LFW, etc.).

---

**⭐ If you find this useful, please give us a star!**
