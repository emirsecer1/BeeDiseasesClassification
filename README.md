# 🐝 Honey Bee Disease Classification with Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A comprehensive deep learning study for automated honey bee disease detection.**  
*Swin Transformer achieves 98.74% accuracy — surpassing the existing literature record.*

[Dataset](https://www.kaggle.com/datasets/emirsecer/beediseasesdataset) • [Notebook 1](https://www.kaggle.com/code/emirsecer/notebook-1-eda) • [Notebook 2](https://www.kaggle.com/code/emirsecer/notebook-2-baseline-efficientnet) • [Notebook 3](https://www.kaggle.com/code/emirsecer/notebook-3-hybrid-cnn-boost) • [Notebook 4](https://www.kaggle.com/code/emirsecer/notebook-4-swin-transformer) • [Notebook 5](https://www.kaggle.com/code/emirsecer/notebook-5-final-ensemble)

</div>

---

## 📊 Results at a Glance

| Model | Accuracy | F1 Score | First on Dataset |
|-------|----------|----------|-----------------|
| 🏆 **Swin Transformer** | **98.74%** | **98.75%** | ✅ Yes |
| VGG-19 [Liang, 2022] — *prev. record* | 98.65% | — | — |
| **Final Ensemble** | **98.65%** | **98.65%** | ✅ Yes |
| **EfficientNetV2-S** | **98.55%** | **98.55%** | ✅ Yes |
| VGG-19 [Kaplan Berkaya, 2021] | 98.07% | 94.19% | — |
| CNN+MLFB [Metlek, 2021] | 95.04% | 95.04% | — |
| Hybrid CNN+GB | 91.01% | 90.61% | ✅ Yes |

---

## 📁 Repository Structure

```
BeeDiseasesClassification/
│
├── README.md
│
├── report/
│   └── bee_diseases_paper.tex        # IEEE-format LaTeX paper (Turkish)
│
├── part_one/
│   └── notebook_1_eda.ipynb          # Exploratory Data Analysis
│
├── part_two/
│   └── notebook_2_baseline_efficientnet.ipynb   # EfficientNetV2-S
│
├── part_three/
│   └── notebook_3_hybrid_cnn_boost.ipynb        # Hybrid CNN + Gradient Boosting
│
├── part_four/
│   └── notebook_4_swin_transformer.ipynb        # Swin Transformer ⭐
│
└── part_five/
    └── notebook_5_final_ensemble.ipynb          # Final Ensemble
```

---

## 🗂️ Dataset

**BeeImage Dataset** — Jenny Yang, Harvard Dataverse (2018)  
📦 Kaggle: [`emirsecer/beediseasesdataset`](https://www.kaggle.com/datasets/emirsecer/beediseasesdataset/settings)

| Class | Label | Images | Weight |
|-------|-------|--------|--------|
| 0 | Ant Problems | 457 | 1.89× |
| 1 | Small Hive Beetles | 579 | 1.49× |
| 2 | Healthy | 3,384 | 0.25× |
| 3 | Robbed Hive | 251 | 3.43× |
| 4 | Missing Queen | 29 | **5.00×** |
| 5 | Varroa | 472 | 1.83× |
| — | **Total** | **5,172** | — |

> ⚠️ Severe class imbalance: Healthy vs. Missing Queen ratio = **116:1**

---

## 🔬 Notebooks

### Part 1 — Exploratory Data Analysis
> `part_one/notebook_1_eda.ipynb`

- Class distribution & imbalance analysis
- Per-class image quality inspection (blur, resolution)
- Color channel statistics (mean, std per class)
- Class weight computation: `[1.89, 1.49, 0.25, 3.43, 5.00, 1.83]`
- Augmentation strategy design based on EDA findings

---

### Part 2 — EfficientNetV2-S Baseline
> `part_two/notebook_2_baseline_efficientnet.ipynb`

**Architecture:**
```
EfficientNetV2-S (ImageNet-1K pretrained)
└── Dropout(0.2) → Linear(1280→512) → BN → ReLU → Dropout(0.3) → Linear(512→6)
```

**Key techniques:**
- 2-stage fine-tuning (backbone frozen for 5 epochs → full unfreeze)
- Class-weighted CrossEntropy + Label Smoothing (λ=0.05)
- Automatic Mixed Precision (AMP)
- Gradient clipping (max_norm=1.0)
- 5× Test Time Augmentation (TTA)
- Grad-CAM visualizations

**Result: 98.55% accuracy | 98.55% F1**

---

### Part 3 — Hybrid CNN + Gradient Boosting
> `part_three/notebook_3_hybrid_cnn_boost.ipynb`

**Architecture:**  
EfficientNetV2-S as feature extractor (1,280-dim embeddings) → Gradient Boosting classifiers

| Classifier | Optimization | Trials |
|------------|-------------|--------|
| XGBoost | Optuna | 50 |
| LightGBM | Optuna | 50 |
| CatBoost | Optuna | 30 |

Soft voting ensemble with F1-weighted combination.

> 💡 XGBoost fix for new versions: `tree_method='hist'` + `device='cuda'` (deprecated: `gpu_hist`)

**Result: 91.01% accuracy | 90.61% F1**

---

### Part 4 — Swin Transformer ⭐
> `part_four/notebook_4_swin_transformer.ipynb`

**First application of Swin Transformer on BeeImage dataset.**

**Architecture:**
```
swin_small_patch4_window7_224 (timm, ImageNet-22K pretrained)
└── LayerNorm(768) → Dropout(0.2) → Linear(768→512) → GELU → Dropout(0.3) → Linear(512→6)
```

**Key techniques:**
- OneCycleLR scheduler (10% warmup + cosine annealing) → CosineAnnealingLR after unfreeze
- Warmup phase critical for stable Transformer training
- 5× TTA

**Result: 98.74% accuracy | 98.75% F1 — 🏆 New literature record**

---

### Part 5 — Final Ensemble
> `part_five/notebook_5_final_ensemble.ipynb`

EfficientNetV2-S + Swin Transformer heterogeneous ensemble with 5 combination strategies:

| Strategy | Weight | Accuracy |
|----------|--------|----------|
| **Equal Average** | 0.5 / 0.5 | **98.65%** ✅ Best |
| F1-Weighted | 0.49 / 0.51 | 98.55% |
| Swin-Heavy | 0.4 / 0.6 | 98.45% |
| Max Probability | — | 98.26% |
| Geometric Mean | — | 98.36% |

> 🐛 **Bug fixed:** Missing `classifier.4` (Dropout) in `BeeEfficientNet` class caused checkpoint loading failure.

**Result: 98.65% accuracy | 98.65% F1 — only 14/1,035 misclassified**

---

## ⚙️ Environment

```
Python        3.12
PyTorch       2.0
CUDA          11.8
timm          0.9+
XGBoost       2.0+
LightGBM      4.0+
CatBoost      1.2+
Optuna        3.0+
Platform      Kaggle (Tesla T4 × 2)
Random seed   42
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/emirsecer/BeeDiseasesClassification.git
cd BeeDiseasesClassification
```

### 2. Install dependencies
```bash
pip install torch torchvision timm xgboost lightgbm catboost optuna
```

### 3. Run on Kaggle
Add the dataset to your Kaggle notebook:
```python
# Dataset path
DATA_PATH = "https://www.kaggle.com/datasets/emirsecer/beediseasesdataset"

# EfficientNetV2-S checkpoint
EFFNET_CKPT = "https://www.kaggle.com/datasets/emirsecer/bee-efficientnet"

# Swin Transformer checkpoint
SWIN_CKPT = "https://www.kaggle.com/datasets/emirsecer/bee-swin"
```

---

## 📈 Evaluation Protocol

- **Cross-validation:** 5-fold Stratified K-Fold (results on Fold 0)
- **Train / Val split:** 4,137 / 1,035
- **Test Time Augmentation:** 5 transforms
- **Metrics:** Accuracy, Weighted F1, Per-class Precision/Recall/F1

---

## 🏆 Novel Contributions

1. **First Swin Transformer** application on BeeImage dataset → new SOTA (98.74%)
2. **First EfficientNetV2-S** evaluation on this dataset (98.55%)
3. **First CNN+Transformer heterogeneous ensemble** for bee disease detection
4. **Hybrid CNN + XGBoost/LightGBM/CatBoost** with Optuna hyperparameter optimization
5. **Grad-CAM interpretability analysis** for all six disease classes

---

## 📄 Paper

This work is written up as an IEEE two-column format academic paper in Turkish.

```
Seçer, M. E. (2025).
Bal Arısı Hastalıklarının Görüntü Tabanlı Derin Öğrenme ile Sınıflandırılması:
Çok Modelli Karşılaştırmalı Bir Çalışma.
Fırat Üniversitesi, Mekatronik Mühendisliği Bölümü.
```

---

## 📬 Contact

**Muhammed Emir Seçer**  
Mechatronics Engineering, Fırat University  
📧 muhammedemirsecer@gmail.com  
🔗 [Kaggle](https://www.kaggle.com/emirsecer)

---

<div align="center">
<sub>Made with ❤️ and a lot of 🐝</sub>
</div>
