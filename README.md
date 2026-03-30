# 🧠 Brain Tumor Segmentation using UNet & ResUNet

<div align="center">

![Medical AI](https://img.shields.io/badge/Medical%20Imaging-AI-blue?style=for-the-badge&logo=brain)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge&logo=pytorch)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Segmentation-green?style=for-the-badge&logo=opencv)

**Accurate Brain Tumor Segmentation using ROI-based Preprocessing and Deep Learning**

</div>

---

## 🌟 **Project Overview**

Brain tumor segmentation is a critical task in medical imaging that helps in **early diagnosis, treatment planning, and survival prediction**.

This project implements:
- 🧠 **UNet**
- 🔁 **ResUNet (Residual UNet)**

combined with:
- 🎯 **ROI-based preprocessing pipeline**
- 🧪 Evaluation on **BraTS 2018, 2019, and 2020 datasets**

Inspired by the research paper:  
📄 :contentReference[oaicite:0]{index=0}

---

## 🚨 **Problem Statement**

Brain MRI images are:
- Low contrast  
- Noisy  
- Highly variable across scanners  

👉 This makes tumor detection difficult even for experts.

Additionally:
- Tumor regions overlap with healthy tissue
- Multi-modal MRI data adds complexity
- Full-image training increases computation and noise impact

---

## ✅ **Proposed Solution**

✔️ Use **Region of Interest (ROI) extraction** to focus only on tumor regions  
✔️ Apply **UNet and ResUNet architectures**  
✔️ Leverage **multi-modal MRI data (T1, T1ce, T2, FLAIR)**  
✔️ Improve segmentation accuracy and efficiency  

---

## 🧠 **Models Implemented**

### 🔷 UNet
- Encoder-decoder architecture
- Skip connections for feature fusion
- Strong baseline for medical segmentation

### 🔶 ResUNet
- Residual blocks added to UNet
- Solves vanishing gradient problem
- Better feature preservation and convergence

---

## 📊 **Datasets Used**

### 🧪 BraTS Datasets
- BraTS 2018  
- BraTS 2019  
- BraTS 2020  

### 📌 Key Details
- Multi-institution MRI data  
- 4 modalities per patient:
  - T1
  - T1ce
  - T2
  - FLAIR  

- Image size: **240 × 240 × 155**

### 🧬 Tumor Classes
- **WT** – Whole Tumor  
- **ET** – Enhancing Tumor  
- **TC** – Tumor Core  

📌 These datasets include expert-annotated ground truth masks :contentReference[oaicite:1]{index=1}

---

## 🧠 **MRI Modalities Explained**

| Modality | Purpose |
|----------|--------|
| **T1** | Healthy tissue structure |
| **T1ce** | Tumor boundaries (contrast-enhanced) |
| **T2** | Edema detection |
| **FLAIR** | Highlights tumor vs CSF |

---

## 🔍 **Key Innovation: ROI-Based Preprocessing**

Instead of training on the entire image:

👉 The model focuses only on **tumor-relevant regions**

### ⚙️ Pipeline:
1. Normalize MRI intensities
2. Use FLAIR modality for ROI detection
3. Apply:
   - Morphological operations
   - Top-hat / bottom-hat transforms
4. Noise removal (median filtering)
5. Background suppression
6. ROI extraction
7. Apply same ROI across all modalities

### 🚀 Benefits:
- Reduced computation
- Better accuracy
- Less noise influence
- Avoids overfitting :contentReference[oaicite:2]{index=2}

---

## 🏗️ **Project Structure**

```bash
BRAIN_TUMOR_SEGMENTATION/
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   ├── preprocessing_resunet.py
│   │   ├── dataset_inspection.py
│   │   └── dataset_inspection_resunet.py
│   │
│   ├── models/
│   │   ├── unet_model.py
│   │   └── resunet_model.py
│   │
│   ├── training/
│   │   ├── train_unet.py
│   │   ├── train_resunet.py
│   │   ├── run_experiments.py
│   │   ├── run_experiments_resunet.py
│   │   ├── run_folds.py
│   │   └── run_folds_resunet.py
│   │
│   ├── evaluation/
│   │   ├── loss_metrics.py
│   │   ├── final_unet_results.py
│   │   └── final_visualization.py
│   │
│   └── utils/
│       └── verify_datasets.py
│
├── Output_UNet.png
├── requirements.txt
└── README.md
```

---

## 📈 Evaluation Metrics

The model is evaluated using:

- 🎯 Dice Score  
- 📊 Jaccard Index  
- 🎯 Accuracy  
- 🔍 Precision  
- 📈 Sensitivity (Recall)  
- 🛡️ Specificity  

### 📌 Key Formula (Dice)

```math
Dice = 2|A ∩ B| / (|A| + |B|)
```

---

## 📊 Experimental Insights

### 🔵 UNet
- Good baseline performance  
- Struggles with fine details  
- Suffers from semantic gap  

### 🔴 ResUNet
- Better feature preservation  
- Improved convergence  
- Higher segmentation accuracy  

### 🎯 Key Observation
> ROI-based preprocessing significantly improves segmentation performance and reduces computational cost.

---

## 🧪 Results

- High Dice scores across BraTS datasets  
- Better tumor boundary detection  
- Improved segmentation of:
  - Enhancing tumor (ET)  
  - Tumor core (TC)  

📌 Comparable to state-of-the-art methods  

---

## 🚀 Getting Started

### 📋 Prerequisites
- Python 3.8+  
- PyTorch  
- NumPy  
- OpenCV  
- Matplotlib  

---

### ⚡ Installation
```bash
# Clone the repository
git clone https://github.com/your-username/brain-tumor-segmentation.git
cd brain-tumor-segmentation

# Install dependencies
pip install -r requirements.txt
```
---

### ▶️ Run Training
```bash
# Train UNet
python src/training/train_unet.py

# Train ResUNet
python src/training/train_resunet.py
```
---

## 📊 Run Experiments
```bash
python src/training/run_experiments.py
```
---

## 🎯 Research Impact

### 🔬 Why This Matters
- Improves **medical diagnosis accuracy**  
- Reduces **manual annotation workload**  
- Enables **automated decision support**  

### 📊 Applications
- Tumor detection  
- Treatment planning  
- Radiology AI systems  

---

## 🚀 Future Work
- 🔁 3D UNet implementation  
- 🧠 Attention-based segmentation  
- 📊 Transformer-based models  
- ⚡ Real-time clinical deployment  

---

<div align="center">

🧠 **Accurate Segmentation Starts with Better Preprocessing**

</div>