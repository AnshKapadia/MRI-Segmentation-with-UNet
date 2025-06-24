# 3D Brain Tumor Segmentation using MONAI (BraTS Dataset)

This project implements a 3D U-Net using MONAI to segment brain tumors from MRI volumes in the BraTS dataset. The model identifies tumor subregions (enhancing tumor, tumor core, edema) across multiple MRI modalities (T1, T1Gd, T2, FLAIR).

## 🧠 Dataset
- **Source:** BraTS 2020 via MONAI `DecathlonDataset` (`Task01_BrainTumour`)
- **Input:** 4-channel 3D MRI volumes per patient
- **Labels:** 3-class segmentation mask

## 🛠️ Tools & Frameworks
- [MONAI](https://monai.io/) for medical imaging workflows
- PyTorch for model training
- NiBabel for NIfTI handling
- Optional: Google Colab / Kaggle GPU

## 🔄 Preprocessing
- Convert all volumes to channel-first format
- Z-score intensity normalization
- Region of interest (ROI) cropping
- Data caching with `CacheDataset`
- Light augmentations (random flips, intensity shifts)

## 🧮 Model
- **Architecture:** 3D U-Net (`monai.networks.nets.UNet`)
- **Patch size:** 128×128×128
- **Batch size:** 1 (adjustable based on GPU VRAM)
- **Loss:** Dice Loss (+ optional CE)
- **Metrics:** Per-class Dice Score
- **Training time:** ~6–12 hrs on 8–16 GB GPU (50 epochs)

## ✅ Results
- Achieved Dice scores close to BraTS benchmarks
- Supports full volume inference using sliding window
- Output masks are visualized alongside MRI slices

## 📁 Structure
```
├── data/             # Instructions or symbolic link to BraTS dataset
├── src/              # Training & evaluation scripts
├── checkpoints/      # Saved model weights
├── results/          # Sample predictions
└── README.md         # Project summary
```

## 🚀 Quickstart
```bash
pip install -r requirements.txt

# Download dataset via MONAI
python scripts/download_brats.py

# Train
python src/train.py

# Validate
python src/evaluate.py
```

## 💡 Notes
- Trained on Google Colab (T4, 16 GB) in ~8 hours
- Optimization tricks: ROI cropping, AMP, caching
- Can run with fewer samples or patch-based training on low-end GPUs
