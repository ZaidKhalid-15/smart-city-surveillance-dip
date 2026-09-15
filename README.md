# Smart City Surveillance System — Digital Image Processing CCP

**Course:** CS 406 — Digital Image Processing
**Project:** Enhancing Smart City Surveillance through Digital Image Processing: Filtering, Restoration, Segmentation, and Compression Techniques
**Author:** Zaid Khalid — Department of Computer Science, University of Management and Technology, Lahore, Pakistan

## Overview

This project implements and evaluates a complete six-stage digital image processing pipeline addressing the problem of poor image quality, noise, and variable lighting in smart-city public-space surveillance systems (modeled on Islamabad's CCTV network). Using a Custom Face Mask Dataset (14,536 images across three classes) as a practical proxy for object identification, the pipeline covers:

1. **Spatial-domain filtering** — Mean, Median, and Laplacian filters (from-scratch and vectorized implementations)
2. **Frequency-domain filtering** — Ideal, Butterworth, and Gaussian filters (low-pass and high-pass)
3. **Restoration** — Inverse and Wiener filtering for motion-blur recovery
4. **Segmentation** — Edge-based (Sobel, Canny, Laplacian-of-Gaussian) and region-based (Region Growing, Watershed)
5. **Compression** — JPEG (DCT-based) and Wavelet compression, rate-distortion analysis
6. **Object Recognition** — Fine-tuned YOLOv8n classifier, with FPS benchmarking across all pipeline stages

Each stage is evaluated quantitatively (PSNR, SSIM, IoU, Dice coefficient, Precision/Recall/F1, FPS) and discussed in terms of the five conflicting requirements from the project brief: real-time performance vs. processing quality, resource constraints vs. performance, noise removal vs. detail preservation, compression vs. image fidelity, and security/privacy vs. system accessibility.

## Repository Structure

```
smart-city-surveillance-dip/
├── notebooks/
│   ├── 01_preprocessing.ipynb        # Dataset download, sampling, noisy pair generation
│   ├── 02_spatial_filtering.ipynb    # Mean/Median/Laplacian filters + PSNR/SSIM
│   ├── 03_frequency_filtering.ipynb  # FFT, frequency filters, restoration
│   ├── 04_segmentation.ipynb         # Edge/region-based segmentation + IoU/Dice
│   ├── 05_compression.ipynb          # JPEG/Wavelet compression + rate-distortion
│   └── 06_recognition.ipynb          # YOLOv8n fine-tuning + FPS benchmarking
├── report/
│   └── main.tex                      # Full IEEE-format report source
├── results/
│   ├── figures/                      # All generated plots and comparison images
│   └── tables/                       # CSV result tables
└── README.md
```

## How to Run

All notebooks are designed for **Google Colab** and expect Google Drive to be mounted for persistent storage between notebooks.

1. **Run `01_preprocessing.ipynb` first.** This downloads the Face Mask Dataset from Kaggle (requires a Kaggle API token), samples and preprocesses the images, generates synthetic noisy test pairs, and saves everything to `MyDrive/smart-city-surveillance-dip/` in Google Drive.
2. **Run `02` through `05` in any order.** Each mounts Drive and loads the data saved by notebook 01 — no re-downloading needed.
3. **Run `06_recognition.ipynb` last**, on a GPU runtime (Runtime → Change runtime type → T4 GPU in Colab), since it fine-tunes a YOLOv8n classifier.

### Requirements

- Python 3.10+
- `opencv-python`, `numpy`, `scipy`, `scikit-image`, `PyWavelets`, `pandas`, `matplotlib`
- `ultralytics` (for notebook 06 only)
- A Kaggle account and API token (for notebook 01 only)

## Key Results Summary

| Stage | Key Finding |
|---|---|
| Spatial Filtering | Median filtering improves PSNR by +15.8 dB over unfiltered salt-and-pepper noise |
| Restoration | Wiener filtering recovers +13.2 dB over naive inverse filtering |
| Segmentation | Mask-representation correction raised Canny's IoU from 0.080 to 0.384 |
| Compression | Wavelet compression achieves higher SSIM than JPEG at comparable compression ratios |
| Recognition | Fine-tuned YOLOv8n reaches 95% precision, recall, and F1-score at 246.9 FPS |

Full quantitative results, per-class breakdowns, and discussion are in [`report/main.tex`](report/main.tex).

## Deliverables

- **Report:** `report/main.tex` (compiled PDF available via the Overleaf link below)
- **Overleaf (editable):** _[add your Overleaf share link here]_
- **Video presentation:** _[add your video link here]_

## Author

Zaid Khalid
Department of Computer Science, University of Management and Technology, Lahore, Pakistan
f2023266005@umt.edu.pk
