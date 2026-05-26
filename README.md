# AI-Generated vs Real Product Image Classification

> Binary image classification to detect AI-generated product listings in e-commerce — achieving an **F1-score of 0.95** using transfer learning with EfficientNet-B3 and ConvNeXt-Base.

---

## Overview

As AI-generated imagery becomes increasingly indistinguishable from real photographs, platforms like Etsy face a growing challenge: fraudulent product listings using fake images. This project builds a deep learning pipeline to automatically distinguish **AI-generated product images** from **authentic photographs**, using a real-world e-commerce dataset provided by Etsy.

The work covers the full ML pipeline — from data preprocessing and augmentation through model fine-tuning, ensembling, and threshold optimisation — with a focus on maximising F1-score on an imbalanced binary classification task.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV, torchvision |
| ML Utilities | Scikit-learn |
| Architectures | EfficientNet-B3, ConvNeXt-Base |

---

## Methodology

### 1. Data Preprocessing
- Loaded and inspected the Etsy e-commerce dataset (AI-generated vs real product images)
- Applied class-imbalance analysis to understand label distribution
- Normalised images and applied standard resizing for model input

### 2. Data Augmentation
- **CutMix augmentation** — mixes patches from two images to improve generalisation and reduce overfitting
- Standard augmentations: random flips, colour jitter, random crops

### 3. Model Architecture & Fine-Tuning
- Fine-tuned two pretrained transfer learning backbones:
  - **EfficientNet-B3** — efficient scaling of depth, width, and resolution
  - **ConvNeXt-Base** — modern CNN architecture competitive with Vision Transformers
- Final classification head replaced and trained on the binary task

### 4. Ensemble & Post-Processing
- **Snapshot Ensembling** — saves model checkpoints at multiple learning rate cycle points and averages predictions, improving robustness without additional training cost
- **Test-Time Augmentation (TTA)** — applies augmentations at inference and averages predictions to reduce variance
- **Threshold Optimisation** — tuned the decision boundary away from the default 0.5 to maximise F1-score on the validation set, particularly important given class imbalance

### 5. Evaluation
- Primary metric: **F1-Score** (harmonic mean of precision and recall)
- Secondary metrics: Precision, Recall, Confusion Matrix
- Focus on minimising **false negatives** (missed AI-generated images) in the fraud-detection context

---

## Key Takeaways

- Transfer learning with modern CNN architectures is highly effective even on domain-specific binary tasks
- Ensemble methods (snapshot ensembling + TTA) provide meaningful gains with no additional labelling cost
- Threshold optimisation is an underrated post-processing step that significantly improved F1 on the imbalanced dataset
- CutMix augmentation helped prevent overfitting on visually similar classes


## Author

**Soheb Khan**
MSc Computing (Artificial Intelligence) — Dublin City University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/soheb-khan-101519311/)
