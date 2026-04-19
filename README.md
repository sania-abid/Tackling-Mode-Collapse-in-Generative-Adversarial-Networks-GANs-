#  Tackling Mode Collapse in GANs: DCGAN vs WGAN‑GP

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue)](https://www.kaggle.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-App-orange)](https://gradio.app/)

> A complete implementation of **DCGAN** and **Wasserstein GAN with Gradient Penalty (WGAN‑GP)** for anime face generation.  
> **Goal:** Demonstrate how advanced loss functions eliminate mode collapse and improve image diversity.

---

##  Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Training Details](#training-details)
- [Results](#results)
- [How to Run](#how-to-run)
- [Gradio App](#gradio-app)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [References](#references)
- [License](#license)

---

## Overview

Generative Adversarial Networks (GANs) suffer from **mode collapse** – the generator produces only a few types of images. This project implements two popular GAN variants on **64×64 anime faces**:

- **DCGAN** (baseline) – uses convolutional layers, BatchNorm, and BCE loss.
- **WGAN‑GP** (improved) – replaces the discriminator with a **critic** (no sigmoid), uses **Wasserstein loss** and a **gradient penalty** (λ=10) to enforce Lipschitz constraint.

The results clearly show that WGAN‑GP generates more diverse and higher‑quality images, effectively reducing mode collapse.

---

## Key Features

-  **Dual GPU support** (Kaggle T4×2) with `DataParallel`
- **Mixed precision training** (`torch.cuda.amp`) for speed and memory efficiency
- **Complete training pipeline** for both DCGAN and WGAN‑GP
- **Checkpoint saving** every 5 epochs
- **Loss curves** (Generator, Discriminator/Critic, Gradient Penalty)
- **Interactive Gradio app** with:
  - Single model generation (adjustable seed, temperature, number of images)
  - Side‑by‑side comparison (same latent vector)
  - Diversity analysis (64‑image grid to visually detect mode collapse)
  - Training metrics (loss curves + optional FID placeholder)

---

## Dataset

Two datasets are supported (auto‑detected):

1. **Anime Faces (64×64)** – [Kaggle Link](https://www.kaggle.com/datasets/soumikrakshit/anime-faces)
2. **Pokémon Sprites** – [Kaggle Link](https://www.kaggle.com/datasets/jackemartin/pokemon-sprites)

Images are:
- Resized to `64×64`
- Normalized to `[-1, 1]`
- Subset of 15,000 images used for faster training (adjustable)

---

## Model Architectures

### DCGAN

| Generator | Discriminator |
|-----------|----------------|
| Transposed Conv (4×4) | Conv2D (4×4) |
| BatchNorm2d | BatchNorm2d |
| ReLU | LeakyReLU (0.2) |
| Output: Tanh | Output: Sigmoid + BCEWithLogitsLoss |

### WGAN‑GP

| Generator (same as DCGAN) | Critic (no sigmoid) |
|---------------------------|----------------------|
| Transposed Conv | Conv2D |
| BatchNorm2d | **InstanceNorm2d** |
| ReLU | LeakyReLU (0.2) |
| Output: Tanh | Output: scalar (no activation) |

**Loss functions:**
- Wasserstein loss: `C(fake).mean() - C(real).mean()`
- Gradient penalty: `λ * (‖∇D(interpolated)‖₂ - 1)²` with `λ = 10`
- **Critic updates per generator update = 5**

---

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 0.0002 |
| Betas | (0.5, 0.999) |
| Batch size | 32 (adjustable for memory) |
| Epochs (DCGAN) | 10 |
| Epochs (WGAN‑GP) | 10 |
| Noise dimension (z) | 100 |
| Mixed precision | Enabled (`torch.cuda.amp`) |
| Gradient penalty λ | 10 |
| Critic iterations | 5 |

All training was performed on **Kaggle GPU T4×2** with memory optimizations (no OOM).

---

## Results

### Generated Samples (after 10 epochs)

| DCGAN | WGAN‑GP |
|-------|---------|
| Some mode collapse visible (repetitive faces) | More diverse expressions, hair colours, accessories |

### Loss Curves

- **DCGAN**: Generator and discriminator losses oscillate; mode collapse risk.
- **WGAN‑GP**: Critic loss correlates with image quality; gradient penalty stays around 10.

### Mode Collapse Detection

- **DCGAN 64‑image grid** often shows repeated faces.
- **WGAN‑GP 64‑image grid** exhibits much greater variation.

>  **Observation:** WGAN‑GP successfully reduces mode collapse thanks to the Wasserstein distance and gradient penalty.

---

## How to Run

### Option 1: Run on Kaggle (recommended)

1. Create a new Kaggle notebook with **GPU T4×2** accelerator.
2. Add the dataset(s):
   - [Anime Faces](https://www.kaggle.com/datasets/soumikrakshit/anime-faces)
   - (Optional) [Pokémon Sprites](https://www.kaggle.com/datasets/jackemartin/pokemon-sprites)
3. Copy all code cells from the provided Jupyter notebook into the Kaggle notebook.
4. Run cells in order (1 → 2 → 3 → ... → 11).
5. The Gradio app will launch with a public link.

### Option 2: Run Locally (requires GPU)

git clone https://github.com/your-username/gan-mode-collapse.git
cd gan-mode-collapse
pip install torch torchvision gradio matplotlib tqdm pillow
jupyter notebook GAN_DCGAN_WGAN_GP.ipynb

# **Gradio App**

Launch the app after training by running the last notebook cell. It provides an interactive interface for generating and comparing results.

| Tab | Functionality |
|-----|--------------|
| Generate Images | Choose model, set seed, temperature, and number of images |
| Compare Models | Same latent vector → side-by-side outputs from both models |
| Diversity Analysis | Generate 64 images in one grid to visually inspect mode collapse |
| Training Metrics | Loss curves + optional placeholder for FID score |

![Gradio App Screenshot](outputs/gradio_screenshot.png)  
*Add your own screenshot after running the app.*

---

# **Project Structure**

├── GAN_DCGAN_WGAN_GP.ipynb      # Complete notebook
├── outputs/                     # Generated images, loss curves, trained models
│   ├── dcgan/
│   ├── wgan/
│   ├── loss_curves.png
│   ├── real_samples.png
│   ├── side_by_side.png
│   ├── dcgan_final.pth
│   └── wgan_final.pth
├── checkpoints/                 # Intermediate training checkpoints
├── README.md
└── requirements.txt

## Requirements

```text
torch>=2.0.0
torchvision>=0.15.0
gradio>=3.40.0
matplotlib>=3.5.0
tqdm>=4.64.0
Pillow>=9.0.0
numpy>=1.21.0
```

 ## Install dependencies:

pip install -r requirements.txt
```

---

## References

- DCGAN Paper (Radford et al., 2015)  
- WGAN-GP Paper (Gulrajani et al., 2017)  
- PyTorch DCGAN Tutorial



## License

MIT License — free to use, modify, and distribute for educational purposes.
