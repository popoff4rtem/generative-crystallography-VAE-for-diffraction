# Generative Crystallography: VAE for Diffraction

Official repository for the paper on **structuring and analyzing latent spaces of Variational Autoencoders (VAE)** for 2D neutron diffraction data.

📂 **Dataset (Hugging Face):**  
https://huggingface.co/datasets/popoff4rtem/Diffraction-Dataset-For-Generative-Models  

📄 (Paper link will be added after publication)

---

## 🔬 About the Project

This work investigates the **geometry and structure of latent spaces in VAE models** trained on 2D neutron diffraction patterns.

We study:

- The effect of **KL divergence** (with free bits regularization)
- The influence of additional latent regularizers:
  - **Pull component** (reduces intra-class variance)
  - **Push component** (increases inter-class separation)
- Differences between:
  - **Unconditional VAE**
  - **Conditional VAE**

We demonstrate that:

- In the **Unconditional VAE**, strong clustering requires high KL values, which breaks latent continuity.
- In the **Conditional VAE**, clustering emerges more naturally, but excessive KL leads to degradation of generative quality.
- VAE is better interpreted as a **structured latent encoder** rather than a standalone unconditional generator for multi-class physical data.

The repository also includes:
- Vision Transformer (ViT) based VAE architectures
- CNN-based baselines
- Latent-space clustering evaluation tools
- Physical diffraction metrics
- Generative quality metrics (PSNR, SSIM, Inception Score, per-class FID)
- A latent diffusion transformer (DiT) trained on VAE latent space

![Spatial Vit VAE Architecture](Figs/Figure%203.png)


---

# 📁 Repository Structure

---

## `Experiments/`

Contains all experimental training configurations and notebooks used in the paper.

### CNN-based models
- `CNN_UnConditional_VAEs.py`  
  CNN-based VAE architectures.

### Vector latent ViT VAE
- `UnConditional_Vector_ViT_VAE_model.py`  
- `Conditional_Vector_ViT_VAE_model.py`  
- `Conditional_Vector_ViT_VAE_train.ipynb`  
- `Conditional_Vector_ViT_VAE_tests.ipynb`

### Spatial (multi-dimensional) latent ViT VAE
- `Sapatial_UnConditional_ViT_VAE_model.py`  
- `Spatial_UnConditional_ViT_VAE_train.ipynb`  
- `Spatial_Conditional_ViT_VAE_model.py`  
- `Spatial_Conditional_ViT_VAE_train.ipynb`  
- `Sapatial_ViT_VAE_tests.ipynb`

### Diffusion Transformer in latent space
- `dit_model.py`  
- `Latent_DiT_train.ipynb`

### Metrics classifier
- `Classifier_for_ViT_metrics.ipynb`  
  Training of ResNet-18 classifier used for:
  - Inception (ResNet) Score
  - FID computation

⚠️ **If you want to reproduce all training experiments, use the notebooks in `Experiments/`.**

---

## `models/`

Contains pretrained weights:

- VAE models (various configurations)
- ResNet-18 classifier trained on diffraction dataset
- Latent diffusion model weights

---

## Results folders

Contain visualizations and experimental outputs:

- `Results Vector VAE/`
- `VAE Results/`
- `VAE_tests_diffraction_data_d_space/`
- `ViT VAE/`

These include:

- Latent space visualizations (t-SNE projections)
- Clustering metric plots
- Generated diffraction samples
- Reconstruction comparisons
- Physical metric evaluations

---

# 📊 Evaluation & Metrics

The repository provides a full evaluation pipeline.

---

## Latent Space Evaluation

`tests.py` contains:

- `LatentSpaceBasicStats`  
  Basic statistics of latent representations

- `LatentSpaceVisualizer`  
  Dimensionality reduction + visualization

- `LatentSpaceClusteringEval`  
  Clustering metrics:
  - Silhouette Score
  - Calinski-Harabasz Index
  - Davies-Bouldin Index
  - Linear Probe Accuracy
  - Mutual Information Proxy

---

## Generative Quality Metrics

`Generative_Quality_Metrics.py` contains:

- `PSNR_SSIM_Evaluator`  
  PSNR & SSIM with bootstrap confidence intervals

- `InceptionScoreEvaluator`  
  ResNet-based Inception Score with bootstrap

- `FID_Evaluator`  
  **Per-class FID** + mean FID with bootstrap

> We use per-class FID because feature distributions of different crystal classes may occupy very different regions of feature space, making global averaging misleading.

---

## Physical Diffraction Metrics

- `Diffraction_metrics_Calculator.py`
- `physical_diffraction_metrics.py`

These compute domain-specific physical divergence metrics between real and generated diffraction patterns.

---

# 📦 Dataset

The dataset used in the paper is available at:

https://huggingface.co/datasets/popoff4rtem/Diffraction-Dataset-For-Generative-Models

To create dataloaders:

```
from create_dataloaders import create_dataloaders
```
The script:

- Downloads / reads the dataset
- Prepares PyTorch dataloaders
- Applies required preprocessing

---

# 🧠 Training (Beta API)

The following files provide modular training pipelines (currently in beta):

- `Spatial_loss_functions.py`
- `Vector_loss_functions.py`
- `spatial_trainer.py`
- `decoder_finetuner.py`
  
⚠️ For exact reproduction of paper results, use the notebooks in Experiments/.

---

# 🚀 Tutorial

`tutorial.ipynb`

Step-by-step guide on how to:

- Load dataset
- Train VAE
- Evaluate latent space
- Compute generative metrics

---
  
# 🧩 Key Features of This Work

- Vision Transformer VAE for diffraction data
- Structured spatial latent space (not just flat vector)
- Channel-wise free bits regularization
- Pull/Push latent structuring components
- Conditional VAE analysis
- Per-class FID evaluation
- Latent diffusion transformer integration

---
# 📖 Citation
If you use this code in your research, please cite the corresponding article:
```
bibtex@article{Popoff2025SwinWNet,
  title   = {Design and Analysis of the Latent Space for VAE-Based Neutron Diffraction Data Generation},
  authors  = {Popov A.I., Antropov N.O., Smirnov A.A., Kravtsov E.A., Ogorodnikov I.N.},
  journal = {PRX Intelligence},
  year    = {2025}
}
```

---

# ⚠️ Disclaimer
This project is provided for research purposes only.

# 🤝 Acknowledgements
This work builds upon advances in:

* Transformer-based Generative Models
* Latent Design
* Physics-aware machine learning

---

# 📬 Contact
For questions or collaboration:
Artem Popov popoff4rtem@gmail.com

