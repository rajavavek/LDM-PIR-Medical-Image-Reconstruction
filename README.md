# Lightweight Physics Conditioned Diffusion Multi-Model for Medical Image Reconstruction

[![GitHub](https://img.shields.io/badge/GitHub-Code-blue?logo=github)](https://github.com/rajavavek/DAugSindhi) [![Paper](https://img.shields.io/badge/Paper-DOI:10.53388/BMEC2026012-red?logo=read-the-docs)](https://doi.org/10.53388/BMEC2026012) [![Web](https://img.shields.io/badge/Web-LDM-PIR-grey)](https://colab.research.google.com/drive/1EMqx8C7PeRID5q0EKlBAnHPbmRf1Jsjx?usp=sharing) 


<p align="center" style="color:blue; font-size:24px;">
   Raja Vavekanand, Ganesh Kumar, Shakhlokhon Kurbanova
</p>

This repository presents LDM-PIR (Lightweight Physics-Conditioned Diffusion Multi-Model for Medical Image Reconstruction), a novel deep learning approach for enhancing medical imaging tasks such as MRI and CT reconstruction. The model integrates physics-based diffusion processes with a multi-task architecture, unifying denoising, inpainting, and super-resolution tasks under a shared framework. It efficiently addresses challenges in image reconstruction, such as noise, undersampling, and modality generalization.
**Key Features:**
- **Physics-Conditioned Diffusion:** Embeds acquisition physics (e.g., Fourier/Radon transforms) and noise models into the reconstruction process, ensuring images align with real-world physical constraints.
- **Multi-Task Architecture:** A shared model for multiple tasks, reducing the need for task-specific retraining and improving generalization.
- **Self-Supervised Fine-Tuning:** Adapt to new imaging modalities or conditions using a small number of annotated samples (e.g., 5 images), making it scalable for clinical applications.
- **Lightweight Design:** With just 2.1M parameters, the model achieves fast inference times (0.8s/image on GPU), suitable for real-time applications in medical imaging.

  <img width="560" height="217" alt="image" src="https://github.com/user-attachments/assets/63c5c26d-873c-4ad8-a291-c197585bb96f" />
  
The model is trained on diverse datasets like fastMRI and LIDC-IDRI, applying various noise and undersampling conditions to simulate real-world scenarios. It leverages a UNet architecture combined with Krylov subspace modules (KSM) for efficient inverse problem solving. The forward and reverse diffusion processes guide the image restoration, incorporating the measurement process to improve fidelity.
<img width="806" height="433" alt="image" src="https://github.com/user-attachments/assets/e41168b8-397e-4b9b-b378-712f1428fc3d" />

The model achieves state-of-the-art performance across MRI and CT tasks, outperforming traditional iterative methods and task-specific deep learning models in terms of PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index). It handles extreme undersampling and low SNR conditions, making it robust for clinical settings where data acquisition is constrained by time or radiation dose.

<img width="906" height="273" alt="image" src="https://github.com/user-attachments/assets/244f5d2e-492b-4875-a887-93a14ca767eb" />

LDM-PIR represents a significant advancement in medical image reconstruction by combining physics-conditioned diffusion with a lightweight, multi-task architecture. The model excels in handling various imaging tasks, such as denoising, inpainting, and super-resolution, while maintaining efficiency and high image quality. With its self-supervised fine-tuning capabilities, it adapts to new tasks with minimal annotated data, making it highly suitable for real-world clinical applications. The results demonstrate its potential for improving medical imaging, particularly in environments constrained by data acquisition limitations.

