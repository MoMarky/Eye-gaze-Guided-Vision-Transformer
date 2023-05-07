# EG-ViT: Eye-gaze-Guided-Vision-Transformer for Rectifying Shortcut Learning

<div align="center">
    <img src="/res/title.png">
</div>

This repository contains the official PyTorch training and evluation codes for EG-ViT (Eye-gaze-Guided-Vision-Transformer for Rectifying Shortcut Learning).


# Pipeline

<div align="center">
    <img src="/res/pipeline.jpg">
</div>

The eye-gaze points are collected and pre-processed to generate the eye-gaze heatmap. Then, the original image and the corresponding heatmap are randomly cropped with a smaller size. The cropped image is divided into several image patches for patch embedding. The eye-gaze mask is then applied to screen out the patches that may be unrelated to pathology and radiologists’ interest. The masked image patches (green rectangle) are treated as input to the transformer encoder layer. Note that to maintain the information of all patches including those been masked, we add an additional residual connection (highlighted by the red arrow) from the input and the last encoder layer.


# Data Collection System

<div align="center">
    <img src="/res/overall.png">
</div>

We design a new eye-gaze collection system basen on a public DICOM viewer. It supports the radiologists to adjust the image freely while completing the eye-gaze data collection. We modified the source code of the project so that the software can output real-time status parameters of medical images, such as position, zoom scale, contrast parameters, etc. The code of collection system can be found at [eye-tracking-system-for-radiologists
](https://github.com/MoMarky/eye-tracking-system-for-radiologists/tree/main).


# Installatoin

The following descriptions all refer to the EG-ViT project. If you want to build your own eye-gaze collection system, please refer to our code above. In addition, we provide a RELEASE package, you can download and install it directly, and we also reserve some interfaces for you to DIY.


**1. System.**

The data collection system is running on a Windows 10 platform. The EG-ViT project is running on a Linux 16.04 platform. 

**2. Create an environment.**

`conda env create -f environment.yml`



# Data Processing

**2. Prepare the dataset.**

We use two public datasets INbreast and SIIM-ACR for training and testing. The source image files of INbreast dataset can be found at [INbreast Kaggle](https://www.kaggle.com/datasets/martholi/inbreast?resource=download), and the source images of SIIM-ACR can be found at [SIIM-ACR-Gaze](https://github.com/HazyResearch/observational).

For detailed data pre-pocessing, 


**2. Training on INbreast.**

`python main_inbreast.py`

**3. Training on SIIM-ACR.**

`python main_siim.py`
