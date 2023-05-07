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

