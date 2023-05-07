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

The EG-ViT project is running on a Linux 16.04 platform. 

**2. Create an environment.**

`conda env create -f environment.yml`



# Data Processing

**1. Prepare the dataset.**

We use two public datasets INbreast and SIIM-ACR for training and testing. The source image files of INbreast dataset can be found at [INbreast Kaggle](https://www.kaggle.com/datasets/martholi/inbreast?resource=download), and the source images of SIIM-ACR can be found at [SIIM-ACR-Gaze](https://github.com/HazyResearch/observational).


**2. Convert DICOM file to JPG file.**

Implemented by .\data processing\ [dcm_to_jpg.py]().

Change your path of DICOM files and path for saving jpgs.

    dcm_root_path = "your dcm root path" 
    jpg_root_path = "your saving root path" 


**3. Process eye-gaze data.**

Implemented by .\data processing\ [process_train_test_data.py]().

Function 'gen_train_test_split()' is to generate the random split of train and test split. Function 'gen_trantest_csv()' is to generate the csv file for exp_settings. 

You also have to change the corresponding path for data saving.

We also public our processed dataset, please download it at [Baidu Netdisk](https://pan.baidu.com/s/1nLpWgE6mFOuc4gvWSylk9Q) with extract key: 'c5dt'.

**4. Plot the eye-gaze heatmap.**

Implemented by .\data processing\ [draw_heatmap.py]().

Change the 'img_root_path', 'gaze_root_path' and 'Save root'.

	img_root_path = "root path for your img"
    gaze_root_path = "root path of gaze.csv files"
    
    array_save_root = r'Save root'
    heat_map_save_root = r'Save root'

# Training and Evaluation

**1. Training / Evaluation on INbreast.**

`python main_inbreast.py`

**2. Training / Evaluation on SIIM-ACR.**

`python main_siim.py`


