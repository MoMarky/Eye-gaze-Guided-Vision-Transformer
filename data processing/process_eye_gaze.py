import matplotlib.pyplot as plt
import cv2
import pydicom
import pickle as pkl
import SimpleITK as sitk
import os
import os.path as op
import numpy as np
from tqdm import tqdm
import json
from matplotlib import pyplot, image
import numpy
import random

def read_gaze_pkl():
 	imgs_gaze = pkl.load(open(r'Path for cxr_gaze_data.pkl', 'rb'))

 	save_csv_root = "Path for saving csv file"
	os.makedirs(save_csv_root, exist_ok=True)

 	for img_name in imgs_gaze:
 		dcm_path = op.join(r'G:\archive\siim\dicom-images-train', img_name)
        data = sitk.ReadImage(dcm_path)
        img = toGray8(sitk.GetArrayFromImage(data)[0, :])
        (h, w) = img.shape

        gaze_data = imgs_gaze[img_name]

        save_csv_path = op.join(save_csv_root, '{}.csv'.format(img_name.split('.')[0]))
        list_writer = csv.writer(open(save_csv_path, 'a+', encoding='utf-8', newline=""))

 		for gaze_item in gaze_data:
 			gaze_x = int(gaze_item[0]*w)
            gaze_y = int(gaze_item[1]*h)

            list_writer.writerow([gaze_x, gaze_y])

        list_writer.close()
