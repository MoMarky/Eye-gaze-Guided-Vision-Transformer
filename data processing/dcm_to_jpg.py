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


def toGray8(img):
    result = img.copy()
    cv2.normalize(img, result, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)


def change_dcm_to_img():

	dcm_root_path = "your dcm root path"
	jpg_root_path = "your saving root path"

	for dcm_name in tqdm(os.listdir(root_path)):
		data = sitk.ReadImage(op.join(root_path, dcm_name))
		img = toGray8(sitk.GetArrayFromImage(data)[0, :])

		new_name = dcm_name.replace('.dcm', '.jpg')
		cv2.imwrite(op.join(jpg_root_path, new_name), img)


if __name__ == '__main__':
	change_dcm_to_img()
