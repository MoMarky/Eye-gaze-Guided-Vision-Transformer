import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import os.path as op
import cv2
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import logging
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import torchvision.transforms as transforms
import timm
import torch
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from timm.optim.optim_factory import create_optimizer
from types import SimpleNamespace
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
import numpy as np
from timm.models.vision_transformer import VisionTransformer
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch.distributed as dist
import matplotlib.pyplot as plt

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

"""No mask"""
class MyDataset_INbreast_Train(Dataset):
    def __init__(self, args):
        self.data_list = open(args.train_data, 'r').readlines()
        self.data_size = len(self.data_list)
        self.gaze_input_norm = args.gaze_input_norm
        self.gaze_mask_num = args.gaze_mask_num
        self.to_tensor = ToTensorV2()
        self.patch_num = args.img_size // args.patch_size
        self.transform1 = A.Compose([
            A.GaussNoise(p=0.2),
            A.Resize(height=224, width=224, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            # A.ElasticTransform(p=0.2),
            A.Flip(p=0.2),
            # A.ElasticTransform(p=0.2),
            # A.Lambda(p=0.2),
            # A.MaskDropout(p=0.2),
            # A.OpticalDistortion(p=0.2),
            # A.CLAHE(p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.3, 0.3, 0.3), p=1.0),
            # ToTensorV2(),
        ])

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):

        data_index = idx % self.data_size
        item = self.data_list[data_index]

        img_path, label = item.rstrip().split(',')
        label = int(label)

        """get bk"""
        src_img = cv2.imread(img_path)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

        """after transform"""
        transformed = self.transform1(image=src_img)
        img_after = transformed['image']  # (224, 224)
        tensor_trans = self.to_tensor(image=img_after)
        img_after = tensor_trans['image']

        sample = torch.tensor(img_after)
        label = torch.tensor(label)

        return sample, label, torch.tensor(idx), torch.tensor(idx)
class MyDataset_INbreast_Test(Dataset):
    def __init__(self, args):
        self.data_list = open(args.test_data, 'r').readlines()
        self.data_size = len(self.data_list)
        self.gaze_input_norm = args.gaze_input_norm
        self.to_tensor = ToTensorV2()
        self.patch_num = args.img_size // args.patch_size
        self.transform1 = A.Compose([
            A.Resize(height=224, width=224, p=1.0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.3, 0.3, 0.3), p=1.0),
        ])

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):

        data_index = idx % self.data_size
        item = self.data_list[data_index]

        img_path, label = item.rstrip().split(',')
        label = int(label)

        """get bk"""
        src_img = cv2.imread(img_path)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

        """after transform"""
        transformed = self.transform1(image=src_img)
        img_after = transformed['image']  # (224, 224)

        tensor_trans = self.to_tensor(image=img_after)
        img_after = tensor_trans['image']

        sample = torch.tensor(img_after)
        label = torch.tensor(label)

        return sample, label, torch.tensor(idx), torch.tensor(idx)


"""data aug version v3"""
class MyDataset_INbreast_Gaze_Mask49_Train_v3(Dataset):
    def __init__(self, args):
        self.data_list = open(args.train_data, 'r').readlines()
        self.data_size = len(self.data_list)
        self.gaze_input_norm = args.gaze_input_norm
        self.gaze_mask_num = args.gaze_mask_num
        self.to_tensor = ToTensorV2()
        self.patch_num = args.img_size // args.patch_size
        self.transform1 = A.Compose([
            A.GaussNoise(p=0.2),
            A.Resize(height=224, width=224, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            # A.ElasticTransform(p=0.2),
            A.Flip(p=0.2),
            # A.ElasticTransform(p=0.2),
            # A.Lambda(p=0.2),
            # A.MaskDropout(p=0.2),
            # A.OpticalDistortion(p=0.2),
            # A.CLAHE(p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.3, 0.3, 0.3), p=1.0),
            # ToTensorV2(),
        ])

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):

        data_index = idx % self.data_size
        item = self.data_list[data_index]

        img_path, label = item.rstrip().split(',')
        label = int(label)

        img_name = img_path.split('/')[-1]
        org_img_mid = img_name.split('r')[0][:-1]
        roi_info = img_name.split('r')[1].split('.')[0]
        tmp = roi_info.split('_')

        if label == 1 or label == 2:  # r0_1053_1024_2077_736_230_827_343_c0_e0_b0_1_1
            npy_name = '{}.npy'.format(org_img_mid)
            roi_x, roi_y, roi_x1, roi_y1 = int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3])
            box_x, box_y, box_x1, box_y1 = int(tmp[4]), int(tmp[5]), int(tmp[6]), int(tmp[7])
        elif label == 0:  # r31_2030_1055_3054_c0_e0_0_0
            npy_name = '{}.npy'.format(org_img_mid)
            roi_x, roi_y, roi_x1, roi_y1 = int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3])
        gaze_mask_path = r'Gaze_heatmap_path_of_INbreast_dataset/{}'.format(npy_name)

        """get bk"""
        src_img = cv2.imread(img_path)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        bk_mask = np.load(gaze_mask_path)
        roi_mask = bk_mask[roi_y:roi_y1, roi_x:roi_x1]  # .T
        # lowbound = np.mean(bk_mask[bk_mask > 0])
        # bk_mask[bk_mask < lowbound] = 0

        norm_img = np.zeros(roi_mask.shape)
        cv2.normalize(roi_mask, norm_img, 0, 255, cv2.NORM_MINMAX)
        heat_img = np.asarray(norm_img, dtype=np.uint8)

        """after transform"""
        transformed = self.transform1(image=src_img, mask=heat_img)
        img_after = transformed['image']  # (224, 224)
        heat_img_after = transformed['mask']  # (224, 224)
        heat_img_196 = cv2.resize(heat_img_after, (self.patch_num, self.patch_num))  # (14, 14)
        tensor_trans = self.to_tensor(image=img_after, mask=heat_img_196)
        img_after = tensor_trans['image']
        heat_img_196 = tensor_trans['mask']

        sample = torch.tensor(img_after)
        heat_img_196 = torch.tensor(heat_img_196).flatten(0)
        if self.gaze_input_norm:
            heat_img_196 = (heat_img_196 - heat_img_196.min()) / (heat_img_196.max() - heat_img_196.min() if heat_img_196.max() - heat_img_196.min() != 0 else 1.0)
        label = torch.tensor(label)

        """generate mask"""
        mask_indexes = torch.argsort(heat_img_196, descending=True)[:self.gaze_mask_num]
        new_zeros = torch.zeros((self.patch_num*self.patch_num), dtype=torch.uint8)
        new_gaze_mask = new_zeros == 1
        new_gaze_mask[mask_indexes] = True

        if new_gaze_mask.sum() != self.gaze_mask_num:
            print('new_gaze_mask.sum() != args.gaze_mask_num')
            exit(0)

        return sample, label, torch.tensor(idx), new_gaze_mask, heat_img_196
class MyDataset_INbreast_Gaze_Mask49_Test_v3(Dataset):
    def __init__(self, args):
        self.data_list = open(args.test_data, 'r').readlines()
        self.data_size = len(self.data_list)
        self.gaze_input_norm = args.gaze_input_norm
        self.to_tensor = ToTensorV2()
        self.patch_num = args.img_size // args.patch_size
        self.transform1 = A.Compose([
            A.Resize(height=224, width=224, p=1.0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.3, 0.3, 0.3), p=1.0),
        ])

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):

        data_index = idx % self.data_size
        item = self.data_list[data_index]

        img_path, label = item.rstrip().split(',')
        label = int(label)

        img_name = img_path.split('/')[-1]
        org_img_mid = img_name.split('r')[0][:-1]
        roi_info = img_name.split('r')[1].split('.')[0]
        tmp = roi_info.split('_')

        if label == 1 or label == 2:  # r0_1053_1024_2077_736_230_827_343_c0_e0_b0_1_1
            npy_name = '{}.npy'.format(org_img_mid)
            roi_x, roi_y, roi_x1, roi_y1 = int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3])
            box_x, box_y, box_x1, box_y1 = int(tmp[4]), int(tmp[5]), int(tmp[6]), int(tmp[7])
        elif label == 0:  # r31_2030_1055_3054_c0_e0_0_0
            npy_name = '{}.npy'.format(org_img_mid)
            roi_x, roi_y, roi_x1, roi_y1 = int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3])
        gaze_mask_path = r'Gaze_heatmap_path_of_INbreast_dataset/{}'.format(npy_name)

        """get bk"""
        src_img = cv2.imread(img_path)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        bk_mask = np.load(gaze_mask_path)
        roi_mask = bk_mask[roi_y:roi_y1, roi_x:roi_x1]  # .T

        norm_img = np.zeros(roi_mask.shape)
        cv2.normalize(roi_mask, norm_img, 0, 255, cv2.NORM_MINMAX)
        heat_img = np.asarray(norm_img, dtype=np.uint8)

        """after transform"""
        transformed = self.transform1(image=src_img, mask=heat_img)
        img_after = transformed['image']  # (224, 224)
        heat_img_after = transformed['mask']  # (224, 224)

        if self.gaze_input_norm:
            heat_img_after = cv2.resize(heat_img_after, (14, 14))
            heat_img_after = (heat_img_after - heat_img_after.min()) / (heat_img_after.max() - heat_img_after.min())
            # norm_heat_img_after = torch.tensor(norm_heat_img_after)

        tensor_trans = self.to_tensor(image=img_after, mask=heat_img_after)
        img_after = tensor_trans['image']
        heat_img_after = tensor_trans['mask']

        heat_img_after = torch.tensor(heat_img_after)
        if self.gaze_input_norm:
            heat_img_after = heat_img_after.flatten(0)
        sample = torch.tensor(img_after)
        label = torch.tensor(label)

        return sample, label, torch.tensor(idx), heat_img_after

