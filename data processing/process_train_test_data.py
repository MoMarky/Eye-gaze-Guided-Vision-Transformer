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


def gen_train_test_split():
	all_img_root = r'root path for your imgs'

	img_list = os.listdir(all_img_root)

	random.shuffle(img_list)

	test_list = random.sample(img_list, (len(img_list)//10)*2)

	train_list = []
	for img_name in img_list:
		if img_name not in test_list:
			train_list.append(img_name)

	data_split = {'train': train_list, 'test': test_list}

	json.dump(data_split,
                  open(r'/data_split.json', 'w'))


def gen_trantest_csv():
    all_img_root = r'root path for your imgs'
    split_dict = json.load(open(r'data_split.json', 'r'))

    # Training set
    save_train_csv_path = 'train_list.csv'
    train_list_writer = csv.writer(open(save_train_csv_path, 'a+', encoding='utf-8', newline=""))

    for img_name in split_dict['train']:
    	img_path = op.join(all_img_root, img_name)
    	img_label = img_name.split('.')[0][-1]

    	train_list_writer.writerow([img_path, label])

    # Test set
    save_test_csv_path = 'test_list.csv'
    test_list_writer = csv.writer(open(save_test_csv_path, 'a+', encoding='utf-8', newline=""))

    for img_name in split_dict['test']:
    	img_path = op.join(all_img_root, img_name)
    	img_label = img_name.split('.')[0][-1]

    	test_list_writer.writerow([img_path, label])


if __name__ == '__main__':
	gen_train_test_split()
	
	# gen_trantest_csv()