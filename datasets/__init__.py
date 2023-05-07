from .INbreast import *


def build_dataset(data_name, args):

    """INbreast"""
    if 'INbreast' in data_name:
        if data_name == 'INbreast_v3_gaze':
            training_dataset = MyDataset_INbreast_Gaze_Mask49_Train_v3(args)
            testing_dataset = MyDataset_INbreast_Gaze_Mask49_Test_v3(args)
    	if data_name == 'INbreast_v3_org':
            training_dataset = MyDataset_INbreast_Train(args)
            testing_dataset = MyDataset_INbreast_Test(args)

    if 'SIIM' in data_name:
        if data_name == 'SIIM_v3_org':
        	training_dataset = MyDataset_SIIM_Train(args)
            testing_dataset = MyDataset_SIIM_Test(args)
        if data_name == 'SIIM_v3_gaze':
            training_dataset = MyDataset_SIIM_Gaze_Mask49_Train_v3(args)
            testing_dataset = MyDataset_SIIM_Gaze_Mask49_Test_v3(args)
            