## Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
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
import timm
from timm.loss import LabelSmoothingCrossEntropy
import torch
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from timm.optim.optim_factory import create_optimizer
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
import numpy as np
from timm.models.vision_transformer import VisionTransformer
from utils import *
import datasets
from models import *

def get_args_parser():
    parser = argparse.ArgumentParser('EG-ViT', add_help=False)

    parser.add_argument('--model_version', type=int, default=1)
    parser.add_argument('--res_layer', type=int, default=11)

    parser.add_argument('--forward_with', default="gaze")  # grad   gaze
    parser.add_argument('--mask_G_or_S', default="S")  # G :use mask_filter  S: no use filter
    parser.add_argument('--cross_loss_para', default='sum')  # mean   sum
    parser.add_argument('--gaze_input_norm', default=False, help='Norm gaze hm')  # False   True
    parser.add_argument('--gaze_mask_num', type=int, default=49)
    parser.add_argument('--random_use_mask', type=float, default=0.4)  # 0.3  0.5  0.7

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--warm_up', default=8, type=int)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)

    parser.add_argument('--backbone_name', default='vit_small_patch16_224', type=str)
    parser.add_argument('--pre_trained', default=True)
    parser.add_argument('--resume', default="")

    """INbreast"""
    parser.add_argument('--num_classes', default=3, type=int)
    # parser.add_argument('--data_name', default='INbreast_v3_org', type=str)
    parser.add_argument('--data_name', default='INbreast_v3_gaze', type=str)
    parser.add_argument('--train_data', default="./exp_settings/INbreast/train_list.csv")
    parser.add_argument('--test_data', default='./exp_settings/INbreast/test_list.csv')
    parser.add_argument('--output_dir', default='./save/')

    parser.add_argument('--model_save_interval', default=10, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--exp_documentation', default=exp_documentation)

    parser.add_argument('--eval', default=False)  # False  True
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--num_workers', default=2, type=int)

    return parser



def main_INbreast(args):

    os.makedirs(args.output_dir, exist_ok=True)
    # define logging
    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler("{}/log.txt".format(args.output_dir), mode='a', encoding='UTF-8'),
            logging.StreamHandler()
        ]
    )

    logging.info("##################\tconfig\t##################")
    for opt_key in vars(args).keys():
        logging.info("{}: {}".format(opt_key, vars(args)[opt_key]))

    """model"""
    if args.model_version == 1:
        logging.info("define model v1")
        mymodel = build_model_v1(args.res_layer, num_class=args.num_classes)  # add to the last layer, can change
    pretrain_weight = timm.create_model(args.backbone_name, num_classes=args.num_classes, pretrained=True).state_dict()
    mymodel.load_state_dict(pretrain_weight, strict=False)

    optimizer = create_optimizer(args, mymodel)
    scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, lr_min=1e-7, decay_rate=0.1, warmup_t=args.warm_up)
    loss_fn = LabelSmoothingCrossEntropy(0.1)

    setup_seed(args.seed)

    logging.info("define dataset")
    training_dataset, testing_dataset = datasets.build_dataset(args.data_name, args)
    training_dataloader = DataLoader(training_dataset, args.batch_size, shuffle=True)
    testing_dataloader = DataLoader(testing_dataset, args.batch_size)

    accelerator = Accelerator()
    device = accelerator.device
    mymodel.to(device)
    my_model, my_optimizer, my_training_dataloader = accelerator.prepare(mymodel, optimizer, training_dataloader)
    my_testing_dataloader = accelerator.prepare(testing_dataloader)

    """train"""
    logging.info("##################\tstart training\t##################")
    best_acc, best_epoch, best_auc, best_f1 = 0, 0, 0, 0
    epoch_loss, mask_loss, cont_loss, epoch_acc = None, None, None, None
    # test_data_len_size = 0
    start_time = time.time()

    """orgViT  without mask"""
    # train_epoch = train_one_epoch_without_mask

    """gaze mask"""
    train_epoch = train_one_epoch_with_gaze_mask_random_OUT

    """test"""
    test_epoch = evaluate_without_mask

    logging.info("train use {}".format(train_epoch))
    logging.info("test use {}".format(test_epoch))
    for epoch in range(args.epochs):
        logging.info('Epoch {}, lr={}'.format(epoch, scheduler.optimizer.param_groups[0]['lr']))

        """Train"""
        epoch_loss, epoch_acc = train_epoch(my_training_dataloader, my_model, device, loss_fn, optimizer, scheduler, args)

        """test"""
        novel_test_acc, auc, f1, data_len_size = test_epoch(my_testing_dataloader, my_model, device)

        if epoch_loss != None and mask_loss == None and cont_loss == None and epoch_acc != None:
            logging.info('Loss:%.4f   TrainAcc:%.4f   TestAcc:%.4f' % (epoch_loss, epoch_acc, novel_test_acc))
        elif epoch_loss != None and mask_loss != None and cont_loss == None and epoch_acc != None:
            logging.info('Loss:%.4f    MaskLoss:%.4f  TrainAcc:%.4f   TestAcc:%.4f'
                % (epoch_loss, mask_loss, epoch_acc, novel_test_acc))
        elif epoch_loss != None and mask_loss == None and cont_loss != None and epoch_acc != None:
            logging.info('Loss:%.4f    cont_loss:%.4f  TrainAcc:%.4f   TestAcc:%.4f'
                 % (epoch_loss, cont_loss, epoch_acc, novel_test_acc))
        elif epoch_loss != None and mask_loss != None and cont_loss != None and epoch_acc != None:
            logging.info('Loss:%.4f    MaskLoss:%.4f   CosLoss:% .4f   TrainAcc:%.4f   TestAcc:%.4f'
                         % (epoch_loss, mask_loss, cont_loss, epoch_acc, novel_test_acc))

        logging.info("data len={}, ACC={}, AUC={}, F1={}".format(data_len_size, novel_test_acc, auc, f1))
        scheduler.step(epoch)

        if novel_test_acc > best_acc:
            best_checkpoint_path = op.join(args.output_dir, 'deit_small_best.pth')
            save_on_master(
                {'model': my_model.state_dict(),
                 'args': args,
                 'optimizer': optimizer.state_dict(),
                 'lr_scheduler': scheduler.state_dict(),
                 'epoch': epoch}, best_checkpoint_path)
            logging.info('### save best ### {}, {}'.format(novel_test_acc, best_checkpoint_path))
            best_acc = round(novel_test_acc, 6)
            best_epoch = epoch
            best_auc = round(auc, 6)
            best_f1 = round(f1, 6)

        if (epoch+1)%args.model_save_interval==0:
            checkpoint_path = op.join(args.output_dir, 'deit_small_{}_epoch.pth'.format(epoch+1))
            save_on_master(
                {'model': my_model.state_dict(),
                'args': args,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'epoch': epoch},checkpoint_path)

    logging.info('\n###\nBest at epoch {}, acc={}, auc={}, f1={}'.format(best_epoch, best_acc, best_auc, best_f1))
    logging.info("test data len={}, Time Cost {}".format(data_len_size, str(datetime.timedelta(seconds=int(time.time()-start_time)))))
    logging.info('save at {}'.format(best_checkpoint_path))



"""without mask"""
def train_one_epoch_without_mask(dataloader,model,device,criterion,optimizer, lr_scheduler, args,max_norm=-1):
    model.train()
    criterion.train()
    criterion = criterion.to(device)
    epoch_accuracy,epoch_loss=0,0
    start = time.time()
    for idx, (data, label, imgpath, imgpath) in enumerate(dataloader):
        if idx % 10 == 0:
            print(idx, time.time() - start)

        label=torch.tensor(label).to(device)
        data = data.to(device)

        output = model(data)
        loss = criterion(output, label)
        model.zero_grad()
        loss.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        acc = (output.argmax(dim=1) == label).float().mean().item()
        epoch_accuracy += acc / len(dataloader)
        epoch_loss += (loss.item()) / len(dataloader)
        torch.cuda.empty_cache()
    return epoch_loss, epoch_accuracy


"""Gaze Mask"""
def train_one_epoch_with_gaze_mask_random_OUT(dataloader,model,device,criterion,optimizer, lr_scheduler, args, max_norm=-1):
    model.train()
    criterion.train()
    criterion=criterion.to(device)
    model=model.to(device)
    # model.backbone.patch_embed.register_backward_hook(backward_hook)
    epoch_accuracy,epoch_loss=0,0
    start = time.time()
    for idx, (data, label, imgpath, gaze_mask, gaze_hm) in enumerate(dataloader):
        # grad_block=list()

        if idx % 10 == 0:
            print(idx, time.time()-start)

        label=torch.tensor(label).to(device)
        data=data.to(device)
        mask = gaze_mask.to(device)

        if random.random() > args.random_use_mask:
            out = model(data)
        else:
            out = model.forward_with_mask(data, mask)
        loss = criterion(out, label)
        model.zero_grad()
        loss.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        acc = (out.argmax(dim=1) == label).float().mean().item()
        epoch_accuracy += acc / len(dataloader)
        epoch_loss += (loss.item()) / len(dataloader)

        del mask
        # del inputs
    return epoch_loss, epoch_accuracy


def evaluate_without_mask(dataloader,model,device):
    model.eval()
    with torch.no_grad():
        one_hot_num = 0
        data_len_size = 0
        confusion_label, confusion_pred = [], []
        output_list = []
        softmax = nn.Softmax()
        for data, label, imgpath, gaze_mask in dataloader:
            label=torch.tensor([item for item in label]).to(device)
            data = data.to(device)
            B=data.shape[0]
            output = model(data)

            acc = (output.argmax(dim=1) == label).float().sum().item()
            one_hot_num += acc  # / len(dataloader)
            data_len_size += B

            output_list.append(output)
            confusion_pred += list(output.argmax(dim=1).squeeze().cpu().numpy())
            confusion_label += list(label.cpu().numpy())

    ACC = one_hot_num / data_len_size

    out_cat = softmax(torch.cat(output_list, dim=0)).cpu().numpy()
    auc = roc_auc_score(confusion_label, out_cat, multi_class='ovo')
    f1 = f1_score(confusion_label, confusion_pred, average='weighted')

    return ACC, auc, f1, data_len_size


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EG_ViT', parents=[get_args_parser()])
    args = parser.parse_args()

    args.output_dir = '{}/{}'.format(args.output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    if args.output_dir and args.eval != True:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main_INbreast(args)




