import os
import math
import time

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset

import torch.nn.functional as F
import random
import torch, torchvision
import torch.nn as nn

import matplotlib.pyplot as plt
import math
import cv2
from typing import List, Optional

from Model_PAAE import OneDRAC_Pose
from Dataset_1DRAC import SkeletonData_Density_Train, SkeletonData_Density_Valid
from Dataset_Syn_1 import SkeletonSynDataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
torch.cuda.set_device(0)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/freq3/fu9_cro_60')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
class FrequencyLoss(nn.Module):
    def __init__(self):
        super(FrequencyLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, x_encoder, groundtruth):
        # 计算傅里叶变换
        x_encoder_fft = torch.fft.fft(x_encoder, dim=-1)
        groundtruth_fft = torch.fft.fft(groundtruth, dim=-1)

        # 提取幅度
        x_encoder_fft_abs = x_encoder_fft.abs()
        groundtruth_fft_abs = groundtruth_fft.abs()

        # 计算幅度的 MSE 损失
        loss = self.mse_loss(x_encoder_fft_abs, groundtruth_fft_abs)

        return loss


def training_loop(n_epochs,
                  model,
                  train_set,
                  val_set,
                  batch_size,
                  lr=0.001,
                  ckpt_name='ckpt',
                  use_x_encoder_error=False,
                  saveCkpt=True,#true
                  train=False,
                  validate=True,
                  gradient_accumulation_steps=2,
                  lastCkptPath = None):
    prevEpoch = 0
    trainLosses = []
    valLosses = []

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


    if lastCkptPath != None:
        print("loading checkpoint")
        checkpoint = torch.load(lastCkptPath)
        prevEpoch = checkpoint['epoch']
        trainLosses = checkpoint['trainLosses']
        valLosses = checkpoint['valLosses']

        model.load_state_dict(checkpoint['state_dict'], strict=True)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        del checkpoint

    model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    lossMSE = nn.MSELoss()
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=0,
                              shuffle=True)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            num_workers=0,
                            drop_last=False,
                            shuffle=True)

    if validate and not train:
        currEpoch = prevEpoch
    else:
        currEpoch = prevEpoch + 1


    for epoch in tqdm(range(currEpoch, n_epochs + currEpoch)):
        if train:
            testOBO = []
            testMAE = []
            pbar = tqdm(train_loader, total=len(train_loader))
            mae = 0
            i = 0
            for X, y, gt in pbar:
                torch.cuda.empty_cache()
                model.train()
                acc = 0
                X = X.to(device).float()
                # y是生成的密度图
                y = y.to(device).float()
                gt = gt.to(device).float()
                # 密度图输出，中途embedding输出， 降维前的output
                origin_feature, x1, x_encoder, x_decoder = model(X)

                x_encoder = x_encoder.squeeze(-1)

                loss1 = lossMSE(x1, x_decoder)
                loss2 = lossMSE(y, x_encoder)
                # print(gt.shape)
                mae += loss1.item()

                if use_x_encoder_error:
                    loss = loss1 * 0.6 + loss2 * 0.4
                else:
                    loss = loss2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                trainLosses.append(loss.item())

                del X, y, gt, x1, x_encoder, x_decoder

                i += batch_size

                pbar.set_postfix({'Epoch': epoch,
                                  'MAE_period': (mae),
                                  'Mean Tr Loss': np.mean(trainLosses[-i + 1:])})

            writer.add_scalar('Metrics/Train/loss', np.mean(trainLosses[-1]), epoch)

        if validate:
            testOBO = []
            testMAE = []
            with torch.no_grad():
                model.eval()
                pbar = tqdm(val_loader, total=len(val_loader))
                print(f"Length of val_loader: {len(val_loader)}")

                for X, gt, _ in pbar:
                    if X.size(0) == 0 or gt.size(0) == 0:
                        print("Empty batch found!")

                    acc = 0
                    torch.cuda.empty_cache()
                    X = X.to(device).float()
                    gt = gt.to(device).float()

                    _, x1, x_encoder, x_decoder = model(X)


                    # 使用解码器的输出进行计算
                    min_val, _ = x_encoder.min(dim=1, keepdim=True)
                    max_val, _ = x_encoder.max(dim=1, keepdim=True)
                    projection = (x_encoder - min_val) / (max_val - min_val)
                    # 计算平均值
                    y_line = projection.mean(dim=1, keepdim=True)
                    # 计算每个时间步的差值
                    diff = (projection[:, :-1] - y_line) * (projection[:, 1:] - y_line)
                    # 找到交叉点
                    count_intersections = (diff < 0).sum(dim=1).float()
                    # 计算交叉点数量并存储
                    predict_count = count_intersections / 2.0


                    print('countpred', predict_count.shape, [round(x, 2) for x in predict_count.cpu().detach().numpy().reshape(-1)])
                    print('gt', gt.shape, [round(x, 2) for x in gt.cpu().detach().numpy().reshape(-1)])
                    mae = torch.sum(torch.div(torch.abs(predict_count - gt), (gt + 1e-1)))/batch_size
                    gaps = torch.sub(predict_count, gt).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    # print('-----gaps-----------', gaps)
                    for item in gaps:
                        if abs(item) <= 1:
                            acc += 1

                    OBO = acc / predict_count.flatten().shape[0]
                    testOBO.append(OBO)
                    MAE = mae.item()
                    testMAE.append(MAE)

                    del X, gt
                    pbar.set_postfix({'Epoch': epoch,
                                      'MAE_count': (np.mean(testMAE)),
                                      'OBO_count': (np.mean(testOBO))})

                print("MAE:{0},OBO:{1}".format(np.mean(testMAE), np.mean(testOBO)))
                # print(testMAE)
                print("valid-------len(testMAE),len(testOBO)", len(testMAE), len(testOBO))
                writer.add_scalar('Metrics/Val/MAE_count', np.mean(testMAE), epoch)
                writer.add_scalar('Metrics/Val/OBO_count', np.mean(testOBO), epoch)

        # save checkpoint
        if saveCkpt and (np.mean(testMAE) <= 0.40 or epoch % 10 == 0):
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainLosses': trainLosses,
                'valLosses': valLosses
            }
            ckpt_num = f"epoch_{epoch}_{np.mean(testMAE):.2f}_{np.mean(testOBO):.2f}"
            torch.save(checkpoint, ckpt_name + ckpt_num + '.pt')


    return trainLosses, valLosses


def train_loop_skeleton(if_syn):
    NUM_FRAME = 128

    data_type = 'pose'
    root_path = '../../RepCountA_Dataset'
    video_root_path = 'RepCountA_Video'
    pose_root_path = 'RepCountA_Skeleton_PF'
    label_root_path = 'annotation'

    train_choice = 'train'
    valid_choice = 'test'
    train_label_video = 'train.csv'
    valid_label = 'test.csv'

    if if_syn:
        trainDataset = SkeletonSynDataset(
            '../../RepCountA_Dataset/RepCountA_Skeleton_PF',
            'train',
            2000,
            128,
            2,
            32
        )

    else:
        trainDataset = SkeletonData_Density_Train(
            root_path,
            video_root_path,
            pose_root_path,
            label_root_path,
            train_choice,
            train_label_pose,
            train_label_video,
            num_frame=NUM_FRAME,
            data_type='pose'
    )
    validDataset = SkeletonData_Density_Valid(
        root_path,
        video_root_path,
        pose_root_path,
        label_root_path,
        valid_choice,
        valid_label,
        num_frame=NUM_FRAME,
        data_type=data_type
    )

    model = OneDRAC_Pose(NUM_FRAME)
    model = model.to(device)

    sampleDatasetA = torch.utils.data.Subset(trainDataset, range(0, len(trainDataset)))
    sampleDatasetB = torch.utils.data.Subset(validDataset, range(0, len(validDataset)))

    print('len(sampleDatasetA', len(sampleDatasetA))
    print(len(sampleDatasetB))
    trLoss, valLoss = training_loop(200,
                                    model,
                                    sampleDatasetA,
                                    sampleDatasetB,
                                    16,
                                    3e-6,
                                    'checkpoint/fu9/',
                                    use_x_encoder_error=True,
                                    saveCkpt=1,
                                    train=1,
                                    validate=1,
                                    gradient_accumulation_steps=1,
                                    lastCkptPath=None,
                                    )


train_loop_skeleton(if_syn=True)