import numpy as np
import math
from scipy import integrate
import csv
import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from datetime import datetime

import cv2
import glob
import random
from tqdm import tqdm
from random import randrange, randint
import math, base64, io, os, time
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def is_in_cycle(frame_idx, cycles):
    for start, end in zip(cycles[::2], cycles[1::2]):
        if start <= frame_idx <= end:
            return 1
    return 0


def PDF(x, u, sig):
    """计算高斯分布的概率密度函数值"""
    return np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)

def get_integrate(x_1, x_2, avg, sig):
    """计算在区间 [x_1, x_2] 上高斯分布的积分值"""
    y, err = integrate.quad(PDF, x_1, x_2, args=(avg, sig))
    return y

def normalize_label(y_frame, y_length):
    """将关键帧转换为归一化的标签数组"""
    y_label = np.zeros(y_length)
    for i in range(0, len(y_frame), 2):
        x_a = y_frame[i]
        x_b = y_frame[i + 1]
        avg = (x_a + x_b) / 2
        sig = (x_b - x_a) / 6
        num = x_b - x_a + 1
        if num != 1:
            # 计算当前区间的最大积分值
            max_integrate = max([get_integrate(x_a - 0.5 + j, x_a + 0.5 + j, avg, sig) for j in range(num)])
            for j in range(num):
                x_1 = x_a - 0.5 + j
                x_2 = x_a + 0.5 + j
                y_ing = get_integrate(x_1, x_2, avg, sig) / max_integrate  # 归一化为区间内的最大值
                y_label[x_a + j] = y_ing
        else:
            y_label[x_a] = 1
    return y_label

def preprocess(video_frame_length, time_points, num_frames, sampled_indexes=None):
    """
    处理标签（.csv）以生成密度图标签
    Args:
        video_frame_length: 视频总帧数，例如 1024 帧
        time_points: 标签时间点，例如 [1, 23, 23, 40, 45, 70,.....] 或 [0]
        num_frames: 64
        sampled_indexes: 采样索引，如果有的话
    Returns:
        归一化的标签数组，例如 [0.1, 0.8, 0.1, .....]
    """
    new_crop = []
    for i in range(len(time_points)):
        item = min(
            math.ceil(
                (float(time_points[i]) / float(video_frame_length)) * num_frames
            ),
            num_frames - 1,
        )
        new_crop.append(item)
    new_crop = np.sort(new_crop)
    label = normalize_label(new_crop, num_frames)
    return label


def get_labels_list(names_path, counts_path, valid=True):
    labels_list = []

    if valid:
        with open(counts_path, encoding='utf-8') as counts_file:
            counts_csv = csv.DictReader(counts_file)
            for row in counts_csv:
                if 'name' in row and 'count' in row:
                    if not row['count']:
                        print(row['name'] + ' error')
                        labels_list.append({
                            'name': row['name'],
                            'count': 0,
                        })
                    else:
                        labels_list.append({
                            'name': row['name'],
                            'count': int(row['count']),
                        })
        return labels_list

    # 读取第一个文件，提取 name 字段
    names = set()
    with open(names_path, encoding='utf-8') as names_file:
        names_csv = csv.DictReader(names_file)
        for row in names_csv:
            if 'name' in row and row['name']:
                names.add(row['name'])
            else:
                print("缺少 'name' 字段或 'name' 值为空。")

    print(names)

    # 读取第二个文件，根据 name 字段查找 count 值
    with open(counts_path, encoding='utf-8') as counts_file:
        counts_csv = csv.DictReader(counts_file)
        for row in counts_csv:
            if 'name' in row and 'count' in row and row['name'] in names:
                cycle = [int(row[key]) for key in row.keys() if 'L' in key and row[key] != '']
                if not row['count']:
                    labels_list.append({
                        'name': row['name'],
                        'count': 0,
                        'cycle': [0, 0]
                    })
                else:
                    labels_list.append({
                        'name': row['name'],
                        'count': int(row['count']),
                        'cycle': cycle
                    })
    return labels_list


def load_keypoints(path):
    """从 .npz 文件中加载骨骼关键点"""
    if path is None:
        path = None
    data = np.load(path)
    keypoints = data['keypoints']  # 形状为 (num_frames, num_joints, 3)
    return keypoints



def crop_frame(pose_path, num_frame):
    frames = load_keypoints(pose_path)  # 加载视频的所有帧
    pose_tensor = []
    frame_length = len(frames)

    if num_frame <= frame_length:
        # 如果帧数足够，均匀采样
        for i in range(num_frame):
            frame_index = i * frame_length // num_frame
            pose_tensor.append(frames[frame_index])
    else:
        # 如果帧数不足，添加所有可用帧
        for i in range(frame_length):
            pose_tensor.append(frames[i])
        # 通过重复最后一帧填充列表，直到有 self.num_frames 帧
        for i in range(frame_length, num_frame):
            pose_tensor.append(frames[frame_length - 1])

    # 将帧列表转换为张量
    Frame_Tensor = torch.as_tensor(np.stack(pose_tensor))

    return Frame_Tensor, frame_length


def get_frame(video_path):
    # print('video_path----------', video_path)
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    # self.frame_length = len(frames)
    return frames


def crop_frame_img(labels_dict, video_path, video_name, num_frame):
    frames = get_frame(video_path)  # 加载视频的所有帧
    frames_tensor = []
    frame_length = len(frames)
    # 获取重复片段信息
    video_info = labels_dict.get(video_name, {})
    cycles = video_info.get('cycle', [])
    count = video_info.get('count', 0)

    if num_frame <= frame_length:
        # 如果帧数足够，均匀采样
        for i in range(num_frame):
            frame_index = i * frame_length // num_frame
            frame = frames[frame_index]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))  # [3, 224, 224]
            frame = transforms.ToTensor()(frame)
            frames_tensor.append(frame)
    else:
        # 如果帧数不足，添加所有可用帧
        for i in range(frame_length):
            frame = frames[i]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))  # [3, 224, 224]
            frame = transforms.ToTensor()(frame)
            frames_tensor.append(frame)
        # 通过重复最后一帧填充列表，直到有 self.num_frames 帧
        for i in range(frame_length, num_frame):
            frame = frames[frame_length - 1]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))  # [3, 224, 224]
            frame = transforms.ToTensor()(frame)
            frames_tensor.append(frame)
            frames_tensor.append(frame)
            # labels_tensor.append(is_in_cycle(self.frame_length - 1, cycles))

    # 将帧列表转换为张量
    Frame_Tensor = torch.as_tensor(np.stack(frames_tensor))
    count = torch.as_tensor(count)

    # print('Frame_Tensor.shape', Frame_Tensor.shape)
    # print('Labels_Tensor.shape', Labels_Tensor.shape)

    return Frame_Tensor, frame_length, cycles, count


def crop_frame_valid(pose_path, num_frame):
    frames = load_keypoints(pose_path)  # 加载视频的所有帧
    pose_tensor = []
    frame_length = len(frames)

    if num_frame <= frame_length:
        # 如果帧数足够，均匀采样
        for i in range(num_frame):
            frame_index = i * frame_length // num_frame
            pose_tensor.append(frames[frame_index])
            # labels_tensor.append(is_in_cycle(frame_index, cycles))
    else:
        # 如果帧数不足，添加所有可用帧
        for i in range(frame_length):
            pose_tensor.append(frames[i])
            # labels_tensor.append(is_in_cycle(i, cycles))
        # 通过重复最后一帧填充列表，直到有 self.num_frames 帧
        for i in range(frame_length, num_frame):
            pose_tensor.append(frames[frame_length - 1])
            # padding_frame = np.zeros_like(frames[0])  # 生成一个与第一帧相同形状的零张量
            # pose_tensor.append(padding_frame)
            # labels_tensor.append(is_in_cycle(self.frame_length - 1, cycles))

    # 将帧列表转换为张量
    Frame_Tensor = torch.as_tensor(np.stack(pose_tensor))


    return Frame_Tensor

def crop_frame_img_valid(labels_dict, video_path, video_name, num_frame):
    frames = get_frame(video_path)  # 加载视频的所有帧
    frames_tensor = []
    frame_length = len(frames)
    # 获取重复片段信息
    video_info = labels_dict.get(video_name, {})
    count = video_info.get('count', 0)

    if num_frame <= frame_length:
        # 如果帧数足够，均匀采样
        for i in range(num_frame):
            frame_index = i * frame_length // num_frame
            frame = frames[frame_index]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))  # [3, 224, 224]
            frame = transforms.ToTensor()(frame)
            frames_tensor.append(frame)
    else:
        # 如果帧数不足，添加所有可用帧
        for i in range(frame_length):
            frame = frames[i]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))  # [3, 224, 224]
            frame = transforms.ToTensor()(frame)
            frames_tensor.append(frame)
        # 通过重复最后一帧填充列表，直到有 self.num_frames 帧
        for i in range(frame_length, num_frame):
            frame = frames[frame_length - 1]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))  # [3, 224, 224]
            frame = transforms.ToTensor()(frame)
            frames_tensor.append(frame)
            frames_tensor.append(frame)
            # labels_tensor.append(is_in_cycle(self.frame_length - 1, cycles))

    # 将帧列表转换为张量
    Frame_Tensor = torch.as_tensor(np.stack(frames_tensor))
    count = torch.as_tensor(count)


    return Frame_Tensor, count




class SkeletonData_Density_Train(Dataset):
    def __init__(self, root, video_root, pose_root, label_root, train_or_not,
                 label_train_pose, label_train_video, num_frame, data_type):
        self.video_root = os.path.join(root, video_root)  # 'Dataset/RepCountA/RepCountA_Video'
        self.pose_root = os.path.join(root, pose_root)   # 'Dataset/RepCountA/RepCountA_Skeleton_MP'
        self.train_or_not = train_or_not  # 'train'
        self.label_pose = os.path.join(root, label_root, label_train_pose)   # 'Dataset/RepCountA/annotation/pose_train.csv'
        self.label_video = os.path.join(root, label_root, label_train_video)   # 'Dataset/RepCountA/annotation/train.csv'

        # print('len(self.all_video_dir)', len(self.all_video_dir))
        self.num_frame = num_frame
        self.data_type = data_type
        if self.data_type == 'pose':
            self.label_dict = get_labels_list(self.label_pose, self.label_video, valid = False)  # get all labels
            print(self.label_dict)
        else:
            self.label_dict = get_labels_list(self.label_video, self.label_video, valid = True)  # get all labels

    def __getitem__(self, inx):

        if self.data_type == 'pose':
            video_name = self.label_dict[inx].get('name').replace('.mp4','')
            # print('video_name=============', video_name)
            gt_count = self.label_dict[inx].get('count')
            gt_count = torch.tensor(gt_count).unsqueeze(0)
            time_points = self.label_dict[inx].get('cycle')

            pose_path = os.path.join(self.pose_root, self.train_or_not, video_name,
                                     'output_3D', '3D_keypoints.npz')
            frame_tensor, frames_len = crop_frame(pose_path, self.num_frame)
            label_tensor = torch.as_tensor(preprocess(frames_len, time_points, self.num_frame))
            # save_skeleton_video(frame_tensor, os.path.join('test_sup/', video_name[:-4] + '.mp4'), )

            return frame_tensor, label_tensor, gt_count


    def __len__(self):
        """返回数据集的大小"""
        return len(self.label_dict)



class SkeletonData_Density_Valid(Dataset):
    def __init__(self, root, video_root, pose_root, label_root, train_or_not,
                 label_valid, num_frame, data_type):
        self.video_root = os.path.join(root, video_root)  # 'Dataset/RepCountA/RepCountA_Video'
        self.pose_root = os.path.join(root, pose_root)  # 'Dataset/RepCountA/RepCountA_Skeleton_MP'
        self.train_or_not = train_or_not  # 'train'
        self.label_pose = os.path.join(root, label_root,
                                       label_valid)  # 'Dataset/RepCountA/annotation/valid.csv'
        self.num_frame = num_frame
        self.data_type = data_type
        self.label_dict = []
        if self.data_type == 'pose':
            self.label_dict = get_labels_list(self.label_pose, self.label_pose, valid=True)  # get all labels
            print(self.label_dict)

    def __getitem__(self, inx):
        if self.data_type == 'pose':
            video_name = self.label_dict[inx].get('name').replace('.mp4','')
            gt_count = self.label_dict[inx].get('count')
            gt_count = torch.tensor(gt_count).unsqueeze(0)

            pose_path = os.path.join(self.pose_root, self.train_or_not, video_name,
                                     'output_3D', '3D_keypoints.npz')

            frame_tensor = crop_frame_valid(pose_path, self.num_frame)
            # gt_count = torch.as_tensor(gt_count)
            return frame_tensor, gt_count, video_name

    def __len__(self):
        """返回数据集的大小"""
        return len(self.label_dict)
