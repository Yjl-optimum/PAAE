import numpy as np
import csv
import os
import torch


def load_keypoints(path):
    """从 .npz 文件中加载骨骼关键点"""
    if path is None:
        path = None
    data = np.load(path)
    keypoints = data['keypoints']  # 形状为 (num_frames, num_joints, 3)
    return keypoints



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


def crop_frame_valid(pose_path, num_frame, if_seg=False):
    frames = load_keypoints(pose_path)  # 加载视频的所有帧
    frame_length = len(frames)
    frames_tensor_list = []

    def sample_frames(frames, num_frame, frame_length):
        frame_tensor = []
        if num_frame <= frame_length:
            # 如果帧数足够，均匀采样
            for i in range(num_frame):
                frame_index = i * frame_length // num_frame
                frame_tensor.append(frames[frame_index])
        else:
            # 如果帧数不足，添加所有可用帧
            frame_tensor.extend(frames)
            # 通过重复最后一帧填充列表，直到有 num_frame 帧
            for i in range(frame_length, num_frame):
                frame_tensor.append(frames[frame_length - 1])
        return torch.as_tensor(np.stack(frame_tensor))

    if if_seg:
        frames_per_segment = 800
        for start_frame in range(0, frame_length, frames_per_segment):
            segment_frames = frames[start_frame:start_frame + frames_per_segment]
            segment_length = len(segment_frames)
            frame_tensor = sample_frames(segment_frames, num_frame, segment_length)
            frames_tensor_list.append(frame_tensor)
        # print('frames_tensor_list.length', len(frames_tensor_list))
        return frames_tensor_list

    else:
        frame_tensor = sample_frames(frames, num_frame, frame_length)
        return frame_tensor
