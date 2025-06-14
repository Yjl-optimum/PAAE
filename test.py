import os
import math
import time
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
if __name__ == '__main__':
    from torch.utils.data import DataLoader, ConcatDataset
    import torch.nn.functional as F
    import random
    import torch, torchvision
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import math
    import cv2
    from typing import List, Optional
    from counter import count
    from scipy.signal import find_peaks
    from scipy.fft import fft, fftfreq
    from Model_PAAE import OneDRAC_Pose
    from Dataset_1DRAC import SkeletonData_Density_Valid

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    torch.cuda.set_device(0)


    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    set_seed(42)

    def testing_loop(n_epochs, model, test_set, txt_path, batch_size, CkptDir=''):
        # 设置目录路径和文件前缀
        ckpt_dir = CkptDir
        prefix = os.listdir(ckpt_dir)
        prefix = ['epoch_92_0.33_0.56.pt']

        train_loader = DataLoader(test_set,
                                  batch_size=batch_size,
                                  num_workers=1,
                                  shuffle=False
                                  )

        best_mae = float('inf')
        best_obo = float('inf')
        best_mae_ckpt = ''
        best_obo_ckpt = ''
        results = []
        for ckpt_name in prefix:
            # for ckpt_name in [prefix]:
            lastCkptPath = os.path.join(ckpt_dir, ckpt_name)
            print(f'Found checkpoint: {lastCkptPath}')
            currEpoch = 0
            model = model.to(device)
            if lastCkptPath != None:
                print("loading checkpoint")
                checkpoint = torch.load(lastCkptPath)
                currEpoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'], strict=True)
                del checkpoint

            for epoch in tqdm(range(currEpoch, n_epochs + currEpoch)):
                Predict = []
                GT = []
                namelist = []
                with torch.no_grad():
                    model.eval()
                    batch_idx = 0
                    pbar = tqdm(train_loader, total=len(train_loader))
                    print('len(train_loader)', len(train_loader))

                    for X, gt, video_name in pbar:
                        acc = 0

                        torch.cuda.empty_cache()
                        X = X.to(device).float()
                        gt = gt.to(device).float()

                        xret, x1, x_encoder, x_decoder = model(X)


                        min_val, _ = x_encoder.min(dim=1, keepdim=True)
                        max_val, _ = x_encoder.max(dim=1, keepdim=True)
                        projection = (x_encoder - min_val) / (max_val - min_val)

                        pre = projection[0].detach().cpu().numpy().flatten().tolist()
                        predict_count = count(pre,4)

                        Predict.append(math.ceil(predict_count))
                        GT.append(gt[0])

                    acc = 0
                    Predict = torch.tensor(Predict)
                    GT = torch.tensor(GT)
                    print(len(namelist),namelist)
                    print(len(Predict),Predict)
                    print(len(GT),GT)
                    mae = torch.sum(torch.div(torch.abs(Predict - GT), (GT + 1e-1))) / len(train_loader)
                    gaps = torch.sub(Predict, GT).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    for item in gaps:
                        if abs(item) <= 1:
                            acc += 1
                    OBO = acc / Predict.flatten().shape[0]
                    # 记录每个 ckpt 的结果
                    results.append(f"Checkpoint: {ckpt_name}, MAE: {mae:.4f}, OBO: {OBO:.4f}")

                    if mae < best_mae:
                        best_mae = mae
                        best_mae_ckpt = lastCkptPath

                    if OBO < best_obo:
                        best_obo = OBO
                        best_obo_ckpt = lastCkptPath

                    print("MAE:{0},OBO:{1}".format(mae, OBO))


            print("len(Predict),len(GT)", len(Predict), len(GT))
            print(f"Best MAE: {best_mae} from {best_mae_ckpt}")
            print(f"Best OBO: {best_obo} from {best_obo_ckpt}")

            # 将最佳 MAE 和 OBO 信息写入文件
        with open(txt_path, "w") as f:
            for result in results:
                f.write(result + "\n")
            f.write(f"Best MAE: {best_mae} from {best_mae_ckpt}\n")
            f.write(f"Best OBO: {best_obo} from {best_obo_ckpt}\n")

        return None


    def test_loop():
        NUM_FRAME = 128

        # #RepcountA
        data_type = 'pose'
        root_path = '../../RepCountA_Dataset'
        video_root_path = 'RepCountA_Video'
        pose_root_path = 'RepCountA_Skeleton_PF'
        label_root_path = 'annotation'
        valid_choice = 'test'
        valid_label = 'test.csv'


        testDataset = SkeletonData_Density_Valid(
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

        sampleDataset = torch.utils.data.Subset(testDataset, range(0, len(testDataset)))

        print('len(sampleDatasetA', len(sampleDataset))

        testing_loop(
            1,
            model,
            sampleDataset,
            'Model_fu9' + "best_metrics.txt",
            1,
            'checkpoint/fu9/',
        )

    test_loop()