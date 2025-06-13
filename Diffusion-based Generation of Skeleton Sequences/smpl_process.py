import pickle
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from smplx import SMPL,SMPLX
import cv2
# from process import save_skeleton_video
from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_matrix, rotation_6d_to_matrix)

def ax_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    print('mat',mat.shape)
    ax = matrix_to_axis_angle(mat)
    return ax


def motion_data_load_process(motionfile):
    if motionfile.split(".")[-1] == "pkl":
        pkl_data = pickle.load(open(motionfile, "rb"))
        smpl_poses = pkl_data["smpl_poses"]
        modata = np.concatenate((pkl_data["smpl_trans"], smpl_poses), axis=1)
        if modata.shape[1] == 69:
            hand_zeros = np.zeros([modata.shape[0], 90], dtype=np.float32)
            modata = np.concatenate((modata, hand_zeros), axis=1)
        assert modata.shape[1] == 159
        modata[:, 1] = modata[:, 1] + 0  # + 1.25
        return modata
    elif motionfile.split(".")[-1] == "npy":
        modata = np.load(motionfile)
        if len(modata.shape) == 3 and modata.shape[1] % 8 == 0:
            print("modata has 3 dim , reshape the batch to time!!!")
            modata = modata.reshape(-1, modata.shape[-1])
        if modata.shape[-1] == 315:
            print("modata.shape is:", modata.shape)
            rot6d = torch.from_numpy(modata[:, 3:])
            T, C = rot6d.shape
            rot6d = rot6d.reshape(-1, 6)
            axis = ax_from_6v(rot6d).view(T, -1).detach().cpu().numpy()
            modata = np.concatenate((modata[:, :3], axis), axis=1)
            print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 319:
            print("modata.shape is:", modata.shape)
            modata = modata[:, 4:]
            rot6d = torch.from_numpy(modata[:, 3:])
            T, C = rot6d.shape
            rot6d = rot6d.reshape(-1, 6)
            axis = ax_from_6v(rot6d).view(T, -1).detach().cpu().numpy()
            modata = np.concatenate((modata[:, :3], axis), axis=1)
            # print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 159:
            print("159modata.shape is:", modata.shape)
            print("159modata.shape is:", modata.shape)
        elif modata.shape[-1] == 135:
            print("modata.shape is:", modata.shape)
            if len(modata.shape) == 3 and modata.shape[0] == 1:
                modata = modata.squeeze(0)
            rot6d = torch.from_numpy(modata[:, 3:])
            T, C = rot6d.shape
            rot6d = rot6d.reshape(-1, 6)
            axis = ax_from_6v(rot6d).view(T, -1).detach().cpu().numpy()
            hand_zeros = torch.zeros([T, 90]).to(rot6d).detach().cpu().numpy()
            modata = np.concatenate((modata[:, :3], axis, hand_zeros), axis=1)
            print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 139:
            print("139modata.shape is:", modata.shape)
            modata = modata[:, 4:]
            print("139modata.shape is:", modata.shape)
            rot6d = torch.from_numpy(modata[:, 3:])
            T, C = rot6d.shape
            print("rot6d.shape is:", rot6d.shape)
            rot6d = rot6d.reshape(-1, 6)
            print("rot6d.shape is:", rot6d.shape)
            axis = ax_from_6v(rot6d).view(T, -1).detach().cpu().numpy()
            print("axis.shape is:", axis.shape)
            hand_zeros = torch.zeros([T, 90]).to(rot6d).detach().cpu().numpy()
            print("hand_zeros.shape is:", hand_zeros.shape)
            modata = np.concatenate((modata[:, :3], axis, hand_zeros), axis=1)
            print(modata[:, :3].shape,axis.shape, hand_zeros.shape)
            print("139modata.shape is:", modata.shape)
        else:
            raise ("shape error!")

        modata[:, 1] = modata[:, 1] + 0  # + 1.25
        return modata

def extract_smpl_joints_smplx(motion_data, smpl_model):
    seq_rot = torch.tensor(motion_data, dtype=torch.float32).to('cuda')
    output = smpl_model.forward(
    #     betas=torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
    #     transl=seq_rot[:, :3],
    #     global_orient=seq_rot[:, 3:6],
    #     body_pose=torch.cat([seq_rot[:, 6:69], seq_rot[:, 69:72], seq_rot[:, 114:117]], dim=1)
        betas=torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
        # transl = motion[:,:3],
        transl=seq_rot[:, :3],
        global_orient=seq_rot[:, 3:6],
        body_pose=seq_rot[:, 6:69],
        jaw_pose=torch.zeros([seq_rot.shape[0], 3]).to(seq_rot),
        leye_pose=torch.zeros([seq_rot.shape[0], 3]).to(seq_rot),
        reye_pose=torch.zeros([seq_rot.shape[0], 3]).to(seq_rot),
        left_hand_pose=torch.zeros([seq_rot.shape[0], 45]).to(seq_rot),
        right_hand_pose=torch.zeros([seq_rot.shape[0], 45]).to(seq_rot),
        expression=torch.zeros([seq_rot.shape[0], 10]).to(seq_rot),
    )

    joints = output.joints.detach().cpu().numpy()
    return joints

def extract_smpl_joints_smpl(motion_data, smpl_model):
    seq_rot = torch.tensor(motion_data, dtype=torch.float32).to('cuda')
    print('seq',seq_rot.shape)
    output = smpl_model.forward(
            betas=torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
            transl=seq_rot[:, :3],
            global_orient=seq_rot[:, 3:6],
            body_pose=torch.cat([seq_rot[:, 6:69], seq_rot[:, 69:72], seq_rot[:, 114:117]], dim=1)
    )
    joints = output.joints.detach().cpu().numpy()
    # print("out",output)
    return joints

def show3Dpose1(vals, ax):
    # 定义新骨架连接，去掉移除的关节后重新组织
    I = np.array([0, 0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14,6,6])
    J = np.array([1, 2, 4, 5, 0, 7, 8, 3, 6, 9, 13, 14, 15, 16,11,12])

    # 颜色标识左侧和右侧
    LR = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1,0,1], dtype=bool)
    ax.view_init(elev=15., azim=70)

    lcolor = (0, 0, 1)  # 蓝色表示左侧
    rcolor = (1, 0, 0)  # 红色表示右侧

    for i in np.arange(len(I)):
        x, z, y = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)

    RADIUS = 0.72  # 骨架图的尺度
    RADIUS_Z = 0.7

    xroot, zroot, yroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS_Z + zroot, RADIUS_Z + zroot])
    ax.set_aspect('auto')  # 自动调整比例

    white = (1.0, 1.0, 1.0, 0.0)  # 设置背景颜色
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)

def save_skeleton_keyframes(frames, filename, fps=16):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (640, 480))

    for i in range(frames.shape[0]):
        ax.cla()  # Clear the current axes

        show3Dpose1(frames[i], ax)

        # Convert plot to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


        # Write the frame to the video
        out.write(cv2.resize(img, (640, 480)))
        if i % fps == 0:
             print('save the keyframes-------------------')

    out.release()
    plt.close(fig)

def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor = (0, 0, 1)
    rcolor = (1, 0, 0)

    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)

    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS_Z + zroot, RADIUS_Z + zroot])
    ax.set_aspect('auto')  # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)


def save_skeleton_video(frames, filename, fps=16):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (640, 480))
    i = 0
    for keypoints_3d in frames:
        ax.cla()  # Clear the current axes

        show3Dpose(keypoints_3d, ax)

        # Convert plot to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


        # Write the frame to the video
        out.write(cv2.resize(img, (640, 480)))
        print('save the video-------------------')

    out.release()
    plt.close(fig)

def process_motion_data(motion_file, smpl_path="./data/SMPL_NEUTRAL.pkl"):
    motion_data = motion_data_load_process(motion_file)
    print("Motion data shape:", motion_data.shape)

    smpl = SMPL(smpl_path).to('cuda').eval()
    joints_smpl = extract_smpl_joints_smpl(motion_data, smpl)
    print("Joint SMPL data shape:", joints_smpl.shape)

    remove_indices = [20, 21, 8, 7, 6, 13, 14]
    data = np.delete(joints_smpl[:, :24, :], remove_indices, axis=1)
    data[:, 6, :] = (data[:, 11, :] + data[:, 12, :]) / 2
    print("Processed data shape:", data.shape)

    return data

if __name__ == '__main__':
    SMPL_PATH = "./data/SMPL_NEUTRAL.pkl"
    # SMPLX_path = "./data/SMPLX_NEUTRAL.npz"
    # motion_file = os.path.join('.', 'experiments', 'Local_Module', 'FineDance_FineTuneV2_Local',
    #                            'samples_dod_2999_299_inpaint_soft_ddim_notranscontrol_2024-10-23-01-34-24',
    #                            'concat', 'npy', '211.npy')
    motion_file = os.path.join('.', 'experiments', 'Local_Module', 'FineDance_FineTuneV2_Local',
                               'samples_dod_2999_299_inpaint_soft_ddim_notranscontrol_2025-03-10-21-37-12',
                               'dod_39_147g009g_l003.npy')

    motion_data = motion_data_load_process(motion_file)
    print(motion_data.shape)
    # Load SMPL model
    smpl = SMPL(SMPL_PATH).to('cuda').eval()
    # smplx = SMPLX(SMPLX_path, use_pca=False, flat_hand_mean=True).eval()
    # smplx.to('cuda').eval()

    # joints_smplx = extract_smpl_joints_smplx(motion_data, smplx)
    # print("Joint_smplx data shape:", joints_smplx.shape)
    joints_smpl = extract_smpl_joints_smpl(motion_data, smpl)
    print("Joint_smpl data shape:", joints_smpl.shape)
    remove_indices = [20, 21, 8, 7, 6, 13, 14]
    data = np.delete(joints_smpl[:, :24, :], remove_indices, axis=1)
    data[:, 6, :] = (data[:, 11, :] + data[:, 12, :]) / 2
    print("Processed data shape:", data.shape)

    save_skeleton_keyframes(data, 'dod.mp4', fps=16)


