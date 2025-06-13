from smpl_process import process_motion_data
import numpy as np
import os
import pandas as pd
import math
import ast
import matplotlib.pyplot as plt
import cv2
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
if __name__ == "__main__":
    motion_file = os.path.join('.', 'experiments', 'Local_Module', 'FineDance_FineTuneV2_Local',
                               'samples_dod_2999_299_inpaint_soft_ddim_notranscontrol_2025-03-10-21-37-12')
    output_file = os.path.join(motion_file, 'finish')

    output_df = pd.DataFrame(columns=['name', 'gt', 'intervals'])
    # output_df.to_csv(os.path.join(output_file, 'output_process.csv'), index=False)

    df = pd.read_csv(os.path.join(motion_file, 'output.csv'))
    for index, row in df.iterrows():
        name = row['name']
        gt = row['gt']
        intervals = row['intervals']
        # 将字符串形式的 intervals 转换为列表形式
        if isinstance(intervals, str):
            intervals = ast.literal_eval(intervals)
        # print("in",intervals)
        processed_intervals = [
            (math.ceil((x[0]) * 2 / 2), math.floor((x[1]) * 2 / 2))
            for x in intervals
        ]
        # print("out",processed_intervals)
        output_df = output_df.append({'name': name, 'gt': gt, 'intervals': processed_intervals}, ignore_index=True)
        motion_path = os.path.join(motion_file, name)
        motion = process_motion_data(motion_path)
        motion = motion[::2]
        index_mapping = [0, 2, 5, 8, 1, 4, 7, 3, 6, 9, 10, 11, 13, 15, 12, 14, 16]
        motion = motion[:, index_mapping, :]
        motion_swapped = motion.copy()
        motion_swapped[:, :, [1, 2]] = motion[:, :, [2, 1]]
        # print("motion shape: ", motion.shape)
        np_path = os.path.join(output_file, name)
        np.save(np_path, motion_swapped)
        # break

    output_df.to_csv(os.path.join(output_file, 'output_process.csv'), index=False)