import os
import sys
import cv2
import time
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from support import *
from detection import *
from visualizations import *

pose_pairs = np.array([[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16], [2, 8], [5, 11]])
              
colors = [[0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], 
         [0,0,0], [0,0,0], [0,255,0], [0,255,0]]

keypoints_mapping = ['Nose', 'Neck', 'Right Sholder', 'Right Elbow', 'Right Wrist', 'Left Sholder', 
                    'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee', 'Right Ankle', 'Left Hip', 
                    'Left Knee', 'Left Ankle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear']

videos_dir = "../Videos/"
data_dir = "../Data/"

video_name_ext = "Diogo.mp4"
file_name_1 = "Diogo_T.data"
file_name_2 = "Diogo_GT.data"

video_name = (video_name_ext).split(sep='.')[0]
file_dir = data_dir + video_name + '/'
if not os.path.exists(file_dir):
    os.makedirs(file_dir)

file_path_1 = file_dir + file_name_1
file_path_2 = file_dir + file_name_2

metadata, keypoints = readFrameDATA(file_path_1, frame_n=0)

n_frames, fps = metadata["n_frames"], metadata["fps"]
frame_height, frame_width = metadata["frame_height"], metadata["frame_width"]
joint_pairs = metadata["joint_pairs"]

pairs = []
for j in joint_pairs:
    pairs.append(pose_pairs[j])
joints = np.unique(pairs)

Et_keypoints = np.zeros(keypoints.shape[0])
Et = 0

Et_keypoints_vec = np.zeros([n_frames, keypoints.shape[0]])
Et_vec = np.zeros(n_frames)

_, keypoints_1 = readAllFramesDATA(file_path_1)
_, keypoints_2 = readAllFramesDATA(file_path_2)

for i in range(n_frames):
    E_tmp = np.power((keypoints_1[i] - keypoints_2[i]),2)
    E_tmp = np.sum(E_tmp, axis=1)
    E_tmp = np.sqrt(E_tmp)
    E = np.sum(E_tmp, axis=0)

    Et_keypoints += E_tmp/n_frames
    Et += E/n_frames
    Et_vec[i] = E
    Et_keypoints_vec[i] = E_tmp

print(Et_keypoints)
print(Et)

# plt.figure()
# plt.grid(True)
# plt.plot(Et_keypoints_vec[:, 1])

joint_n = 3

plt.figure()
plt.title("Comparison X {}".format(keypoints_mapping[joints[joint_n]]))
plt.grid(True)
plt.plot(keypoints_1[:, joint_n, 0], label="X_pred")
plt.plot(keypoints_2[:, joint_n, 0], label="X_gt")
plt.legend()

plt.figure()
plt.title("Comparison Y {}".format(keypoints_mapping[joints[joint_n]]))
plt.grid(True)
plt.plot(keypoints_1[:, joint_n, 1], label="Y_pred")
plt.plot(keypoints_2[:, joint_n, 1], label="Y_gt")
plt.legend()

plt.show()

