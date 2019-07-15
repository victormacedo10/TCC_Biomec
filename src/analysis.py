import os
import sys
import cv2
import time
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
sr = ['Right Sholder', 'Right Elbow', 'Right Wrist', 'Right Hip', 'Right Knee', 'Right Ankle']
sl = ['Left Sholder', 'Left Elbow', 'Left Wrist', 'Left Hip', 'Left Knee', 'Left Ankle']

km = ['Nose', 'Neck', 'RS', 'RE', 'RW', 'LS', 
                    'E', 'LW', 'RH', 'RK', 'RA', 'LH', 
                    'LK', 'LA', 'RE', 'LE', 'RE', 'LE']

videos_dir = "../Videos/"
data_dir = "../Data/"

def saveData(video_name, file_names, file_ref, plot_name='teste', pose='Saggital Right'):
    if pose=='Saggital Right':
        for p in sr:
            plotTrajectory(video_name, file_names, file_ref, p, 'x', plot_name + 'ptx' + p, saveplot=True)
            plotTrajectory(video_name, file_names, file_ref, p, 'y', plot_name + 'pty' + p, saveplot=True)

            showMetrics(video_name, file_names, file_ref, p, error_type='Error Graph', def_error=False, 
                        plot_name=plot_name + 'epj' + p, saveplot=True)
    showMetrics(video_name, file_names, file_ref, p, error_type='Error Graph', def_error=False, 
                        plot_name=plot_name + 'et', saveplot=True)
    showMetrics(video_name, file_names, file_ref, p, error_type='Error DF', def_error=True, 
                        plot_name=plot_name + 'dfe', saveplot=True)
    showMetrics(video_name, file_names, file_ref, p, error_type='False Negatives DF', def_error=True, 
                        plot_name=plot_name + 'dffn', saveplot=True)

def plotTrajectory(video_name, file_names, file_ref, point, coord='x', plot_name='teste', saveplot=False):
    video_name = (video_name).split(sep='.')[0]
    file_dir = data_dir + video_name + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    if(file_names[0]=='None' and file_ref=='None'):
        print("No file selected")
        return
    elif(file_names[0]=='None'):
        file_path = file_dir + file_ref
    else:
        file_path = file_dir + file_names[0]

    metadata, keypoints = readFrameDATA(file_path, frame_n=0)
    n_frames, fps = metadata["n_frames"], metadata["fps"]
    joint_pairs = metadata["joint_pairs"]

    pairs = []
    for j in joint_pairs:
        pairs.append(pose_pairs[j])
    joints = np.unique(pairs)

    point_n = keypoints_mapping.index(point)
    joints = joints.tolist()
    if point_n in joints:
        point_idx = joints.index(point_n)
    else:
        print("Joint not found")
        return

    if(coord=='x'):
        plt.figure(figsize=[9,6])
        plt.title("Comparison in the X coordinate for the {}".format(point))
        plt.grid(True)
        if(file_ref != 'None'):
            file_path = file_dir + file_ref
            _, keypoints = readAllFramesDATA(file_path)
            keypoints = np.where(keypoints==-1, np.nan, keypoints)
            plt.plot(keypoints[:, point_idx, 0], label="Referência")
        for file_name in file_names:
            file_path = file_dir + file_name
            _, keypoints = readAllFramesDATA(file_path)
            keypoints = np.where(keypoints==-1, np.nan, keypoints)
            plt.plot(keypoints[:, point_idx, 0], label=(file_name).split(sep='.')[0])
            plt.legend()
    else:
        if not saveplot:
            plt.figure(figsize=[9,6])
            plt.title("Comparison Y {}".format(point))
            plt.grid(True)
        if(file_ref != 'None'):
            file_path = file_dir + file_ref
            _, keypoints = readAllFramesDATA(file_path)
            keypoints = np.where(keypoints==-1, np.nan, keypoints)
            plt.plot(keypoints[:, point_idx, 1], label="Referência")
        for file_name in file_names:
            file_path = file_dir + file_name
            _, keypoints = readAllFramesDATA(file_path)
            keypoints = np.where(keypoints==-1, np.nan, keypoints)
            plt.plot(keypoints[:, point_idx, 1], label=(file_name).split(sep='.')[0])
        plt.legend()

    if saveplot:
        plt.savefig(data_dir + video_name + '/' + plot_name + '.png')
        with open(data_dir + video_name + '/' + plot_name + '.json', "w") as write_file:
            json.dump(keypoints.tolist(), write_file)
    else:
        plt.show()

def showMetrics(video_name, file_names, file_ref, point, error_type='Error Graph', def_error=False, plot_name='teste', saveplot=False):
    if(file_names[0]=='None'):
        print('No file selected')
        return
    
    video_name = (video_name).split(sep='.')[0]
    file_dir = data_dir + video_name + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = file_dir + file_ref
    metadata, keypoints = readFrameDATA(file_path, frame_n=0)
    n_frames, fps = metadata["n_frames"], metadata["fps"]
    joint_pairs = metadata["joint_pairs"]

    pairs = []
    for j in joint_pairs:
        pairs.append(pose_pairs[j])
    joints = np.unique(pairs)

    point_n = keypoints_mapping.index(point)
    joints = joints.tolist()
    if point_n in joints:
        point_idx = joints.index(point_n)
    else:
        if def_error:
            print("Joint not found")
            return

    _, keypoints_ref = readAllFramesDATA(file_path)

    if(error_type=='Error Graph'):
        plt.figure(figsize=[9,6])
        if def_error:
            plt.title("Error x Frame ({})".format(point))
        else:
            plt.title("Error x Frame (Total)")
        plt.grid(True)
    else:
        data = np.zeros([len(file_names), len(joints)+1])
        data_fn = np.zeros([len(file_names), len(joints)+1])

    j = 0
    for file_name in file_names:
        fn_T = 0
        n_frames = metadata["n_frames"]
        Et = 0
        Et_keypoints = np.zeros(keypoints.shape[0])
        Et_keypoints_vec = np.zeros([n_frames, keypoints.shape[0]])
        Et_vec = np.zeros(n_frames)

        fn_vec = np.zeros(len(joints))

        file_path_cmp = file_dir + file_name
        _, keypoints_cmp = readAllFramesDATA(file_path_cmp)

        for i in range(n_frames):
            fn = 0
            E_tmp = np.power((keypoints_ref[i] - keypoints_cmp[i]),2)
            E_tmp = np.sum(E_tmp, axis=1)
            E_tmp = np.sqrt(E_tmp)

            if -1 in keypoints_cmp[i]:
                for k in range(len(keypoints_cmp[i])):
                    if -1 in keypoints_cmp[i,k]:
                        fn+=1
                        fn_vec[k]+=1
                        E_tmp[k] = 0
                E = np.sum(E_tmp, axis=0)/(len(joints)-fn)
                Et_keypoints += E_tmp
                E_tmp = np.where(E_tmp==0, np.nan, E_tmp)
            else:        
                E = np.sum(E_tmp, axis=0)/len(joints)
                Et_keypoints += E_tmp
            Et += E/n_frames 
            Et_vec[i] = E
            Et_keypoints_vec[i] = E_tmp
        n_frames_fn = n_frames - fn_vec
        Et_keypoints = np.divide(Et_keypoints, n_frames_fn)
        fn_T = np.sum(fn_vec, axis=0)

        if(error_type=='Error Graph'):
            if def_error:
                plt.plot(Et_keypoints_vec[:, point_idx], label=(file_name).split(sep='.')[0])
            else:
                plt.plot(Et_vec, label=(file_name).split(sep='.')[0])
        else:
            data[j, :len(joints)] = Et_keypoints
            data[j,-1] = Et
            data_fn[j, :len(joints)] = fn_vec
            data_fn[j,-1] = fn_T
        j+=1

    if(error_type=='Error Graph'):
        plt.legend()
        if saveplot:
            plt.savefig(data_dir + video_name + '/' + plot_name + '.png')
            with open(data_dir + video_name + '/' + plot_name + '.json', "w") as write_file:
                json.dump(Et_keypoints_vec.tolist(), write_file)
                json.dump(Et_vec.tolist(), write_file)
        else:
            plt.show()
    elif(error_type=='Error DF'):
        col = []
        for joint in joints:
            col.append("$E_{" + km[joint] + "}$")
        col.append("$E_{Total}$")
        row = []
        for file_name in file_names:
            row.append((file_name).split(sep='.')[0])
        df = pd.DataFrame(data=data,columns=col, index=row)
        with pd.option_context('display.float_format', '{:0.2f}'.format):
            if saveplot:
                latex = df.to_latex(index=False)
                file1 = open(data_dir + video_name + '/' + plot_name.tabtex, "w") 
                file1.write(latex) 
                file1.close()
            else:
                display(df)
                
    elif(error_type=='False Negatives DF'):
        col = []
        for joint in joints:
            col.append("$FN_{" + km[joint] + "}$")
        col.append("$FN_{Total}$")
        row = []
        for file_name in file_names:
            row.append((file_name).split(sep='.')[0])
        df = pd.DataFrame(data=data_fn,columns=col, index=row)
        with pd.option_context('display.float_format', '{:0.0f}'.format):
            if saveplot:
                latex = df.to_latex(index=False)
                file1 = open(data_dir + video_name + '/' + plot_name.tabtex, "w") 
                file1.write(latex) 
                file1.close()
            else:
                display(df)
