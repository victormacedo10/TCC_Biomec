import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from numpy import ma
import json
from support import *
from detection import *

videos_dir = "../Videos/"
data_dir = "../Data/"

n_points = 18

map_idx = np.array([[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], 
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], 
          [47,48], [49,50], [53,54], [51,52], [55,56], 
          [37,38], [45,46]])

pose_pairs = np.array([[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16], [2, 8], [5, 11]])

colors = [[0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], 
         [0,0,0], [0,0,0], [0,255,0], [0,255,0]]

def organizeBiggestPerson(personwise_keypoints, keypoints_list):
    biggest_dict = {}
    unsorted_keypoints = -1*np.ones([len(personwise_keypoints), n_points, 2])
    sorted_keypoints = -1*np.ones([len(personwise_keypoints), n_points, 2])
    for n in range(len(personwise_keypoints)):
        for i in range(n_points):
            index = personwise_keypoints[n][i]
            if index == -1:
                continue
            unsorted_keypoints[n][i] = keypoints_list[int(personwise_keypoints[n][i])][0:2]
        biggest_dict[rectangularArea(unsorted_keypoints[n])] = n
    biggest_values = sorted(biggest_dict, reverse=True)
    n = 0
    for i in biggest_values:
        index = biggest_dict[int(i)]
        sorted_keypoints[n] = unsorted_keypoints[index]
        n += 1
    return sorted_keypoints

def fillwLast(keypoints_vector):
    for n in range(len(keypoints_vector)):
        if(n>0):
            keypoints_vector[n] = np.where(keypoints_vector[n]<0, keypoints_vector[n-1], keypoints_vector[n])
    return keypoints_vector

def missingDataKalman(X, t):
    X = ma.where(X==-1, ma.masked, X)

    dt = t[2] - t[1]

    F = [[1,  dt,   0.5*dt*dt], 
        [0,   1,          dt],
        [0,   0,           1]]  

    H = [1, 0, 0]

    Q = [[   1,     0,     0], 
        [   0,  1e-4,     0],
        [   0,     0,  1e-6]] 

    R = [0.01]
    
    if(X[0]==ma.masked):
        X0 = [0,0,0]
    else:
        X0 = [X[0],0,0]
    P0 = [[ 10,    0,   0], 
        [  0,    1,   0],
        [  0,    0,   1]]
    n_tsteps = len(t)
    n_dim_state = 3
    filtered_state_means = np.zeros((n_tsteps, n_dim_state))
    filtered_state_covariances = np.zeros((n_tsteps, n_dim_state, n_dim_state))
    kf = KalmanFilter(transition_matrices = F, 
                    observation_matrices = H, 
                    transition_covariance = Q, 
                    observation_covariance = R, 
                    initial_state_mean = X0, 
                    initial_state_covariance = P0)
    for t in range(n_tsteps):
        if t == 0:
            filtered_state_means[t] = X0
            filtered_state_covariances[t] = P0
        else:
            filtered_state_means[t], filtered_state_covariances[t] = (
            kf.filter_update(filtered_state_means[t-1],
                            filtered_state_covariances[t-1],
                            observation = X[t]))
    return filtered_state_means[:, 0] 

def fillwKalman(keypoints_vector, t):
    for i in range(keypoints_vector.shape[1]):
        keypoints_vector[:,i,0] = missingDataKalman(keypoints_vector[:, i, 0], t)
    return keypoints_vector

def missingDataInterpolation(X, interp='cubic'):
    X = np.where(X==-1, np.nan, X)
    X = pd.Series(X)
    X_out = X.interpolate(limit_direction='both', kind=interp)
    return X_out

def fillwInterp(keypoints_vector):
    for i in range(keypoints_vector.shape[1]):
        keypoints_vector[:,i,0] = missingDataInterpolation(keypoints_vector[:, i, 0])
        keypoints_vector[:,i,1] = missingDataInterpolation(keypoints_vector[:, i, 1])
    return keypoints_vector.astype(int)

def removePairs(main_keypoints, joint_pairs):
    out_keypoints = -1*np.ones(main_keypoints.shape)
    pairs = []
    for j in joint_pairs:
        pairs.append(pose_pairs[j])
    n = np.unique(pairs)
    for i in n:
        out_keypoints[i] = main_keypoints[i]
    return out_keypoints

def removePairsFile(main_keypoints, joints):
    out_keypoints = []
    for i in joints:
        out_keypoints.append(main_keypoints[i])
    return np.array(out_keypoints)

def saveJointFile(video_name_ext, file_name, output_name, joint_pairs, summary, miss_points):
    print(joint_pairs)
    if(video_name_ext == "None"):
        print("No video found")
        return
    if(file_name == "None"):
        print("No JSON found")
        return
    video_name = (video_name_ext).split(sep='.')[0]
    file_dir = data_dir + video_name + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = file_dir + file_name
    metadata, data = readFrameJSON(file_path, frame_n=0)
    n_frames = metadata["n_frames"]
    fps = metadata["fps"]
    frame_height, frame_width = metadata["frame_height"], metadata["frame_width"]
    
    output_path = file_dir + output_name + ".data"
    video_path = file_dir + output_name + ".mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    vid_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width,frame_height))
    
    metadata["joint_pairs"] = joint_pairs
    metadata["summary"] = str(summary)
    
    pairs = []
    for j in joint_pairs:
        pairs.append(pose_pairs[j])
    joints = np.unique(pairs)

    with open(output_path, 'w') as f:
        f.write(json.dumps(metadata))
        f.write('\n')

    keypoints_vector = np.zeros((n_frames, len(joints), 2))
    print("Organizing data...")
    for n in range(n_frames):
        t = time.time()
        _, data = readFrameJSON(file_path, frame_n=n)
        output = np.array(data["output"]).astype(float)
        try:
            metadata["threshold"]
        except KeyError:
            threshold = 0.1
            n_interp_samples = 10
            paf_score_th = 0.1
            conf_th = 0.7
        else:
            threshold = metadata["threshold"]
            n_interp_samples = metadata["n_interp_samples"]
            paf_score_th = metadata["paf_score_th"]
            conf_th = metadata["conf_th"]
            
        personwise_keypoints, keypoints_list = keypointsFromHeatmap(output, frame_width, frame_height, threshold,
                                                                    n_interp_samples, paf_score_th, conf_th)
        sorted_keypoints = organizeBiggestPerson(personwise_keypoints, keypoints_list)
        main_keypoints = sorted_keypoints[0]

        main_keypoints = removePairsFile(main_keypoints, joints)
        keypoints_vector[n] = main_keypoints
        time_taken = time.time() - t
        print("[{0:d}/{1:d}] {2:.1f} seconds/frame".format(n+1, n_frames, time_taken), end="\r")

    if(miss_points == 'Fill w/ Last'):
        print("Interpolating Last...")
        keypoints_vector = fillwLast(keypoints_vector)
    elif(miss_points == 'Fill w/ Kalman'):
        print("Interpolating Kalman...")
        t = np.linspace(0, len(keypoints_vector)/fps, len(keypoints_vector))
        keypoints_vector = fillwKalman(keypoints_vector, t)
    elif(miss_points == 'Fill w/ Interp'):
        print("Interpolating Interpolation...")
        keypoints_vector = fillwInterp(keypoints_vector)

    print("Saving...")
    for n in range(n_frames):
        t = time.time()
        main_keypoints = keypoints_vector[n]
        file_data = {
            'keypoints': main_keypoints.tolist()
        }
        
        with open(output_path, 'a') as f:
            f.write(json.dumps(file_data))
            f.write('\n')
        
        frame, _, _ = getFrame(video_name_ext, n)
        
        for i in joint_pairs:
            pose_pairs[i][0]
            a_idx = (joints.tolist()).index(pose_pairs[i][0])
            b_idx = (joints.tolist()).index(pose_pairs[i][1])
            A = tuple(main_keypoints[a_idx].astype(int))
            B = tuple(main_keypoints[b_idx].astype(int))
            if (-1 in A) or (-1 in B):
                continue
            cv2.line(frame, (A[0], A[1]), (B[0], B[1]), colors[i], 3, cv2.LINE_AA)
                
        vid_writer.write(frame)
       
        time_taken = time.time() - t
        print("[{0:d}/{1:d}] {2:.1f} seconds/frame".format(n+1, n_frames, time_taken), end="\r")
    
    vid_writer.release()
    print()
    print("Done")
    
    