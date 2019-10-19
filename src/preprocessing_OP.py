import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def organizeBiggestPerson(pose_keypoints):
    try:
        biggest_dict = {}
        sorted_keypoints = np.zeros(pose_keypoints.shape)
        # print(pose_keypoints)
        for n in range(pose_keypoints.shape[0]):
            area = rectangularArea(pose_keypoints[n,:,:2])
            # print(area)
            biggest_dict[area] = n
        # print("dict: {}".format(biggest_dict))
        biggest_values = sorted(biggest_dict, reverse=True) 
        # print("values: {}".format(biggest_values))
        n = 0
        for i in biggest_values:
            index = int(biggest_dict[i])
            sorted_keypoints[n] = pose_keypoints[index]
            n += 1
    except:
        return pose_keypoints
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
    X = np.where(X==0, np.nan, X)
    X = pd.Series(X)
    X_out = X.interpolate(limit_direction='both', kind=interp)
    return X_out

def fillwInterp(keypoints_vector):
    for i in range(keypoints_vector.shape[1]):
        keypoints_vector[:,i,0] = missingDataInterpolation(keypoints_vector[:, i, 0])
        keypoints_vector[:,i,1] = missingDataInterpolation(keypoints_vector[:, i, 1])
    return keypoints_vector.astype(int)

def selectJoints(pose_keypoints, old_joints, new_joints):
    try:
        out_keypoints = np.zeros([pose_keypoints.shape[0], len(new_joints), pose_keypoints.shape[2]])
        for joint in new_joints:
            out_keypoints[:, new_joints.index(joint), :] = pose_keypoints[:, old_joints.index(joint), :]
    except:
        return pose_keypoints
    return out_keypoints
