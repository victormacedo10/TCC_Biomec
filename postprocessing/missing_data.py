import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pykalman import KalmanFilter
from numpy import ma
sys.path.append('../src/')
from support import *

def missingDataInterpolation(X, interp='cubic'):
    X = np.where(X==-1, np.nan, X)
    X = pd.Series(X)
    X_out = X.interpolate(limit_direction='both', kind=interp)
    return X_out
                
def missingDataKalman(X, t):
    # Filter Configuration
    X = ma.where(X==-1, ma.masked, X)
    # t step
    dt = t[2] - t[1]

    # transition_matrix  
    F = [[1,  dt,   0.5*dt*dt], 
        [0,   1,          dt],
        [0,   0,           1]]  

    # observation_matrix   
    H = [1, 0, 0]

    # transition_covariance 
    Q = [[   1,     0,     0], 
        [   0,  1e-4,     0],
        [   0,     0,  1e-6]] 

    # observation_covariance 
    R = [0.01] # max error = 0.6m

    # initial_state_mean
    if(X[0]==ma.masked):
        X0 = [0,0,0]
    else:
        X0 = [X[0],0,0]

    # initial_state_covariance
    P0 = [[ 10,    0,   0], 
        [  0,    1,   0],
        [  0,    0,   1]]

    n_tsteps = len(t)
    n_dim_state = 3

    filtered_state_means = np.zeros((n_tsteps, n_dim_state))
    filtered_state_covariances = np.zeros((n_tsteps, n_dim_state, n_dim_state))

    # Kalman-Filter initialization
    kf = KalmanFilter(transition_matrices = F, 
                    observation_matrices = H, 
                    transition_covariance = Q, 
                    observation_covariance = R, 
                    initial_state_mean = X0, 
                    initial_state_covariance = P0)


    # iterative estimation for each new measurement
    for t in range(n_tsteps):
        if t == 0:
            filtered_state_means[t] = X0
            filtered_state_covariances[t] = P0
        else:
            filtered_state_means[t], filtered_state_covariances[t] = (
            kf.filter_update(
                filtered_state_means[t-1],
                filtered_state_covariances[t-1],
                observation = X[t])
            )

    return filtered_state_means    

def fillwInterp(keypoints_vector):
    for i in range(keypoints_vector.shape[1]):
        keypoints_vector[:,i,0] = missingDataInterpolation(keypoints_vector[:, i, 0])
        keypoints_vector[:,i,1] = missingDataInterpolation(keypoints_vector[:, i, 1])
    return keypoints_vector.astype(int)

file_path = "../Data/Diogo/Diogo.data"
metadata, keypoints_vector = readAllFramesDATA(file_path)
main_keypoints = keypoints_vector[0]
fps = metadata["fps"]
joint_pairs = metadata["joint_pairs"]
n_frames = metadata["n_frames"]
#t = np.linspace(0, len(keypoints_vector)/fps, len(keypoints_vector))
#x = keypoints_vector[:, 0, 0]
#y = keypoints_vector[:, 0, 1]
#
#filtered_state_means = missingDataKalman(x,t)
#X_kalman = filtered_state_means[:, 0]
#X_interp = missingDataInterpolation(x, interp='cubic')
#
#plt.figure()
#plt.plot(t, X_interp, label="Interpolation", markersize=1)
#plt.plot(t, X_kalman, label="Kalman", markersize=1)
#plt.plot(t, x, label="Missing", markersize=1)
#plt.grid()
#plt.xlabel("t (s)")
#plt.ylabel("Position (x)")
#plt.legend()
#
#filtered_state_means = missingDataKalman(y,t)
#Y_kalman = filtered_state_means[:, 0]
#Y_interp = missingDataInterpolation(y, interp='cubic')
#
#plt.figure()
#plt.plot(t, Y_interp, label="Interpolation", markersize=1)
#plt.plot(t, Y_kalman, label="Kalman", markersize=1)
#plt.plot(t, y, label="Missing", markersize=1)
#plt.grid()
#plt.xlabel("t (s)")
#plt.ylabel("Position (y)")
#plt.legend()

keypoints_test = np.zeros((n_frames, 6, 2))

keypoints_vector = fillwInterp(keypoints_vector)
