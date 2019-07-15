from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('../src/')
from support import *

def processing_function(file_path, joints):
    _, keypoints = readAllFramesDATA(file_path)
    
    for i in range(len(joints)):
        measurements = np.copy(keypoints[:, i])

        initial_state_mean = [measurements[0, 0], 0,
                            measurements[0, 1], 0]

        transition_matrix = [[1, 1, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]]

        observation_matrix = [[1, 0, 0, 0],
                            [0, 0, 1, 0]]

        kf1 = KalmanFilter(transition_matrices = transition_matrix,
                        observation_matrices = observation_matrix,
                        initial_state_mean = initial_state_mean)

        kf1 = kf1.em(measurements, n_iter=5)
        (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)

        keypoints[:, i, 0] = smoothed_state_means[:, 0]
        keypoints[:, i, 1] = smoothed_state_means[:, 2]
        
    return keypoints