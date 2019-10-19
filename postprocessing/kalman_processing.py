from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('../src/')
from support import *

def kalmanFilter(measurements):

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

	kf2 = KalmanFilter(transition_matrices = transition_matrix,
						observation_matrices = observation_matrix,
						initial_state_mean = initial_state_mean,
						observation_covariance = 10*kf1.observation_covariance,
						em_vars=['transition_covariance', 'initial_state_covariance'])

	kf2 = kf2.em(measurements, n_iter=5)
	(smoothed_state_means, smoothed_state_covariances)  = kf2.smooth(measurements)
	
	return smoothed_state_means[:, 0], smoothed_state_means[:, 2]

def processing_function(keypoints_vec):
	# _, keypoints_vec = readAllFramesDATA(file_path)
	
	for i in range(keypoints_vec.shape[1]):
		measurements = np.copy(keypoints_vec[:, i])
		keypoints_vec[:, i, 0], keypoints_vec[:, i, 1] = kalmanFilter(measurements)
	return keypoints_vec