from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('../src/')
from support import *

keypoints_mapping = ['Nose', 'Neck', 'Right Sholder', 'Right Elbow', 'Right Wrist', 'Left Sholder', 
                    'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee', 'Right Ankle', 'Left Hip', 
                    'Left Knee', 'Left Ankle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear']

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

def processing_function(file_path, joints):
	_, keypoints = readAllFramesDATA(file_path)
	joints_names_list = ['Neck', 'Right Sholder', 'Left Sholder', 'Right Hip', 'Right Knee', 'Right Ankle', 'Left Hip', 
                    'Left Knee', 'Left Ankle']
	joints_list = []
	for joint_name in joints_names_list:
		joints_list.append(joints.tolist().index(keypoints_mapping.index(joint_name)))
	for i in joints_list:
		measurements = np.copy(keypoints[:, i])
		keypoints[:, i, 0], keypoints[:, i, 1] = kalmanFilter(measurements)
	return keypoints