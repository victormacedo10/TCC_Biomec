import numpy as np
import matplotlib.pyplot as plt
import cv2
from pykalman import KalmanFilter
import time
import sys
sys.path.append('../src/')
from support import *

def processing_function(file_path, n):
	pass

def processing_lib(file_path, n):
	_, keypoints = readAllFramesDATA(file_path)
	measurements = np.asarray([(399,293),(403,299),(409,308),(416,315),(418,318),(420,323),(429,326),(423,328),(429,334),(431,337),(433,342),(434,352),(434,349),(433,350),(431,350),(430,349),(428,347),(427,345),(425,341),(429,338),(431,328),(410,313),(406,306),(402,299),(397,291),(391,294),(376,270),(372,272),(351,248),(336,244),(327,236),(307,220)])

	initial_state_mean = [measurements[0, 0],
												0,
												measurements[0, 1],
												0]

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

	plt.figure(1)
	times = range(measurements.shape[0])
	plt.plot(times, measurements[:, 0], 'bo',
					times, measurements[:, 1], 'ro',
					times, smoothed_state_means[:, 0], 'b--',
					times, smoothed_state_means[:, 2], 'r--',)
	plt.show()