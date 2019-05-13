import numpy as np
import matplotlib.pyplot as plt
import cv2
from support import *

def processing_function(file_path, n):
	s = 2

    _, main_keypoints = readFrameDATA(file_path, frame_n=n)

    kp = main_keypoints[2, :]
    if(n==0):
    	state = np.array([kp[0],kp[1],0,0], dtype='float64')
		kalman = cv2.KalmanFilter(4,2,0)	
		kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
		                                [0., 1., 0., .1],
		                                [0., 0., 1., 0.],
		                                [0., 0., 0., 1.]])
		kalman.measurementMatrix = 1. * np.eye(2, 4)
		kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
		kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
		kalman.errorCovPost = 1e-1 * np.eye(4, 4)
		kalman.statePost = state
		measurement = np.array([kp[0], kp[1]], dtype='float64')
		posterior = kalman.correct(measurement)
    elif(n%s):
    	measurement = np.array([kp[0], kp[1]], dtype='float64')
    	posterior = kalman.correct(measurement)
    else:
	    prediction = kalman.predict()
	    measurement = np.array([kp[0], kp[1]], dtype='float64') 
	    posterior = kalman.correct(measurement)

    return np.where(main_keypoints<0, -1, out_keypoints)