import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../src/')
from support import *

def processing_function(file_path, n):
    _, main_keypoints = readFrameDATA(file_path, frame_n=n)
    noise = 10*np.random.rand(main_keypoints.shape[0], main_keypoints.shape[1])
    out_keypoints = main_keypoints + noise

    return np.where(main_keypoints<0, -1, out_keypoints)