# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
import pandas as pd

# Import Libs
openpose_path = '/home/megasxlr/VictorM/openpose/'
repo_path = '/home/megasxlr/VictorM/TCC_Biomec/'

# Change these variables to point to the correct folder (Release/x64 etc.) 
sys.path.append(openpose_path + 'build/python')
from openpose import pyopenpose as op
sys.path.append(repo_path + 'src')
from visualizations_OP import poseDATAtoFrame, rectAreatoFrame, showFrame
from parameters import *
from support import *
sys.path.append(repo_path + 'postprocessing')
from kalman_processing import processing_function

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

output_path = "/home/megasxlr/VictorM/TCC_Biomec/Data/Lucas/Lucas_Teste.data"
_, keypoints_all = readAllFramesDATA(output_path)
keypoints_vec = fillwInterp(keypoints_all)
keypoints_vec = processing_function(keypoints_vec)