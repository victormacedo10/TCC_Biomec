import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from support import *
from detection import *
from preprocessing import *
from processing import *
from visualizations import *

def colorizeImage(input_video, arguments, joint_pairs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], summary="Colorized image test"):
    video_name = (input_video).split(sep='.')[0]
    output_name = video_name
    if arguments != None:
        if 'D' in arguments:
            print("Initilize detection")
            videoInference(input_video, summary, output_name, threshold=0.1, n_interp_samples=10, paf_score_th=0.1, conf_th=0.7)
        if 'P' in arguments:
            print("Initilize preparation")
            saveJointFile(input_video, output_name + '.json', output_name + "_I", joint_pairs, summary, miss_points='Fill w/ Interp')
        if 'F' in arguments:
            print("Initilize filtering")
            saveProcessedFileAll(input_video, output_name + "_I" + ".data", output_name + "_ISK", "selective_kalman_processing.py", summary)
    else:
        print("Begin all steps")
        print("Initilize detection")
        videoInference(input_video, summary, output_name, threshold=0.1, n_interp_samples=10, paf_score_th=0.1, conf_th=0.7)
        print("Initilize preparation")
        saveJointFile(input_video, output_name + '.json', output_name + "_I", joint_pairs, summary, miss_points='Fill w/ Interp')
        print("Initilize filtering")
        saveProcessedFileAll(input_video, output_name + "_I" + ".data", output_name + "_ISK", "selective_kalman_processing.py", summary)
    frame = visualizeColoredVideo(video_name, output_name + "_ISK.data")
    plt.figure(figsize=[9,6])
    plt.imshow(frame[:,:,[2,1,0]])
    plt.axis("off")
    plt.savefig("../Data/" + video_name + "/" + output_name + "_ISK" + "_CI" + ".png")