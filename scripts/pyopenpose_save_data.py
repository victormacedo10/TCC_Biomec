# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np

# Import Libs
openpose_path = '/home/megasxlr/VictorM/openpose/'
repo_path = '/home/megasxlr/VictorM/TCC_Biomec/'

# Change these variables to point to the correct folder (Release/x64 etc.) 
sys.path.append(openpose_path + 'build/python')
from openpose import pyopenpose as op
sys.path.append(repo_path + 'src')
from preprocessing_OP import organizeBiggestPerson, selectJoints
from visualizations_OP import poseDATAtoFrame, rectAreatoFrame, showFrame
from parameters import *

def save_data():
    # Flags
    parser = argparse.ArgumentParser()
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = openpose_path + "models/"

    try:
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        datum = op.Datum()

        video_name = "Remo"
        video_name_ext = video_name + ".mp4"
        file_name_ext = video_name + ".data"

        if(video_name_ext == "None"):
            print("No video found")
            return
        if(file_name_ext == "None"):
            print("No JSON found")
            return

        file_dir = data_dir + video_name + '/'
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file_path = file_dir + file_name_ext
        # n_frames = metadata["n_frames"]
        # fps = metadata["fps"]
        # frame_height, frame_width = metadata["frame_height"], metadata["frame_width"]
        
        # output_path = file_dir + output_name + ".data"
        # video_out_path = file_dir + output_name + ".mp4"
        
        video_path = repo_path + videos_dir + video_name_ext
        video_path = 0
        cap = cv2.VideoCapture(video_path)
        print(video_path)
        
        if(cap.isOpened() == False):
            print("Error opening video stream or file")

        while(cap.isOpened):
            # Process Image
            _, imageToProcess = cap.read()
            # Start timer
            print(imageToProcess.shape)
            timer = cv2.getTickCount()

            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop([datum])
            pose_keypoints = datum.poseKeypoints
            pose_keypoints = selectJoints(pose_keypoints, keypoints_mapping_BODY_25, SL_mapping)
            pose_keypoints = organizeBiggestPerson(pose_keypoints)

            img_out = poseDATAtoFrame(imageToProcess, pose_keypoints, [0], SL_mapping, SL_pairs, thickness=3, color = -1)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Display FPS on frame
            cv2.putText(img_out, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
            
            # Display Image
            cv2.namedWindow('OpenPose', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('OpenPose', 640,480)
            cv2.imshow("OpenPose", img_out)
            # while True:
            #     if(cv2.waitKey(25) & 0xFF == ord('q')):
            #         break
            if(cv2.waitKey(25) & 0xFF == ord('q')):
                break

    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(-1)

if __name__ == "__main__":
    save_data()