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

SL_joints = ['Left Shoulder', 'Left Elbow', 'Left Wrist', 'Left Hip', 'Left Knee', 'Left Ankle']
SR_joints = ['Right Shoulder', 'Right Elbow', 'Right Wrist', 'Right Hip', 'Right Knee', 'Right Ankle']

try:
    # Change these variables to point to the correct folder (Release/x64 etc.) 
    sys.path.append(openpose_path + 'build/python')
    from openpose import pyopenpose as op
    sys.path.append(repo_path + 'src')
    from preprocessing_OP import organizeBiggestPerson
    from visualizations_OP import poseDATAtoFrame, rectAreatoFrame
except ImportError as e:
    print('Error: Could not import libs, check folder')
    raise e

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

    if(video_name_ext == "None"):
        print("No video found")
        return
    if(file_name_ext == "None"):
        print("No JSON found")
        return

    video_name = (video_name_ext).split(sep='.')[0]
    file_dir = data_dir + video_name + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = file_dir + file_name_ext
    metadata, data = readFrameJSON(file_path, frame_n=0)
    #n_frames = metadata["n_frames"]
    fps = metadata["fps"]
    frame_height, frame_width = metadata["frame_height"], metadata["frame_width"]
    
    output_path = file_dir + output_name + ".data"
    video_path = file_dir + output_name + ".mp4"
    
    video_path = repo_path + videos_dir + "Treino_08082019_2.mp4"
    #video_path = 0
    cap = cv2.VideoCapture(video_path)
    
    if(cap.isOpened() == False):
        print("Error opening video stream or file")

    while(cap.isOpened):
        # Process Image
        _, imageToProcess = cap.read()

        # Start timer
        timer = cv2.getTickCount()

        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])
        pose_keypoints_mp = datum.poseKeypoints

        pose_keypoints_mp = selectJoints(pose_keypoints_mp, keypoints_mapping_BODY_25, SL_joints)
        pose_keypoints = organizeBiggestPerson(pose_keypoints_mp)

        

        img_out = poseDATAtoFrame(imageToProcess, pose_keypoints, -1, 'BODY_25', -1, thickness=3, color = -1)
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Display FPS on frame
        cv2.putText(img_out, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        
        # Display Image
        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        # print(type(datum.poseKeypoints))
        cv2.imshow("OpenPose", img_out)
        while True:
            if(cv2.waitKey(25) & 0xFF == ord('q')):
                break

except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(-1)