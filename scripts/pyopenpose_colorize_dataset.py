# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

# Import Libs
openpose_path = '/home/megasxlr/VictorM/openpose/'
repo_path = '/home/megasxlr/VictorM/TCC_Biomec/'

# Change these variables to point to the correct folder (Release/x64 etc.) 
sys.path.append(openpose_path + 'build/python')
from openpose import pyopenpose as op
sys.path.append(repo_path + 'src')
from preprocessing_OP import organizeBiggestPerson, selectJoints, fillwInterp
from visualizations_OP import poseDATAtoFrame, rectAreatoFrame, showFrame, poseDATAtoCI
from parameters import *
from support import *
sys.path.append(repo_path + 'postprocessing')
from kalman_processing import processing_function

def processKeypointsData(video_name, output_name, nn_model="BODY_25", pose_model="BODY_25", process_params="BIK", 
                        save_video=True, show_video=True, save_data=True, show_frame=False, fitted=False):
    if fitted:
        width, height = get_curr_screen_geometry()
    else:
        width, height = 640, 480
    
    nn_mapping, new_mapping, new_pairs = getPoseParameters(nn_model, pose_model)

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = openpose_path + "models/"

    try:
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        datum = op.Datum()
        
        video_path = getVideoPath(video_name)
        video_out_path, output_path = setOutputPath(video_name, output_name)
        file_metadata = setMetadata(video_name, new_mapping, new_pairs)
        frame_width, frame_height, fps = file_metadata["frame_width"], file_metadata["frame_height"], file_metadata["fps"]
        
        cap = cv2.VideoCapture(video_path)
        if save_data:
            writeToDATA(output_path, file_metadata, write_mode='w')
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            vid_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (frame_width,frame_height))

        if(cap.isOpened() == False):
            print("Error opening video stream or file")
        while(cap.isOpened):
            # Process Image
            ret, imageToProcess = cap.read()
            if not ret:
                break

            # Start timer
            timer = cv2.getTickCount()

            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop([datum])
            pose_keypoints = datum.poseKeypoints
            try:
                pose_keypoints = pose_keypoints[:,:, :2]
            except:
                continue
            pose_keypoints = selectJoints(pose_keypoints, nn_mapping, new_mapping)
            if 'B' in process_params:
                pose_keypoints = organizeBiggestPerson(pose_keypoints)
                pose_keypoints = pose_keypoints[0]
            if show_video or save_video or show_frame:
                if 'B' in process_params:
                    img_out = poseDATAtoFrame(imageToProcess, pose_keypoints, 0, new_mapping, new_pairs, thickness=3, color = -1)
                else:
                    img_out = poseDATAtoFrame(imageToProcess, pose_keypoints, [0], new_mapping, new_pairs, thickness=3, color = -1)
                if save_video:
                    vid_writer.write(img_out)
            if save_data:
                pose_keypoints = np.round(pose_keypoints).astype(int)
                file_data = {
                'keypoints': pose_keypoints.tolist()
                }
                writeToDATA(output_path, file_data, write_mode='a')

            if show_video or show_frame:
                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                # Display FPS on frame
                cv2.putText(img_out, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                
                # Display Image
                cv2.namedWindow('OpenPose', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('OpenPose', (width, height))
                cv2.imshow("OpenPose", img_out)
            if show_frame:
                while True:
                    if(cv2.waitKey(25) & 0xFF == ord('q')):
                        break
            else:
                if(cv2.waitKey(25) & 0xFF == ord('q')):
                    break

    except KeyboardInterrupt:
        cap.release()
        vid_writer.release()
        cv2.destroyAllWindows()
        sys.exit(-1)

    cap.release()
    vid_writer.release()
    cv2.destroyAllWindows()

    if 'I' or 'K' in process_params:
        _, keypoints_vector = readAllFramesDATA(output_path)
        if 'I' in process_params:
            keypoints_vec = fillwInterp(keypoints_vector)
        if 'K' in process_params:
            keypoints_vec = processing_function(keypoints_vec)

        video_path = getVideoPath(video_name)
        video_out_path, output_path = setOutputPath(video_name, output_name + process_params)
        cap = cv2.VideoCapture(video_path)
        if save_data:
            writeToDATA(output_path, file_metadata, write_mode='w')
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            vid_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (frame_width,frame_height))

        if(cap.isOpened() == False):
            print("Error opening video stream or file")
        for i in range(len(keypoints_vec)):
            # Process Image
            ret, imageToProcess = cap.read()
            if not ret:
                break

            # Start timer
            timer = cv2.getTickCount()

            pose_keypoints = keypoints_vec[i]

            if show_video or save_video or show_frame:
                if 'B' in process_params:
                    img_out = poseDATAtoFrame(imageToProcess, pose_keypoints, 0, new_mapping, new_pairs, thickness=3, color = -1)
                else:
                    img_out = poseDATAtoFrame(imageToProcess, pose_keypoints, [0], new_mapping, new_pairs, thickness=3, color = -1)
                if save_video:
                    vid_writer.write(img_out)
            if save_data:
                file_data = {
                'keypoints': pose_keypoints.tolist()
                }
                writeToDATA(output_path, file_data, write_mode='a')

            if show_video or show_frame:
                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                # Display FPS on frame
                cv2.putText(img_out, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                
                # Display Image
                cv2.namedWindow('OpenPose', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('OpenPose', (width, height))
                cv2.imshow("OpenPose", img_out)
            if show_frame:
                while True:
                    if(cv2.waitKey(25) & 0xFF == ord('q')):
                        break
            else:
                if(cv2.waitKey(25) & 0xFF == ord('q')):
                    break

        cap.release()
        vid_writer.release()
        cv2.destroyAllWindows()

        frame_CI = poseDATAtoCI(img_out, keypoints_vec)

        plt.figure(figsize=[9,6])
        plt.imshow(frame_CI[:,:,[2,1,0]])
        plt.axis("off")
        plt.savefig("../Data/" + video_name + "/" + output_name + "_CI" + ".png")

if __name__ == "__main__":
    for i in range(19,20):
        name = "Erro_" + str(i)
        processKeypointsData(name, name, pose_model="Tennis", process_params="BIK", show_video=False)
    sys.exit(-1)