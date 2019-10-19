import os
import sys
import cv2
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from support import *
from detection import *
from preprocessing_OP import organizeBiggestPerson, selectJoints, fillwInterp
from visualizations_OP import poseDATAtoFrame, rectAreatoFrame, showFrame
from parameters import *
from kinematics import *
from sys import platform
import argparse

# Change these variables to point to the correct folder (Release/x64 etc.) 
sys.path.append(openpose_path + 'build/python')
from openpose import pyopenpose as op
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
                pose_keypoints = np.zeros([1, len(new_mapping), 2])
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
        print("Start Processing")
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
            angles = inverseKinematicsRowing(pose_keypoints)

            if show_video or save_video or show_frame:
                if 'B' in process_params:
                    img_out = poseDATAtoFrame(imageToProcess, pose_keypoints, 0, new_mapping, new_pairs, thickness=3, color = -1)
                else:
                    img_out = poseDATAtoFrame(imageToProcess, pose_keypoints, [0], new_mapping, new_pairs, thickness=3, color = -1)
                if save_video:
                    vid_writer.write(img_out)
            if save_data:
                file_data = {
                'keypoints': pose_keypoints.tolist(),
                'angles': angles.tolist()
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

def angle3pt(a, b, c):
    if b in (a, c):
        raise ValueError("Undefined angle, two identical points", (a, b, c))
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    if ang < 0:
        ang += 360
    if ang > 180:
        ang = 360 - ang
    return ang

def idxFromName(point_name, joints):
    point_n = keypoints_mapping.index(point_name)
    joints = joints.tolist()
    if point_n in joints:
        point_idx = joints.index(point_n)
    else:
        print("Joint not found")
        return None
    return point_idx

def saveProcessedAngles(video_name_ext, file_name, output_name, summary):
    
    print("Processing...")
    
    if(video_name_ext == "None"):
        print("No video found")
        return
    if(file_name == "None"):
        print("No DATA found")
        return
    video_name = (video_name_ext).split(sep='.')[0]
    file_dir = data_dir + video_name + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = file_dir + file_name
    metadata, data = readFrameDATA(file_path, frame_n=0)
    n_frames, fps = metadata["n_frames"], metadata["fps"]
    frame_height, frame_width = metadata["frame_height"], metadata["frame_width"]
    
    joint_pairs = metadata["joint_pairs"]
    pairs = []
    for j in joint_pairs:
        pairs.append(pose_pairs[j])
    joints = np.unique(pairs)

    output_path = file_dir + output_name + ".data"
    video_path = file_dir + output_name + ".mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    vid_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width,frame_height))

    metadata["summary"] = str(summary)

    for n in range(n_frames):
        
        _, main_keypoints = readFrameDATA(file_path, frame_n=n)
        frame, _, _ = getFrame(video_name_ext, n)

        a_idx = idxFromName('Right Ankle', joints)
        b_idx = idxFromName('Right Knee', joints)
        c_idx = idxFromName('Right Hip', joints)

        A = tuple(main_keypoints[a_idx].astype(int))
        B = tuple(main_keypoints[b_idx].astype(int))
        C = tuple(main_keypoints[c_idx].astype(int))
        
        cv2.line(frame, (A[0], A[1]), (B[0], B[1]), colors[0], 3, cv2.LINE_AA)
        cv2.line(frame, (B[0], B[1]), (C[0], C[1]), colors[0], 3, cv2.LINE_AA)

        cv2.circle(frame, A, 3, [0,0,255], -1, cv2.LINE_AA)
        cv2.circle(frame, B, 3, [0,0,255], -1, cv2.LINE_AA)
        cv2.circle(frame, C, 3, [0,0,255], -1, cv2.LINE_AA)

        angle = angle3pt(A, B, C)

        cv2.putText(frame, "Angle = {:.2f} degrees".format(angle), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
                        
        vid_writer.write(frame)
    
    vid_writer.release()
    print()
    print("Done")

def saveProcessedFileAll(video_name_ext, file_name, output_name, function_ext, summary):
    
    print("Processing...")
    
    if(video_name_ext == "None"):
        print("No video found")
        return
    if(file_name == "None"):
        print("No DATA found")
        return
    video_name = (video_name_ext).split(sep='.')[0]
    file_dir = data_dir + video_name + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = file_dir + file_name
    metadata, data = readFrameDATA(file_path, frame_n=0)
    n_frames, fps = metadata["n_frames"], metadata["fps"]
    frame_height, frame_width = metadata["frame_height"], metadata["frame_width"]
    
    joint_pairs = metadata["joint_pairs"]
    pairs = []
    for j in joint_pairs:
        pairs.append(pose_pairs[j])
    joints = np.unique(pairs)

    output_path = file_dir + output_name + ".data"
    video_path = file_dir + output_name + ".mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    vid_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width,frame_height))

    metadata["summary"] = str(summary)
    with open(output_path, 'w') as f:
        f.write(json.dumps(metadata))
        f.write('\n')
    
    sys.path.append('../postprocessing/')
    function = (function_ext).split(sep='.')[0]

    processing = __import__(function)
    processing_function = getattr(processing, 'processing_function')
    keypoints = processing_function(file_path, joints)

    for n in range(n_frames):
        main_keypoints = keypoints[n]
        file_data = {
            'keypoints': main_keypoints.tolist()
        }
        
        with open(output_path, 'a') as f:
            f.write(json.dumps(file_data))
            f.write('\n')
        
        frame, _, _ = getFrame(video_name_ext, n)
        
        for i in joint_pairs:
            pose_pairs[i][0]
            a_idx = (joints.tolist()).index(pose_pairs[i][0])
            b_idx = (joints.tolist()).index(pose_pairs[i][1])
            A = tuple(main_keypoints[a_idx].astype(int))
            B = tuple(main_keypoints[b_idx].astype(int))
            if (-1 in A) or (-1 in B):
                continue
            cv2.line(frame, (A[0], A[1]), (B[0], B[1]), colors[i], 3, cv2.LINE_AA)
                
        vid_writer.write(frame)
    
    vid_writer.release()
    print()
    print("Done")

def saveProcessedFileOnline(video_name_ext, file_name, output_name, function_ext, summary):
    
    print("Processing...")
    
    if(video_name_ext == "None"):
        print("No video found")
        return
    if(file_name == "None"):
        print("No DATA found")
        return
    video_name = (video_name_ext).split(sep='.')[0]
    file_dir = data_dir + video_name + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = file_dir + file_name
    metadata, data = readFrameDATA(file_path, frame_n=0)
    n_frames, fps = metadata["n_frames"], metadata["fps"]
    frame_height, frame_width = metadata["frame_height"], metadata["frame_width"]
    
    joint_pairs = metadata["joint_pairs"]
    pairs = []
    for j in joint_pairs:
        pairs.append(pose_pairs[j])
    joints = np.unique(pairs)

    output_path = file_dir + output_name + ".data"
    video_path = file_dir + output_name + ".mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    vid_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width,frame_height))

    metadata["summary"] = str(summary)
    with open(output_path, 'w') as f:
        f.write(json.dumps(metadata))
        f.write('\n')
    
    sys.path.append('../postprocessing/')
    function = (function_ext).split(sep='.')[0]

    for n in range(n_frames):
        t = time.time()

        processing = __import__(function)
        processing_function = getattr(processing, 'processing_function')
        main_keypoints = processing_function(file_path, n)
        file_data = {
            'keypoints': main_keypoints.tolist()
        }
        
        with open(output_path, 'a') as f:
            f.write(json.dumps(file_data))
            f.write('\n')
        
        frame, _, _ = getFrame(video_name_ext, n)
        
        for i in joint_pairs:
            pose_pairs[i][0]
            a_idx = (joints.tolist()).index(pose_pairs[i][0])
            b_idx = (joints.tolist()).index(pose_pairs[i][1])
            A = tuple(main_keypoints[a_idx].astype(int))
            B = tuple(main_keypoints[b_idx].astype(int))
            if (-1 in A) or (-1 in B):
                continue
            cv2.line(frame, (A[0], A[1]), (B[0], B[1]), colors[i], 3, cv2.LINE_AA)
                
        vid_writer.write(frame)
       
        time_taken = time.time() - t
        print("[{0:d}/{1:d}] {2:.1f} seconds/frame".format(n+1, n_frames, time_taken), end="\r")
    
    vid_writer.release()
    print()
    print("Done")