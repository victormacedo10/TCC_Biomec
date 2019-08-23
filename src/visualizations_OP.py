import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from support import *
from detection import *
from preprocessing import *
from parameters import *

colors_t = [[0,255,0], [0,0,255], [0,0,0], [255,255,255]]

def keypointsDATAtoFrame(image, keypoints, thickness=3, color = -1):
    for i in range(len(keypoints)):
        A = tuple(keypoints[i].astype(int))
        if (-1 in A) or (0 in A):
            continue
        cv2.circle(image, (A[0], A[1]), thickness, colors_2[i], -1)
    return image

def visualizeFrame(video_name, n):
    n = int(np.round(n))
    if(video_name == "None"):
        print("Choose a video")
    else:
        image, _, _ = getFrame(video_name, n)
        plt.figure(figsize=[14,10])
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")

def visualizeMainKeypoints(frame, pose_keypoints, persons, joint_pairs):    
    frame_out = frame.copy()
    
    if (persons[0] == -1) or (max(persons) >= len(pose_keypoints)):
        persons = np.arange(len(pose_keypoints))
    
    try:
        joint_pairs[0]
    except IndexError:
        pass
    else:
        if (joint_pairs[0] == -1):
            joint_pairs = np.arange(len(pose_pairs)-2)
    
    
    for n in persons:
        for i in joint_pairs:
            A = tuple(pose_keypoints[n][joint_pairs[i][0]].astype(int))
            B = tuple(pose_keypoints[n][joint_pairs[i][1]].astype(int))
            if (-1 in A) or (-1 in B):
                continue
            cv2.line(frame_out, (A[0], A[1]), (B[0], B[1]), colors[i], 3, cv2.LINE_AA)

    plt.figure(figsize=[9,6])
    plt.imshow(frame_out[:,:,[2,1,0]])
    plt.axis("off")

def rectAreatoFrame(frame, pose_keypoints):
    try:
        print(pose_keypoints.shape[0])
        for n in range(pose_keypoints.shape[0]):
            area = rectangularArea(pose_keypoints[n,:,:2])
            max_x , min_x, max_y, min_y = getVertices(pose_keypoints[n,:,:2])
            print(max_x , min_x, max_y, min_y)
            cv2.rectangle(frame, (min_y, min_x), (max_y, max_x), colors_t[n], 2, 1)
            print("{0}: {1}".format(n, area))
        return frame
    except IndexError:
        return frame

def poseDATAtoFrame(frame, pose_keypoints, persons, model, joint_pairs, thickness=3, color = -1):
    if model == "BODY_25":
        pose_pairs_MODEL = pose_pairs_BODY_25
    elif model == "BODY_21":
        pose_pairs_MODEL = pose_pairs_BODY_21
    elif model == "COCO":
        pose_pairs_MODEL = pose_pairs_COCO
    if joint_pairs == -1:
        joint_pairs = np.arange(pose_pairs_MODEL.shape[0])
    
    try:
        if persons == -1:
            persons = np.arange(pose_keypoints.shape[0])

    except IndexError:
        return frame

    pairs = []
    for j in joint_pairs:
        pairs.append(pose_pairs_MODEL[j])
    joints = np.unique(pairs)
    
    for n in persons:
        for i in joint_pairs:
            pose_pairs_MODEL[i][0]
            a_idx = (joints.tolist()).index(pose_pairs_MODEL[i][0])
            b_idx = (joints.tolist()).index(pose_pairs_MODEL[i][1])
            A = tuple(pose_keypoints[n][a_idx][:2].astype(int))
            B = tuple(pose_keypoints[n][b_idx][:2].astype(int))
            if (0 in A) or (0 in B):
                continue
            if(color == -1):
                cv2.line(frame, (A[0], A[1]), (B[0], B[1]), colors_25[i], thickness, cv2.LINE_AA)
            else:
                cv2.line(frame, (A[0], A[1]), (B[0], B[1]), colors_2[color], thickness, cv2.LINE_AA)
    return frame

def showFrame(frame, save=False):
    plt.figure(figsize=[9,6])
    plt.imshow(frame[:,:,[2,1,0]])
    plt.axis("off")
    plt.show()