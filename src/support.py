import cv2
import time
import numpy as np
import json
import math

keypoints_mapping = np.array(['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
                    'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 
                    'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear'])

videos_dir = "../Videos/"
allvid_dir = "../Others/"
data_dir = "../Data/"

n_points = 18

def changeKeypointsVector(personwise_keypoints, keypoints_list):
    unsorted_keypoints = -1*np.ones([len(personwise_keypoints), n_points, 2])
    for n in range(len(personwise_keypoints)):
        for i in range(n_points):
            index = personwise_keypoints[n][i]
            if index == -1:
                continue
            unsorted_keypoints[n][i] = keypoints_list[int(personwise_keypoints[n][i])][0:2]
    return unsorted_keypoints

def readFrameJSON(file_path, frame_n=0):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i==0:
                metadata = json.loads(line)
            elif i==frame_n+1:
                data = json.loads(line)
    return metadata, data

def readFrameDATA(file_path, frame_n=0):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i==0:
                metadata = json.loads(line)
            elif i==frame_n+1:
                data = json.loads(line)
    keypoints = np.array(data["keypoints"]).astype(float)
    return metadata, keypoints

def readMultipleFramesDATA(file_path, frames=[0]):
    keypoints_vector = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if (i+1) in frames:
                data = json.loads(line)
                keypoints_vector.append(data["keypoints"])
    keypoints_vector = np.array(keypoints_vector).astype(float)
    return keypoints_vector

def readAllFramesDATA(file_path):
    keypoints_vector = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i==0:
                metadata = json.loads(line)
            else:
                data = json.loads(line)
                keypoints_vector.append(data["keypoints"])
    keypoints_vector = np.array(keypoints_vector).astype(float)
    return metadata, keypoints_vector

def getJoint(keypoints_list, personwise_keypoints, person, joint_name):
    joint_n = keypoints_mapping.tolist().index(joint_name)
    index = personwise_keypoints[person][joint_n]
    X = np.int32(keypoints_list[index.astype(int), 0])
    Y = np.int32(keypoints_list[index.astype(int), 1])
    return (X, Y)

def getFrame(video_name, n, allvid=False):
    if allvid:
        input_source = allvid_dir + video_name
    else:
        input_source = videos_dir + video_name
    
    cap = cv2.VideoCapture(input_source)

    cap.set(cv2.CAP_PROP_POS_FRAMES, n)

    has_frame, image = cap.read()
    
    cap.release()
    
    frame_width = image.shape[1]
    frame_height = image.shape[0]
    
    return image, frame_width, frame_height

def getFrames(video_name, n):
    input_source = videos_dir + video_name
    
    cap = cv2.VideoCapture(input_source)
    
    frames = []
    
    for i in n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        has_frame, image = cap.read()
        frames.append(image)
    
    cap.release()
    
    return frames

def angle3pt(a, b, c):
    if b in (a, c):
        raise ValueError("Undefined angle, two identical points", (a, b, c))
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    if ang < 0:
        ang += 360
    if ang > 180:
        ang = 360 - ang
    return ang

def rectangularArea(person):
    max_x , min_x, max_y, min_y = getVertices(person)
    return (max_x - min_x)*(max_y - min_y)

def getVertices(person):
    try:
        # print(person)
        max_x = max(person[:, 1])
        min_x = min([n for n in person[:, 1] if n>0])
        max_y = max(person[:, 0])
        min_y = min([n for n in person[:, 0] if n>0])
    except:
        print(person)
        return 0
    return (max_x - min_x)*(max_y - min_y)
