import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from support import *
from detection import *

videos_dir = "../Videos/"
data_dir = "../Data/"

n_points = 18

map_idx = np.array([[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], 
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], 
          [47,48], [49,50], [53,54], [51,52], [55,56], 
          [37,38], [45,46]])

pose_pairs = np.array([[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16], [2, 8], [5, 11]])

colors = [[0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], 
         [0,0,0], [0,0,0], [0,255,0], [0,255,0]]

def organizeBiggestPerson(personwise_keypoints, keypoints_list):
    biggest_dict = {}
    unsorted_keypoints = -1*np.ones([len(personwise_keypoints), n_points, 2])
    sorted_keypoints = -1*np.ones([len(personwise_keypoints), n_points, 2])
    for n in range(len(personwise_keypoints)):
        for i in range(n_points):
            index = personwise_keypoints[n][i]
            if index == -1:
                continue
            unsorted_keypoints[n][i] = keypoints_list[int(personwise_keypoints[n][i])][0:2]
        biggest_dict[rectangularArea(unsorted_keypoints[n])] = n
    biggest_values = sorted(biggest_dict, reverse=True)
    n = 0
    for i in biggest_values:
        index = biggest_dict[int(i)]
        sorted_keypoints[n] = unsorted_keypoints[index]
        n += 1
    return sorted_keypoints

def fillMissingPoints(main_keypoints, last_keypoints):
    main_keypoints = np.where(main_keypoints<0, last_keypoints, main_keypoints)
    return main_keypoints

def removePairs(main_keypoints, joint_pairs):
    out_keypoints = -1*np.ones(main_keypoints.shape)
    pairs = []
    for j in joint_pairs:
        pairs.append(pose_pairs[j])
    n = np.unique(pairs)
    for i in n:
        out_keypoints[i] = main_keypoints[i]
    return out_keypoints

def removePairsFile(main_keypoints, joints):
    out_keypoints = []
    for i in joints:
        out_keypoints.append(main_keypoints[i])
    return np.array(out_keypoints)

def saveJointFile(video_name_ext, file_name, output_name, joint_pairs, summary, miss_points):
    
    print("Saving...")
    
    if(video_name_ext == "None"):
        print("No video found")
        return
    if(file_name == "None"):
        print("No JSON found")
        return
    video_name = (video_name_ext).split(sep='.')[0]
    file_dir = data_dir + video_name + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = file_dir + file_name
    metadata, data = readFrameJSON(file_path, frame_n=0)
    n_frames, fps = metadata["n_frames"], metadata["fps"]
    frame_height, frame_width = metadata["frame_height"], metadata["frame_width"]
    
    output_path = file_dir + output_name + ".data"
    video_path = file_dir + output_name + ".mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    vid_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width,frame_height))
    
    metadata["joint_pairs"] = joint_pairs
    metadata["summary"] = str(summary)
    
    pairs = []
    for j in joint_pairs:
        pairs.append(pose_pairs[j])
    joints = np.unique(pairs)

    with open(output_path, 'w') as f:
        f.write(json.dumps(metadata))
        f.write('\n')
    
    for n in range(n_frames):
        t = time.time()
        _, data = readFrameJSON(file_path, frame_n=n)
        output = np.array(data["output"]).astype(float)
        try:
            metadata["threshold"]
        except KeyError:
            threshold = 0.1
            n_interp_samples = 10
            paf_score_th = 0.1
            conf_th = 0.7
        else:
            threshold = metadata["threshold"]
            n_interp_samples = metadata["n_interp_samples"]
            paf_score_th = metadata["paf_score_th"]
            conf_th = metadata["conf_th"]
            
        personwise_keypoints, keypoints_list = keypointsFromHeatmap(output, frame_width, frame_height, threshold,
                                                                    n_interp_samples, paf_score_th, conf_th)
        sorted_keypoints = organizeBiggestPerson(personwise_keypoints, keypoints_list)
        main_keypoints = sorted_keypoints[0]

        if(miss_points == 'Fill w/ Last'):
            if (n>0):
                main_keypoints = fillMissingPoints(main_keypoints, last_keypoints)
            last_keypoints = np.copy(main_keypoints)

        # elif(miss_points == 'Fill w/ Kalman'):
        #     if (n>0):
        #         main_keypoints = fillMissingPoints(main_keypoints, last_keypoints)
        #     last_keypoints = np.copy(main_keypoints)

        main_keypoints = removePairsFile(main_keypoints, joints)

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
    
    