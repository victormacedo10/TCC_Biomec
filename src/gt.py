import os
import sys
import cv2
import time
import cv2
import json
import numpy as np
from IPython.display import display, HTML
import sys
sys.path.append('../src/')
from support import *
from detection import *
from visualizations import *

videos_dir = "../Videos/"
data_dir = "../Data/"

n_points = 18
 
pose_pairs = np.array([[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16], [2, 8], [5, 11]])
              
colors = [[0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], 
         [0,0,0], [0,0,0], [0,255,0], [0,255,0]]

keypoints_mapping = ['Nose', 'Neck', 'Right Sholder', 'Right Elbow', 'Right Wrist', 'Left Sholder', 
                    'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee', 'Right Ankle', 'Left Hip', 
                    'Left Knee', 'Left Ankle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear']

def gtRoutine(video_name_ext = "Diogo.mp4", file_name = "Diogo_BRF.data", output_name = "Diogo_GT", summary = "Ground Truth"):

  video_name = (video_name_ext).split(sep='.')[0]
  file_dir = data_dir + video_name + '/'
  if not os.path.exists(file_dir):
      os.makedirs(file_dir)
  file_path = file_dir + file_name
  metadata, keypoints_vector = readAllFramesDATA(file_path)
  n_frames, fps = metadata["n_frames"], metadata["fps"]
  frame_height, frame_width = metadata["frame_height"], metadata["frame_width"]

  joint_pairs = metadata["joint_pairs"]

  output_path = file_dir + output_name + ".data"
  video_path = file_dir + output_name + ".mp4"

  fourcc = cv2.VideoWriter_fourcc(*'X264')
  vid_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width,frame_height))

  metadata["summary"] = summary
  with open(output_path, 'w') as f:
      f.write(json.dumps(metadata))
      f.write('\n')

  pairs = []
  for j in joint_pairs:
      pairs.append(pose_pairs[j])
  joints = np.unique(pairs)

  n = 0
  joint_n = 0
  k = 1

  while(1):
    # Capture frame-by-frame
    frame, _, _ = getFrame(video_name_ext, n)
    main_keypoints = keypoints_vector[n]

    key = cv2.waitKey(30) 
    if key == ord('q'):
      print("Saving...")
      break
    elif key == 81: #left
      main_keypoints[joint_n, 0] -= k
    elif key == 82: #up
      main_keypoints[joint_n, 1] -= k
    elif key == 83: #right
      main_keypoints[joint_n, 0] += k
    elif key == 84: #down
      main_keypoints[joint_n, 1] += k
    elif key == 171: #+
      k+=1
    elif key == 173: #-
      if(k>1):
        k-=1
    elif (key == 32): #space
      if(joint_n>=len(joints)-1):
        joint_n = 0
      else:
        joint_n += 1
    if (key == 13 or key==141): #enter
      if(n>=n_frames-1):
        n=0
      else:
        n+=1
    elif(key == 8):
      if(n<=0):
        n=n_frames-1
      else:
        n-=1
    else:
      keypoints_vector[n] = main_keypoints

    keypointsDATAtoFrame(frame, main_keypoints, joint_pairs, 1)
    
    for i in range(len(main_keypoints)):
      A = tuple(main_keypoints[i].astype(int))
      if(i==joint_n):
        cv2.circle(frame, (A[0], A[1]), 1, (255,255,255), -1)
      else:
        cv2.circle(frame, (A[0], A[1]), 1, (0,0,0), -1)

    frame = cv2.copyMakeBorder(frame,50,0,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
    cv2.putText(frame, "[{}/{}]".format(n, n_frames), (10, 30), 
                cv2.FONT_HERSHEY_COMPLEX, .8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, "{}".format(keypoints_mapping[joints[joint_n]]), (200, 30), 
                cv2.FONT_HERSHEY_COMPLEX, .8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, "k: {}".format(k), (450, 30), 
                cv2.FONT_HERSHEY_COMPLEX, .8, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    frame[40:45,450:450+k,:] = 255

    cv2.imshow('Frame', frame)

  for n in range(n_frames):
    frame, _, _ = getFrame(video_name_ext, n)
    main_keypoints = keypoints_vector[n]
    file_data = {
          'keypoints': main_keypoints.tolist()
      }
      
    with open(output_path, 'a') as f:
        f.write(json.dumps(file_data))
        f.write('\n')

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

  cv2.destroyAllWindows()
  vid_writer.release()
  print("Done")

#gtRoutine()