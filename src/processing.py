import cv2
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from inference import *
from preprocessing import *

videos_dir = "../Videos/"
data_dir = "../Data/"

n_points = 18

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

def processVideo(input_video, summary, output_name):
    video_path = videos_dir + input_video
    video_name = (input_video).split(sep='.')[0]

    if(video_name == "None"):
        print("No video found")
        return
    
    file_dir = data_dir + video_name + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    file_path = file_dir + output_name + '.json'
    output_path = file_dir + output_name + ".mp4"

    cap = cv2.VideoCapture(video_path)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    has_frame, image = cap.read()
    frame_width = image.shape[1]
    frame_height = image.shape[0]

    cap.set(2,0.0)

    file_metadata = {
        'video_name': video_name,
        'n_frames': length,
        'n_points': n_points,
        'frame_width': frame_width,
        'frame_height': frame_height,
        'fps': fps,
        'summary': summary
    }

    with open(file_path, 'w') as f:
        f.write(json.dumps(file_metadata))
        f.write('\n')

    fourcc = cv2.VideoWriter_fourcc(*'X264')
    vid_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width,frame_height))

    print("Processing...")

    for fn in range(length):
        t = time.time()

        has_frame, image = cap.read()

        output, _ = SingleFrameInference(image, print_time=False)

        # Gerar lista classificando os pontos para cada pessoa identificada
        personwise_keypoints, keypoints_list = processKeypoints(output, frame_width, frame_height)

        frame_clone = image.copy()

        for i in range(n_points-1):
            for n in range(len(personwise_keypoints)):
                index = personwise_keypoints[n][np.array(pose_pairs[i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])
                cv2.line(frame_clone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

        vid_writer.write(frame_clone)
        
        file_data = {
            'output': output.tolist()
        }

        with open(file_path, 'a') as f:
            f.write(json.dumps(file_data))
            f.write('\n')

        time_taken = time.time() - t
        print("[{0:d}/{1:d}] {2:.1f} seconds/frame".format(fn+1, length, time_taken), end="\r")

    vid_writer.release()
    cap.release()
    print()
    print("Done")