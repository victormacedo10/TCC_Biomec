import cv2
import time
import numpy as np

proto_file = "../Models/Openpose/coco/pose_deploy_linevec.prototxt"
weights_file = "../Models/Openpose/coco/pose_iter_440000.caffemodel"

def SingleFrameInference(frame, print_time=True):
    t = time.time()
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
    
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    
    # Fix the input Height and get the width according to the Aspect Ratio
    in_height = 368
    in_width = int((in_height/frame_height)*frame_width)

    inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (in_width, in_height),
                              (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inp_blob)
    net_output = net.forward()
    time_taken = time.time() - t
    if print_time:
        print("Time Taken = {}".format(time_taken))
    return net_output, time_taken