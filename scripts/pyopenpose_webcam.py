# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time

# Import Openpose
openpose_path = '/home/megasxlr/VictorM/openpose/'

try:
    # Change these variables to point to the correct folder (Release/x64 etc.) 
    sys.path.append(openpose_path + 'build/python')
    # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
    # sys.path.append('/usr/local/python')
    from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = openpose_path + "models/"

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    datum = op.Datum()

    cap = cv2.VideoCapture(0)
    
    if(cap.isOpened() == False):
        print("Error opening video stream or file")

    while(cap.isOpened):
        # Process Image
        _, imageToProcess = cap.read()

        # Start timer
        timer = cv2.getTickCount()

        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        img_out = datum.cvOutputData

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Display FPS on frame
        cv2.putText(img_out, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        # Display Image
        print("Body keypoints: \n" + str(datum.poseKeypoints))
        print(type(datum.poseKeypoints))
        cv2.imshow("OpenPose", img_out)
        if(cv2.waitKey(25) & 0xFF == ord('q')):
            break

except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(-1)