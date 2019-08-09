# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time

# Import Libs
openpose_path = '/home/megasxlr/VictorM/openpose/'
repo_path = '/home/megasxlr/VictorM/TCC_Biomec/'

try:
    # Change these variables to point to the correct folder (Release/x64 etc.) 
    sys.path.append(openpose_path + 'build/python')
    from openpose import pyopenpose as op
    sys.path.append(repo_path + 'src')
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