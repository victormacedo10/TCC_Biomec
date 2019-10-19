# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np

# Import Libs
openpose_path = '/home/megasxlr/VictorM/openpose/'
repo_path = '/home/megasxlr/VictorM/TCC_Biomec/'

# Change these variables to point to the correct folder (Release/x64 etc.) 
sys.path.append(openpose_path + 'build/python')
from openpose import pyopenpose as op
sys.path.append(repo_path + 'src')
from preprocessing_OP import organizeBiggestPerson, selectJoints
from visualizations_OP import poseDATAtoFrame, rectAreatoFrame, showFrame
from parameters import *
from support import *
from kinematics import *

# get and save pose dimensions
def poseManualAdjustmentInteface(webcam_number=1, nn_model="BODY_25", pose_model="SL", fitted=True):
    folder_path = saveSelectedFolder()
    try:
        if fitted:
            width, height = get_curr_screen_geometry()
        else:
            width, height = 640, 480
        cap = cv2.VideoCapture(webcam_number)
        if not (cap.isOpened):
            print("Error opening video stream or file")
        while(cap.isOpened):
            # Process Image
            ret, imageToProcess = cap.read()
            if not ret:
                break

            # Display Image
            cv2.namedWindow('OpenPose', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('OpenPose', (width, height))
            cv2.imshow("OpenPose", imageToProcess)
            key = cv2.waitKey(25)
            if(key & 0xFF == ord('q')):
                break
            elif(key == 13):
                cap.release()
                #cv2.destroyAllWindows()
                doPoseAdjustment(imageToProcess, folder_path)
                break
            elif(key != -1):
                print(key)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        sys.exit(-1)

def doPoseAdjustment(imageToProcess, folder_path, webcam_number=1, nn_model="BODY_25", pose_model="SL", fitted=True):

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
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])
        pose_keypoints = datum.poseKeypoints
        try:
            pose_keypoints = pose_keypoints[:,:, :2]
        except:
            pose_keypoints = np.zeros([1, len(new_mapping), 2])

        pose_keypoints = selectJoints(pose_keypoints, nn_mapping, new_mapping)
        pose_keypoints = organizeBiggestPerson(pose_keypoints)
        pose_keypoints = pose_keypoints[0]
        for i in range(len(pose_keypoints)):
            if 0 in pose_keypoints[i]:
                pose_keypoints[i, 0], pose_keypoints[i, 1] = imageToProcess.shape[0]/2, imageToProcess.shape[1]/2 

        joint_n = 0
        k = 1
        while True:
            image = np.copy(imageToProcess)
            img_out = poseDATAtoFrame(image, pose_keypoints, 0, new_mapping, new_pairs, thickness=1, color = -1)
            for i in range(len(pose_keypoints)):
                A = tuple(pose_keypoints[i].astype(int))
                if(i==joint_n):
                    cv2.circle(img_out, (A[0], A[1]), 1, (255,255,255), -1)
                else:
                    cv2.circle(img_out, (A[0], A[1]), 1, (0,0,0), -1)
            img_processed = np.copy(img_out)
            img_out = cv2.copyMakeBorder(img_out,50,0,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
            cv2.putText(img_out, "{}".format(new_mapping[joint_n]), (200, 30), 
                        cv2.FONT_HERSHEY_COMPLEX, .8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(img_out, "k: {}".format(k), (450, 30), 
                        cv2.FONT_HERSHEY_COMPLEX, .8, (255, 255, 255), 2, lineType=cv2.LINE_AA)

            img_out[40:45,450:450+k,:] = 255
            cv2.namedWindow('OpenPose', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('OpenPose', (width, height))
            cv2.imshow("OpenPose", img_out)
            key = cv2.waitKey(30) 
            if key == ord('q'):
                print("Not saving...")
                break
            elif key == 81: #left
                pose_keypoints[joint_n, 0] -= k
            elif key == 82: #up
                pose_keypoints[joint_n, 1] -= k
            elif key == 83: #right
                pose_keypoints[joint_n, 0] += k
            elif key == 84: #down
                pose_keypoints[joint_n, 1] += k
            elif key == 171: #+
                k+=1
            elif key == 173: #-
                if(k>1):
                    k-=1
            elif (key == 32): #space
                if(joint_n>=pose_keypoints.shape[0]-1):
                    joint_n = 0
                else:
                    joint_n += 1
            elif(key == 8):
                if(joint_n<=0-1):
                    joint_n = pose_keypoints.shape[0]
                else:
                    joint_n -= 1
            elif(key == 13 or key==141): #enter
                print("Saving dimensions")
                cv2.destroyAllWindows()
                mmppx, mmppy = getMmppInterface(img_processed)
                saveCalibrations(img_processed, pose_keypoints, new_mapping, folder_path, mmppx, mmppy, mode="manual")
                break 
    except KeyboardInterrupt:
        print("Not saving...")
        cv2.destroyAllWindows()
        sys.exit(-1)

def saveCalibrations(image_output, pose_keypoints, mapping, folder_path, mmppx, mmppy, mode="auto", thickness=1, textsize=0.3, fitted=True, show_plot=True):
    if fitted:
        width, height = get_curr_screen_geometry()
    else:
        width, height = 640, 480
    file_path = folder_path.split("/")
    file_path = file_path[-1]
    file_path = folder_path + "/" + file_path

    pose_keypoints_xy = getKeypointsCoord(pose_keypoints, image_output.shape[0], mmppx=mmppx, mmppy=mmppy)
    distances = getBoneDimensions(pose_keypoints_xy)
    showRowingChainDistancesPlot(pose_keypoints_xy, file_path + "_" + mode + "_plot.png", show_plot)
    file_data = {'mapping': mapping,
                'pixel_conversion' : [mmppx, mmppy],
                'distances': distances.tolist()}
    writeToDATA(file_path + "_" + mode + ".calib", file_data, write_mode='w')

    adj=2
    for i in range(len(pose_keypoints)-1):
        A = pose_keypoints[i]
        B = pose_keypoints[i+1]
        D = (int((A[0] + B[0])/2), int((A[1] + B[1])/2))
        str_to_write = str(int(round(distances[i])))
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_len = cv2.getTextSize(str_to_write, font, textsize, thickness)[0]
        cv2.rectangle(image_output, (D[0]-adj, D[1]+adj), (D[0] + text_len[0]+adj, D[1] - text_len[1]-adj), (255,0,0), -1)
        cv2.putText(image_output, str_to_write, D, font, textsize,(0,0,0),thickness,cv2.LINE_AA)
        i+=1

    if show_plot:
        cv2.namedWindow('OpenPose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('OpenPose', (width, height))
        cv2.imshow("OpenPose", image_output)
    cv2.imwrite(file_path + "_" + mode + "_image.png" ,image_output)

def getMmppInterface(frame):
    # Select ROI
    r = cv2.selectROI(frame)
    cv2.destroyAllWindows()
    x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])
    side = True
    while True:
        # Crop image
        imCrop = frame[y:y+h, x:x+w]
        # Display cropped image
        key = cv2.waitKey(25)
        if key == ord('q'):
            print("Not saving...") 
            break
        elif key == 81: #left
            if side:
                x -= 1
                w += 1
            else:
                w -= 1
        elif key == 82: #up
            if side:
                y -= 1
                h += 1
            else:
                h -= 1
        elif key == 83: #right
            if side:
                x += 1
                w -= 1
            else:
                w += 1
        elif key == 84: #down
            if side:
                y += 1
                h -= 1
            else:
                h += 1
        elif (key == 32): #space
            side = not(side)
        elif(key == 13 or key==141): #enter
            print("Saving conversion")
            cv2.destroyAllWindows()
            break
        cv2.imshow("Image", imCrop)
    p_x = w
    p_y = h

    mmppx = object_x_mm/p_x
    mmppy = object_y_mm/p_y
    return mmppx, mmppy

def findGreatesContour(contours):
    largest_area = 0
    largest_contour_index = -1
    i = 0
    total_contours = len(contours)
    while (i < total_contours ):
        area = cv2.contourArea(contours[i])
        if(area > largest_area):
            largest_area = area
            largest_contour_index = i
        i+=1
            
    return largest_area, largest_contour_index

def getMarkerRegion(frame):

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of yellow color in HSV
    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([40,255,255])

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    contours, _ =  cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    _, largest_contour_index = findGreatesContour(contours)

    cv2.drawContours(frame, contours[largest_contour_index], -1, (0, 0, 255), 3)
    region = np.array(contours[largest_contour_index])
    #img_out = np.zeros(frame.shape)
    #cv2.rectangle(img_out, (min(region[:, :, 0]), min(region[:, :, 1])), (max(region[:, :, 0]), max(region[:, :, 1])), (255,0,0), -1)
    x, y, w, h = min(region[:, :, 1]), min(region[:, :, 0]), max(region[:, :, 1]) - min(region[:, :, 1]), max(region[:, :, 0]) - min(region[:, :, 0])
    return int(x), int(y), int(w), int(h)

def getMmppAutomatically(frame):
    x, y, w, h = getMarkerRegion(frame)
    p_x = w
    p_y = h

    mmppx = object_x_mm/p_x
    mmppy = object_y_mm/p_y
    return mmppx, mmppy

def poseAutomaticAdjustmentInterface(n_mean=5, enter_folder=True, webcam_number=1, nn_model="BODY_25", pose_model="SL", fitted=True):
    if fitted:
        width, height = get_curr_screen_geometry()
    else:
        width, height = 640, 480
    if enter_folder:
        folder_path = saveSelectedFolder()
    else:
        folder_path = repo_path + "Calib/" + "Tmp"
    print("Get set...")
    time.sleep(10)
    print("Started")
    
    nn_mapping, new_mapping, new_pairs = getPoseParameters(nn_model, pose_model)

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = openpose_path + "models/"

    try:
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        cap = cv2.VideoCapture(webcam_number)
        if(cap.isOpened() == False):
            print("Error opening video stream or file")

        keep_running = True
        while keep_running:
            ret, imageToProcess = cap.read()
            frame = np.copy(imageToProcess)
            if not ret:
                break
            datum = op.Datum()
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop([datum])
            pose_keypoints = datum.poseKeypoints
            try:
                pose_keypoints = pose_keypoints[:,:, :2]
            except:
                pose_keypoints = np.zeros([1, len(new_mapping), 2])

            pose_keypoints = selectJoints(pose_keypoints, nn_mapping, new_mapping)
            pose_keypoints = organizeBiggestPerson(pose_keypoints)
            pose_keypoints = pose_keypoints[0]
            img_processed = poseDATAtoFrame(imageToProcess, pose_keypoints, 0, new_mapping, new_pairs, thickness=1, color = -1)
            for i in range(len(pose_keypoints)):
                if 0 in pose_keypoints[i]:
                    print("Wait till all joints are found")
                else:
                    keep_running = False
            if not keep_running:
                print("Joints found")

    except KeyboardInterrupt:
        print("Not saving...")
        cv2.destroyAllWindows()
        sys.exit(-1)

    mmppx, mmppy = getMmppAutomatically(frame)
    saveCalibrations(img_processed, pose_keypoints, new_mapping, folder_path, mmppx, mmppy, mode="automatic", show_plot=False)

if __name__ == "__main__":
    #poseManualAdjustmentInteface()    
    poseAutomaticAdjustmentInterface()
