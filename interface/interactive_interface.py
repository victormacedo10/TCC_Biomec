import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import ipywidgets as wg
from IPython.display import display, HTML
import sys
from preview_video_interface import *
from choose_video_interface import *
from detect_keypoints_interface import *
from pre_process_interface import *
from processing_interface import *
from gt_interface import *
from analyze_data_interface import *
sys.path.append('../src/')
from preparations import *
from detection import *
from visualizations import *

videos_dir = "../Videos/"
allvid_dir = "../Others/"
data_dir = "../Data/"
post_dir = "../postprocessing/"

keypoints_mapping = ['Nose', 'Neck', 'Right Sholder', 'Right Elbow', 'Right Wrist', 'Left Sholder', 
                    'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee', 'Right Ankle', 'Left Hip', 
                    'Left Knee', 'Left Ankle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear']

def interactiveInterface():

    videos_list = os.listdir(videos_dir)
    video_options = ["None"]
    for video in videos_list:
        video_options.append(video)
    video_dropdown = wg.Dropdown( options=video_options,
                            description='Video:',
                            disabled=False)
       
    allvid_list = os.listdir(allvid_dir)
    allvid_options = ["None"]
    for allvid in allvid_list:
        allvid_options.append(allvid)
    allvid_dropdown = wg.Dropdown( options=allvid_options,
                            description='All Videos:',
                            disabled=False)

    video_path = videos_dir + video_dropdown.value
    video_name = (video_dropdown.value).split(sep='.')[0]

    file_dir = data_dir + video_name + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    files_list = os.listdir(file_dir)
    
    json_list = ["None"]
    for names in files_list:
        if names.endswith(".json"):
            json_list.append(names)
    mp4_list = ["None"]
    for names in files_list:
        if names.endswith(".mp4"):
            mp4_list.append(names)
    data_list = ["None"]
    for names in files_list:
        if names.endswith(".data"):
            data_list.append(names)

    json_dropdown = wg.Dropdown(options=json_list,
                            description='File:',
                            disabled=False)   
    mp4_dropdown = wg.Dropdown(options=mp4_list,
                            description='Processed:',
                            disabled=False)  
    data_dropdown = wg.Dropdown(options=data_list,
                            description='Data:',
                            disabled=False)  
    data_ref = wg.Dropdown(options=data_list,
                            description='Reference:',
                            disabled=False) 
    
    frame_n = wg.Play(
        value=0,
        min=0,
        max=100,
        step=1,
        description="Press play",
        disabled=False
    )

    def onVideoChange(change):
        if(video_dropdown.value == "None"):
            print("Choose a video")
            return
        
        video_path = videos_dir + video_dropdown.value
        video_name = (video_dropdown.value).split(sep='.')[0]

        file_dir = data_dir + video_name + '/'
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        files_list = os.listdir(file_dir)
        json_list = ["None"]
        for names in files_list:
            if names.endswith(".json"):
                json_list.append(names)
        mp4_list = ["None"]
        for names in files_list:
            if names.endswith(".mp4"):
                mp4_list.append(names)
        data_list = ["None"]
        for names in files_list:
            if names.endswith(".data"):
                data_list.append(names)
        json_dropdown.options = json_list
        mp4_dropdown.options = mp4_list
        data_dropdown.options = data_list
        data_ref.options = data_list
        cap = cv2.VideoCapture(videos_dir + video_dropdown.value)
        framewidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_n.max = n_frames - 1
        cap.release()
    
    video_dropdown.observe(onVideoChange, names='value')

    tabs = ['Preview Video', 'Choose Video', 'Detect Keypoints', 'Pre Processing', 'Processing', 'Ground Truth', 'Analyze Data']
    children = []
    children.append(videoPreviewInterface(video_dropdown, json_dropdown, data_dropdown))
    children.append(chooseVideoInterface(video_dropdown, allvid_dropdown, frame_n))
    children.append(detectKeypointsInterface(video_dropdown, json_dropdown, data_dropdown, frame_n))
    children.append(preProcessingInterface(video_dropdown, json_dropdown, data_dropdown, frame_n))
    children.append(processingInterface(video_dropdown, json_dropdown, data_dropdown, frame_n))
    children.append(gtInterface(video_dropdown, data_dropdown))
    children.append(analyzeDataInterface(video_dropdown, data_dropdown, data_ref, frame_n))
    tab = wg.Tab()
    tab.children = children
    for i in range(len(children)):
        tab.set_title(i, tabs[i])
    display(tab)
