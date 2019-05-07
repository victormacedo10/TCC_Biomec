import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import ipywidgets as wg
from IPython.display import display, HTML
import sys
sys.path.append('../src/')
from preparations import *
from detection import *
from visualizations import *

videos_dir = "../Videos/"
data_dir = "../Data/"
post_dir = "../postprocessing/"

keypoints_mapping = ['Nose', 'Neck', 'Right Sholder', 'Right Elbow', 'Right Wrist', 'Left Sholder', 
                    'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee', 'Right Ankle', 'Left Hip', 
                    'Left Knee', 'Left Ankle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear']

def analyzeDataInterface(video_dropdown, json_dropdown, frame_n):
    persons = wg.RadioButtons(options=['Main', 'Custom', 'Unsorted', 'All'],
                                value='Main',
                                rows=4,
                                description='Persons',
                                disabled=False)
    
    custom = wg.BoundedIntText( value=0,
                                min=0,
                                max=10,
                                step=1,
                                description='Custom:',
                                disabled=False)

    joint_pose = wg.RadioButtons(options=['Sagittal Left', 'Sagittal Right', 'Whole Body'],value='Sagittal Left',
                                    rows=3,
                                    description='Pose',
                                    disabled=False)

    frame_slider = wg.IntSlider()

    video_display = wg.interactive_output(keypointsFromJSON, {"video_name": video_dropdown,
                                                  "file_name": json_dropdown,
                                                  "persons": persons,
                                                  "custom": custom,
                                                  "joint_pose": joint_pose,
                                                  "frame_n": frame_n})

    wg.jslink((frame_n, 'value'), (frame_slider, 'value'))
    
    hbox_play = wg.HBox([video_dropdown, json_dropdown, frame_n, frame_slider])
    vbox_res = wg.VBox([persons, custom, joint_pose])
    hbox_res = wg.HBox([video_display, vbox_res])
    vbox_vid = wg.VBox([hbox_play, hbox_res])
    
    return vbox_vid