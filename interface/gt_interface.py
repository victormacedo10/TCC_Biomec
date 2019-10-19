import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import ipywidgets as wg
from IPython.display import display, HTML
import sys
sys.path.append('../src/')
from gt import *
from preparations import *
from detection import *
from visualizations import *
from processing import *

videos_dir = "../Videos/"
data_dir = "../Data/"
post_dir = "../postprocessing/"

keypoints_mapping = ['Nose', 'Neck', 'Right Sholder', 'Right Elbow', 'Right Wrist', 'Left Sholder', 
                    'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee', 'Right Ankle', 'Left Hip', 
                    'Left Knee', 'Left Ankle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear']

def gtInterface(video_dropdown, data_dropdown):
    def onStartClicked(b): 
        gtRoutine(video_dropdown.value, data_dropdown.value, 
                output_name.value, summary.value)
    
    label_button = wg.Button(description='Start Labeling')
   
    summary = wg.Textarea(value='',placeholder='description',description='Summary:',disabled=False)

    output_name = wg.Text(value='',placeholder='File output name',description='Output:',disabled=False)

    label_button.on_click(onStartClicked)
    
    vbox = wg.VBox([video_dropdown, data_dropdown, output_name, summary, label_button]) 
    
    return vbox