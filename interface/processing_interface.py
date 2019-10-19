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
from processing import *

videos_dir = "../Videos/"
data_dir = "../Data/"
post_dir = "../postprocessing/"

keypoints_mapping = ['Nose', 'Neck', 'Right Sholder', 'Right Elbow', 'Right Wrist', 'Left Sholder', 
                    'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee', 'Right Ankle', 'Left Hip', 
                    'Left Knee', 'Left Ankle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear']

def processingInterface(video_dropdown, json_dropdown, data_dropdown, frame_n):
    def onPosProcessClicked(b): 
        if(option.value=='Online'):
            saveProcessedFileOnline(video_dropdown.value, data_dropdown.value, output_name.value, 
                        function.value, summary.value)
        elif(option.value=='All Data'):
            saveProcessedFileAll(video_dropdown.value, data_dropdown.value, output_name.value, 
                        function.value, summary.value)

    def posprocessView(video_name, file_name, frame_n): 
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
        data_list = ["None"]
        for names in files_list:
            if names.endswith(".data"):
                data_list.append(names)
        json_dropdown.options = json_list
        data_dropdown.options = data_list
        keypointsFromDATA(video_name, file_name, frame_n)

    folder_files = os.listdir(post_dir)
    py_list = []
    for names in folder_files:
        if names.endswith(".py"):
            py_list.append(names)
    function = wg.Dropdown(options=py_list,
                        value='post_processing_template.py',
                        description='Algorithm:',
                        disabled=False)

    option = wg.RadioButtons(options=['Online', 'All Data', 'Batch'],value='Online',
                                 rows=3,description='Options',disabled=False,
                                 layout=wg.Layout(display='flex',flex_flow='line',width='90%'))

    frame_slider = wg.IntSlider()
    wg.jslink((frame_n, 'value'), (frame_slider, 'value'))
    
    posprocess_vid = wg.Button(description='Pos Process Video')
   
    summary = wg.Textarea(value='',placeholder='description',description='Summary:',disabled=False)

    output_name = wg.Text(value='',placeholder='File output name',description='Output:',disabled=False)

    posprocess_vid.on_click(onPosProcessClicked)
    
    batch = wg.IntSlider(value=1,min=1,max=100,step=1,description='Batch size',disabled=False)

    video_display = wg.interactive_output(posprocessView, {"video_name": video_dropdown,
                                                     "file_name": data_dropdown,
                                                     "frame_n": frame_n})
    
    hbox_input = wg.HBox([video_dropdown, data_dropdown])
    hbox_play = wg.HBox([frame_n, frame_slider])
    vbox_config = wg.VBox([function, option, batch, output_name, summary, posprocess_vid]) 
    vbox_vid = wg.VBox([video_display, hbox_play])
    hbox_res = wg.HBox([vbox_vid, vbox_config])
    vbox_res = wg.VBox([hbox_input, hbox_res])
    
    return vbox_res