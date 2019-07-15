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
allvid_dir = "../Others/"
data_dir = "../Data/"
post_dir = "../postprocessing/"

keypoints_mapping = ['Nose', 'Neck', 'Right Sholder', 'Right Elbow', 'Right Wrist', 'Left Sholder', 
                    'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee', 'Right Ankle', 'Left Hip', 
                    'Left Knee', 'Left Ankle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear']

def chooseVideoInterface(video_dropdown, allvid_dropdown, frame_n):
    slider_range = wg.IntRangeSlider(
        value=[0, 100],
        min=0,
        max=100,
        step=1,
        description='Cut:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
    )
    frame_slider = wg.IntSlider(value=0,
        min=0,
        max=100,
        step=1,
        description='Frame:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )
    width_range = wg.IntRangeSlider(
        value=[0, 100],
        min=0,
        max=100,
        step=1,
        description='Width:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
    )
    height_range = wg.IntRangeSlider(
        value=[0, 100],
        min=0,
        max=100,
        step=1,
        description='Heigth:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
    )
    resolution = wg.IntSlider(value=0,
        min=0,
        max=480,
        step=1,
        description='Resolution:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )
    output_name = wg.Text(
            value='',
            placeholder='File output name',
            description='Output:',
            disabled=False
    )
    save_vid = wg.Button(description='Save Video')

    wg.jslink((frame_n, 'value'), (frame_slider, 'value'))
    
    def onValueChange(change):
        frame_slider.min, frame_slider.max = slider_range.value[0], slider_range.value[1]

    def onVideoChange(change):
        if(video_dropdown.value != "None"):
            cap = cv2.VideoCapture(videos_dir + video_dropdown.value)
            framewidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_n.max = n_frames - 1
            frame_slider.max = n_frames - 1
            slider_range.max = n_frames - 1
            slider_range.value = (0, n_frames - 1)
            height_range.max, width_range.max = frameheight, framewidth
            height_range.value, width_range.value = (0, frameheight), (0, framewidth)
            resolution.max = frameheight
            resolution.value = frameheight
            cap.release()

    def onResChange(change):
        cap = cv2.VideoCapture(videos_dir + video_dropdown.value)
        framewidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        in_height = resolution.value
        in_width = int((in_height/frameheight)*framewidth)
        height_range.max, width_range.max = in_height, in_width
        height_range.value, width_range.value = (0, in_height), (0, in_width)
        cap.release()

    def onSaveVidClicked(b):
        if(output_name.value==''):
            print("Choose a file name", end="\r")
        else:
            output_path = videos_dir + output_name.value
            print("Saving video at: {}".format(output_path))
            editVideo(video_dropdown.value, allvid_dropdown.value, 0, r = slider_range.value, x=width_range.value, 
                       y=height_range.value, in_height=resolution.value, 
                       save_vid=True, output_path=output_path)
            videos_list = os.listdir(videos_dir)
            video_options = ["None"]
            for video in videos_list:
                video_options.append(video)
            video_dropdown.options = video_options
    
    slider_range.observe(onValueChange, names='value')    
    video_dropdown.observe(onVideoChange, names='value')
    resolution.observe(onResChange, names='value')
    save_vid.on_click(onSaveVidClicked)

    if(video_dropdown.value != "None"):
        cap = cv2.VideoCapture(videos_dir + video_dropdown.value)
        framewidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(n_frames)
        frame_n.max = n_frames - 1
        frame_slider.max = n_frames - 1
        slider_range.max = n_frames - 1
        slider_range.value = (0, n_frames - 1)
        height_range.max, width_range.max = frameheight, framewidth
        height_range.value, width_range.value = (0, frameheight), (0, framewidth)
        resolution.max = frameheight
        resolution.value = frameheight
        cap.release()

    video_display = wg.interactive_output(editVideo, {"video_name":video_dropdown, 
                                            "allvid_name":allvid_dropdown, "n":frame_n, 
                                            "r": slider_range, "x":width_range, 
                                            "y":height_range, "in_height":resolution})

    hbox_input = wg.HBox([video_dropdown, allvid_dropdown])
    hbox_player = wg.HBox([frame_n, frame_slider])
    video_player = wg.VBox([video_display, hbox_player])
    vbox_config = wg.VBox([slider_range, width_range, height_range, resolution, output_name, save_vid])
    hbox_res = wg.HBox([video_player, vbox_config])
    vbox_res = wg.VBox([hbox_input, hbox_res])
    return vbox_res