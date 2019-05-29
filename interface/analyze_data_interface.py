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

def analyzeDataInterface(video_dropdown, data_dropdown, frame_n):
    def overlapKeypoints(show, video_name, data_drop_2, frame_n, show_point, point, thickness):
        file_names = list(data_dropdown_2.options)
        if(show == 'Video'):
            keypointsFromDATACompare(video_name, file_names, frame_n, show_point, point, thickness)
        elif(show == 'Error'):
            keypointsFromDATACompare(video_name, file_names, frame_n, show_point, point)
        else:
            keypointsFromDATACompare(video_name, file_names, frame_n, show_point, point)

    data_dropdown_2 = wg.Dropdown(options=['None'],
                            description='Added:',
                            disabled=False)  

    def addDATAClicked(b):
        opt_tmp = list(data_dropdown_2.options)

        if(opt_tmp[0] == 'None'):
            opt_tmp = [data_dropdown.value]
        else:
            opt_tmp.append(data_dropdown.value)
        
        data_dropdown_2.options = opt_tmp
        data_dropdown_2.value = data_dropdown.value

    def resetDATAClicked(b):
        data_dropdown_2.options = ['None']
        data_dropdown_2.value = 'None'

    add_data = wg.Button(description='Add DATA')
    reset_data = wg.Button(description='Reset DATA')


    show = wg.ToggleButtons(options=['Video', 'Error', 'Trajectory'],
                                value='Video',
                                disabled=False,
                                button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                tooltips=['Show Video', 'Show Graphs', 'Show Trajectory'],
                            #     icons=['check'] * 3
                                )

    frame_slider = wg.IntSlider()
    thickness = wg.IntSlider(value=1,min=1,max=20,step=1,description='Thickness',disabled=False)

    point = wg.Dropdown(options=keypoints_mapping, value='Nose', description='Joint:',disabled=False)

    show_point = wg.Checkbox(value=False, description='Show Point',disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='80%'))

    wg.jslink((frame_n, 'value'), (frame_slider, 'value'))

    vbox_view = wg.VBox([point, show_point, thickness])

    tabs = ['Video', 'Error', 'Trajectory']
    children = []
    children.append(vbox_view)
    children.append(point)
    children.append(point)
    tab = wg.Tab()
    tab.children = children
    for i in range(len(children)):
        tab.set_title(i, tabs[i])

    video_display = wg.interactive_output(overlapKeypoints, {"show": show,
                                                            "video_name": video_dropdown,
                                                            "data_drop_2": data_dropdown_2, 
                                                            "frame_n": frame_n,
                                                            "show_point": show_point,
                                                            "point": point,
                                                            "thickness": thickness})
        
    hbox_add = wg.HBox([data_dropdown, add_data])
    hbox_remove = wg.HBox([data_dropdown_2, reset_data])

    add_data.on_click(addDATAClicked)
    reset_data.on_click(resetDATAClicked)

    vbox_input = wg.VBox([hbox_add, hbox_remove])

    hbox_input = wg.HBox([video_dropdown, vbox_input])
    hbox_play = wg.HBox([frame_n, frame_slider])
    vbox_vid = wg.VBox([show, video_display, hbox_play])
    hbox_res = wg.HBox([vbox_vid, tab])
    vbox_res = wg.VBox([hbox_input, hbox_res])

    return vbox_res