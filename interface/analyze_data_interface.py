import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import ipywidgets as wg
from IPython.display import display, HTML
import sys
sys.path.append('../src/')
from analysis import *
from preparations import *
from detection import *
from visualizations import *

videos_dir = "../Videos/"
data_dir = "../Data/"
post_dir = "../postprocessing/"

keypoints_mapping = ['Nose', 'Neck', 'Right Sholder', 'Right Elbow', 'Right Wrist', 'Left Sholder', 
                    'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee', 'Right Ankle', 'Left Hip', 
                    'Left Knee', 'Left Ankle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear']

def analyzeDataInterface(video_dropdown, data_dropdown, data_ref, frame_n):

    def overlapKeypoints(show, video_name, file_added, file_ref, frame_n, 
                        show_point, point, thickness, coord, error_type, def_error):
        file_names = list(data_added.options)
        if(show == 'Video'):
            keypointsFromDATACompare(video_name, file_names, file_ref, frame_n, 
                                    show_point, point, thickness)
        elif(show == 'Metrics'):
            showMetrics(video_name, file_names, file_ref, point, error_type, def_error)
        else:
            plotTrajectory(video_name, file_names, file_ref, point, coord)

    data_added = wg.Dropdown(options=['None'],
                            description='Compare to:',
                            disabled=False)  

    def addDATAClicked(b):
        opt_tmp = list(data_added.options)

        if(opt_tmp[0] == 'None'):
            opt_tmp = [data_dropdown.value]
        else:
            opt_tmp.append(data_dropdown.value)
        
        data_added.options = opt_tmp
        data_added.value = data_dropdown.value

    def resetDATAClicked(b):
        data_added.options = ['None']
        data_added.value = 'None'
    
    def saveDATAClicked(b):
        saveData(video_dropdown.value, list(data_added.options), data_ref, plot_name=output_name, pose='Saggital Right')

    add_data = wg.Button(description='Add DATA')
    reset_data = wg.Button(description='Reset DATA')
    save_data = wg.Button(description='Save DATA')

    output_name = wg.Text(value='',placeholder='File output name',description='Comparison name:',disabled=False)

    show = wg.ToggleButtons(options=['Video', 'Metrics', 'Trajectory'],
                                value='Video',
                                disabled=False,
                                button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                tooltips=['Show Video', 'Show Metrics', 'Show Trajectory'])

    coord = wg.RadioButtons(options=['x', 'y'], value='x',
                            description='Coordinate:', disabled=False)
    error_type = wg.RadioButtons(options=['Error DF', 'False Negatives DF', 'Error Graph'], value='Error DF',
                            description='Type:', disabled=False)

    frame_slider = wg.IntSlider()
    thickness = wg.IntSlider(value=1,min=1,max=20,step=1,description='Thickness',disabled=False)

    point = wg.Dropdown(options=keypoints_mapping, value='Nose', description='Joint:',disabled=False)

    show_point = wg.Checkbox(value=False, description='Show Point',disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='80%'))

    def_error = wg.Checkbox(value=False, description='Per joint',disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='80%'))
    def_error.style.button_width=''

    wg.jslink((frame_n, 'value'), (frame_slider, 'value'))

    vbox_view = wg.VBox([point, show_point, thickness])
    vbox_metrcis = wg.VBox([error_type, point, def_error])
    vbox_traj = wg.VBox([point, coord])

    tabs = ['Video', 'Metrics', 'Trajectory']
    children = []
    children.append(vbox_view)
    children.append(vbox_metrcis)
    children.append(vbox_traj)
    tab = wg.Tab()
    tab.children = children
    for i in range(len(children)):
        tab.set_title(i, tabs[i])

    video_display = wg.interactive_output(overlapKeypoints, {"show": show,
                                                            "video_name": video_dropdown,
                                                            "file_added": data_added, 
                                                            "file_ref": data_ref,
                                                            "frame_n": frame_n,
                                                            "show_point": show_point,
                                                            "point": point,
                                                            "thickness": thickness,
                                                            "coord": coord,
                                                            "error_type": error_type,
                                                            "def_error": def_error})
        
    hbox_add = wg.HBox([data_dropdown, add_data])
    hbox_remove = wg.HBox([data_added, reset_data])
    hbox_save = wg.HBox([output_name, save_data])

    add_data.on_click(addDATAClicked)
    reset_data.on_click(resetDATAClicked)

    vbox_input = wg.VBox([hbox_add, hbox_remove])
    vbox_input_2 = wg.VBox([video_dropdown, data_ref])

    hbox_input = wg.HBox([vbox_input_2, vbox_input])
    hbox_play = wg.HBox([frame_n, frame_slider])
    vbox_vid = wg.VBox([show, video_display, hbox_play])
    hbox_res = wg.HBox([vbox_vid, tab])
    vbox_res = wg.VBox([hbox_input, hbox_res])

    return vbox_res