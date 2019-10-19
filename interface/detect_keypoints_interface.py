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

def detectKeypointsInterface(video_dropdown, json_dropdown, data_dropdown, frame_n):
    model_nn = wg.Dropdown( options=["None", "Openpose Model"],
                            value="None",
                            description='Inference:',
                            disabled=False)
    summary = wg.Textarea(value='',placeholder='description',description='Summary:',disabled=False)
    output_name = wg.Text(
            value='',
            placeholder='File output name',
            description='Output:',
            disabled=False
        )
    process_vid = wg.Button(description='Process Video')

    persons = wg.RadioButtons(options=['All','Custom'],value='Custom',
                              rows=2,description='Choose:',disabled=False)
    
    custom = wg.BoundedIntText(value=0,min=0,max=10,step=1,description='Person:',disabled=False,
                              layout=wg.Layout(display='flex',flex_flow='line',
                                               align_items='flex-start',justify_content='flex-start',width='90%'))

    frame_slider = wg.IntSlider()
    
    detect_vid = wg.Button(description='Save Keypoints')

    paf_score_th = wg.FloatText(value=0.1, step=0.1, description='PAF Thres:',
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='90%'))
    conf_th = wg.FloatText(value=0.7, step=0.1, description='Conf. Thres:',
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='90%'))
    n_interp_samples = wg.IntText(value=10, description='Samples Interpolated:',
                     layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                      justify_content='flex-start',width='90%'))    
    joint_n = wg.Dropdown(options=keypoints_mapping, value='Nose', description='Joint:',disabled=False)
    threshold = wg.FloatText(value=0.1, step=0.1, description='Threshold:',
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='80%'))
    alpha = wg.FloatText(value=0.6, step=0.1, description='Transparency:',
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='80%'))
    show_point = wg.Checkbox(value=False, description='Show Point',disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='80%'))
    binary = wg.Checkbox(value=False, description='Binary',disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='80%'))
    
    wg.jslink((frame_n, 'value'), (frame_slider, 'value'))
    
    show_heatmap = wg.ToggleButtons(options=['Keypoints', 'Heatmap'],
                                    value='Keypoints',
                                    disabled=False,
                                    button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                    tooltips=['Show keypoints', 'Show heatmap'],
                                #     icons=['check'] * 3
                                )
   
    summary = wg.Textarea(value='',placeholder='description',description='Summary:',disabled=False)

    output_name = wg.Text(value='',placeholder='File output name',description='Output:',disabled=False)
    
    def DetectView(show_heatmap, video_name, file_name, persons, custom,
                       joint_n, alpha, binary, threshold, n_interp_samples, paf_score_th, 
                       conf_th, frame_n, show_point):

        joint_pairs = [-1]

        if(persons == 'Custom'):
            persons = 'Unsorted'
        if(show_heatmap == 'Keypoints'):   
            keypointsFromJSON(video_name, file_name, persons, custom, joint_pairs, frame_n, threshold, 
                              n_interp_samples, paf_score_th, conf_th)
        else:
            heatmapFromJSON(video_name, file_name, keypoints_mapping.index(joint_n), threshold, 
                            alpha, binary, n_interp_samples, paf_score_th, conf_th, frame_n, show_point)
    
    video_display = wg.interactive_output(DetectView, {"show_heatmap": show_heatmap,
                                                     "video_name": video_dropdown,
                                                     "file_name": json_dropdown,
                                                     "persons": persons,
                                                     "custom": custom,
                                                     "joint_n": joint_n,
                                                     "alpha": alpha,
                                                     "binary": binary,
                                                     "threshold": threshold, 
                                                     "n_interp_samples": n_interp_samples, 
                                                     "paf_score_th": paf_score_th,
                                                     "conf_th": conf_th,
                                                     "frame_n": frame_n,
                                                     "show_point": show_point})

    def onProcessClicked(b):
        if(model_nn.value=='None'):
            print("Select inference method")
        elif(model_nn.value=='Openpose Model'):
            videoInference(video_dropdown.value, summary.value, output_name.value,
                          threshold.value, n_interp_samples.value, paf_score_th.value, conf_th.value)
        else:
            print("No such inference")

        video_name = (video_dropdown.value).split(sep='.')[0]
        file_dir = data_dir + video_name + '/'
        files_list = os.listdir(file_dir)
        json_list = ["None"]
        for names in files_list:
            if names.endswith(".json"):
                json_list.append(names)
        json_dropdown.options = json_list

    process_vid.on_click(onProcessClicked)
    
    hbox_input = wg.HBox([video_dropdown, json_dropdown])
    hbox_play = wg.HBox([frame_n, frame_slider])
    ht_vbox_1 = wg.VBox([joint_n, show_point, alpha, binary, threshold], 
                        layout=wg.Layout(display='flex',flex_flow='line',
                                                          align_items='flex-start',justify_content='flex-start'))
    vbox_params = wg.VBox([paf_score_th, conf_th, n_interp_samples])
    vbox_per = wg.VBox([persons, custom],layout=wg.Layout(display='flex',flex_flow='column',
                                                          align_items='flex-start',justify_content='flex-start'))

    tabs = ['Person', 'Heatmap', 'PAF']
    children = []
    children.append(vbox_per)
    children.append(ht_vbox_1)
    children.append(vbox_params)
    tab = wg.Tab()
    tab.children = children
    for i in range(len(children)):
        tab.set_title(i, tabs[i])

    vbox1= wg.VBox([output_name, model_nn, summary, process_vid])
    #hbox_out = wg.HBox([vbox1, summary, process_vid])
    vbox_video = wg.VBox([video_display, hbox_play])
    vbox_config = wg.VBox([show_heatmap, tab, vbox1])
    hbox_res = wg.HBox([vbox_video, vbox_config])
    vbox_res = wg.VBox([hbox_input, hbox_res])
    
    return vbox_res