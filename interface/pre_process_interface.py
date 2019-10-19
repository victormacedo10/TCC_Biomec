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

pairs_mapping = ['Neck - R-Sho', 'Neck - L-Sho', 'R-Sho - R-Elb', 'R-Elb - R-Wr',    
                'L-Sho - L-Elb', 'L-Elb - L-Wr', 'Neck - R-Hip' ,'R-Hip - R-Knee',  
                'R-Knee - R-Ank', 'Neck - L-Hip', 'L-Hip - L-Knee', 'L-Knee - L-Ank',  
                'Neck - Nose', 'Nose-R - Eye', 'R-Eye - R-Ear', 'Nose - L-Eye',
                'L-Eye - L-Ear', 'R-Sho - L-Ear', 'L-Sho - R-Ear', 'R-Sho - R-Hip', 'L-Sho - L-Hip']

joint_pairs = []

def preProcessingInterface(video_dropdown, json_dropdown, data_dropdown, frame_n):

    def onPreProcessClicked(b): 
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
        
        saveJointFile(video_dropdown.value, json_dropdown.value, output_name.value, joint_pairs, summary.value,
                      miss_points.value)

    def preprocessView(video_name, file_name, persons, custom_person, joint_pose, frame_n,
                      p1_2, p1_5, p2_3, p3_4, p5_6, p6_7, p1_8, p8_9, p9_10, p1_11, p11_12,
                      p12_13, p1_0, p0_14, p14_16, p0_15, p15_17):

        joint_p = []
        if joint_pose == 'Sagittal Left':
            #joint_p = [1,5,9,10,11,12,15,16,4]
            joint_p = [5,10,11,4,20]
        elif joint_pose == 'Sagittal Right':
            #joint_p = [0,3,6,7,8,12,13,14,2]
            joint_p = [3,7,8,2,19]
        elif joint_pose == 'Whole Body':
            joint_p = [-1]
        else:
            if p1_2:
                joint_p.append(0)
            if p1_5:
                joint_p.append(1)
            if p2_3:
                joint_p.append(2)
            if p3_4:
                joint_p.append(3)
            if p5_6:
                joint_p.append(4)
            if p6_7:
                joint_p.append(5)
            if p1_8:
                joint_p.append(6)
            if p8_9:
                joint_p.append(7)
            if p9_10:
                joint_p.append(8)
            if p1_11:
                joint_p.append(9)
            if p11_12:
                joint_p.append(10)
            if p12_13:
                joint_p.append(11)
            if p1_0:
                joint_p.append(12)
            if p0_14:
                joint_p.append(13)
            if p14_16:
                joint_p.append(14)
            if p0_15:
                joint_p.append(15)
            if p15_17:
                joint_p.append(16)

        if(joint_pairs != joint_p):
            joint_pairs.clear()
            for p in joint_p:
                joint_pairs.append(p)

        keypointsFromJSON(video_name, file_name, persons, custom_person, joint_pairs, 
                          frame_n, read_file = True)

    persons = wg.RadioButtons(options=['Biggest', 'Unsorted', 'All'],value='Biggest',
                              rows=3,description='',disabled=False)  
    custom_person = wg.BoundedIntText(value=0,min=0,max=10,step=1,description='Choose:',disabled=False,
                              layout=wg.Layout(display='flex',flex_flow='line',
                                               align_items='flex-start',justify_content='flex-start',width='90%'))
    joint_pose = wg.RadioButtons(options=['Sagittal Left', 'Sagittal Right', 'Whole Body', 'Custom'],value='Sagittal Left',
                                 rows=4,description='',disabled=False,
                                 layout=wg.Layout(display='flex',flex_flow='line',width='90%'))
    p1_2 = wg.ToggleButton(value=False, description=pairs_mapping[0],disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='60%'))
    p1_5 = wg.ToggleButton(value=False, description=pairs_mapping[1],disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='60%'))
    p2_3 = wg.ToggleButton(value=False, description=pairs_mapping[2],disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='60%'))
    p3_4 = wg.ToggleButton(value=False, description=pairs_mapping[3],disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='60%'))
    p5_6 = wg.ToggleButton(value=False, description=pairs_mapping[4],disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='60%'))
    p6_7 = wg.ToggleButton(value=False, description=pairs_mapping[5],disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='60%'))
    p1_8 = wg.ToggleButton(value=False, description=pairs_mapping[6],disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='60%'))
    p8_9 = wg.ToggleButton(value=False, description=pairs_mapping[7],disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='60%'))
    p9_10 = wg.ToggleButton(value=False, description=pairs_mapping[8],disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='60%'))
    p1_11 = wg.ToggleButton(value=False, description=pairs_mapping[9],disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='60%'))
    p11_12 = wg.ToggleButton(value=False, description=pairs_mapping[10],disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='60%'))
    p12_13 = wg.ToggleButton(value=False, description=pairs_mapping[11],disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='60%'))
    p1_0 = wg.ToggleButton(value=False, description=pairs_mapping[12],disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='60%'))
    p0_14 = wg.ToggleButton(value=False, description=pairs_mapping[13],disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='60%'))
    p14_16 = wg.ToggleButton(value=False, description=pairs_mapping[14],disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='60%'))
    p0_15 = wg.ToggleButton(value=False, description=pairs_mapping[15],disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='60%'))
    p15_17 = wg.ToggleButton(value=False, description=pairs_mapping[16],disabled=False,
                         layout=wg.Layout(display='flex',flex_flow='line',align_items='flex-start',
                                          justify_content='flex-start',width='60%'))
    miss_points = wg.RadioButtons(options=['None', 'Fill w/ Last', 'Fill w/ Kalman', 'Fill w/ Interp'],value='None',
                                 rows=4,description='',disabled=False,
                                 layout=wg.Layout(display='flex',flex_flow='line',width='90%'))
    frame_slider = wg.IntSlider()
    preprocess_vid = wg.Button(description='Pre Process Video')
    summary = wg.Textarea(value='',placeholder='description',description='Summary:',disabled=False)
    output_name = wg.Text(value='',placeholder='File output name',description='Output:',disabled=False)
    video_display = wg.interactive_output(preprocessView, {"video_name": video_dropdown,
                                                     "file_name": json_dropdown,
                                                     "persons": persons,
                                                     "custom_person": custom_person,
                                                     "joint_pose": joint_pose,
                                                     "frame_n": frame_n,
                                                     "p1_2": p1_2,"p1_5": p1_5,"p2_3": p2_3,"p3_4": p3_4,
                                                     "p5_6": p5_6,"p6_7": p6_7,"p1_8": p1_8,"p8_9": p8_9,
                                                     "p9_10": p9_10,"p1_11": p1_11,"p11_12": p11_12,
                                                     "p12_13": p12_13,"p1_0": p1_0,"p0_14": p0_14,
                                                     "p14_16": p14_16,"p0_15": p0_15,"p15_17": p15_17})

    preprocess_vid.on_click(onPreProcessClicked)
    wg.jslink((frame_n, 'value'), (frame_slider, 'value'))

    vbox_per = wg.VBox([persons, custom_person],layout=wg.Layout(display='flex',flex_flow='column',
                                                          align_items='flex-start',justify_content='flex-start'))

    hbox_pose1 = wg.HBox([p1_2, p1_5, p2_3])
    hbox_pose2 = wg.HBox([p3_4, p5_6, p6_7])
    hbox_pose3 = wg.HBox([p1_8, p8_9, p9_10])
    hbox_pose4 = wg.HBox([p1_11, p11_12, p12_13])
    hbox_pose5 = wg.HBox([p1_0, p0_14, p14_16])
    hbox_pose6 = wg.HBox([p0_15, p15_17 ])
    vbox_pose1 = wg.VBox([joint_pose, hbox_pose1, hbox_pose2, hbox_pose3, 
                          hbox_pose4, hbox_pose5, hbox_pose6])

    hbox_player = wg.HBox([frame_n, frame_slider])
    
    tabs = ['Person', 'Pose', 'Missing Data']
    children = []
    children.append(vbox_per)
    children.append(vbox_pose1)
    children.append(miss_points)

    tab = wg.Tab()
    tab.children = children
    for i in range(len(children)):
        tab.set_title(i, tabs[i])
    
    video_player = wg.VBox([video_display, hbox_player])
    hbox_filein = wg.HBox([video_dropdown, json_dropdown])
    vbox_fileout = wg.VBox([output_name, miss_points])

    vbox_config = wg.VBox([tab, output_name, summary, preprocess_vid])
    hbox_res = wg.HBox([video_player, vbox_config])
    vbox_res = wg.VBox([hbox_filein, hbox_res])
    
    return vbox_res