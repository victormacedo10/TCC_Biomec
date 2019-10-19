import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import ipywidgets as wg
from IPython.display import display, HTML
from preparations import *
from detection import *
from visualizations import *

videos_dir = "../Videos/"
data_dir = "../Data/"
post_dir = "../postprocessing/"

keypoints_mapping = ['Nose', 'Neck', 'Right Sholder', 'Right Elbow', 'Right Wrist', 'Left Sholder', 
                    'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee', 'Right Ankle', 'Left Hip', 
                    'Left Knee', 'Left Ankle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear']

def videoPreviewInterface(video_dropdown):

    def showVideo(video_dropdown):
        if(video_dropdown == "None"):
            print("Choose a video")
            return
        video_path = videos_dir + video_dropdown
        display(HTML("""<video width="640" height="360" controls="">
                    <source src="{0}"></video>""".format(video_path)))
    
    video_wg = wg.interactive_output(showVideo, {"video_dropdown":video_dropdown})
    video_menu = wg.VBox([video_dropdown, video_wg])
    return video_menu

def chooseVideoInterface(video_dropdown, frame_n):
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
    slider = wg.IntSlider(value=0,
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

    wg.jslink((frame_n, 'value'), (slider, 'value'))
    
    def onValueChange(change):
        slider.min, slider.max = slider_range.value[0], slider_range.value[1]

    def onVideoChange(change):
        if(video_dropdown.value != "None"):
            cap = cv2.VideoCapture(videos_dir + video_dropdown.value)
            framewidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_n.max = n_frames - 1
            slider.max = n_frames - 1
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
            editVideo(video_dropdown.value, 0, r = slider_range.value, x=width_range.value, 
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
        slider.max = n_frames - 1
        slider_range.max = n_frames - 1
        slider_range.value = (0, n_frames - 1)
        height_range.max, width_range.max = frameheight, framewidth
        height_range.value, width_range.value = (0, frameheight), (0, framewidth)
        resolution.max = frameheight
        resolution.value = frameheight
        cap.release()

    y = wg.interactive_output(editVideo, {"video_name":video_dropdown, "n":frame_n, "r": slider_range,
                                           "x":width_range, "y":height_range, "in_height":resolution})

    hbox1 = wg.HBox([video_dropdown, frame_n])
    hbox2 = wg.VBox([output_name, save_vid])
    vbox1 = wg.VBox([slider, slider_range, width_range, height_range, resolution, hbox2])
    hbox3 = wg.HBox([y, vbox1])
    vbox2 = wg.VBox([hbox1, hbox3])
    return vbox2        

def detectKeypointsInterface(video_dropdown, json_dropdown, mp4_dropdown):
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

    def onProcessClicked(b):
        if(model_nn.value=='None'):
            print("Select inference method")
        elif(model_nn.value=='Openpose Model'):
            processVideo(video_dropdown.value, summary.value, output_name.value)
        else:
            print("Post processing not implemented yet")

        video_name = (video_dropdown.value).split(sep='.')[0]
        file_dir = data_dir + video_name + '/'
        files_list = os.listdir(file_dir)
        json_list = ["None"]
        for names in files_list:
            if names.endswith(".json"):
                json_list.append(names)
        json_dropdown.options = json_list

    process_vid.on_click(onProcessClicked)

    vbox1 = wg.VBox([model_nn, summary, output_name, process_vid])
    
    def showVideo(mp4_dropdown):
        if(video_dropdown == "None"):
            print("Choose a video")
            return
        
        video_name = (video_dropdown.value).split(sep='.')[0]
        file_dir = data_dir + video_name + '/'
        video_path = file_dir + mp4_dropdown
        display(HTML("""<video width="540" height="302" controls="">
                    <source src="{0}"></video>""".format(video_path)))
    
    video_wg = wg.interactive_output(showVideo, {"mp4_dropdown":mp4_dropdown})
    
    hbox1 = wg.HBox([video_wg, vbox1])
    hbox2 = wg.HBox([video_dropdown, mp4_dropdown])
    
    video_menu = wg.VBox([hbox2, hbox1])
    
    return video_menu

def preProcessingInterface(video_dropdown, json_dropdown, data_dropdown, frame_n):
    persons = wg.RadioButtons(options=['Biggest', 'Unsorted', 'All'],value='Biggest',
                              rows=3,description='Choose:',disabled=False)
    
    custom = wg.BoundedIntText(value=0,min=0,max=10,step=1,description='Person:',disabled=False,
                              layout=wg.Layout(display='flex',flex_flow='line',
                                               align_items='flex-start',justify_content='flex-start',width='90%'))

    joint_pose = wg.RadioButtons(options=['Sagittal Left', 'Sagittal Right', 'Whole Body'],value='Sagittal Left',
                                 rows=4,description='Pose: ',disabled=False,
                                 layout=wg.Layout(display='flex',flex_flow='line',width='90%'))

    frame_slider = wg.IntSlider()
    
    preprocess_vid = wg.Button(description='Pre Process Video')

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
        
        saveJointFile(video_dropdown.value, json_dropdown.value, output_name.value, threshold.value, 
                      n_interp_samples.value, paf_score_th.value, conf_th.value, summary.value)

    preprocess_vid.on_click(onPreProcessClicked)
    
    ht_vbox_1 = wg.VBox([joint_n, show_point, alpha, binary, threshold], 
                        layout=wg.Layout(display='flex',flex_flow='line',
                                                          align_items='flex-start',justify_content='flex-start'))
    
    hbox_play = wg.HBox([video_dropdown, json_dropdown, frame_n, frame_slider])
    vbox_params = wg.VBox([paf_score_th, conf_th, n_interp_samples])
    vbox_per = wg.VBox([persons, custom, joint_pose],layout=wg.Layout(display='flex',flex_flow='column',
                                                          align_items='flex-start',justify_content='flex-start'))
    
    def preprocessView(show_heatmap, video_name, file_name, persons, custom, joint_pose,
                       joint_n, alpha, binary, threshold, n_interp_samples, paf_score_th, 
                       conf_th, frame_n, show_point):
        if(show_heatmap == 'Keypoints'):   
            keypointsFromJSON(video_name, file_name, persons, custom, joint_pose, threshold, 
                              n_interp_samples, paf_score_th, conf_th, frame_n)
        else:
            heatmapFromJSON(video_name, file_name, keypoints_mapping.index(joint_n), threshold, 
                            alpha, binary, n_interp_samples, paf_score_th, conf_th, frame_n, show_point)
    
    out_res = wg.interactive_output(preprocessView, {"show_heatmap": show_heatmap,
                                                     "video_name": video_dropdown,
                                                     "file_name": json_dropdown,
                                                     "persons": persons,
                                                     "custom": custom,
                                                     "joint_pose": joint_pose,
                                                     "joint_n": joint_n,
                                                     "alpha": alpha,
                                                     "binary": binary,
                                                     "threshold": threshold, 
                                                     "n_interp_samples": n_interp_samples, 
                                                     "paf_score_th": paf_score_th,
                                                     "conf_th": conf_th,
                                                     "frame_n": frame_n,
                                                     "show_point": show_point})
    
    tabs = ['Choose Person', 'Heatmap', 'PAF Parameters']
    children = []
    children.append(vbox_per)
    children.append(ht_vbox_1)
    children.append(vbox_params)
    tab = wg.Tab()
    tab.children = children
    for i in range(len(children)):
        tab.set_title(i, tabs[i])
    
    hbox_out = wg.HBox([output_name, summary, preprocess_vid])
    vbox_res = wg.VBox([show_heatmap, tab])
    hbox_res = wg.HBox([out_res, vbox_res])
    vbox_vid = wg.VBox([hbox_play, hbox_res, hbox_out])
    
    return vbox_vid

def processingInterface(video_dropdown, json_dropdown, data_dropdown, frame_n):
    folder_files = os.listdir(post_dir)
    py_list = []
    for names in folder_files:
        if names.endswith(".py"):
            py_list.append(names)
    function = wg.Dropdown(options=py_list,
                        description='Algorithm:',
                        disabled=False)

    joint_pose = wg.RadioButtons(options=['Sagittal Left', 'Sagittal Right', 'Whole Body'],value='Sagittal Left',
                                 rows=4,description='Pose: ',disabled=False,
                                 layout=wg.Layout(display='flex',flex_flow='line',width='90%'))

    frame_slider = wg.IntSlider()
    
    posprocess_vid = wg.Button(description='Pos Process Video')
    
    wg.jslink((frame_n, 'value'), (frame_slider, 'value'))

   
    summary = wg.Textarea(value='',placeholder='description',description='Summary:',disabled=False)

    output_name = wg.Text(value='',placeholder='File output name',description='Output:',disabled=False)
    
    def onPosProcessClicked(b): 
        saveJointFile(video_dropdown.value, json_dropdown.value, output_name.value, threshold.value, 
                      n_interp_samples.value, paf_score_th.value, conf_th.value, summary.value)

    posprocess_vid.on_click(onPosProcessClicked)
    
    hbox_play = wg.HBox([video_dropdown, data_dropdown, frame_n, frame_slider])
    vbox_per = wg.VBox([joint_pose, function, output_name, summary, posprocess_vid],
                       layout=wg.Layout(display='flex',flex_flow='column',
                                                          align_items='flex-start',justify_content='flex-start'))
    
    def posprocessView(video_name, file_name, joint_pose, frame_n): 
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
        keypointsFromDATA(video_name, file_name, joint_pose)
    
    out_res = wg.interactive_output(posprocessView, {"video_name": video_dropdown,
                                                     "file_name": data_dropdown,
                                                     "joint_pose": joint_pose,
                                                     "frame_n": frame_n})
    
    hbox_out = wg.HBox([output_name, summary, posprocess_vid])
    hbox_res = wg.HBox([out_res, vbox_per])
    vbox_vid = wg.VBox([hbox_play, hbox_res])
    
    return vbox_vid

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

    i_res = wg.interactive_output(keypointsFromJSON, {"video_name": video_dropdown,
                                                  "file_name": json_dropdown,
                                                  "persons": persons,
                                                  "custom": custom,
                                                  "joint_pose": joint_pose,
                                                  "frame_n": frame_n})

    wg.jslink((frame_n, 'value'), (frame_slider, 'value'))
    
    hbox_play = wg.HBox([video_dropdown, json_dropdown, frame_n, frame_slider])
    vbox_res = wg.VBox([persons, custom, joint_pose])
    hbox_res = wg.HBox([i_res, vbox_res])
    vbox_vid = wg.VBox([hbox_play, hbox_res])
    
    return vbox_vid

def interactiveInterface():

    videos_list = os.listdir(videos_dir)
    video_options = ["None"]
    for video in videos_list:
        video_options.append(video)
    video_dropdown = wg.Dropdown( options=video_options,
                            description='Video:',
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
        cap = cv2.VideoCapture(videos_dir + video_dropdown.value)
        framewidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_n.max = n_frames - 1
        cap.release()
    
    video_dropdown.observe(onVideoChange, names='value')

    tabs = ['Choose Video', 'Detect Keypoints', 'Pre Processing', 'Processing', 'Analyze Data']
    children = []
    children.append(chooseVideoInterface(video_dropdown, frame_n))
    children.append(detectKeypointsInterface(video_dropdown, json_dropdown, mp4_dropdown))
    children.append(preProcessingInterface(video_dropdown, json_dropdown, data_dropdown, frame_n))
    children.append(processingInterface(video_dropdown, json_dropdown, data_dropdown, frame_n))
    children.append(analyzeDataInterface(video_dropdown, json_dropdown, frame_n))
    tab = wg.Tab()
    tab.children = children
    for i in range(len(children)):
        tab.set_title(i, tabs[i])
    display(tab)


