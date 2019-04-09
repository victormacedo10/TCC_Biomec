import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import Markdown as md
import os
import ipywidgets as wg
from IPython.display import display
from matplotlib import cm
import math
import json
from random import randint
from IPython.display import HTML

proto_file = "../Models/Openpose/coco/pose_deploy_linevec.prototxt"
weights_file = "../Models/Openpose/coco/pose_iter_440000.caffemodel"
videos_dir = "../Videos/"
data_dir = "../Data/"

n_points = 18
# COCO Output Format
keypoints_mapping = np.array(['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
                    'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 
                    'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear'])

pose_pairs = np.array([[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16]])

# index of pafs correspoding to the pose_pairs
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
map_idx = np.array([[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], 
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], 
          [47,48], [49,50], [53,54], [51,52], [55,56], 
          [37,38], [45,46]])

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

# Find the Keypoints using Non Maximum Suppression on the Confidence Map
def getKeypoints(prob_map, threshold=0.1):
    
    map_smooth = cv2.GaussianBlur(prob_map,(3,3),0,0)

    map_mask = np.uint8(map_smooth>threshold)
    keypoints = []
    
    #find the blobs
    _, contours, _ = cv2.findContours(map_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #for each blob find the maxima
    for cnt in contours:
        blob_mask = np.zeros(map_mask.shape)
        blob_mask = cv2.fillConvexPoly(blob_mask, cnt, 1)
        masked_prob_map = map_smooth * blob_mask
        _, max_val, _, max_loc = cv2.minMaxLoc(masked_prob_map)
        keypoints.append(max_loc + (prob_map[max_loc[1], max_loc[0]],))

    return keypoints

def getKeypointsList(output, frame_width, frame_height, threshold=0.1, print_list=False):
    detected_keypoints = []
    keypoints_list = np.zeros((0,3))
    keypoint_id = 0
    for part in range(n_points):
        prob_map = output[0,part,:,:]
        prob_map = cv2.resize(prob_map, (frame_width, frame_height))
        keypoints = getKeypoints(prob_map, threshold)
        if print_list:
            print("Keypoints - {} : {}".format(keypoints_mapping[part], keypoints))
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)
    
    return keypoints_list, detected_keypoints

# Find valid connections between the different joints of a all persons present
def getValidPairs(output, n_interp_samples=10, paf_score_th=0.1, conf_th=0.7, print_list=False):
    valid_pairs = []
    invalid_pairs = []
    # loop for every POSE_PAIR
    for k in range(len(map_idx)):
        # A->B constitute a limb
        pafA = output[0, map_idx[k][0], :, :]
        pafB = output[0, map_idx[k][1], :, :]
        pafA = cv2.resize(pafA, (frame_width, frame_height))
        pafB = cv2.resize(pafB, (frame_width, frame_height))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[pose_pairs[k][0]]
        candB = detected_keypoints[pose_pairs[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB 
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid
        
        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                max_score = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ]) 
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)
                    
                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair  
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > max_score:
                            max_j = j
                            max_score = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:            
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], max_score]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            invalid_pairs.append(k)
            valid_pairs.append([])
    if print_list:
        print("Valid Pairs:")
        for i in range(len(valid_pairs)):
            print("{}. {}".format(i, valid_pairs[i]))
        print("Invalid Pairs:")
        print(invalid_pairs)
    return valid_pairs, invalid_pairs

# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
# It finds the person and index at which the joint should be added. This can be done since we have an id for each joint
def getPersonwiseKeypoints(valid_pairs, invalid_pairs, print_list=False):
    # the last number in each row is the overall score 
    personwise_keypoints = -1 * np.ones((0, 19))

    for k in range(len(map_idx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(pose_pairs[k])

            for i in range(len(valid_pairs[k])): 
                found = 0
                person_idx = -1
                for j in range(len(personwise_keypoints)):
                    if personwise_keypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwise_keypoints[person_idx][indexB] = partBs[i]
                    personwise_keypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score 
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwise_keypoints = np.vstack([personwise_keypoints, row])
    if print_list:
        print("Personwise Keypoints:")
        for i in range(len(personwise_keypoints)):
            print("Person {}".format(i))
            print(personwise_keypoints[i])
    return personwise_keypoints

def getFrame(video_name, n):
    input_source = videos_dir + video_name
    
    cap = cv2.VideoCapture(input_source)

    cap.set(cv2.CAP_PROP_POS_FRAMES, n)

    has_frame, image = cap.read()
    
    cap.release()
    
    frame_width = image.shape[1]
    frame_height = image.shape[0]
    
    return image, frame_width, frame_height

def visualizeFrame(video_name, n):
    
    image, _, _ = getFrame(video_name, n)
    
    plt.figure(figsize=[14,10])
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    
def SingleFrameInference(frame, print_time=True):
    t = time.time()
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
    
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    
    # Fix the input Height and get the width according to the Aspect Ratio
    in_height = 368
    in_width = int((in_height/frame_height)*frame_width)

    inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (in_width, in_height),
                              (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inp_blob)
    net_output = net.forward()
    time_taken = time.time() - t
    if print_time:
        print("Time Taken = {}".format(time_taken))
    return net_output, time_taken

def visualizeHeatmap(image, nn_output, j, threshold=0.1, alpha=0.6, binary=False):
    frame_width = image.shape[1]
    frame_height = image.shape[0]
    prob_map = output[0, j, :, :]
    prob_map = cv2.resize(prob_map, (frame_width, frame_height))
    if(binary == False):
        prob_map = np.where(prob_map<threshold, 0.0, prob_map)
    else:
        prob_map = np.where(prob_map<threshold, 0.0, prob_map.max())
    plt.figure(figsize=[14,10])
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.imshow(prob_map, alpha=alpha, vmax=prob_map.max(), vmin=0.0)
    plt.colorbar()
    plt.axis("off")
    
def visualizeResult(frame, personwise_keypoints, keypoints_list, persons, joint_pairs):
    
    frame_out = frame.copy()
    
    if (persons[0] == -1):
        persons = np.arange(len(personwise_keypoints))
        
    if (joint_pairs[0] == -1):
        joint_pairs = np.arange(len(pose_pairs)-2)
        
    for i in joint_pairs:
        for n in persons:
            index = personwise_keypoints[n][np.array(pose_pairs[i])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            cv2.line(frame_out, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

    plt.figure(figsize=[15,15])
    plt.imshow(frame_out[:,:,[2,1,0]])
    plt.axis("off")
    
def angle3pt(a, b, c):
    if b in (a, c):
        raise ValueError("Undefined angle, two identical points", (a, b, c))
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    if ang < 0:
        ang += 360
    if ang > 180:
        ang = 360 - ang
    return ang

def getJoint(keypoints_list, personwise_keypoints, person, joint_name):
    joint_n = keypoints_mapping.tolist().index(joint_name)
    index = personwise_keypoints[person][joint_n]
    X = np.int32(keypoints_list[index.astype(int), 0])
    Y = np.int32(keypoints_list[index.astype(int), 1])
    return (X, Y)

def readFrameJSON(file_path, frame_n=0):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i==0:
                metadata = json.loads(line)
            elif i==frame_n+1:
                data = json.loads(line)
    return metadata, data

def getVideos(input_video):
    video_path = videos_dir + input_video
    display(HTML("""<video width="840" height="460" controls="">
                <source src="{0}"></video>""".format(video_path)))

def interactiveResult(video_name, file_name, persons, joint_pose, frame_n=0):
    video_name = (video_name).split(sep='.')[0]
    file_dir = data_dir + video_name + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = file_dir + file_name
    metadata, data = readFrameJSON(file_path, frame_n=frame_n)
    personwise_keypoints = np.array(data["personwise_keypoints"]).astype(float)
    keypoints_list = np.array(data["keypoints_list"]).astype(float)
    if persons == 'Main':
        p = [0]
    else:
        p = [-1]
    
    if joint_pose == 'Sagittal':
        joint_pairs = [1,5,9,10,11,12,15,16,4]
    else:
        joint_pairs = [-1]
    
    video_name_ext = [filename for filename in os.listdir(videos_dir) if filename.startswith(metadata["video_name"])]
    image, _, _ = getFrame(video_name_ext[0], frame_n)
    visualizeResult(image, personwise_keypoints, keypoints_list, p, joint_pairs)

def interactiveInterface():
    videos = os.listdir(videos_dir)
    input_video = wg.Dropdown( options=videos,
                            value="Remo_Lite_480p.mp4",
                            description='Video:',
                            disabled=False)

    y = wg.interactive_output(getVideos, {"input_video":input_video})
    vbox_video = wg.VBox([input_video, y])     

    persons = wg.RadioButtons(options=['Main', 'All'],
                                value='Main',
                                rows=2,
                                description='Persons',
                                disabled=False)

    joint_pose = wg.RadioButtons(options=['Sagittal', 'Whole Body'],
                                    value='Sagittal',
                                    rows=2,
                                    description='Pose',
                                    disabled=False)

    video_path = videos_dir + input_video.value
    video_name = (input_video.value).split(sep='.')[0]

    file_dir = data_dir + video_name + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    files_list = os.listdir(file_dir)
    json_list = []
    for names in files_list:
        if names.endswith(".json"):
            json_list.append(names)

    json_dropdown = wg.Dropdown(options=json_list,
                            value=json_list[0],
                            description='File:',
                            disabled=False)

    frame_n= wg.IntText(value=0,description='Frame:')

    hbox = wg.HBox([json_dropdown, frame_n])

    vbox_res = wg.VBox([persons, joint_pose])

    i_res = wg.interactive_output(interactiveResult, {"video_name": input_video,
                                                      "file_name": json_dropdown,
                                                      "persons": persons,
                                                      "joint_pose": joint_pose,
                                                      "frame_n": frame_n})

    vbox_vid = wg.VBox([hbox, i_res])

    hbox_res = wg.HBox([vbox_vid, vbox_res])

    tabs = ['Videos', 'Data']
    children = []
    children.append(vbox_video)
    children.append(hbox_res)
    tab = wg.Tab()
    tab.children = children
    for i in range(len(children)):
        tab.set_title(i, tabs[i])
    display(tab)