import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from support import *
from preprocessing import *

videos_dir = "../Videos/"
data_dir = "../Data/"

n_points = 18

map_idx = np.array([[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], 
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], 
          [47,48], [49,50], [53,54], [51,52], [55,56], 
          [37,38], [45,46]])

pose_pairs = np.array([[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16]])

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

def organizeBiggestPerson(personwise_keypoints, keypoints_list):
    biggest_dict = {}
    unsorted_keypoints = -1*np.ones([len(personwise_keypoints), n_points, 2])
    sorted_keypoints = -1*np.ones([len(personwise_keypoints), n_points, 2])
    for n in range(len(personwise_keypoints)):
        for i in range(n_points):
            index = personwise_keypoints[n][i]
            if index == -1:
                continue
            unsorted_keypoints[n][i] = keypoints_list[int(personwise_keypoints[n][i])][0:2]
        biggest_dict[rectangularArea(unsorted_keypoints[n])] = n
    biggest_values = sorted(biggest_dict, reverse=True)
    n = 0
    for i in biggest_values:
        index = biggest_dict[int(i)]
        sorted_keypoints[n] = unsorted_keypoints[index]
        n += 1
    return sorted_keypoints

# Find the Keypoints using Non Maximum Suppression on the Confidence Map
def defineKeypoints(prob_map, threshold=0.1):
    
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

def processKeypoints(output, frame_width, frame_height, threshold=0.1, n_interp_samples=10, paf_score_th=0.1, conf_th=0.7):
    detected_keypoints = []
    keypoints_list = np.zeros((0,3))
    keypoint_id = 0
    for part in range(n_points):
        prob_map = output[0,part,:,:]
        prob_map = cv2.resize(prob_map, (frame_width, frame_height))
        keypoints = defineKeypoints(prob_map, threshold)
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)

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

    return personwise_keypoints, keypoints_list

def saveJointFile(video_name, file_name, output_name, threshold, n_interp_samples, paf_score_th, conf_th, summary):
    if(video_name == "None"):
        print("No video found")
        return
    if(file_name == "None"):
        print("No JSON found")
        return
    video_name = (video_name).split(sep='.')[0]
    file_dir = data_dir + video_name + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = file_dir + file_name
    metadata, data = readFrameJSON(file_path, frame_n=frame_n)
    n_frames, fps = metadata["n_frames"], metadata["fps"]
    frame_height, frame_width = metadata["frame_height"], metadata["frame_width"]
    
    output_path = file_dir + output_name + ".data"
    video_path = file_dir + output_name + ".mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    vid_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width,frame_height))
    
    metadata["summary"] = summary
    with open(output_path, 'w') as f:
        f.write(json.dumps(metadata))
        f.write('\n')
    
    for n in range(n_frames):
        _, data = readFrameJSON(file_path, frame_n=n)
        output = np.array(data["output"]).astype(float)
        personwise_keypoints, keypoints_list = processKeypoints(output, frame_width, frame_height, threshold, 
                         n_interp_samples, paf_score_th, conf_th)
        sorted_keypoints = organizeBiggestPerson(personwise_keypoints, keypoints_list)
        main_keypoints = sorted_keypoints[0]
        
        file_data = {
            'keypoints': main_keypoints.tolist(),
            'angles': []
        }
        
        with open(output_path, 'a') as f:
            f.write(json.dumps(file_data))
            f.write('\n')
        
        frame, _, _getFrame(video_name, n)
        
        for i in range(n_points-1):
            B = int(main_keypoints[i, 0])
            A = int(main_keypoints[i, 1])
            if -1 in (A,B):
                cv2.line(frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
                
        vid_writer.write(frame)
       
    vid_writer.release()
    print("Done")
    
    