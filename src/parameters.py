import numpy as np

openpose_path = '/home/deskema/VictorM/openpose/'
repo_path = '/home/deskema/VictorM/TCC_Biomec/'

proto_file = "Models/Openpose/coco/pose_deploy_linevec.prototxt"
weights_file = "Models/Openpose/coco/pose_iter_440000.caffemodel"
videos_dir = "Videos/"
data_dir = "Data/"

colors = [[0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], 
         [0,0,0], [0,0,0], [0,255,0], [0,255,0]]

colors_2 = [[0,255,0], [255,0,0], [0,0,255], [0,255,255],[255,255,0], 
         [255,0,255], [0,255,0], [255,200,100], [200,255,100],
         [100,255,200], [255,100,200], [100,200,255], [200,100,255],
         [200,200,0], [200,0,200],[0,200,200]]

colors_25 = [[0,255,0], [255,0,0], [0,0,255], [0,255,255],[255,255,0], 
         [255,0,255], [0,255,0], [255,200,100], [200,255,100],
         [100,255,200], [255,100,200], [100,200,255], [200,100,255],
         [200,200,0], [200,0,200],[0,200,200], [0,255,0], [255,0,0], [0,0,255],
         [100,255,200], [255,100,200], [100,200,255], [200,100,255],
         [0,0,255], [0,255,255],[255,255,0]]

indep_colors = [[1,0,0], [0,1,0], [0,0,1], 
            [0,1,1],[0,0.25,0.5], [1, 0, 0.75], 
            [1,1,1], [0, 0.5, 1], [1, 0.75, 0],
            [0.5, 0.5, 0.5], [0, 0.25, 0],[0.1, 0.1, 0],
            [0, 0, 0.25], [0.1, 0, 0.1], [0.6, 1, 0.6],
            [0.5, 0, 0.25], [0.6, 1, 1], [1, 0.75, 1],
            [1, 1, 0.6], [0.25,0.25,0.6], [0.6,0.25,0.25], 
            [0.6,0.6,0.1], [0.1,0.6,0.6], [0.25,0.6,0.25]]

keypoints_mapping_COCO = ['Nose', 'Neck', 'Right Shoulder', 'Right Elbow', 'Right Wrist', 'Left Shoulder', 
                    'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee', 'Right Ankle', 'Left Hip', 
                    'Left Knee', 'Left Ankle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear']

keypoints_mapping_BODY_25 = ['Nose', 'Neck', 'Right Shoulder', 'Right Elbow', 'Right Wrist', 'Left Shoulder', 
                    'Left Elbow', 'Left Wrist', 'Middle Hip', 'Right Hip', 'Right Knee', 'Right Ankle', 'Left Hip', 
                    'Left Knee', 'Left Ankle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear', 'Left Big Toe', 
                    'Left Small Toe', 'Left Heel', 'Right Big Toe', 'Right Small Toe', 'Right Heel', 'Background']

Tennis_mapping = ['Neck', 'Right Shoulder', 'Right Elbow', 'Right Wrist', 'Left Shoulder', 
                    'Left Elbow', 'Left Wrist', 'Middle Hip', 'Right Hip', 'Right Knee', 'Right Ankle', 'Left Hip', 
                    'Left Knee', 'Left Ankle']

SL_mapping = ['Left Ankle', 'Left Knee', 'Left Hip', 'Left Shoulder', 'Left Elbow', 'Left Wrist']
SR_mapping = ['Right Ankle', 'Right Knee', 'Right Hip', 'Right Shoulder', 'Right Elbow', 'Right Wrist']

SL_pairs = [['Left Shoulder', 'Left Elbow'], ['Left Elbow', 'Left Wrist'], ['Left Shoulder', 'Left Hip'], 
            ['Left Hip', 'Left Knee'], ['Left Knee', 'Left Ankle']]
SR_pairs = [['Right Shoulder', 'Right Elbow'], ['Right Elbow', 'Right Wrist'], ['Right Shoulder', 'Right Hip'], 
            ['Right Hip', 'Right Knee'], ['Right Knee', 'Right Ankle']]

BODY_25_pairs = [['Neck', 'Right Shoulder'], ['Neck', 'Left Shoulder'], ['Right Shoulder', 'Right Elbow'], 
                ['Right Elbow', 'Right Wrist'], ['Left Shoulder', 'Left Elbow'], ['Left Elbow', 'Left Wrist'], 
                ['Neck', 'Middle Hip'], ['Middle Hip', 'Right Hip'], ['Right Hip', 'Right Knee'], 
                ['Left Hip', 'Left Knee'], ['Middle Hip', 'Left Hip'], ['Right Knee', 'Right Ankle'], 
                ['Right Ankle', 'Right Heel'], ['Right Ankle', 'Right Big Toe'], ['Right Big Toe', 'Right Small Toe'], 
                ['Left Knee', 'Left Ankle'], ['Left Ankle', 'Left Heel'], ['Left Ankle', 'Left Big Toe'], 
                ['Left Big Toe', 'Left Small Toe'], ['Neck', 'Nose'], ['Nose', 'Right Eye'], ['Nose', 'Left Eye'], 
                ['Right Eye', 'Right Ear'], ['Left Eye', 'Left Ear']]

Tennis_pairs = [['Neck', 'Right Shoulder'], ['Neck', 'Left Shoulder'], ['Right Shoulder', 'Right Elbow'], 
                ['Right Elbow', 'Right Wrist'], ['Left Shoulder', 'Left Elbow'], ['Left Elbow', 'Left Wrist'], 
                ['Neck', 'Middle Hip'], ['Middle Hip', 'Right Hip'], ['Right Hip', 'Right Knee'], 
                ['Left Hip', 'Left Knee'], ['Middle Hip', 'Left Hip'], ['Right Knee', 'Right Ankle'], ['Left Knee', 'Left Ankle']]

pose_pairs_COCO = np.array([[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
            [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
            [1,0], [0,14], [14,16], [0,15], [15,17],
            [2,17], [5,16], [2, 8], [5, 11]])
      
pose_pairs_BODY_21 = np.array([[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
            [1,8], [8,9], [9,10], [12,13],
            [1,0], [0,14], [14,16], [0,15], [15,17],
            [2,17], [5,16], [2, 8], [5, 11]])

pose_pairs_BODY_25 = np.array([[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
            [1,8], [8,9], [9,10], [12,13], [8,12], [10,11], 
            [11,24], [11,22], [22,23], [13,14], [14,21], [14,19], [19,20],
            [1,0], [0,15], [0,16], [15,17], [16,18]])

pose_pairs_BODY_25 = np.array([[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
            [1,8], [8,9], [9,10], [12,13], [8,12], [10,11], 
            [11,24], [11,22], [22,23], [13,14], [14,21], [14,19], [19,20],
            [1,0], [0,15], [0,16], [15,17], [16,18]])

object_x_mm = 75.0
object_y_mm = 75.0

