import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def getPointsDistance(A, B):
    distance = np.sqrt(np.sum(np.power((A - B),2)))
    return distance
                      
def getBoneDimensions(keypoints_xy):
    """ Expected joint order
    if orientation == "Sagittal Right":
        joints_order = ["Right Ankle", "Right Knee", "Right Hip", "Right Shoulder", "Right Elbow", "Right Wrist"]
    else:
        joints_order = ["Left Ankle", "Left Knee", "Left Hip", "Left Shoulder", "Left Elbow", "Left Wrist"]
    """
    distances = np.zeros(5)
    for i in range(5):
        distances[i] = getPointsDistance(keypoints_xy[i], keypoints_xy[i+1])
    return distances

def getPixel(coord_xy, f_height, mmppx=1, mmppy=1):
    j = int(round(coord_xy[0]/mmppx))
    i = int(round(f_height - (coord_xy[1]/mmppy)))
    return np.array([j, i])

def getCoord(pixel_ji, f_height, mmppx=1, mmppy=1):
    x = pixel_ji[0]*mmppx
    y = (f_height - pixel_ji[1])*mmppy
    return np.array([x, y])

def getKeypointsPixels(keypoints_xy, f_height, mmppx=1, mmppy=1):
    keypoints_ji = np.zeros(keypoints_xy.shape)
    for i in range(len(keypoints)):
        keypoints_ji[i] = list(getPixel(keypoints_xy[i], f_height, mmppx, mmppy))
    return keypoints_ji

def getKeypointsCoord(keypoints_ji, f_height, mmppx=1, mmppy=1):
    keypoints_xy = np.zeros(keypoints_ji.shape)
    for i in range(len(keypoints_xy)):
        keypoints_xy[i] = list(getCoord(keypoints_ji[i], f_height, mmppx, mmppy))
    return keypoints_xy

def showFrame(frame):
    plt.figure(figsize=[9,6])
    plt.imshow(frame[:,:,[2,1,0]])
    plt.axis("off")
    plt.show()

def drawCircle(img, coord, thickness=5):
    coord = tuple(coord.astype(int))
    cv2.circle(img, coord, thickness, (0,0,1), -1)
    return img

def drawLine(img, A, B, color = (0,0,0), thickness = 2):
    A = tuple(A.astype(int))
    B = tuple(B.astype(int))
    cv2.line(img, A, B, color,thickness)
    return img

def getAngle(A, B, O):
    x1, y1, x2, y2 = A[0], A[1], B[0], B[1]
    angle = int(math.atan((y1-y2)/(x2-x1))*180/math.pi)
    return angle

def getAngleLimited(A, B, O):
    try:
        ang = math.degrees(math.atan2(B[1]-O[1], B[0]-O[0]) - math.atan2(A[1]-O[1], A[0]-O[0]))
        if ang < 0:
            ang += 360
        if ang > 180:
            ang = 360 - ang
    except:
        ang = 0
    return ang

def drawAngle(img, O, angle, thickness=10, textsize=1):
    O = tuple(O.astype(int))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.circle(img, O, thickness, (1,0.8,0.8), -1)
    cv2.putText(img,str(int(round(angle))),(O[0]-round(thickness/2),O[1]-round(thickness/2)), font, textsize,(1,0,0),2,cv2.LINE_AA)
    return img

def showAngleImage(frame, A, B, O):
    frame = drawLine(frame, A, O)
    frame = drawLine(frame, O, B)
    frame = drawCircle(frame, A)
    frame = drawCircle(frame, B)
    frame = drawCircle(frame, O)
    angle = getAngleLimited(A, B, O)
    frame = drawAngle(frame, O, angle)
    showFrame(frame)

def showAnglePlot(A, B, O):
    angle = getAngleLimited(A, B, O)
    plt.figure()
    plt.plot([A[0], O[0]], [A[1], O[1]], 'ro-')
    plt.plot([O[0], B[0]], [O[1], B[1]], 'ro-')
    plt.text(O[0], O[1], "{}".format(round(angle)), bbox=dict(facecolor='red', alpha=0.5))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.show()

def showJointsImage(frame, keypoints_ji, joint_pairs, thickness=5, textsize=0.8):
#     joint_names_abv = ["RA", "RK", "RH", "RS", "RE", "RW"]
#     i = 0
    for pair in joint_pairs:
        A = keypoints_ji[pair[0]]
        B = keypoints_ji[pair[1]]
        frame = drawLine(frame, A, B)
        frame = drawCircle(frame, A)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(frame,joint_names_abv[i],(int(A[0]-round(thickness/2)),int(A[1]-round(thickness/2))), font, textsize,(0,0,1),2,cv2.LINE_AA)
#         i+=1
    showFrame(frame)
    
def showJointsPlot(keypoints_xy, joint_pairs):
    plt.figure()
    for pair in joint_pairs:
        A = keypoints_xy[pair[0]]
        B = keypoints_xy[pair[1]]
        plt.plot([A[0], B[0]], [A[1], B[1]], 'ro-', color = "black")
        circle1 = plt.Circle((A[0], A[1]), 1, color='r')
        plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.show()

def inverseKinematicsRowing(keypoints):
    """ Expected joint order
    if orientation == "Sagittal Right":
        joints_order = ["Right Ankle", "Right Knee", "Right Hip", "Right Shoulder", "Right Elbow", "Right Wrist"]
    else:
        joints_order = ["Left Ankle", "Left Knee", "Left Hip", "Left Shoulder", "Left Elbow", "Left Wrist"]
    """
    angles = np.zeros(5)
    for i in range(5):
        if i == 0:
            O = keypoints[i]
            B = keypoints[i+1]
            C = keypoints[i+2]
            A = np.array([C[0], O[1]])
        else:
            A = keypoints[i-1]
            O = keypoints[i]
            B = keypoints[i+1]
            
        angles[i] = getAngleLimited(A, B, O)
        if i==3:
            if A[0] > B[0]:
                angles[i] = -angles[i]
    return angles

def fowardKinematicsRowing(root_xy, angles, distances, orientation = "Sagittal Left"):
    keypointsFK_xy = np.zeros([6, 2])
    keypointsFK_xy[0] = root_xy
    if orientation == "Sagittal Left":
        sign = 1
    else:
        sign = -1
    for i in range (1, 6):
        if i == 1:
            arg = math.radians(angles[i-1])
            keypointsFK_xy[i, 0] = keypointsFK_xy[i-1, 0] - (sign)*distances[i-1]*math.cos(arg)
            keypointsFK_xy[i, 1] = keypointsFK_xy[i-1, 1] + distances[i-1]*math.sin(arg)
        elif i == 2:
            arg = math.pi - arg - math.radians(angles[i-1])
            keypointsFK_xy[i, 0] = keypointsFK_xy[i-1, 0] - (sign)*distances[i-1]*math.cos(arg)
            keypointsFK_xy[i, 1] = keypointsFK_xy[i-1, 1] - distances[i-1]*math.sin(arg)
        elif i == 3:
            #arg = arg - math.radians(angles[i-1])
            arg = math.pi - arg - math.radians(angles[i-1])
            keypointsFK_xy[i, 0] = keypointsFK_xy[i-1, 0] - (sign)*distances[i-1]*math.cos(arg)
            keypointsFK_xy[i, 1] = keypointsFK_xy[i-1, 1] + distances[i-1]*math.sin(arg)
        elif i == 4:
            #arg = math.pi - arg - math.radians(angles[i-1])
            arg = arg - math.radians(angles[i-1])
            keypointsFK_xy[i, 0] = keypointsFK_xy[i-1, 0] + (sign)*distances[i-1]*math.cos(arg)
            keypointsFK_xy[i, 1] = keypointsFK_xy[i-1, 1] - distances[i-1]*math.sin(arg)
        elif i == 5:
            arg = math.pi - arg - math.radians(angles[i-1])
            keypointsFK_xy[i, 0] = keypointsFK_xy[i-1, 0] + (sign)*distances[i-1]*math.cos(arg)
            keypointsFK_xy[i, 1] = keypointsFK_xy[i-1, 1] + distances[i-1]*math.sin(arg)
            
    return keypointsFK_xy

def showRowingChainAnglesImage(frame, keypoints_ji, textsize=0.5):
    """ Expected joint order
    if orientation == "Sagittal Right":
        joints_order = ["Right Ankle", "Right Knee", "Right Hip", "Right Shoulder", "Right Elbow", "Right Wrist"]
    else:
        joints_order = ["Left Ankle", "Left Knee", "Left Hip", "Left Shoulder", "Left Elbow", "Left Wrist"]
    """    
    angles = inverseKinematicsRowing(keypoints_ji)
    for i in range(5):
        O = keypoints_ji[i]
        B = keypoints_ji[i+1]       
        frame = drawLine(frame, O, B)
        frame = drawAngle(frame, O, angles[i], 5, textsize)
        
    showFrame(frame)
    
def showRowingChainAnglesPlot(keypoints_xy): 
    """ Expected joint order
    if orientation == "Sagittal Right":
        joints_order = ["Right Ankle", "Right Knee", "Right Hip", "Right Shoulder", "Right Elbow", "Right Wrist"]
    else:
        joints_order = ["Left Ankle", "Left Knee", "Left Hip", "Left Shoulder", "Left Elbow", "Left Wrist"]
    """    
    angles = inverseKinematicsRowing(keypoints_xy)
    
    plt.figure()
    for i in range(5):
        O = keypoints_xy[i]
        B = keypoints_xy[i+1]          
        plt.plot([O[0], B[0]], [O[1], B[1]], 'ro-', color = "black")
        circle1 = plt.Circle((O[0], O[1]), 1, color='r')
        plt.text(O[0], O[1], "{}".format(int(round(angles[i]))), bbox=dict(facecolor='red', alpha=0.5))
        plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.show()

def showRowingChainDistancesPlot(keypoints_xy, plot_path=None, show_plot=True): 
    """ Expected joint order
    if orientation == "Sagittal Right":
        joints_order = ["Right Ankle", "Right Knee", "Right Hip", "Right Shoulder", "Right Elbow", "Right Wrist"]
    else:
        joints_order = ["Left Ankle", "Left Knee", "Left Hip", "Left Shoulder", "Left Elbow", "Left Wrist"]
    """    
    distances = getBoneDimensions(keypoints_xy)
    if show_plot:
        plt.figure()
    for i in range(5):
        A = keypoints_xy[i]
        B = keypoints_xy[i+1]
        O = (A+B)/2.0
               
        plt.plot([A[0], B[0]], [A[1], B[1]], 'ro-', color = "black")
        circle1 = plt.Circle((A[0], A[1]), 1, color='r')
        plt.text(O[0], O[1], "{}".format(int(round(distances[i]))), bbox=dict(facecolor='blue', alpha=0.5))
        plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    if not (plot_path == None):
        plt.savefig(plot_path)
    if show_plot:
        plt.show()
    else:
        plt.show(False)
    
def showRowingChainAnglesDistancesPlot(keypoints_xy): 
    """ Expected joint order
    if orientation == "Sagittal Right":
        joints_order = ["Right Ankle", "Right Knee", "Right Hip", "Right Shoulder", "Right Elbow", "Right Wrist"]
    else:
        joints_order = ["Left Ankle", "Left Knee", "Left Hip", "Left Shoulder", "Left Elbow", "Left Wrist"]
    """    
    angles = inverseKinematicsRowing(keypoints_xy)
    
    plt.figure()
    for i in range(5):
        A = keypoints_xy[i]
        B = keypoints_xy[i+1]
        O = (A+B)/2.0
               
        plt.plot([A[0], B[0]], [A[1], B[1]], 'ro-', color = "black")
        circle1 = plt.Circle((A[0], A[1]), 1, color='r')
        plt.text(O[0], O[1], "{}".format(int(round(distances[i]))), bbox=dict(facecolor='blue', alpha=0.5))
        plt.text(A[0], A[1], "{}".format(int(round(angles[i]))), bbox=dict(facecolor='red', alpha=0.5))
        plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.show()