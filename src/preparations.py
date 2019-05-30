import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from support import getFrame

videos_dir = "../Videos/"
allvid_dir = "../Others/"
data_dir = "../Data/"

def calibrateVideo():
    pass

def editVideo(video_name, allvid_name, n, r, x=[0,-1], y=[0,-1], in_height=-1, 
            save_vid=False, output_path='None'):
    allvid = False
    if(video_name == "None" and allvid_name == "None"):
        print("Choose a video")
        return
    elif(allvid_name != "None"):
        video_name = allvid_name
        input_source = allvid_dir + video_name
        allvid = True
    else:
        input_source = videos_dir + video_name

    n = int(np.round(n))
    
    if save_vid:            
        cap = cv2.VideoCapture(input_source)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        image, frame_width, frame_height = getFrame(video_name, 0)

        if(in_height != frame_height):
            in_width = int((in_height/frame_height)*frame_width)
            image = cv2.resize(image, (in_width, in_height), interpolation = cv2.INTER_AREA)
        image = image[y[0]:y[1],x[0]:x[1]]
        
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        
        if not output_path.endswith(".mp4"):
            output_path = output_path + ".mp4"
            
        out = cv2.VideoWriter(output_path,fourcc, fps, (image.shape[1],image.shape[0]))
        print("Saving video...", end="\r")
        cap.set(cv2.CAP_PROP_POS_FRAMES, r[0])
        for fn in range(r[1] - r[0]):
            t = time.time()
            ret, image = cap.read()
            if ret==True:
                if(in_height != frame_height):
                    in_width = int((in_height/frame_height)*frame_width)
                    image = cv2.resize(image, (in_width, in_height), interpolation = cv2.INTER_AREA)
                image = image[y[0]:y[1],x[0]:x[1]]
                cv2.imshow('frame',image)
                out.write(image)
            else:
                break
            time_taken = time.time() - t
            print("                                                                   ", end="\r")
            print("[{0:d}/{1:d}] {2:.1f} seconds/frame".format(fn+1, r[1] - r[0], time_taken), end="\r")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print()
        print("Video saved")
    else:
        image, frame_width, frame_height = getFrame(video_name, n, allvid)
        if(in_height != frame_height):
            in_width = int((in_height/frame_height)*frame_width)
            image = cv2.resize(image, (in_width, in_height), interpolation = cv2.INTER_AREA)
        image = image[y[0]:y[1],x[0]:x[1]]
        plt.figure(figsize=[8,5])
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")