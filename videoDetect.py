import cv2
# import numpy as np
# import pandas as pd
import yolo1
import openpose
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"


video_path = 'testfolder/short.mp4' #-- 사용할 영상 경로

def detect_to_csv(video_path,label):
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    data = []
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        
        x,y,w,h = yolo1.detect(frame,frame_num)
        x_data, y_data = openpose.pose_save(frame,frame_num,x,y,w,h)
        
        data.append(x_data + y_data + [frame_num] + [label])
        frame_num += 1
        #-- q 입력시 종료
        if cv2.waitKey(1) & 0xFF == ord('q') & frame_num == 150:
            break
        
    # df = pd.DataFrame(data)
    # df.to_csv('output.csv',index=False)
    # print('saved')
    return data