import cv2
import yolo1
import openpose

def count_zero(arr):
    count = 0
    for i in arr:
        if i == 0:
            count += 1
    return count

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    data = []
    drunken = False
    if not cap.isOpened:
        print('Video failed --- break')
        exit(0)
    
    while True:
        ret, frame = cap.read()
        
        if frame is None:
            print('frame not detected --- break')
            break
        # drunk detect 전
        if not drunken:
            x,y,w,h = yolo1.detect(frame,frame_num)
            
            if w == 0 and h == 0:
                print('person not detected --- ')
                
            else:
                x_data,y_data = openpose.pose_save(frame,frame_num,x,y,w,h)
                pose_data = x_data + y_data
                # 포즈 데이터에 0이 많으면 제대로 인식이 안됬다고 가정
                if count_zero(pose_data) >= 20:
                    continue
                else:
                    data.append(pose_data)
            frame_num += 1
        # drunk detect 후
        else:
            print()
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    