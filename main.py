import cv2
import yolo1
import openpose
import tensorflow as tf

model = tf.keras.models.load_model('drunk.h5') # DL 모델
def count_zero(arr):
    count = 0
    for i in arr:
        if i == 0:
            count += 1
    return count

def main(video_path):
    output_file = 'final_result.mp4' 
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)
    
    frame_num = 0
    data = [[]]
    result = [[0]]
    drunken = False
    tracking_data = [[0,0,0,0]]*60
    d_f = 0
    detect = False
    if not cap.isOpened:
        print('Video failed --- break')
        exit(0)
    
    while True:
        print(frame_num)
        ret, frame = cap.read()
        if frame is None:
            print('frame not detected --- break')
            break
        # drunk detect 전
        if not drunken and not detect:
            x,y,w,h = yolo1.detect(frame,frame_num)
            cv2.rectangle(frame,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),2,1)
            if w == 0 and h == 0:
                print('person not detected --- ')
                
            else:
                x_data,y_data = openpose.pose_save(frame,frame_num,x,y,w,h)
                pose_data = x_data + y_data
                # 포즈 데이터에 0이 많으면 제대로 인식이 안됬다고 가정
                if count_zero(pose_data) <= 10:
                    data[0].append(pose_data)
            # 100프레임 고정
            if len(data[0]) > 100:
                data = [data[0][1:]]
                result = model.predict(data)
                print('pose >:',result[0][0])
            elif len(data[0]) == 100:
                result = model.predict(data)
                print('pose =:',result[0][0])
            frame_num += 1
            # tracker 설정
            if result[0][0] >= 0.5:
                print('trans')
                drunken = True
                tracker = cv2.TrackerMIL_create()
                isInit = tracker.init(frame,(x,y,w,h))
        # drunk detect 후
        else:
            ok, bbox = tracker.update(frame)
            (x,y,w,h) = bbox
            if ok:
                cv2.rectangle(frame,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,255),2,1)
                cv2.putText(frame,'drunk',(x,y-10),1,1,(0,255,255),2,1)
                tracking_data.append([x,y,w,h])
            # 목표가 사라졌을 때
            if tracking_data[-49] == tracking_data[-1]:
                car_x,car_y,car_w,car_h = yolo1.detect_car(frame)
                position_x = x + w/2
                position_y = y + h/2
                if car_x <= position_x and position_x <= car_x + car_w and car_y <= position_y and position_y <= car_y + car_h:
                    cv2.rectangle(frame,(int(car_x),int(car_y)),(int(car_x+car_w),int(car_y+car_h)),(0,0,255),2,1)    
                    cv2.putText(frame,'Detected!!!',(int(car_x),int(car_y) - 10),1,1,(0,0,255),2,1)
                    detect = True
                else:
                    drunken = False
                    data = [[]]
                    result = [[0]]
                    tracking_data = [[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        if detect:
            cv2.rectangle(frame,(int(car_x),int(car_y)),(int(car_x+car_w),int(car_y+car_h)),(0,0,255),2,1)    
            cv2.putText(frame,'Detected!!!',(int(car_x),int(car_y) - 10),1,1,(0,0,255),2,1)
        cv2.imshow('drunk detecting...',frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
           
main("test3.mp4")