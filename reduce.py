import cv2
import os

input_file = 'input.mp4'

def reduce_video(input_file,number):
    cap = cv2.VideoCapture(input_file)

    output_file = 'video_'+str(number)+'.mp4'

    output_width = 1280 
    output_height = 720 
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (output_width, output_height))

    while True:
        
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (output_width, output_height))
        out.write(resized_frame)
        cv2.imshow('Resized Video', resized_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    
def reduce_videoTime(input_file,number,start_time,output_path):
    
    end_time = start_time + 50
    output_file = output_path + 'video_'+str(number)+'_'+str(start_time)+'.mp4'
    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if cap.get(cv2.CAP_PROP_POS_MSEC) >= end_time * 1000:
            break

        out.write(frame)
        # cv2.imshow('Trimmed Video', frame)       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    print(output_path+' // finish')
    cv2.destroyAllWindows()
    

def get_files_in_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(file))
    return file_list

folder_path = 'before'
output_path = 'after/'
files = get_files_in_folder(folder_path)

num = 0
for file in files:
    num += 1
    input_file = folder_path+'/'+file
    print(file)
    reduce_videoTime(input_file,num,0,output_path)
    reduce_videoTime(input_file,num,50,output_path)
    reduce_videoTime(input_file,num,100,output_path)
    reduce_videoTime(input_file,num,150,output_path)
    reduce_videoTime(input_file,num,200,output_path)
    reduce_videoTime(input_file,num,250,output_path)
    
# os.system("shutdown /s /t 0")