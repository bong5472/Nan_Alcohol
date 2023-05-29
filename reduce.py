import cv2

input_file = 'C:/Users/User/Desktop/공모전/해커톤/video/input.mp4'

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