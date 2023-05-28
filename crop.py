import cv2

def crop_save(frame,label,frame_num,x,y,w,h):
    x1 = x + w // 2 - 120
    y1 = y + h // 2 - 120
    x2 = x + w //2 + 120
    y2 = y + h //2 + 120
    image_height, image_width, _ = frame.shape
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width, x2)
    y2 = min(image_height, y2)
    
    cropped = frame[y1:y2,x1:x2]
    cv2.imwrite('cropped_image/'+label+'_'+str(frame_num)+'.jpg',cropped)