import cv2

protoFile = "testfolder/file/pose_deploy_linevec.prototxt"
weightsFile = "testfolder/file/pose_iter_160000.caffemodel"

def pose_save(frame,frame_num,x,y,w,h):
    
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    x1 = x + w // 2 - 120
    y1 = y + h // 2 - 120
    x2 = x + w //2 + 120
    y2 = y + h //2 + 120
    
    image_height, image_width, _ = frame.shape
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width, x2)
    y2 = min(image_height, y2)
    
    frame1 = frame[y1:y2,x1:x2]
    
    w = 240
    h = 240
    
    previous_x, previous_y = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    pairs = [[0,1], # head
         [1,2],[1,5], # sholders
         [2,3],[3,4],[5,6],[6,7], # arms
         [1,14],[14,11],[14,8], # hips
         [8,9],[9,10],[11,12],[12,13]] # legs
    
    circle_color, line_color = (0,255,255), (0,255,0)
    
    inpBlob = cv2.dnn.blobFromImage(frame1, 0.00392, (368, 368), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(inpBlob)
    output = net.forward()
    H = output.shape[2]
    W = output.shape[3]
    
    points = []
    x_data, y_data = [], []
    for i in range(15):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        x = (w * point[0]) / W
        y = (h * point[1]) / H
        
        if prob > 0.1:
            points.append((int(x), int(y)))
            x_data.append(x)
            y_data.append(y)
        else :
            points.append((0, 0))
            x_data.append(previous_x[i])
            y_data.append(previous_y[i])
    # print(x_data,y_data)
    for i in range(len(points)):
        cv2.circle(frame1, (points[i][0], points[i][1]), 2, circle_color, -1)
    for pair in pairs:
        partA = pair[0]
        partB = pair[1]
        cv2.line(frame1, points[partA], points[partB], line_color, 1, lineType=cv2.LINE_AA)
    cv2.imwrite('cropped_image/person_'+str(frame_num)+'_skelecton.jpg',frame1)
    cv2.imshow('skelecton',frame1)
    return x_data,y_data

    
    