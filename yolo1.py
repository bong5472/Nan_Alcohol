import cv2
import numpy as np

import os

import crop

model_file = 'yolov3.weights' #-- 본인 개발 환경에 맞게 변경할 것
config_file = 'yolov3.cfg' #-- 본인 개발 환경에 맞게 변경할 것
net = cv2.dnn.readNet(model_file, config_file)

def detect(frame,num):
    min_confidence = 0.5
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    img = cv2.resize(frame,None,fx=1,fy=1)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > min_confidence:
                # 탐지한 객체 박싱
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
               
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    
    for i in range(len(boxes)):
        if i in indexes and classes[class_ids[i]] == 'person':
            x,y,w,h = boxes[i]
            label = classes[class_ids[i]]
            crop.crop_save(frame,label,num,x,y,w,h)
            
    return x,y,w,h

def detect_car(frame,num):
    min_confidence = 0.5
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    img = cv2.resize(frame,None,fx=1,fy=1)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > min_confidence:
                # 탐지한 객체 박싱
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
               
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    check = 0
    for i in range(len(boxes)):
        if i in indexes and classes[class_ids[i]] == 'car':
            check += 1
            x,y,w,h = boxes[i]
            label = classes[class_ids[i]]
            # return x,y,w,h
            # crop.crop_save(frame,label,num,x,y,w,h)
        elif classes[class_ids[i]] == 'truck':
            check += 1
            x,y,w,h = boxes[i]
            label = classes[class_ids[i]]
    if check == 0:
        return 0,0,0,0
    else:
        return x,y,w,h