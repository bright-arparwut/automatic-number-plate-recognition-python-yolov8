from ultralytics import YOLO
import cv2

from sort.sort import *

#load model
coco_model = YOLO('/Users/bright/ml-project/automatic-number-plate-recognition-python-yolov8/model/yolov8n.pt')
license_model = YOLO('/Users/bright/ml-project/automatic-number-plate-recognition-python-yolov8/model/car-licence-best.pt')
cap = cv2.VideoCapture("./sample.mp4")

mot_tracker = Sort()

frame_number = -1
status = True
vechicle_classes = [2, 3, 5, 7]

while status:
    status, frame = cap.read()
    if status and frame_number <10:
        pass

#Detect vechicle
        detection = coco_model(frame)[0]
        detection_list =[]
        for detection in detection.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vechicle_classes:
                detection_list.append([x1, y1, x2, y2, score, class_id])
                
#Track Vechicles
    track_ids = mot_tracker.update(np.asarray(detection_list))
    
#Detect License Plate
    license_plates = license_model(frame)[0]