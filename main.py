from ultralytics import YOLO
import cv2

#load model
coco_model = YOLO('/Users/bright/ml-project/automatic-number-plate-recognition-python-yolov8/model/yolov8n.pt')
license_model = YOLO('/Users/bright/ml-project/automatic-number-plate-recognition-python-yolov8/model/car-licence-best.pt')
cap = cv2.VideoCapture("./sample.mp4")

frame_number = -1
status = True
vechicle_classes = [2, 3, 5, 7]

while status:
    status, frame = cap.read()
    if status and frame_number <10:
        pass

        #detech vechicle
        detection = coco_model(frame)[0]
        detection_list =[]
        for detection in detection.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vechicle_classes:
                detection_list.append([x1, y1, x2, y2, score, class_id])