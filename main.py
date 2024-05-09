from ultralytics import YOLO
import cv2

#load model
model = YOLO('/Users/bright/ml-project/automatic-number-plate-recognition-python-yolov8/model/car-licence-best.pt')
cap = cv2.VideoCapture("./sample.mp4")

frame_number = -1
status = True
while status:
    status, frame = cap.read()
    if status and frame_number <10:
        pass

        #detech vechicle
        detection = model(frame)[0]
        print(detection)