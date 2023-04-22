from ultralytics import YOLO
import cv2
model=YOLO('yolov8n.pt')
result=model('mvpa.mp4',save_txt=True,save=True)
