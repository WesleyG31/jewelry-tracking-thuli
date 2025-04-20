from ultralytics import YOLO
import torch
import cv2
import numpy as np

class RingDetectorYOLO:
    def __init__(self, model_path, confidence_yolo=0.7):     #set the path to the model
        self.model = YOLO(model_path)   # get the model
        self.confidence_yolo = confidence_yolo

    def detect_ring_yolo(self, frame_roi):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        frames_roi= cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB)
        results= self.model.predict(frames_roi, verbose = False, device=device)
        boxes=results[0].boxes

        for box in boxes:
            class_id=int(box.cls[0])
            conf=float(box.conf[0])
            conf=round(conf,2)

            if class_id==0 and conf>=self.confidence_yolo:
                return True  # Ring detected

        return False


    