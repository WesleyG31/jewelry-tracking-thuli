import cv2
import numpy as np

class RingDetectorColors:
    def __init__(self):
        self.lower_gold = np.array([15, 80, 80])        # Normal Ring GOLD
        self.upper_gold = np.array([35, 255, 255])

        self.lower_silver = np.array([0, 0, 150])       # Normal Ring SILVER
        self.upper_silver = np.array([180, 40, 255])
        
        self.lower_blue = np.array([90, 50, 50])        # Custom RING BLUE
        self.upper_blue = np.array([130, 255, 255])

        self.lower_black = np.array([0, 0, 0])          # Custom RING BLACK
        self.upper_black = np.array([180, 255, 50])

    def detect_in_finger_colors(self, frame, box_coords):

        x_min, y_min = box_coords[0]
        x_max, y_max = box_coords[1]

        roi = frame[y_min:y_max, x_min:x_max]   # get the box from the frame
        if roi.size == 0:
            return False  

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) # OpenCV uses BGR we need to convert to HSV to get the colors


        mask_blue = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        mask_black = cv2.inRange(hsv, self.lower_black, self.upper_black)
        mask_gold = cv2.inRange(hsv, self.lower_gold, self.upper_gold)
        mask_silver = cv2.inRange(hsv, self.lower_silver, self.upper_silver)

        combined_mask = cv2.bitwise_or(mask_blue, mask_black) # All the colors we combined all the colors here 

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 1000: 
                return True 

        return False
