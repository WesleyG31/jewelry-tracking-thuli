# hand_tracker.py
import mediapipe as mp #Importing mediapipe for hand tracking
import cv2 # Importing OpenCV for image processing

class HandTracker: 
    def __init__(self): # All variables we need for hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, #real time
                                         max_num_hands=2,
                                         min_detection_confidence=0.7)  # First I'll try with 0.7 confidence
        self.mp_draw = mp.solutions.drawing_utils #to show the hand points on the image

    def detect_hands(self, frame):
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # OpenCV uses BGR format, so we need to convert it to RGB for mediapipe
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks
        return []
