import cv2
import numpy as np
from src.hand_tracker import HandTracker
from src.region_finger import extract_region_finger,extract_rotated_roi_finger,extract_aligned_roi_finger


def main():
    cap = cv2.VideoCapture(1)  # Webcam
    tracker = HandTracker() # Initialize the hand tracker

    while cap.isOpened():
        success, fram = cap.read() # Read the frame from CAP
        if not success:
            break

        frame = cv2.flip(fram, 1)

        landmarks_list = tracker.detect_hands(frame) # Detect hands in the frame

        for hand_landmarks in landmarks_list: # For every hand detected
            h, w, _ = frame.shape # high, width, channels = frame.shape
            for lm in hand_landmarks.landmark:  # For every landmark in the hand

                cx, cy = int(lm.x * w), int(lm.y * h) # Get the coordinates of the landmark
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            #region_finger=extract_region_finger(hand_landmarks, frame, scale=0.4)
            region_finger = extract_aligned_roi_finger(hand_landmarks, frame)
            '''for region in region_finger:
                (x_min, y_min), (x_max, y_max) = region["coords"]
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(frame, region["finger"], (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) '''
            
            for region in region_finger:
                roi = region["roi"]
                finger = region["finger"]
                #(x_min, y_min), (x_max, y_max) = region["coords"]
                
                if roi is not None and roi.size > 0:
                    roi_resized = cv2.resize(roi, (256, 256))
                    cv2.imshow(f"{finger}_ROI", roi_resized)


                # Dibujar rectángulo aproximado (opcional)
                #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                #cv2.putText(frame, finger, (x_min, y_min - 10),
                 #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                for i in range(4):
                    pt1 = tuple(np.int32(region["pts"][i]))
                    pt2 = tuple(np.int32(region["pts"][(i+1)%4]))
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
