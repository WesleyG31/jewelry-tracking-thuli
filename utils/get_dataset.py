import cv2
import os
from src.hand_tracker import HandTracker
from src.region_finger import extract_aligned_roi_finger

SAVE_DIR = "C:\All files\jewelry-tracking-thuli\data\Raw"
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    cap = cv2.VideoCapture(1)  
    tracker = HandTracker()
    counter = 0
    max_images = 100

    print(f"Working...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success or counter >= max_images:
            break

        frame = cv2.flip(frame, 1)
        landmarks_list = tracker.detect_hands(frame)

        for hand_landmarks in landmarks_list:
            regions = extract_aligned_roi_finger(hand_landmarks, frame)

            for region in regions:
                roi = region["roi"]
                finger = region["finger"]

                if roi is None or roi.size == 0:
                    continue

                roi_resized = cv2.resize(roi, (256, 256))
                filename = f"{finger}_rooi_{counter:04d}.jpg"
                filepath = os.path.join(SAVE_DIR, filename)
                cv2.imwrite(filepath, roi_resized)
                counter += 1

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:  
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f" Completado: {counter} ROIs guardadas.")

if __name__ == "__main__":
    main()
