import cv2
from src.hand_tracker import HandTracker

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
            for lm in hand_landmarks.landmark:  # For every landmark in the hand
                h, w, _ = frame.shape # high, width, channels = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h) # Get the coordinates of the landmark
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
