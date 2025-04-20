import cv2
from src.hand_tracker import HandTracker       # hand detection
from src.region_finger import extract_region_finger,extract_aligned_roi_finger  # region of interest
from src.ring_detector_colors import RingDetectorColors     # ring detection


def main():
    cap = cv2.VideoCapture(1)  # webcam (1) or path to the video
    tracker = HandTracker()    # Hand tracker
    detector = RingDetectorColors()  # ring detector

    while cap.isOpened():
        success, fram = cap.read() # Read the frame from CAP
        if not success:
            break

        frame = cv2.flip(fram, 1)

        #frame_height, frame_width = frame.shape[:2]
        landmarks_list = tracker.detect_hands(frame)



        for hand_landmarks in landmarks_list:
            
            #rois = extract_region_finger(hand_landmarks, frame, scale=0.4)
            rois= extract_aligned_roi_finger(hand_landmarks, frame, ring_width_base_on_finger=0.65)

            '''for roi in rois:
                (x_min, y_min), (x_max, y_max) = roi["coords"]
                #finger_name = roi["finger"]

                # Detectar si hay anillo en la región
                has_ring = detector.detect_in_finger_colors(frame, ((x_min, y_min), (x_max, y_max)))

                # Visualización
                color = (0, 255, 0) if has_ring else (0, 0, 255)
                label = f"{'Ring Detected' if has_ring else 'No Ring'}"

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) '''
            for new_roi in rois:
                #finger_name = new_roi["finger"] # name
                roi=new_roi["roi"]  #roi warped
                pts=new_roi["pts"] # coordinates of the rectangle

                # Check if there's a ring
                has_ring=detector.detect_in_finger_colors(roi,((0,0),(roi.shape[1],roi.shape[0])))

                # change the color of the rectangle based on the detection
                color =(0,255,0) if has_ring else (0,0,255)
                label = f"{'Ring Detected' if has_ring else 'No Ring'}"

                # Draw the rectangle around the finger
                pts_int= pts.reshape((-1,1,2)).astype(int) # convert to int
                cv2.polylines(frame, [pts_int], isClosed=True, color=color, thickness=2)
                cv2.putText(frame, label, tuple(pts_int[3][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
