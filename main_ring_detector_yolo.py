import cv2
import os
from src.ring_detector_yolo import RingDetectorYOLO
from src.hand_tracker import HandTracker
from src.region_finger import extract_aligned_roi_finger
from utils.video_utils import save_video

def main():

    base_path = os.path.dirname(__file__)

    video_path= 1  #1 for webcam, 0 for external camera, or path to video file
    cap= cv2.VideoCapture(video_path)
    tracker= HandTracker()

    model_path= os.path.join(base_path,"models","yolov8n_rings.pt")
    detector=RingDetectorYOLO(model_path=model_path, confidence_yolo=0.57)
    
    output_video_frames=[]

    while cap.isOpened():
        success,frame = cap.read()
        if not success:
            break
        
        if video_path == 1:
            frame= cv2.flip(frame,1)

        landmarks_list= tracker.detect_hands(frame)

        for hand_landmarks in landmarks_list:
            rois= extract_aligned_roi_finger(hand_landmarks, frame, ring_width_base_on_finger=0.65)

            for new_roi in rois:
                roi=new_roi["roi"]
                pts=new_roi["pts"]

                detect_ring=detector.detect_ring_yolo(roi)
                color= (0,255,0) if detect_ring else(0,0,255)
                label= f"{'Ring Detected' if detect_ring else 'No Ring'}"

                pts_int= pts.reshape((-1,1,2)).astype(int)
                cv2.polylines(frame,[pts_int],isClosed=True, color=color, thickness=2)
                cv2.putText(frame,label,tuple(pts_int[3][0]),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,thickness=2)

        output_video_frames.append(frame.copy())
        
        cv2.imshow("Ring Detector", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

    output_video_path_avi= os.path.join(base_path,"output","video_output.avi")
    output_video_path_mp4= os.path.join(base_path,"output","video_output.mp4")
    save_video(output_video_frames,output_video_path_avi,output_video_path_mp4)


    print(f"Video saved in: {output_video_path_mp4}")
    print("Processing completed.")

if __name__ == "__main__":
    main()

