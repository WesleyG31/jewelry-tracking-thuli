import streamlit as st
import cv2
import os
import tempfile
from src.ring_detector_yolo import RingDetectorYOLO
from src.hand_tracker import HandTracker
from src.region_finger import extract_aligned_roi_finger
from utils.video_utils import save_video


st.set_page_config(page_title="Jewelry-Tracking", layout="centered")
st.title("Real-Time Ring Detector and Tracking with YOLOv8")


base_path = os.path.dirname(__file__)
os.makedirs(os.path.join(base_path, "output"), exist_ok=True)

modes_availables = ["Real Time","External Webcam", "Analize Video"]

@st.cache_resource
def load_models():
    tracker = HandTracker()
    model_path= os.path.join(base_path,"models","yolov8n_rings.pt")
    detector = RingDetectorYOLO(model_path=model_path, confidence_yolo=0.55)
    return tracker, detector

tracker, detector = load_models()

output_video_frames=[]

def webcam_video(election_video):
    if election_video == "Real Time":
        return election_video # Webcam
    if election_video=="External Webcam":
        return election_video
    elif election_video == "Analize Video":
        uploaded_video = st.file_uploader("Upload the video (.mp4)", type=["mp4"])
        temp_dir = tempfile.mkdtemp()
        if uploaded_video:
            video_path = os.path.join(temp_dir, uploaded_video.name)
        if uploaded_video is not None:
            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())
            return video_path

webcam_video_selected = st.selectbox("Choose if you want to go Real Time or Analize a Video", modes_availables)

run = webcam_video(webcam_video_selected)


frame_placeholder = st.empty()



if run:

    if webcam_video_selected == "Analize Video":
        cap = cv2.VideoCapture(run)
    if run == "Real Time":
        cap = cv2.VideoCapture(1)
    if run == "External Webcam":
        cap = cv2.VideoCapture(0)


    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.success("Video ended")
            break

        if webcam_video_selected == "Real Time":
            frame = cv2.flip(frame, 1)
        
        landmarks_list = tracker.detect_hands(frame)

        for hand_landmarks in landmarks_list:
            rois = extract_aligned_roi_finger(hand_landmarks, frame, ring_width_base_on_finger=0.65)

            for roi_data in rois:
                roi = roi_data["roi"]
                pts = roi_data["pts"]

                detect_ring = detector.detect_ring_yolo(roi)
                color = (0, 255, 0) if detect_ring else (0, 0, 255)
                label = f"{'Ring Detected' if detect_ring else 'No Ring'}"

                pts_int = pts.reshape((-1, 1, 2)).astype(int)
                cv2.polylines(frame, [pts_int], isClosed=True, color=color, thickness=2)
                cv2.putText(frame, label, tuple(pts_int[3][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        output_video_frames.append(frame.copy())

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        
        if not run:
             break


        


    cap.release()
    cv2.destroyAllWindows()

    output_video_path_avi= os.path.join(base_path,"output","video_output.avi")
    output_video_path_mp4= os.path.join(base_path,"output","video_output.mp4")
    save_video(output_video_frames,output_video_path_avi,output_video_path_mp4)


    st.subheader("Download the video")
    with open(output_video_path_mp4, "rb") as f:
        st.download_button("Download video", f.read(), file_name="video_analyzed.mp4")