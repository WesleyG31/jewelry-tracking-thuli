# Hands & Ring tracking and detection (Mediapipe and YOLO)

---

## Overview

Anna is a fashionista with a passion for rings and earrings, and she meticulously coordinates her jewelry with her outfit – matching by color, style, and design. Typically, she wears two rings on her left hand and one on her right. As a social media influencer, Anna creates a video showcasing her jewelry by moving her hands in various directions (to capture the rings and earrings from every angle) and concludes with a full-body view as she rotates the camera. The end goal is to produce a seamless result using either a video synthesis or a 3D modeling approach.

Finger & Jewelry Tracking:
Focus solely on tracking the rings on Anna's fingers.
Experiment with techniques for precise finger and jewelry rendering and localization using computer vision.

---

## Find more information in desing doc

- Assumptions and scope of work.
- Step-by-step implementation details.
- Discussion on failure cases and iterative improvements.
- Experiments, decision-making process, and learning points.

```
├── desing_doc.md                   # Provide a detailed design doc 
```

---

## 📽️ Demo

📽️ [YouTube Part 1](https://youtu.be/q1fTYXFudIs)

📽️ [YouTube Part 2](https://youtu.be/GGSkEnNGisc)


---

## 🚀 Features

- 🧠 **End-to-End AI Pipeline**: From video capture → hand detection → region extraction → YOLO-based detection → visual overlay.
- 🔍 **Finger-Aware Region Extraction**: ROI is aligned to the direction of the finger (MCP–PIP), making it rotation-robust.
- 💡 **YOLOv8n Model**: Custom-trained model to detect rings with real-time performance.
- 🌐 **Streamlit Web App**: Easily test the system in a browser 
- 🧰 **Modular Architecture**: Clean, reusable, and extensible codebase.

---

## 🧠 Technologies Used

| Component                    | Tech Stack                                   |
|------------------------------|----------------------------------------------|
| Hands Detection / Tracking   | MediaPipe                                    |
| Object Detection             | Heuristic Detection / YOLOv8                 |
| Region Extraction            | MCP–PIP aligned / warpPerspective            |
| Deployment                   | Local PC / Streamlit Local / Streamlit Cloud |

---

## 📂 Project Structure

```
├── app.py                          # Streamlit web interface
├── main_ring_detector_yolo.py      # Main Real-time detector
├── main_ring_detector_colors.py    # Heuristic Detection 
├── main_region_detector.py         # To try MCP–PIP aligned / warpPerspective  
├── main_hand_tracker.py            # To Try Google’s MediaPipe Hands
├── src/
│   ├── hand_tracker.py             # Google’s MediaPipe Hands
│   ├── region_finger.py            # Extract the Area 
│   ├── ring_detector_colors.py     # Calculations for heuristic detection
│   ├── ring_detector_yolo.py       # Code for Yolo detection
├── utils/
│   ├── get_dataset.py              # To get your custom dataset
│   └── video_utils.py              # To save the video
├── models/
│   ├── yolov8n_rings.pt            # Custom Yolo model 
├── data/                           # All the data
│   ├── raw/                        # Images from get_dataset.py
    ├── data_clean/                 # Data proccesed
        ├── train/                  # Train data
        ├── test/                   # Test data
        ├── valid/                  # Valid data
        ├── data.yaml               # Data yaml
        ├── train_yolo_v8.py        # Code to train the model
├── output/                         # To save the video proccesed
├── requirements.txt                # All libraries
├── packages.txt                    # Library for streamlit
├── torch-cuda.txt                  # If you have GPU you can install cuda 
├── desing_doc.md                   # Provide a detailed design doc 
└── README.md
```

---

## 🚀 How to Run (Local PC / Streamlit Local / Streamlit Cloud)

### Local PC

1. Clone the repo
```bash
git clone https://github.com/WesleyG31/jewelry-tracking-thuli.git
cd jewelry-tracking-thuli
```

2. (Optional) Create a virtual environment with Anaconda
```bash
conda create -n jewelry-tracking-thuli python=3.10
conda activate jewelry-tracking-thuli
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. (Optional) Install Cuda
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

5. Run the python file locally
```bash
The path can be changed for a video or webcam is this line 
├── main_ring_detector_yolo.py 
    video_path= 1  #1 for webcam, 0 for external camera, or path to video file

Then run this line
python main_ring_detector_yolo.py
```

### Local Streamlit

Repeat the same steps until step 4.

1. Run the streamlit file locally
```bash
streamlit run app.py
```
###  Streamlit cloud

1. No installation required. Just open the link and upload a video.
```bash
https://jewelry-tracking-thuli-gquvljglhikn4l8ow2cz8k.streamlit.app/
PD: Streamlit Cloud only allows video uploads. It does not allow webcam use.
```

---

## 📄 Sample Outputs

- ✅ Annotated video with tracking and detections.

---

## 💼 Why This Project Matters

This project demonstrates:
- End-to-end pipeline (from detection to deployment)
- Real-time processing with computer vision
- Integration of multiple advanced AI components
- Hands-on understanding of AI for Jewelry e-commerce space
- The lightest form of tracking and detection

> ✅ Perfect for companies in mobile-first jewelry marketplace designed for modern consumers

---

## 👨‍💻 Author

**Wesley Gonzales**  
Computer Vision & AI Engineer  
📫 wes.gb31@gmail.com  
🔗 [https://www.linkedin.com/in/wesleygb/](https://www.linkedin.com/in/wesleygb/)  
🤖 [My Github](https://github.com/WesleyG31)
---

## 🪪 License

This project is licensed under the MIT License.