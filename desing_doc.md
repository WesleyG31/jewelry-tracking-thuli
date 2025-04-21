# Hands & Ring tracking and detection – Design Document

## Overview

The challenge was to create a system that can detect rings on moving hands in real-time — knowing the problem is open-ended and ambiguous by design. My goal was to explore a single, critical component (ring tracking) deeply and practically: a pipeline that detects rings worn on fingers using real video input and need to have low latency (mobile-first jewelry marketplace)

Rather than aiming for a perfect end product, I prefered to build an MVP focused on robust and testable solution , learning from every iteration, and documenting the rationale behind each decision.  

---

## Assumptions & Scope

### What I included:
- Hand tracking using MediaPipe
- ROI extraction based on MCP–PIP alignment
- YOLOv8n-based ring detection
- Real-time detection via webcam
- Streamlit app for easy testing and sharing

### What I excluded (due to time constraints):
- Earrings, clothing, full-body tracking

### Assumptions:
- Rings are most commonly worn on index, middle, and ring fingers
- Input could be from a standard webcam, a phone video, external cam.
- Performance must be real-time (low latency), and work under normal lighting

---

## Architecture & Pipeline

### System Architecture
```plaintext
Video Frame
   ↓
HandTracker (MediaPipe)
   ↓
Landmarks → extract_aligned_roi_finger()
   ↓
Perspective-aligned ROI per finger
   ↓
RingDetectorYOLO → YOLOv8 detection
   ↓
Overlay (OpenCV or Streamlit UI)
```

### Pipeline

1. Hand Tracking – hand_tracker.py
Uses MediaPipe’s 21 landmarks per hand

Lightweight and very accurate for fingers

2. ROI Extraction – region_finger.py
Computes the midpoint between MCP and PIP joints

Uses vector math to build a rectangle aligned with finger direction

Applies cv2.getPerspectiveTransform() to obtain clean ROIs


3. (First design) Ring Detection by colors – ring_detector_colors.py
Based on colors (Heuristic Detection)

Works on the small area and has very low latency


4. (Main design) Ring Detection by YOLO – ring_detector_yolo.py
Loads a YOLOv8n model trained specifically on ring data

Works on the small aligned ROI (fast and focused)

Uses confidence threshold = 0.55 for balance

5. Interfaces
main_ring_detector_yolo.py: OpenCV-based real-time detection

app.py: Streamlit web app to test easily in browser


### Design Decisions & Alternatives


| Choise                    | Why? / Why not?                                  |
|------------------------------|----------------------------------------------|
| Mediapipe Hands  | Precise finger/hand detection, Ideal for mobile first               |
| MCP–PIP based ROI  | Aligns with where rings are worn, better than fixed box                  |
| Perspective warping  | Keeps ROI aligned despite hand rotation, improves detection                  |
| Heuristic Detection  | It depends a lot on the color of the rings, it cannot be a single model for several rings. |
| Custom ring model  | Pretrained YOLOs don’t detect rings — I trained my own                |
| YOLOv8n Detection  | Accurate, efficient, and easy to train/fine-tune                  |
| YOLOv8n Tracking             | Too much latency for this case                 |


Choice | Why?
 | 
 | 
 | 
 | 
Streamlit | Easy to test and demo without needing local setup
Fallback to OpenCV | Useful for debugging and real-time experimentation









3. Architecture & Pipeline

 4. 
 5. Iterations & Learnings
6. Technical Trade-offs
7. Future Work / Roadmap
 8. Alignment with Thuli Studios Vision
 9. Summary




Alcance: ¿qué parte resolveré?
Seguimiento de dedos y anillos, Es el núcleo técnico del reto. Tienes experiencia con YOLO y tracking.

Supuestos: ¿qué limitaciones tengo?
1 week
Adaptabilidad mobile,Documentamos cómo el pipeline puede funcionar en tiempo real y adaptarse.

Estrategia: ¿profundizo o abarco?

Primero empezaré con el tracking de anillos y dedos

Stack: ¿qué herramientas y modelos usaré?

Por el tiempo, utilizaré un modelo ya entrenado como mediapipe.
Python	Lenguaje principal	Tu dominio + comunidad + rapidez
OpenCV	Procesamiento de imágenes, video, visualización	Ligero, flexible
MediaPipe Hands	Detección de mano y dedos	Preciso, en tiempo real, ideal para mobile-first
YOLOv5 (opcional)	Detección más precisa de anillos	Solo si lo básico con heurística falla
Google Colab / Local	Desarrollo inicial	Rápido de montar y compartir
Draw.io	Diagramas de arquitectura	Para documentación técnica
Google Drive	Envío de entregables	Solicitado por la empresa


Realice el hand tracking con mediapipe, ahora puedo ver los puntos de la mano (articulaciones) (me piden seguimiento de dedos y joyas)
Ahora el siguiente paso es detectar el anillo: me piden centrarme en el seguimiento y localizacion preciso del anillo.

-Primero sera por landmarks de mediapipe
🧠 ¿Qué hará ring_detector.py?
Recibirá los landmarks de una mano y el frame de imagen.


https://github.com/user-attachments/assets/e1dc3017-fae5-4062-9adf-3b18c2bc83aa


Identificará las zonas de los dedos donde normalmente se usan anillos.

Extraerá esas regiones.

Aplicará una detección basada en color o contorno.

Devolverá True si encuentra algo con características de un anillo.

ELegi 4 colores distintos porque son normales el dorado y plateado pero yo tengo azul y negro
