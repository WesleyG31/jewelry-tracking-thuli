import numpy as np
import cv2

# Usually I use my rings on the index, middle and ring fingers, that's why I use these landmarks
FINGER_PAIRS = {
    "index": (5, 6),    # 5 (MCP) y 6 (PIP)
    "middle": (9, 10),  # 9 = Base, 10 = MID
    "ring": (13, 14),   # 13 = Base, 14 = MID
}


# Now I realize that the ring is usually between the base and the middle of the finger, so I will use the center of those points
# to get the center of the ring and then I will use the distance between them to get the size of the box
# and I will use that size to get the box of the ring
def extract_region_finger(hand_landmarks, frame, scale=0.4):

    h, w, _ = frame.shape

    region_finger = []

    for finger_name, (mcp_id, pip_id) in FINGER_PAIRS.items():  # Get the finger name and the landmarks
        mcp = hand_landmarks.landmark[mcp_id]       
        pip = hand_landmarks.landmark[pip_id]

        x1 = int(mcp.x * w)     # Get the coordinates of the landmarks base on the frame size
        y1 = int(mcp.y * h) 
        x2 = int(pip.x * w)
        y2 = int(pip.y * h)

        cx = int((x1 + x2) / 2)         # Get the center of the box
        cy = int((y1 + y2) *0.47)         # Get the center of the box
        dist = int(np.linalg.norm(np.array([x2 - x1, y2 - y1])))    # Get the distance between the two points

        #box_size = int(scale * dist)    # Get the size of the box, I will use 0.4 of the distance between the two points scale=0.4
        
        box_w = int(scale * dist * 1.3)  # width box    #Rectangle
        box_h = int(scale * dist * 1.4)  # height box


        x_min = max(0, cx - box_w // 2)  # Get the coordinates of the box
        y_min = max(0, cy - box_h // 2)
        x_max = min(w, cx + box_w // 2)
        y_max = min(h, cy + box_h // 2)

        region_finger.append({
            "finger": finger_name,  # FINGER
            "coords": ((x_min, y_min), (x_max, y_max))  # BOX
        })

    return region_finger


def extract_rotated_roi_finger(hand_landmarks, frame, scale=0.4, box_width=40):
    h, w, _ = frame.shape
    rois = []

    for finger_name, (mcp_id, pip_id) in FINGER_PAIRS.items():
        mcp = hand_landmarks.landmark[mcp_id]
        pip = hand_landmarks.landmark[pip_id]

        # Puntos en píxeles
        x1, y1 = int(mcp.x * w), int(mcp.y * h)
        x2, y2 = int(pip.x * w), int(pip.y * h)

        # Centro entre MCP y PIP
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Ángulo del dedo (en grados)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Largo del ROI (según distancia MCP–PIP)
        length = int(np.linalg.norm([x2 - x1, y2 - y1]) * scale)

        # Obtener matriz de rotación y rotar el frame
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated_frame = cv2.warpAffine(frame, M, (w, h))

        # Coordenadas del ROI alineado al dedo
        x_start = int(cx - length / 2)
        y_start = int(cy - box_width / 2)
        x_end = int(cx + length / 2)
        y_end = int(cy + box_width / 2)

        roi = rotated_frame[y_start:y_end, x_start:x_end]

        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            continue

        rois.append({
            "finger": finger_name,
            "roi": roi,
            "coords": ((x_start, y_start), (x_end, y_end)),
            "angle": angle
        })

    return rois


def extract_aligned_roi_finger(hand_landmarks, frame, ring_width_base_on_finger=0.5):
    h, w, _ = frame.shape
    rois = []

    for finger_name, (mcp_id, pip_id) in FINGER_PAIRS.items():
        mcp = hand_landmarks.landmark[mcp_id]   # coordinates of the MCP landmark
        pip = hand_landmarks.landmark[pip_id]   # coordinates of the PIP landmark

        x1, y1 = int(mcp.x * w), int(mcp.y * h) # Get the coordinates of the landmarks base on the frame size
        x2, y2 = int(pip.x * w), int(pip.y * h)

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 #Center of Region

        dx = x2 - x1    # Real Distance euclidean between the two points (sqr(a^2 + b^2))
        dy = y2 - y1
        length = np.hypot(dx, dy)   # hypotenuse = euclidean distance

        ux = dx / length    # Normalized vector (unit vector) in the direction of the finger because we need to follow the finger direction
        uy = dy / length

        roi_height = length * 0.9
        roi_width = length * ring_width_base_on_finger  # width of the rectangle based on the length of the finger

        wx = -uy * roi_width / 2    # Vector 90 grades to the finger direction (perpendicular)
        wy = ux * roi_width / 2     #to the left the vector (wx,wy)
        lx = ux * roi_height / 2
        ly = uy * roi_height / 2

        pts_src = np.array([
            [cx - lx - wx, cy - ly - wy],  # top-left #because wx,wy is to the left and up
            [cx - lx + wx, cy - ly + wy],  # top-right
            [cx + lx + wx, cy + ly + wy],  # bottom-right
            [cx + lx - wx, cy + ly - wy],  # bottom-left
        ], dtype=np.float32)

        dst_width = int(roi_width)      # width of the rectangle
        dst_height = int(roi_height)

        pts_dst = np.array([
            [0, 0],
            [dst_width - 1, 0],
            [dst_width - 1, dst_height - 1],
            [0, dst_height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(pts_src, pts_dst)       #to see the image in the rectangle
        warped = cv2.warpPerspective(frame, M, (dst_width, dst_height))

        rois.append({
            "finger": finger_name,
            "roi": warped,
            "pts": pts_src
        })

    return rois