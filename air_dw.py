import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# ─── Settings ───────────────────────────────────────────────────────
MODEL_PATH         = "hand_landmarker.task"     # ← this file MUST exist here
WINDOW_WIDTH       = 1280
WINDOW_HEIGHT      = 720
BRUSH_THICKNESS    = 7
TRAIL_SMOOTH       = 18
DRAW_COLOR         = (0, 180, 255)              # orange-yellow
ERASER_THICKNESS   = 45
ERASER_COLOR       = (0, 0, 0)

# ─── MediaPipe Setup ────────────────────────────────────────────────
BaseOptions         = python.BaseOptions
HandLandmarker      = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode   = vision.RunningMode

options = HandLandmarkerOptions(
    base_options      = BaseOptions(model_asset_path=MODEL_PATH),
    running_mode      = VisionRunningMode.VIDEO,
    num_hands         = 1,
    min_hand_detection_confidence = 0.65,
    min_hand_presence_confidence  = 0.65,
    min_tracking_confidence       = 0.55
)

landmarker = HandLandmarker.create_from_options(options)

# ─── Camera & Canvas ────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

points_history = deque(maxlen=TRAIL_SMOOTH)
drawing_active = False
is_eraser_mode = False
prev_point     = None

print("Air Drawing ready!")
print(" • Index finger extended              → draw")
print(" • Open palm (all fingers extended)   → erase")
print(" • Fist / fingers curled              → stop")
print(" • Press  c                           → clear canvas")
print(" • Press  q                           → quit\n")

frame_timestamp = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    frame_timestamp += 33  # ≈30 fps
    result = landmarker.detect_for_video(mp_image, frame_timestamp)

    tip_point = None
    drawing_active = False
    is_eraser_mode = False

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        index_tip_y    = landmarks[8].y
        middle_tip_y   = landmarks[12].y
        ring_tip_y     = landmarks[16].y
        pinky_tip_y    = landmarks[20].y

        index_knuckle_y  = landmarks[5].y
        middle_knuckle_y = landmarks[9].y

        tx = int(landmarks[8].x * w)
        ty = int(landmarks[8].y * h)

        # DRAWING condition
        if (index_tip_y < middle_tip_y - 0.05 and
            index_tip_y < ring_tip_y   - 0.04 and
            index_tip_y < pinky_tip_y  - 0.04):
            drawing_active = True
            tip_point = (tx, ty)

        # ERASER condition
        finger_tips_y = [index_tip_y, middle_tip_y, ring_tip_y, pinky_tip_y]
        tip_spread = max(finger_tips_y) - min(finger_tips_y)

        avg_knuckle_y = (index_knuckle_y + middle_knuckle_y) / 2
        avg_tip_y = sum(finger_tips_y) / 4

        if tip_spread < 0.10 and avg_tip_y < avg_knuckle_y - 0.08:
            is_eraser_mode = True
            tip_point = (tx, ty)

        tip_color = (0, 255, 0) if drawing_active else (0, 100, 255)
        if is_eraser_mode:
            tip_color = (60, 60, 255)
        cv2.circle(frame, (tx, ty), 10, tip_color, -1)

    if tip_point:
        points_history.append(tip_point)

        if len(points_history) >= 2:
            smoothed_x = int(sum(p[0] for p in points_history) / len(points_history))
            smoothed_y = int(sum(p[1] for p in points_history) / len(points_history))
            smoothed = (smoothed_x, smoothed_y)

            if prev_point:
                if is_eraser_mode:
                    cv2.line(canvas, prev_point, smoothed,
                             ERASER_COLOR, ERASER_THICKNESS, cv2.LINE_AA)
                elif drawing_active:
                    cv2.line(canvas, prev_point, smoothed,
                             DRAW_COLOR, BRUSH_THICKNESS, cv2.LINE_AA)

            prev_point = smoothed

    else:
        points_history.clear()
        prev_point = None

    display = cv2.addWeighted(frame, 0.45, canvas, 0.95, 0)

    status_text = "DRAWING" if drawing_active else "READY"
    status_color = (0, 220, 0) if drawing_active else (180, 180, 180)
    if is_eraser_mode:
        status_text = "ERASER MODE"
        status_color = (60, 60, 255)

    cv2.putText(display, status_text, (35, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, status_color, 4, cv2.LINE_AA)
    cv2.putText(display, status_text, (35, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Air Drawing – Index Finger Brush", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        canvas.fill(0)
        print("Canvas cleared")

cap.release()
cv2.destroyAllWindows()