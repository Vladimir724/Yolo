from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import time

model = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture(0)

motion_buffer = deque(maxlen=6)
prev_points = None

THRESHOLD = 4.5
DELAY_SEC = 0.2
SMOOTHING_FACTOR = 0.3

last_move_time = 0
current_status = "STILL"
smooth_diff = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    results = model.predict(frame, show=False, verbose=False, stream=True)

    for r in results:
        annotated_frame = r.plot(boxes=False, kpt_radius=3)

        if r.keypoints is not None and len(r.keypoints.data) > 0:
            curr_points = r.keypoints.data.cpu().numpy()[0, :, :2]

            if prev_points is not None and prev_points.shape == curr_points.shape:
                raw_diff = np.mean(np.linalg.norm(curr_points - prev_points, axis=1))

                smooth_diff = (SMOOTHING_FACTOR * raw_diff) + ((1 - SMOOTHING_FACTOR) * smooth_diff)
                motion_buffer.append(smooth_diff)

                avg_motion = np.mean(motion_buffer)

                if avg_motion > THRESHOLD:
                    last_move_time = time.time()
                    current_status = "MOVING"
                else:
                    if time.time() - last_move_time > DELAY_SEC:
                        current_status = "STILL"

            prev_points = curr_points

            color = (0, 255, 0) if current_status == "MOVING" else (0, 0, 255)
            cv2.putText(annotated_frame, f"STATE: {current_status}", (30, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
        else:
            prev_points = None

    cv2.imshow("Fast 0.2s Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()


cv2.destroyAllWindows()
