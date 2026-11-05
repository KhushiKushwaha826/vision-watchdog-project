import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

CAMERA_INDEX = 0
CONF_THRESHOLD = 0.4
BASELINE_FRAMES = 60
WINDOW = 30               # rolling average ke liye frames
DEGRADE_RATIO = 0.6       # baseline ke 60% se kam = model degraded

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Webcam nahi khula. CAMERA_INDEX = 1 try karo.")
    raise SystemExit

conf_hist = deque(maxlen=WINDOW)    # deque = purane items apne aap hat jate hain
count_hist = deque(maxlen=WINDOW)
base_conf_list, base_count_list = [], []
base_conf = base_count = None
frame_no = 0

print("Pehle 60 frames: khud ko ya koi known object (person, bottle) saaf dikhao. 'q' = exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes
    count = len(boxes)
    avg_conf = float(boxes.conf.mean()) if count > 0 else 0.0

    conf_hist.append(avg_conf)
    count_hist.append(count)
    annotated = results[0].plot()

    if base_conf is None:
        base_conf_list.append(avg_conf)
        base_count_list.append(count)
        status = f"Learning baseline {frame_no}/{BASELINE_FRAMES}"
        color = (255, 255, 0)
        if frame_no >= BASELINE_FRAMES:
            base_conf = float(np.mean(base_conf_list))
            base_count = float(np.mean(base_count_list))
            print(f"Baseline: conf={base_conf:.2f}, count={base_count:.2f}")
    else:
        roll_conf = float(np.mean(conf_hist))
        roll_count = float(np.mean(count_hist))
        problems = []
        if base_count >= 1 and roll_count < DEGRADE_RATIO * base_count:
            problems.append("DETECTIONS DROPPED")
        if base_conf > 0 and roll_conf < DEGRADE_RATIO * base_conf:
            problems.append("CONFIDENCE DROPPED")

        if problems:
            status = "MODEL DEGRADED: " + ", ".join(problems)
            color = (0, 0, 255)
        else:
            status = "MODEL HEALTHY"
            color = (0, 255, 0)
        cv2.putText(annotated, f"Rolling conf {roll_conf:.2f} (base {base_conf:.2f})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f"Rolling count {roll_count:.1f} (base {base_count:.1f})", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Vision Watchdog - Layer 3", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()