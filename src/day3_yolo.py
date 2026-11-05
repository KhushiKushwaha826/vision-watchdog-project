import cv2
from ultralytics import YOLO

CAMERA_INDEX = 0
CONF_THRESHOLD = 0.25     # isse kam confidence wale boxes ignore

model = YOLO("yolov8n.pt")   # pehli baar chalane par ~6 MB download hoga

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Webcam nahi khula. CAMERA_INDEX = 1 try karo.")
    raise SystemExit

print("YOLO chal raha hai. 'q' = exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    count = len(boxes)
    if count > 0:
        avg_conf = float(boxes.conf.mean())
    else:
        avg_conf = 0.0

    annotated = results[0].plot()

    cv2.putText(annotated, f"Detections: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated, f"Avg confidence: {avg_conf:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Vision Watchdog - YOLO", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()