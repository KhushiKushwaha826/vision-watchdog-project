import cv2
import numpy as np

# ---------- Settings ----------
CAMERA_INDEX = 0          # webcam na khule to 1 karo
FRAME_W, FRAME_H = 320, 240
BASELINE_FRAMES = 60      # shuru ke itne frames se "healthy" values seekhenge
FROZEN_THRESHOLD = 0.5    # isse kam diff = frame same
FROZEN_COUNT_LIMIT = 15   # itne frames lagatar same = frozen
BLUR_RATIO = 0.4          # sharpness baseline ke is fraction se kam ho to alert

# ---------- Webcam kholo ----------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Webcam nahi khula. CAMERA_INDEX = 1 try karo.")
    raise SystemExit

prev_gray = None
frozen_counter = 0
baseline_sharp = []
baseline_bright = []
base_sharp = None
base_bright = None
frame_no = 0

print("Pehle 60 frames mein camera ko normal dekhne do (healthy baseline). 'q' dabake exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1
    small = cv2.resize(frame, (FRAME_W, FRAME_H))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 1) Sharpness (Laplacian variance)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2) Brightness
    brightness = gray.mean()

    # 3) Frozen check
    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray).mean()
        if diff < FROZEN_THRESHOLD:
            frozen_counter += 1
        else:
            frozen_counter = 0
    else:
        diff = 0.0
    prev_gray = gray
    is_frozen = frozen_counter >= FROZEN_COUNT_LIMIT
    
    # 4) Baseline seekhna
    if frame_no <= BASELINE_FRAMES:
        baseline_sharp.append(sharpness)
        baseline_bright.append(brightness)
        status = f"Learning baseline... {frame_no}/{BASELINE_FRAMES}"
        color = (255, 255, 0)
        if frame_no == BASELINE_FRAMES:
            base_sharp = float(np.mean(baseline_sharp))
            base_bright = float(np.mean(baseline_bright))
            print(f"Baseline ready -> sharpness={base_sharp:.1f}, brightness={base_bright:.1f}")
    else:
        problems = []
        if sharpness < BLUR_RATIO * base_sharp:
            problems.append("BLUR / DIRTY LENS")
        if brightness < 0.5 * base_bright or brightness < 40:
            problems.append("LOW LIGHT / COVERED")
        if brightness > 1.5 * base_bright and brightness > 200:
            problems.append("GLARE")
        if is_frozen:
            problems.append("FROZEN FEED")

        if problems:
            status = "ALERT: " + ", ".join(problems)
            color = (0, 0, 255)
        else:
            status = "HEALTHY"
            color = (0, 255, 0)

    # 5) Screen par dikhao
    cv2.putText(small, f"Sharp: {sharpness:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(small, f"Bright: {brightness:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(small, f"Diff: {diff:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(small, status, (10, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    big = cv2.resize(small, (640, 480))
    cv2.imshow("Vision Watchdog - Day 1", big)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()