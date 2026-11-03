import cv2
import numpy as np

CAMERA_INDEX = 0
FRAME_W, FRAME_H = 320, 240
SHIFT_LIMIT = 15      # itne pixel se zyada shift = camera moved
MIN_MATCHES = 10      # kam se kam itne points match hone chahiye
DARK_LIMIT = 40       # isse dark = lens blocked
SMOOTH_LIMIT = 15     # isse kam sharpness = lens blocked / blur

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Webcam nahi khula. CAMERA_INDEX = 1 try karo.")
    raise SystemExit

orb = cv2.ORB_create(nfeatures=500)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

ref_kp = None
ref_des = None

print("'r' dabao: is frame ko reference (sahi position) banane ke liye. 'q' = exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.resize(frame, (FRAME_W, FRAME_H))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        ref_kp, ref_des = orb.detectAndCompute(gray, None)
        print(f"Reference set. Points mile: {len(ref_kp)}")

    status = "Press 'r' to set reference"
    color = (255, 255, 0)
    shift = 0.0

    if ref_des is not None:
        kp, des = orb.detectAndCompute(gray, None)
        if des is not None and len(kp) > 0:
            matches = matcher.match(ref_des, des)
            if len(matches) >= MIN_MATCHES:
                shifts = []
                for m in matches:
                    x1, y1 = ref_kp[m.queryIdx].pt
                    x2, y2 = kp[m.trainIdx].pt
                    shifts.append(np.hypot(x2 - x1, y2 - y1))
                shift = float(np.median(shifts))
                if shift > SHIFT_LIMIT:
                    status = "ALERT: CAMERA MOVED"
                    color = (0, 0, 255)
                else:
                    status = "HEALTHY: camera stable"
                    color = (0, 255, 0)
            else:
                status = "ALERT: VIEW CHANGED / BLOCKED"
                color = (0, 0, 255)

    cv2.putText(small, f"Shift: {shift:.1f}px", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(small, status, (10, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow("Vision Watchdog - Day 2", cv2.resize(small, (640, 480)))

cap.release()
cv2.destroyAllWindows()