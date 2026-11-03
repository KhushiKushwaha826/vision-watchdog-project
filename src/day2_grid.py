import cv2
import numpy as np

CAMERA_INDEX = 0
FRAME_W, FRAME_H = 320, 240
BASELINE_FRAMES = 60
GRID = 3                 # 3x3 grid
CELL_RATIO = 0.4         # box ki sharpness baseline ke is fraction se kam = problem
MIN_BASE = 5.0           # jin boxes ka baseline isse kam hai (khali deewar), unhe ignore karo

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Webcam nahi khula. CAMERA_INDEX = 1 try karo.")
    raise SystemExit

def cell_sharpness(gray):
    """Frame ko GRID x GRID boxes mein todke har box ki sharpness return karta hai."""
    h, w = gray.shape
    ch, cw = h // GRID, w // GRID
    result = np.zeros((GRID, GRID))
    for r in range(GRID):
        for c in range(GRID):
            cell = gray[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            result[r, c] = cv2.Laplacian(cell, cv2.CV_64F).var()
    return result

baseline_list = []
base = None
frame_no = 0
print("Pehle 60 frames: camera ko saaf aur normal rakho. 'q' = exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1

    small = cv2.resize(frame, (FRAME_W, FRAME_H))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    cells = cell_sharpness(gray)

    h, w = gray.shape
    ch, cw = h // GRID, w // GRID

    if frame_no <= BASELINE_FRAMES:
        baseline_list.append(cells)
        status = f"Learning baseline... {frame_no}/{BASELINE_FRAMES}"
        color = (255, 255, 0)
        if frame_no == BASELINE_FRAMES:
            base = np.mean(baseline_list, axis=0)
            print("Baseline ready. Per-box sharpness:")
            print(np.round(base, 1))
    else:
        bad = []
        for r in range(GRID):
            for c in range(GRID):
                if base[r, c] < MIN_BASE:
                    continue
                if cells[r, c] < CELL_RATIO * base[r, c]:
                    bad.append(r * GRID + c + 1)
                    x1, y1 = c * cw, r * ch
                    cv2.rectangle(small, (x1, y1), (x1 + cw, y1 + ch), (0, 0, 255), 2)

        if bad:
            status = "ALERT: PARTIAL OBSTRUCTION box " + ",".join(map(str, bad))
            color = (0, 0, 255)
        else:
            status = "HEALTHY"
            color = (0, 255, 0)

    cv2.putText(small, status, (5, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.imshow("Vision Watchdog - Day 2 Grid", cv2.resize(small, (640, 480)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()