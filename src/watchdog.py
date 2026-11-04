import cv2
import numpy as np
from checks import (prepare, sharpness, brightness, frame_diff,
                    cell_sharpness, MoveDetector)

# ---------- Settings ----------
CAMERA_INDEX = 0
BASELINE_FRAMES = 60
GRID = 3
BLUR_RATIO = 0.4          # sharpness baseline ke 40% se kam = blur
CELL_RATIO = 0.4          # box ki sharpness baseline ke 40% se kam = obstruction
MIN_BASE = 5.0            # khali deewar wale boxes ignore
DARK_LIMIT = 40           # isse dark = covered
SMOOTH_LIMIT = 15         # isse kam sharpness = blocked/blur
FROZEN_THRESHOLD = 0.5
FROZEN_COUNT_LIMIT = 15
SHIFT_LIMIT = 15          # pixels
MIN_REF_POINTS = 50       # reference mein itne ORB points na hon to moved check skip
S = 2                     # display scale (320x240 -> 640x480)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Webcam nahi khula. CAMERA_INDEX = 1 try karo.")
    raise SystemExit

mover = MoveDetector()


def reset_state():
    return {"frame_no": 0, "sharp": [], "bright": [], "cells": [],
            "base": None, "ref_points": 0}


state = reset_state()
prev_gray = None
frozen_counter = 0
print("Pehle 60 frames camera ko saaf aur stable rakho. 'r' = baseline dobara seekho, 'q' = exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small, gray = prepare(frame)
    sharp = sharpness(gray)
    bright = brightness(gray)
    cells = cell_sharpness(gray, GRID)
    diff = frame_diff(prev_gray, gray)
    prev_gray = gray

    if diff < FROZEN_THRESHOLD:
        frozen_counter += 1
    else:
        frozen_counter = 0

    big = cv2.resize(small, (small.shape[1] * S, small.shape[0] * S))
    score = 100
    problems = []

    if state["base"] is None:
        # ---- Baseline seekhne ka phase ----
        state["frame_no"] += 1
        state["sharp"].append(sharp)
        state["bright"].append(bright)
        state["cells"].append(cells)
        header = f"Learning baseline {state['frame_no']}/{BASELINE_FRAMES}"
        color = (255, 255, 0)

        if state["frame_no"] >= BASELINE_FRAMES:
            state["base"] = {
                "sharp": float(np.mean(state["sharp"])),
                "bright": float(np.mean(state["bright"])),
                "cells": np.mean(state["cells"], axis=0),
            }
            state["ref_points"] = mover.set_reference(gray)
            frozen_counter = 0
            print(f"Baseline ready: sharp={state['base']['sharp']:.1f}, "
                  f"bright={state['base']['bright']:.1f}, ref points={state['ref_points']}")
    else:
        # ---- Monitoring phase ----
        base = state["base"]
        dark = bright < 0.5 * base["bright"] or bright < DARK_LIMIT
        blurry = sharp < SMOOTH_LIMIT or sharp < BLUR_RATIO * base["sharp"]

        if frozen_counter >= FROZEN_COUNT_LIMIT:
            problems.append("FROZEN FEED")
            score -= 50

        if dark:
            problems.append("LOW LIGHT / COVERED")
            score -= 40
        elif bright > 1.5 * base["bright"] and bright > 200:
            problems.append("GLARE")
            score -= 20

        if blurry and not dark:
            problems.append("BLUR / DIRTY LENS")
            score -= 30

        if not dark and not blurry:
            # Partial obstruction (grid)
            bad = []
            ch, cw = gray.shape[0] // GRID, gray.shape[1] // GRID
            for r in range(GRID):
                for c in range(GRID):
                    if base["cells"][r, c] < MIN_BASE:
                        continue
                    if cells[r, c] < CELL_RATIO * base["cells"][r, c]:
                        bad.append(r * GRID + c + 1)
                        x1, y1 = c * cw * S, r * ch * S
                        cv2.rectangle(big, (x1, y1), (x1 + cw * S, y1 + ch * S), (0, 0, 255), 2)
            if bad:
                problems.append("PARTIAL OBSTRUCTION box " + ",".join(map(str, bad)))
                score -= 20

            # Camera moved
            if state["ref_points"] >= MIN_REF_POINTS:
                shift = mover.shift(gray)
                if shift is None:
                    problems.append("VIEW CHANGED")
                    score -= 30
                elif shift > SHIFT_LIMIT:
                    problems.append(f"CAMERA MOVED ({shift:.0f}px)")
                    score -= 30

        score = max(0, score)
        if score >= 80:
            color = (0, 255, 0)
        elif score >= 50:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        header = f"HEALTH SCORE: {score}"

    # ---- Screen par dikhao ----
    cv2.putText(big, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(big, f"Sharp {sharp:.1f}  Bright {bright:.1f}  Diff {diff:.2f}",
                (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y = 65
    for p in problems:
        cv2.putText(big, p, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        y += 28

    cv2.imshow("Vision Watchdog", big)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        state = reset_state()
        frozen_counter = 0
        print("Baseline reset. Dobara seekh raha hai...")

cap.release()
cv2.destroyAllWindows()