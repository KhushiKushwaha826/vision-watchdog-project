print("Starting Vision Watchdog... (YOLO load ho raha hai, 5-10 second wait karo)")

import cv2
import numpy as np
from ultralytics import YOLO
from checks import (prepare, sharpness, brightness, frame_diff,
                    cell_sharpness, MoveDetector, ModelMonitor)

# ---------- Settings ----------
CAMERA_INDEX = 0
BASELINE_FRAMES = 60
GRID = 3
BLUR_RATIO = 0.4
CELL_RATIO = 0.4
MIN_BASE = 5.0
DARK_LIMIT = 40
SMOOTH_LIMIT = 15
FROZEN_THRESHOLD = 0.5
FROZEN_COUNT_LIMIT = 15
SHIFT_LIMIT = 15
MIN_REF_POINTS = 50
YOLO_EVERY = 3
YOLO_CONF = 0.4
MODEL_WINDOW = 10
MODEL_RATIO = 0.6
MODEL_PENALTY = 25
NOISE_RATIO = 2.5
S = 2

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Webcam nahi khula. CAMERA_INDEX = 1 try karo.")
    raise SystemExit

mover = MoveDetector()
monitor = ModelMonitor(YOLO("yolov8n.pt"), conf=YOLO_CONF, window=MODEL_WINDOW, ratio=MODEL_RATIO)


def reset_state():
    return {"frame_no": 0, "sharp": [], "bright": [], "cells": [],
            "base": None, "ref_points": 0}


state = reset_state()
prev_gray = None
frozen_counter = 0
frame_idx = 0
print("Pehle 60 frames: khud ko ya known object saaf dikhao, camera stable rakho. 'r' = baseline reset, 'q' = exit.")
fail_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        fail_count += 1
        if fail_count > 30:
            print("Camera se frame nahi mil raha. Band kar raha hoon.")
            break
        continue
    fail_count = 0
    frame_idx += 1 

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

    if frame_idx % YOLO_EVERY == 0:
        monitor.update(frame)
    monitor.draw(frame)

    big = cv2.resize(frame, (320 * S, 240 * S))
    cam_problems = []
    model_problems = []
    score = 100
    root = ""

    if state["base"] is None:
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
            monitor.finish_baseline()
            frozen_counter = 0
            print(f"Baseline ready: sharp={state['base']['sharp']:.1f}, "
                  f"bright={state['base']['bright']:.1f}, ref points={state['ref_points']}, "
                  f"yolo conf={monitor.base_conf:.2f}, yolo count={monitor.base_count:.2f}")
            if monitor.base_count < 1:
                print("WARNING: baseline mein YOLO ko kuch detect nahi hua. Khud ko dikhake 'r' dabao.")
    else:
        base = state["base"]
        dark = bright < 0.5 * base["bright"] or bright < DARK_LIMIT
        blurry = sharp < SMOOTH_LIMIT or sharp < BLUR_RATIO * base["sharp"]
        noisy = sharp > NOISE_RATIO * base["sharp"]
        if frozen_counter >= FROZEN_COUNT_LIMIT:
            cam_problems.append("FROZEN FEED")
            score -= 50

        if dark:
            cam_problems.append("LOW LIGHT / COVERED")
            score -= 40
        elif bright > 1.5 * base["bright"] and bright > 200:
            cam_problems.append("GLARE")
            score -= 20

        if blurry and not dark:
            cam_problems.append("BLUR / DIRTY LENS")
            score -= 30

        if noisy:
            cam_problems.append("NOISE / SENSOR FAULT")
            score -= 30

        if not dark and not blurry and not noisy:
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
                cam_problems.append("PARTIAL OBSTRUCTION box " + ",".join(map(str, bad)))
                score -= 20

            if state["ref_points"] >= MIN_REF_POINTS:
                shift = mover.shift(gray)
                if shift is None:
                    cam_problems.append("VIEW CHANGED")
                    score -= 30
                elif shift > SHIFT_LIMIT:
                    cam_problems.append(f"CAMERA MOVED ({shift:.0f}px)")
                    score -= 30

        model_problems = monitor.degraded()
        if model_problems:
            score -= MODEL_PENALTY

        if cam_problems and model_problems:
            root = "ROOT CAUSE: camera fault is degrading the AI model"
        elif model_problems and not cam_problems:
            root = "Model degraded, camera OK: possible data drift / new objects"
        elif cam_problems and not model_problems:
            root = "Camera fault detected, model not affected yet"
        else:
            root = "All systems healthy"

        score = max(0, score)
        if score >= 80:
            color = (0, 255, 0)
        elif score >= 50:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        header = f"HEALTH SCORE: {score}"

    cv2.putText(big, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    y = 65
    for p in cam_problems:
        cv2.putText(big, "CAM: " + p, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y += 26
    for p in model_problems:
        cv2.putText(big, "MODEL: " + p, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        y += 26

    if state["base"] is not None:
        rc, rn = monitor.rolling()
        cv2.putText(big, f"YOLO conf {rc:.2f} (base {monitor.base_conf:.2f})  count {rn:.1f} (base {monitor.base_count:.1f})",
                    (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(big, root, (10, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(big, f"Sharp {sharp:.1f}  Bright {bright:.1f}  Diff {diff:.2f}",
                (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Vision Watchdog", big)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        state = reset_state()
        monitor.reset()
        frozen_counter = 0
        print("Baseline reset. Dobara seekh raha hai...")

cap.release()
cv2.destroyAllWindows()
