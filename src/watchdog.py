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
BLUR_RATIO = 0.4          # sharpness below 40% of baseline = blur
NOISE_RATIO = 2.5         # sharpness above 2.5x baseline = noise
BLOCK_RATIO = 0.25        # box sharpness below 25% of its baseline = blocked
BOX_DARK_RATIO = 0.4      # box brightness below 40% of its baseline = blocked
MIN_BASE = 5.0            # ignore plain boxes (almost no edges)
LOWLIGHT_RATIO = 0.75     # brightness below 75% of baseline = low light
DARK_LIMIT = 40           # absolute brightness below this = covered
SMOOTH_LIMIT = 15
FROZEN_THRESHOLD = 0.5
FROZEN_COUNT_LIMIT = 15
SHIFT_LIMIT = 15
MIN_REF_POINTS = 50
PERSIST = 5               # camera alert must last 5 frames in a row
YOLO_EVERY = 3
YOLO_CONF = 0.4
MODEL_WINDOW = 10
MODEL_RATIO = 0.6
MODEL_PENALTY = 25
S = 2


def cell_brightness(gray, grid=3):
    """Average brightness of each grid box."""
    h, w = gray.shape
    ch, cw = h // grid, w // grid
    out = np.zeros((grid, grid))
    for r in range(grid):
        for c in range(grid):
            out[r, c] = gray[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw].mean()
    return out


cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Webcam nahi khula. CAMERA_INDEX = 1 try karo.")
    raise SystemExit

mover = MoveDetector()
monitor = ModelMonitor(YOLO("yolov8n.pt"), conf=YOLO_CONF, window=MODEL_WINDOW, ratio=MODEL_RATIO)


def reset_state():
    return {"frame_no": 0, "sharp": [], "bright": [], "cells": [], "cbright": [],
            "base": None, "ref_points": 0}


state = reset_state()
prev_gray = None
frozen_counter = 0
frame_idx = 0
fail_count = 0
streak = 0
print("Pehle 60 frames: khud ko saaf dikhao, camera stable rakho. 'r' = baseline reset, 'q' = exit.")

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
    cbright = cell_brightness(gray, GRID)
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
    bad_rects = []
    cam_penalty = 0
    root = ""

    if state["base"] is None:
        state["frame_no"] += 1
        state["sharp"].append(sharp)
        state["bright"].append(bright)
        state["cells"].append(cells)
        state["cbright"].append(cbright)
        header = f"Learning baseline {state['frame_no']}/{BASELINE_FRAMES}"
        color = (255, 255, 0)
        score = 100

        if state["frame_no"] >= BASELINE_FRAMES:
            state["base"] = {
                "sharp": float(np.mean(state["sharp"])),
                "bright": float(np.mean(state["bright"])),
                "cells": np.mean(state["cells"], axis=0),
                "cbright": np.mean(state["cbright"], axis=0),
            }
            state["ref_points"] = mover.set_reference(gray)
            monitor.finish_baseline()
            frozen_counter = 0
            streak = 0
            print(f"Baseline ready: sharp={state['base']['sharp']:.1f}, "
                  f"bright={state['base']['bright']:.1f}, ref points={state['ref_points']}, "
                  f"yolo conf={monitor.base_conf:.2f}, yolo count={monitor.base_count:.2f}")
            if monitor.base_count < 1:
                print("WARNING: baseline mein YOLO ko kuch detect nahi hua. Khud ko dikhake 'r' dabao.")
    else:
        base = state["base"]
        dark = bright < 0.5 * base["bright"] or bright < DARK_LIMIT
        lowlight = bright < LOWLIGHT_RATIO * base["bright"]
        blurry = sharp < SMOOTH_LIMIT or sharp < BLUR_RATIO * base["sharp"]
        noisy = sharp > NOISE_RATIO * base["sharp"]

        if frozen_counter >= FROZEN_COUNT_LIMIT:
            cam_problems.append("FROZEN FEED")
            cam_penalty += 50

        if dark:
            cam_problems.append("LOW LIGHT / COVERED")
            cam_penalty += 40
        elif lowlight:
            cam_problems.append("LOW LIGHT")
            cam_penalty += 25
        elif bright > 1.5 * base["bright"] and bright > 200:
            cam_problems.append("GLARE")
            cam_penalty += 20

        if blurry and not lowlight:
            cam_problems.append("BLUR / DIRTY LENS")
            cam_penalty += 30

        if noisy:
            cam_problems.append("NOISE / SENSOR FAULT")
            cam_penalty += 30

        if not lowlight and not blurry and not noisy:
            ch, cw = gray.shape[0] // GRID, gray.shape[1] // GRID
            bad = []
            for r in range(GRID):
                for c in range(GRID):
                    dark_box = (base["cbright"][r, c] > 30 and
                                cbright[r, c] < BOX_DARK_RATIO * base["cbright"][r, c])
                    blur_box = (base["cells"][r, c] >= MIN_BASE and
                                cells[r, c] < BLOCK_RATIO * base["cells"][r, c])
                    if dark_box or blur_box:
                        bad.append(r * GRID + c + 1)
                        bad_rects.append((c * cw * S, r * ch * S, cw * S, ch * S))
            if bad:
                cam_problems.append("PARTIAL BLOCK box " + ",".join(map(str, bad)))
                cam_penalty += 20

            if state["ref_points"] >= MIN_REF_POINTS:
                shift = mover.shift(gray)
                if shift is None:
                    cam_problems.append("VIEW CHANGED")
                    cam_penalty += 30
                elif shift > SHIFT_LIMIT:
                    cam_problems.append(f"CAMERA MOVED ({shift:.0f}px)")
                    cam_penalty += 30

        # ---- persistence: alert only if it lasts PERSIST frames in a row ----
        if cam_problems:
            streak += 1
        else:
            streak = 0
        if streak < PERSIST:
            cam_problems = []
            bad_rects = []
            cam_penalty = 0

        for (x, y, w, h) in bad_rects:
            cv2.rectangle(big, (x, y), (x + w, y + h), (0, 0, 255), 2)

        model_problems = monitor.degraded()
        score = 100 - cam_penalty
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
        streak = 0
        print("Baseline reset. Dobara seekh raha hai...")

cap.release()
cv2.destroyAllWindows()