import cv2
import numpy as np
from faults import add_noise
from checks import prepare, sharpness

cap = cv2.VideoCapture("data/clean.mp4")
sharps, frames = [], []
while True:
    ret, f = cap.read()
    if not ret:
        break
    frames.append(f)
    sharps.append(sharpness(prepare(f)[1]))
cap.release()

base = float(np.mean(sharps[:60]))
print(f"Baseline sharpness: {base:.1f}, alert line (2.5x): {2.5 * base:.1f}")
print(f"Clean video max ratio: {max(sharps) / base:.2f}  (must be below 2.5)")
for sev in (1, 2, 3):
    r = sharpness(prepare(add_noise(frames[150], sev))[1]) / base
    print(f"Noise sev {sev}: ratio {r:.1f} -> {'CAUGHT' if r > 2.5 else 'MISSED'}")