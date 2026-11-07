import cv2
from faults import FAULTS
from checks import prepare, sharpness, brightness

cap = cv2.VideoCapture("data/clean.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

_, g = prepare(frame)
base_s, base_b = sharpness(g), brightness(g)
print(f"CLEAN: sharp={base_s:.1f} bright={base_b:.1f}")
print(f"Blur alert line: sharp < {0.4 * base_s:.1f} | Dark alert line: bright < {0.5 * base_b:.1f}\n")

for name, fn in FAULTS.items():
    for sev in (1, 2, 3):
        _, g = prepare(fn(frame, sev))
        print(f"{name:10s} sev {sev}: sharp={sharpness(g):7.1f}  bright={brightness(g):6.1f}")
