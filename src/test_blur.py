import cv2
from faults import add_blur
from checks import prepare, sharpness

cap = cv2.VideoCapture("data/clean.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)   # jump to frame 100
ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not read the frame")
    raise SystemExit

_, gray = prepare(frame)
print(f"Clean frame sharpness: {sharpness(gray):.1f}")

for severity in (1, 2, 3):
    blurred = add_blur(frame, severity)
    _, g = prepare(blurred)
    print(f"Blur severity {severity}: sharpness = {sharpness(g):.1f}")