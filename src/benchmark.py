import cv2
import numpy as np
from checks import prepare, sharpness, brightness, cell_sharpness
from faults import FAULTS

np.random.seed(0)          # same random grain every run

FAULT_START = 120          # fault begins at this frame
BASELINE_FRAMES = 60
NEEDED_IN_A_ROW = 5       # alert must last 3 frames to count (ignores flukes)
GRID = 3
BLOCK_RATIO = 0.25         # box must fall below 25% of its baseline

# ---------- load video once ----------
cap = cv2.VideoCapture("data/clean.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []
while True:
    ret, f = cap.read()
    if not ret:
        break
    frames.append(f)
cap.release()
print(f"Loaded {len(frames)} frames at {fps:.1f} fps\n")

# ---------- baseline from the first 60 clean frames ----------
s_list, b_list, c_list = [], [], []
for f in frames[:BASELINE_FRAMES]:
    _, g = prepare(f)
    s_list.append(sharpness(g))
    b_list.append(brightness(g))
    c_list.append(cell_sharpness(g, GRID))
base_s = float(np.mean(s_list))
base_b = float(np.mean(b_list))
base_c = np.mean(c_list, axis=0)


def check(frame):
    """Same rules as watchdog.py. Returns a list of problems (empty = healthy)."""
    _, g = prepare(frame)
    s, b, c = sharpness(g), brightness(g), cell_sharpness(g, GRID)
    problems = []

    dark = b < 0.5 * base_b or b < 40
    blurry = s < 15 or s < 0.4 * base_s
    noisy = s > 2.5 * base_s

    if dark:
        problems.append("DARK")
    elif b > 1.5 * base_b and b > 200:
        problems.append("GLARE")
    if blurry and not dark:
        problems.append("BLUR")
    if noisy:
        problems.append("NOISE")

    if not dark and not blurry and not noisy:
        for r in range(GRID):
            for col in range(GRID):
                if base_c[r, col] >= 5.0 and c[r, col] < BLOCK_RATIO * base_c[r, col]:
                    problems.append("PARTIAL BLOCK")
                    return problems
    return problems

def run(fault_fn=None, severity=1):
    """Run the watchdog over the video. Returns the list of problems per frame."""
    results = []
    for i, f in enumerate(frames):
        if fault_fn is not None and i >= FAULT_START:
            f = fault_fn(f, severity)
        results.append(check(f))
    return results


def first_alert(results, start):
    """First frame >= start where an alert lasts NEEDED_IN_A_ROW frames."""
    for i in range(start, len(results) - NEEDED_IN_A_ROW + 1):
        if all(results[i + k] for k in range(NEEDED_IN_A_ROW)):
            return i
    return None


# ---------- 1) false alarms on the clean video ----------
clean = run()
false_frames = sum(1 for r in clean[BASELINE_FRAMES:] if r)
print(f"CLEAN VIDEO: {false_frames} false-alarm frames out of {len(clean) - BASELINE_FRAMES}\n")

# ---------- 2) every fault x every severity ----------
print(f"{'fault':10s} {'sev':3s} {'caught':7s} {'delay(frames)':14s} {'delay(sec)':10s} label")
total, caught_count = 0, 0
for name, fn in FAULTS.items():
    for sev in (1, 2, 3):
        res = run(fn, sev)
        idx = first_alert(res, FAULT_START)
        total += 1
        if idx is None:
            print(f"{name:10s} {sev:<3d} {'NO':7s} {'-':14s} {'-':10s} -")
        else:
            caught_count += 1
            delay = idx - FAULT_START
            print(f"{name:10s} {sev:<3d} {'YES':7s} {delay:<14d} {delay / fps:<10.2f} {','.join(res[idx])}")

print(f"\nRECALL: {caught_count}/{total} = {100 * caught_count / total:.0f}%")
from collections import Counter
causes = Counter()
for r in clean[BASELINE_FRAMES:]:
    for p in r:
        causes[p] += 1
print("\nFalse alarm causes on clean video:", dict(causes))