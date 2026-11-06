import cv2
import numpy as np
from collections import deque


def prepare(frame, w=320, h=240):
    small = cv2.resize(frame, (w, h))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return small, gray


def sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(gray):
    return float(gray.mean())


def frame_diff(prev_gray, gray):
    if prev_gray is None:
        return 0.0
    return float(cv2.absdiff(prev_gray, gray).mean())


def cell_sharpness(gray, grid=3):
    h, w = gray.shape
    ch, cw = h // grid, w // grid
    result = np.zeros((grid, grid))
    for r in range(grid):
        for c in range(grid):
            cell = gray[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            result[r, c] = cv2.Laplacian(cell, cv2.CV_64F).var()
    return result


class MoveDetector:
    def __init__(self, nfeatures=500, min_matches=10):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = min_matches
        self.ref_kp = None
        self.ref_des = None

    def set_reference(self, gray):
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(gray, None)
        return len(self.ref_kp)

    def shift(self, gray):
        if self.ref_des is None:
            return None
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(kp) == 0:
            return None
        matches = self.matcher.match(self.ref_des, des)
        if len(matches) < self.min_matches:
            return None
        shifts = []
        for m in matches:
            x1, y1 = self.ref_kp[m.queryIdx].pt
            x2, y2 = kp[m.trainIdx].pt
            shifts.append(np.hypot(x2 - x1, y2 - y1))
        return float(np.median(shifts))


class ModelMonitor:
    def __init__(self, model, conf=0.4, window=10, ratio=0.6):
        self.model = model
        self.conf = conf
        self.window = window
        self.ratio = ratio
        self.reset()

    def reset(self):
        self.conf_hist = deque(maxlen=self.window)
        self.count_hist = deque(maxlen=self.window)
        self.base_conf_list = []
        self.base_count_list = []
        self.base_conf = None
        self.base_count = None
        self.last_boxes = []

    def update(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        boxes = results[0].boxes
        count = len(boxes)
        avg_conf = float(boxes.conf.mean()) if count > 0 else 0.0

        self.last_boxes = []
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            name = self.model.names[int(b.cls[0])]
            self.last_boxes.append((x1, y1, x2, y2, f"{name} {float(b.conf[0]):.2f}"))

        self.conf_hist.append(avg_conf)
        self.count_hist.append(count)
        if self.base_conf is None:
            self.base_conf_list.append(avg_conf)
            self.base_count_list.append(count)

    def finish_baseline(self):
        if self.base_count_list:
            self.base_conf = float(np.mean(self.base_conf_list))
            self.base_count = float(np.mean(self.base_count_list))

    def rolling(self):
        if not self.conf_hist:
            return 0.0, 0.0
        return float(np.mean(self.conf_hist)), float(np.mean(self.count_hist))

    def degraded(self):
        if self.base_count is None or self.base_count < 1:
            return []
        rc, rn = self.rolling()
        out = []
        if rn < self.ratio * self.base_count:
            out.append("DETECTIONS DROPPED")
        if rc < self.ratio * self.base_conf:
            out.append("CONFIDENCE DROPPED")
        return out

    def draw(self, frame):
        for x1, y1, x2, y2, label in self.last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
