import cv2
import numpy as np


def prepare(frame, w=320, h=240):
    """Frame ko chhota karo aur grayscale banao."""
    small = cv2.resize(frame, (w, h))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return small, gray


def sharpness(gray):
    """Laplacian variance: zyada = saaf image, kam = dhundhli."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(gray):
    """Average brightness (0-255)."""
    return float(gray.mean())


def frame_diff(prev_gray, gray):
    """Pichle frame se average farq. ~0 matlab frozen."""
    if prev_gray is None:
        return 0.0
    return float(cv2.absdiff(prev_gray, gray).mean())


def cell_sharpness(gray, grid=3):
    """Frame ko grid x grid boxes mein todke har box ki sharpness deta hai."""
    h, w = gray.shape
    ch, cw = h // grid, w // grid
    result = np.zeros((grid, grid))
    for r in range(grid):
        for c in range(grid):
            cell = gray[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            result[r, c] = cv2.Laplacian(cell, cv2.CV_64F).var()
    return result


class MoveDetector:
    """ORB se pata lagata hai ki camera reference position se hila ya nahi."""

    def __init__(self, nfeatures=500, min_matches=10):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = min_matches
        self.ref_kp = None
        self.ref_des = None

    def set_reference(self, gray):
        """Reference frame save karo. Kitne points mile wo return karta hai."""
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(gray, None)
        return len(self.ref_kp)

    def shift(self, gray):
        """Median shift in pixels. Compute na ho sake to None."""
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