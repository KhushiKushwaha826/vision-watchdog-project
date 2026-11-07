import cv2
import numpy as np


def add_blur(frame, severity):
    # severity: 1 = mild, 2 = medium, 3 = strong
    k = 2 * (severity * 5) + 1
    return cv2.GaussianBlur(frame, (k, k), 0)


def add_darkness(frame, severity):
    # multiply brightness: 0.6, 0.4, 0.2
    factor = {1: 0.6, 2: 0.4, 3: 0.2}[severity]
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def add_noise(frame, severity):
    # random grain: stronger severity = more grain
    sigma = {1: 10, 2: 25, 3: 50}[severity]
    noise = np.random.normal(0, sigma, frame.shape)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_occlusion(frame, severity):
    # black patch in the top-left corner: bigger severity = bigger patch
    h, w = frame.shape[:2]
    frac = {1: 0.15, 2: 0.3, 3: 0.5}[severity]
    out = frame.copy()
    out[0:int(h * frac), 0:int(w * frac)] = 0
    return out


FAULTS = {
    "blur": add_blur,
    "darkness": add_darkness,
    "noise": add_noise,
    "occlusion": add_occlusion,
}
