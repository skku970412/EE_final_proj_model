"""Utility helpers for cropping and rectifying license plate regions."""

from __future__ import annotations

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return 4 points ordered as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_perspective(img: np.ndarray, pts: np.ndarray, out_h: int = 48, out_w: int = 192) -> np.ndarray:
    """Apply perspective transform using four corner points to produce a normalized crop."""
    rect = order_points(np.array(pts, dtype=np.float32))
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, matrix, (out_w, out_h))


def crop_rect(
    img: np.ndarray,
    xyxy,
    pad: int = 4,
    out_h: int = 48,
    out_w: int = 192,
) -> np.ndarray | None:
    """Crop a padded rectangle from an image and resize to a fixed resolution."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
