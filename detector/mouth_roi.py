"""Boundary-safe padded mouth crop creation."""
from __future__ import annotations

import cv2
import numpy as np
from models.face import BoundingBox
from utils.config import MouthRoiConfig


class MouthROIExtractor:
    def __init__(self, config: MouthRoiConfig) -> None:
        self._config = config

    def extract(self, frame: np.ndarray, points: list[tuple[int, int]]) -> tuple[BoundingBox, np.ndarray] | None:
        if not points:
            return None
        height, width = frame.shape[:2]
        xs, ys = [point[0] for point in points], [point[1] for point in points]
        x1, x2 = max(0, min(xs) - self._config.padding_x), min(width, max(xs) + self._config.padding_x)
        y1, y2 = max(0, min(ys) - self._config.padding_y), min(height, max(ys) + self._config.padding_y)
        box = BoundingBox(x1, y1, x2 - x1, y2 - y1)
        if box.width < self._config.min_width or box.height < self._config.min_height:
            return None
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return box, cv2.resize(roi, self._config.output_size, interpolation=cv2.INTER_LINEAR)
