"""Primary-face association and short-loss tolerance."""
from __future__ import annotations

from models.face import BoundingBox, Face
from utils.config import TrackerConfig


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x + a.width, b.x + b.width), min(a.y + a.height, b.y + b.height)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.area + b.area - intersection
    return intersection / union if union else 0.0


class FaceTracker:
    """Stable primary ID based on IoU; resets after a bounded temporary loss."""
    def __init__(self, config: TrackerConfig) -> None:
        self._config, self._active, self._missed, self._next_id = config, None, 0, 1

    @property
    def is_tracking(self) -> bool:
        return self._active is not None and self._missed == 0

    def update(self, detections: list[Face], frame_shape: tuple[int, ...]) -> Face | None:
        if self._active is not None and detections:
            match = max(detections, key=lambda item: _iou(self._active.bbox, item.bbox))
            if _iou(self._active.bbox, match.bbox) >= self._config.min_iou:
                self._active = Face(self._active.id, match.bbox, match.confidence)
                self._missed = 0
                return self._active
        if self._active is not None:
            self._missed += 1
            if self._missed <= self._config.max_missed_frames:
                return self._active
            self._active = None
        if not detections:
            return None
        # After confirmed loss, prefer the most prominent visible person.
        chosen = max(detections, key=lambda item: item.bbox.area)
        self._active = Face(self._next_id, chosen.bbox, chosen.confidence)
        self._next_id += 1
        self._missed = 0
        return self._active
    
