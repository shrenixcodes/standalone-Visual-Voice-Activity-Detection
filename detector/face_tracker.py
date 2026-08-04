"""Primary-face association and short-loss tolerance."""
from __future__ import annotations

from math import hypot
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
        self._velocity = (0.0, 0.0)
        self._last_box: BoundingBox | None = None

    @property
    def is_tracking(self) -> bool:
        return self._active is not None and self._missed == 0

    def update(self, detections: list[Face], frame_shape: tuple[int, ...]) -> Face | None:
        if self._active is not None and detections:
            predicted = self._predicted_box()
            match = max(detections, key=lambda item: self._association_score(predicted, item.bbox, frame_shape))
            if _iou(predicted, match.bbox) >= self._config.min_iou:
                self._update_velocity(match.bbox)
                self._active = Face(self._active.id, match.bbox, match.confidence)
                self._missed = 0
                return self._active
        if self._active is not None:
            self._missed += 1
            if self._missed <= self._config.max_missed_frames:
                predicted = self._predicted_box()
                self._active = Face(self._active.id, predicted, self._active.confidence)
                return self._active
            previous = self._active.bbox
            self._active = None
        else:
            previous = None
        if not detections:
            return None
        chosen = self._reacquire(detections, previous, frame_shape)
        self._active = Face(self._next_id, chosen.bbox, chosen.confidence)
        self._last_box = chosen.bbox
        self._velocity = (0.0, 0.0)
        self._next_id += 1
        self._missed = 0
        return self._active

    def _predicted_box(self) -> BoundingBox:
        assert self._active is not None
        dx, dy = self._velocity
        box = self._active.bbox
        return BoundingBox(round(box.x + dx), round(box.y + dy), box.width, box.height)

    def _update_velocity(self, current: BoundingBox) -> None:
        if self._last_box is not None:
            previous_x, previous_y = self._last_box.center
            current_x, current_y = current.center
            self._velocity = current_x - previous_x, current_y - previous_y
        self._last_box = current

    def _association_score(self, predicted: BoundingBox, candidate: BoundingBox, shape: tuple[int, ...]) -> float:
        height, width = shape[:2]
        px, py = predicted.center
        cx, cy = candidate.center
        distance = hypot(px - cx, py - cy) / max(width, height)
        return self._config.prediction_weight * _iou(predicted, candidate) - self._config.center_distance_weight * distance

    def _reacquire(self, detections: list[Face], previous: BoundingBox | None, shape: tuple[int, ...]) -> Face:
        if previous is not None and self._config.reacquire_near_previous:
            nearby = max(detections, key=lambda item: self._association_score(previous, item.bbox, shape))
            if _iou(previous, nearby.bbox) >= self._config.min_iou:
                return nearby
        return max(detections, key=lambda item: item.bbox.area)
    
