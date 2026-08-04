"""Lightweight landmark-based face pose quality checks."""
from __future__ import annotations

from utils.config import FaceQualityConfig

_LEFT_CHEEK, _NOSE_TIP, _RIGHT_CHEEK = 234, 1, 454


def estimate_yaw_degrees(points: list[tuple[int, int]]) -> float | None:
    """Estimate horizontal head rotation from cheek and nose landmark positions."""
    if len(points) <= _RIGHT_CHEEK:
        return None
    left_x = points[_LEFT_CHEEK][0]
    nose_x = points[_NOSE_TIP][0]
    right_x = points[_RIGHT_CHEEK][0]
    half_face_width = (right_x - left_x) / 2
    if half_face_width <= 1:
        return None
    return (nose_x - (left_x + right_x) / 2) / half_face_width * 45.0


def has_usable_lips(
    lips: list[tuple[int, int]], image_shape: tuple[int, ...], config: FaceQualityConfig
) -> bool:
    if len(lips) < config.min_lip_landmarks:
        return False
    height, width = image_shape[:2]
    return all(0 <= x < width and 0 <= y < height for x, y in lips)
