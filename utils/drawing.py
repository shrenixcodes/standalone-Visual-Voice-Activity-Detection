from __future__ import annotations

import cv2
import numpy as np
from models.face import BoundingBox, Face


def draw_face(frame: np.ndarray, face: Face) -> None:
    box = face.bbox
    cv2.rectangle(frame, (box.x, box.y), (box.x + box.width, box.y + box.height), (0, 220, 0), 2)
    cv2.putText(frame, f"Face {face.id} ({face.confidence:.2f})", (box.x, max(22, box.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2, cv2.LINE_AA)


def draw_mouth(frame: np.ndarray, points: list[tuple[int, int]], box: BoundingBox) -> None:
    for point in points:
        cv2.circle(frame, point, 1, (0, 180, 255), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (box.x, box.y), (box.x + box.width, box.y + box.height), (255, 180, 0), 2)


def draw_status(frame: np.ndarray, fps: float, status: str) -> None:
    text = f"FPS: {fps:5.1f} | {status}"
    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 1, cv2.LINE_AA)
