"""Face Mesh inference restricted to a padded primary-face crop."""
from __future__ import annotations

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from models.face import Face
from utils.config import MeshConfig

# Outer and inner lip contour vertices from MediaPipe Face Mesh.
MOUTH_LANDMARKS: tuple[int, ...] = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                                    308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
                                    191, 80, 81, 82, 13, 312, 311, 310, 415)


class FaceMeshProcessor:
    def __init__(self, config: MeshConfig) -> None:
        self._padding = config.crop_padding
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=config.model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._mesh = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray, face: Face, timestamp: float) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
        height, width = frame.shape[:2]
        box = face.bbox
        px, py = int(box.width * self._padding), int(box.height * self._padding)
        x1, y1 = max(0, box.x - px), max(0, box.y - py)
        x2, y2 = min(width, box.x + box.width + px), min(height, box.y + box.height + py)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        result = self._mesh.detect_for_video(image, int(timestamp * 1000))
        if not result.face_landmarks:
            return None
        crop_h, crop_w = crop.shape[:2]
        points = [(x1 + int(item.x * crop_w), y1 + int(item.y * crop_h))
                  for item in result.face_landmarks[0]]
        return points, [points[index] for index in MOUTH_LANDMARKS]

    def close(self) -> None:
        self._mesh.close()
