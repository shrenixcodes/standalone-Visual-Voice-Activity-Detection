"""Multi-face detection using MediaPipe's lightweight detector."""
from __future__ import annotations

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from models.face import BoundingBox, Face
from utils.config import DetectionConfig


class FaceDetector:
    def __init__(self, config: DetectionConfig) -> None:
        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=config.model_path),
            min_detection_confidence=config.min_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, frame: np.ndarray) -> list[Face]:
        height, width = frame.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self._detector.detect(image)
        faces: list[Face] = []
        for index, detection in enumerate(result.detections):
            raw_box = detection.bounding_box
            box = BoundingBox(raw_box.origin_x, raw_box.origin_y, raw_box.width, raw_box.height).clipped(width, height)
            if box.area:
                confidence = float(detection.categories[0].score) if detection.categories else 0.0
                faces.append(Face(index, box, confidence))
        return faces

    def close(self) -> None:
        self._detector.close()
