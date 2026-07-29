"""Frame-local geometry feature extraction; no speech classification."""
from __future__ import annotations

import numpy as np
from models.mouth_features import MouthFeatures


class MouthFeatureExtractor:
    def extract(self, timestamp: float, points: list[tuple[int, int]], roi: np.ndarray) -> MouthFeatures | None:
        if not points or roi.size == 0:
            return None
        coordinates = np.asarray(points, dtype=np.float32)
        minimum, maximum = coordinates.min(axis=0), coordinates.max(axis=0)
        mouth_width, mouth_height = float(maximum[0] - minimum[0]), float(maximum[1] - minimum[1])
        return MouthFeatures(timestamp, mouth_width, mouth_height, mouth_width * mouth_height,
                             mouth_height / mouth_width if mouth_width else 0.0,
                             (float(coordinates[:, 0].mean()), float(coordinates[:, 1].mean())), roi)
