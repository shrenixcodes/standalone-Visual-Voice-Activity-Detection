from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray

Point: TypeAlias = tuple[float, float]
Image: TypeAlias = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class MouthFeatures:
    """Geometry and normalized BGR mouth crop for one frame."""
    timestamp: float
    mouth_width: float
    mouth_height: float
    mouth_area: float
    mouth_aspect_ratio: float
    centroid: Point
    roi: Image
