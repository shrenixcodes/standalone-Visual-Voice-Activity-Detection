from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An image-space bounding box represented as x, y, width, height."""
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    def clipped(self, image_width: int, image_height: int) -> "BoundingBox":
        x1, y1 = max(0, self.x), max(0, self.y)
        x2, y2 = min(image_width, self.x + self.width), min(image_height, self.y + self.height)
        return BoundingBox(x1, y1, max(0, x2 - x1), max(0, y2 - y1))


@dataclass(frozen=True, slots=True)
class Face:
    id: int
    bbox: BoundingBox
    confidence: float
    
