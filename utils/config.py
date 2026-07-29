from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    requested_fps: int = 30

@dataclass(slots=True)
class DetectionConfig:
    min_confidence: float = 0.55
    model_path: str = "assets/blaze_face_short_range.tflite"

@dataclass(slots=True)
class TrackerConfig:
    min_iou: float = 0.15
    max_missed_frames: int = 8

@dataclass(slots=True)
class MeshConfig:
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    crop_padding: float = 0.15
    model_path: str = "assets/face_landmarker.task"

@dataclass(slots=True)
class MouthRoiConfig:
    padding_x: int = 12
    padding_y: int = 10
    output_size: tuple[int, int] = (96, 96)

@dataclass(slots=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    mouth_roi: MouthRoiConfig = field(default_factory=MouthRoiConfig)
