"""Run the standalone VvAD mouth-feature extraction demo."""
from __future__ import annotations

import argparse
import asyncio
import logging
import cv2
from detector.camera import Camera
from detector.face_detector import FaceDetector
from detector.face_mesh import FaceMeshProcessor
from detector.face_tracker import FaceTracker
from detector.feature_extractor import MouthFeatureExtractor
from detector.mouth_roi import MouthROIExtractor
from utils.config import AppConfig
from utils.drawing import draw_face, draw_mouth, draw_status
from utils.fps import FPSCounter
from utils.logger import configure_logging

LOGGER = logging.getLogger(__name__)


async def run(config: AppConfig) -> None:
    camera = Camera(config.camera)
    detector = FaceDetector(config.detection)
    tracker = FaceTracker(config.tracker)
    mesh = FaceMeshProcessor(config.mesh)
    roi_extractor = MouthROIExtractor(config.mouth_roi)
    feature_extractor = MouthFeatureExtractor()
    fps_counter = FPSCounter()
    cv2.namedWindow("Visual VAD", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mouth ROI (96x96)", cv2.WINDOW_AUTOSIZE)
    try:
        camera.open()
        async for video_frame in camera.frames():
            frame, display = video_frame.image, video_frame.image.copy()
            face = tracker.update(detector.detect(frame), frame.shape)
            fps, status = fps_counter.update(), "No face"
            if face is not None:
                draw_face(display, face)
                status = "Tracking" if tracker.is_tracking else "Temporarily lost"
                result = mesh.process(frame, face, video_frame.timestamp)
                if result is not None:
                    _, mouth_points = result
                    roi_result = roi_extractor.extract(frame, mouth_points)
                    if roi_result is not None:
                        mouth_box, roi = roi_result
                        features = feature_extractor.extract(video_frame.timestamp, mouth_points, roi)
                        if features is not None:
                            draw_mouth(display, mouth_points, mouth_box)
                            cv2.imshow("Mouth ROI (96x96)", features.roi)
                            status = "Mouth features extracted"
                else:
                    status = "Face mesh unavailable"
            draw_status(display, fps, status)
            cv2.imshow("Visual VAD", display)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    except RuntimeError as error:
        LOGGER.error("Camera pipeline stopped: %s", error)
    finally:
        camera.close()
        detector.close()
        mesh.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone VvAD mouth-feature prototype")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    configure_logging()
    config = AppConfig()
    config.camera.index, config.camera.width, config.camera.height = args.camera, args.width, args.height
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
