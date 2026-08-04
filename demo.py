"""Run the standalone VvAD mouth-feature extraction demo."""
from __future__ import annotations

import argparse
import asyncio
import logging
import cv2
from detector.camera import Camera
from detector.face_detector import FaceDetector
from detector.face_mesh import FaceMeshProcessor
from detector.face_pose import estimate_yaw_degrees, has_usable_lips
from detector.face_tracker import FaceTracker
from detector.feature_extractor import MouthFeatureExtractor
from detector.mouth_roi import MouthROIExtractor
from utils.config import AppConfig
from utils.drawing import draw_face, draw_mouth, draw_status
from utils.fps import FPSCounter
from utils.logger import configure_logging
from visual_vad.speech_detector import SpeechDetectorConfig, VisualSpeechDetector

LOGGER = logging.getLogger(__name__)


async def run(config: AppConfig, speech_config: SpeechDetectorConfig | None = None) -> None:
    camera = Camera(config.camera)
    detector = FaceDetector(config.detection)
    tracker = FaceTracker(config.tracker)
    mesh = FaceMeshProcessor(config.mesh)
    roi_extractor = MouthROIExtractor(config.mouth_roi)
    feature_extractor = MouthFeatureExtractor()
    speech_detector = VisualSpeechDetector(speech_config or SpeechDetectorConfig())
    fps_counter = FPSCounter()
    cv2.namedWindow("Visual VAD", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mouth ROI (96x96)", cv2.WINDOW_AUTOSIZE)
    try:
        camera.open()
        async for video_frame in camera.frames():
            frame, display = video_frame.image, video_frame.image.copy()
            face = tracker.update(detector.detect(frame), frame.shape)
            fps, status = fps_counter.update(), "No face"
            features = None
            if face is not None and tracker.is_tracking:
                draw_face(display, face)
                result = mesh.process(frame, face, video_frame.timestamp)
                if result is not None:
                    points, mouth_points = result
                    yaw = estimate_yaw_degrees(points)
                    if yaw is None or abs(yaw) > config.face_quality.max_yaw_degrees:
                        status = "Profile view"
                    elif not has_usable_lips(mouth_points, frame.shape, config.face_quality):
                        status = "Mouth occluded"
                    else:
                        roi_result = roi_extractor.extract(frame, mouth_points)
                        if roi_result is not None:
                            mouth_box, roi = roi_result
                            features = feature_extractor.extract(video_frame.timestamp, mouth_points, roi)
                            if features is not None:
                                draw_mouth(display, mouth_points, mouth_box)
                                cv2.imshow("Mouth ROI (96x96)", features.roi)
                        else:
                            status = "Invalid mouth ROI"
                else:
                    status = "Face mesh unavailable"
            elif face is not None:
                draw_face(display, face)
                status = "Face temporarily lost"
            event = speech_detector.update(features) if features is not None else speech_detector.update_missing(video_frame.timestamp)
            if features is not None:
                activity = "Talking" if speech_detector.is_talking else "Silent"
                status = f"{activity} | score {speech_detector.last_confidence:.2f}"
            elif speech_detector.is_talking:
                status = f"{status} | holding speech state"
            if event is not None:
                LOGGER.info("%s confidence=%.2f", event.event_type, event.confidence)
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
