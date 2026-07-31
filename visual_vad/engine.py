"""Asynchronous webcam-to-speech-event orchestration."""
from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import AsyncIterator

from detector.camera import Camera, VideoFrame
from detector.face_detector import FaceDetector
from detector.face_mesh import FaceMeshProcessor
from detector.face_tracker import FaceTracker
from detector.feature_extractor import MouthFeatureExtractor
from detector.mouth_roi import MouthROIExtractor
from models.mouth_features import MouthFeatures
from visual_vad.config import VisualVADConfig
from visual_vad.events import SpeechEvent
from visual_vad.speech_detector import VisualSpeechDetector

LOGGER = logging.getLogger(__name__)
_STOP = object()


@dataclass(slots=True)
class _FeatureFrame:
    timestamp: float
    features: MouthFeatures | None


class VisualVAD:
    """Public asynchronous visual speech event source."""
    def __init__(self, config: VisualVADConfig | None = None) -> None:
        self._config = config or VisualVADConfig()

    async def detect(self) -> AsyncIterator[SpeechEvent]:
        camera = Camera(self._config.pipeline.camera)
        detector = FaceDetector(self._config.pipeline.detection)
        tracker = FaceTracker(self._config.pipeline.tracker)
        mesh = FaceMeshProcessor(self._config.pipeline.mesh)
        roi = MouthROIExtractor(self._config.pipeline.mouth_roi)
        extractor = MouthFeatureExtractor()
        speech = VisualSpeechDetector(self._config.speech)
        frames: asyncio.Queue[VideoFrame | object] = asyncio.Queue(self._config.queue_size)
        feature_frames: asyncio.Queue[_FeatureFrame | object] = asyncio.Queue(self._config.queue_size)
        events: asyncio.Queue[SpeechEvent | object] = asyncio.Queue()

        async def capture() -> None:
            try:
                camera.open()
                async for frame in camera.frames():
                    await frames.put(frame)
            finally:
                await frames.put(_STOP)

        async def extract_features() -> None:
            while (item := await frames.get()) is not _STOP:
                frame = item
                assert isinstance(frame, VideoFrame)
                features = self._extract(frame, detector, tracker, mesh, roi, extractor)
                await feature_frames.put(_FeatureFrame(frame.timestamp, features))
            await feature_frames.put(_STOP)

        async def publish() -> None:
            while (item := await feature_frames.get()) is not _STOP:
                packet = item
                assert isinstance(packet, _FeatureFrame)
                if packet.features is None:
                    continue
                event = speech.update(packet.features)
                if event is not None:
                    await events.put(event)
            await events.put(_STOP)

        tasks = [asyncio.create_task(capture()), asyncio.create_task(extract_features()), asyncio.create_task(publish())]
        try:
            while (event := await events.get()) is not _STOP:
                assert isinstance(event, SpeechEvent)
                yield event
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with suppress(asyncio.CancelledError):
                    await task
            camera.close()
            detector.close()
            mesh.close()

    @staticmethod
    def _extract(frame: VideoFrame, detector: FaceDetector, tracker: FaceTracker, mesh: FaceMeshProcessor, roi: MouthROIExtractor, extractor: MouthFeatureExtractor) -> MouthFeatures | None:
        face = tracker.update(detector.detect(frame.image), frame.image.shape)
        if face is None or not tracker.is_tracking:
            return None
        mesh_result = mesh.process(frame.image, face, frame.timestamp)
        if mesh_result is None:
            return None
        _, lips = mesh_result
        roi_result = roi.extract(frame.image, lips)
        return None if roi_result is None else extractor.extract(frame.timestamp, lips, roi_result[1])
