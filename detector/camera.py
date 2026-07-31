"""Robust OpenCV webcam capture with timestamped asynchronous frames."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from time import monotonic
from typing import AsyncIterator
import cv2
import numpy as np
from utils.config import CameraConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VideoFrame:
    image: np.ndarray
    timestamp: float
    sequence: int


class Camera:
    def __init__(self, config: CameraConfig) -> None:
        self._config, self._capture, self._sequence = config, None, 0

    def open(self) -> None:
        backend = self._backend_constant()
        self._capture = cv2.VideoCapture(self._config.index, backend)
        if not self._capture.isOpened() and backend != cv2.CAP_ANY:
            self._capture.release()
            self._capture = cv2.VideoCapture(self._config.index, cv2.CAP_ANY)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open camera index {self._config.index}")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        self._capture.set(cv2.CAP_PROP_FPS, self._config.requested_fps)
        LOGGER.info("Camera %d opened using %s", self._config.index, self._config.backend)

    def _backend_constant(self) -> int:
        backends = {"auto": cv2.CAP_ANY, "directshow": cv2.CAP_DSHOW, "mediafoundation": cv2.CAP_MSMF}
        try:
            return backends[self._config.backend.lower()]
        except KeyError as error:
            raise ValueError(f"Unsupported camera backend: {self._config.backend}") from error

    def _read_sync(self) -> VideoFrame | None:
        if self._capture is None:
            raise RuntimeError("Camera is not open")
        ok, image = self._capture.read()
        if not ok or image is None or image.size == 0:
            LOGGER.warning("Camera returned an empty frame or disconnected")
            return None
        self._sequence += 1
        return VideoFrame(image, monotonic(), self._sequence)

    async def frames(self) -> AsyncIterator[VideoFrame]:
        empty_frames = 0
        while self._capture is not None and self._capture.isOpened():
            frame = await asyncio.to_thread(self._read_sync)
            if frame is None:
                empty_frames += 1
                if empty_frames == 1:
                    LOGGER.warning("Camera returned an empty frame; waiting for stream readiness")
                if empty_frames >= self._config.max_consecutive_empty_frames:
                    LOGGER.error("Camera stopped after %d consecutive empty frames", empty_frames)
                    break
                await asyncio.sleep(0.03)
                continue
            empty_frames = 0
            yield frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            LOGGER.info("Camera closed")
