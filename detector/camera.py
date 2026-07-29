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
        self._capture = cv2.VideoCapture(self._config.index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open camera index {self._config.index}")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        self._capture.set(cv2.CAP_PROP_FPS, self._config.requested_fps)
        LOGGER.info("Camera %d opened", self._config.index)

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
        while self._capture is not None and self._capture.isOpened():
            frame = await asyncio.to_thread(self._read_sync)
            if frame is None:
                break
            yield frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            LOGGER.info("Camera closed")
