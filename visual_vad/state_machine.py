"""Debounced speech-state transitions."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from visual_vad.events import SpeechEvent, SpeechEventType


class SpeechState(StrEnum):
    IDLE = "idle"
    TALKING = "talking"


@dataclass(slots=True)
class StateMachineConfig:
    start_frames: int = 3
    end_frames: int = 5


class SpeechStateMachine:
    def __init__(self, config: StateMachineConfig) -> None:
        if config.start_frames < 1 or config.end_frames < 1:
            raise ValueError("Debounce frame counts must be positive")
        self._config, self._state, self._active_frames, self._inactive_frames = config, SpeechState.IDLE, 0, 0

    @property
    def state(self) -> SpeechState:
        return self._state

    def update(self, active: bool, timestamp: float, confidence: float) -> SpeechEvent | None:
        self._active_frames = self._active_frames + 1 if active else 0
        self._inactive_frames = self._inactive_frames + 1 if not active else 0
        if self._state is SpeechState.IDLE and self._active_frames >= self._config.start_frames:
            self._state, self._inactive_frames = SpeechState.TALKING, 0
            return SpeechEvent(SpeechEventType.SPEECH_START, timestamp, confidence)
        if self._state is SpeechState.TALKING and self._inactive_frames >= self._config.end_frames:
            self._state, self._active_frames = SpeechState.IDLE, 0
            return SpeechEvent(SpeechEventType.SPEECH_END, timestamp, confidence)
        return None
