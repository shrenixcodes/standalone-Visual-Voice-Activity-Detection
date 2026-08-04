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
    start_confidence: float = 0.45
    end_confidence: float = 0.24
    start_hold_seconds: float = 0.25
    end_hold_seconds: float = 2.00
    minimum_talking_seconds: float = 1.00


class SpeechStateMachine:
    def __init__(self, config: StateMachineConfig) -> None:
        if not 0 <= config.end_confidence < config.start_confidence <= 1:
            raise ValueError("Confidence thresholds must satisfy 0 <= end < start <= 1")
        if min(config.start_hold_seconds, config.end_hold_seconds, config.minimum_talking_seconds) < 0:
            raise ValueError("Speech timing values cannot be negative")
        self._config = config
        self._state = SpeechState.IDLE
        self._state_started_at: float | None = None
        self._active_since: float | None = None
        self._silent_since: float | None = None

    @property
    def state(self) -> SpeechState:
        return self._state

    def update(self, confidence: float, timestamp: float) -> SpeechEvent | None:
        """Apply asymmetric thresholds and time-based debounce to a confidence score."""
        if self._state is SpeechState.IDLE:
            return self._update_idle(confidence, timestamp)
        return self._update_talking(confidence, timestamp)

    def _update_idle(self, confidence: float, timestamp: float) -> SpeechEvent | None:
        if confidence < self._config.start_confidence:
            self._active_since = None
            return None
        self._active_since = self._active_since or timestamp
        if timestamp - self._active_since < self._config.start_hold_seconds:
            return None
        self._state = SpeechState.TALKING
        self._state_started_at = timestamp
        self._active_since = None
        return SpeechEvent(SpeechEventType.SPEECH_START, timestamp, confidence)

    def _update_talking(self, confidence: float, timestamp: float) -> SpeechEvent | None:
        if confidence > self._config.end_confidence:
            self._silent_since = None
            return None
        if self._state_started_at is not None and timestamp - self._state_started_at < self._config.minimum_talking_seconds:
            return None
        self._silent_since = self._silent_since or timestamp
        if timestamp - self._silent_since < self._config.end_hold_seconds:
            return None
        self._state = SpeechState.IDLE
        self._state_started_at = None
        self._silent_since = None
        return SpeechEvent(SpeechEventType.SPEECH_END, timestamp, confidence)
