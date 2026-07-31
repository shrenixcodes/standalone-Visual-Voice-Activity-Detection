"""Speech event contracts."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SpeechEventType(StrEnum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass(frozen=True, slots=True)
class SpeechEvent:
    event_type: SpeechEventType
    timestamp: float
    confidence: float
