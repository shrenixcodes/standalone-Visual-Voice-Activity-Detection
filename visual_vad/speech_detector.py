"""Multi-signal visual speech activity scoring."""
from __future__ import annotations

from dataclasses import dataclass, field

from models.mouth_features import MouthFeatures
from visual_vad.filters import FilterConfig, TemporalFilter
from visual_vad.state_machine import SpeechState, SpeechStateMachine, StateMachineConfig
from visual_vad.events import SpeechEvent


@dataclass(slots=True)
class SpeechDetectorConfig:
    filter: FilterConfig = field(default_factory=FilterConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    motion_threshold: float = 0.018
    velocity_threshold: float = 0.75
    acceleration_threshold: float = 8.0
    motion_weight: float = 0.40
    velocity_weight: float = 0.35
    acceleration_weight: float = 0.25


class VisualSpeechDetector:
    """Converts temporal mouth geometry into debounced speech events."""
    def __init__(self, config: SpeechDetectorConfig) -> None:
        self._config = config
        self._filter = TemporalFilter(config.filter)
        self._machine = SpeechStateMachine(config.state_machine)
        self._previous: tuple[float, float, float, float] | None = None
        self._last_confidence = 0.0

    @property
    def last_confidence(self) -> float:
        return self._last_confidence

    @property
    def is_talking(self) -> bool:
        return self._machine.state is SpeechState.TALKING

    def update(self, features: MouthFeatures) -> SpeechEvent | None:
        signature = features.mouth_aspect_ratio + features.mouth_width / max(features.mouth_height + features.mouth_width, 1.0)
        if self._previous is None:
            self._previous = features.timestamp, signature, 0.0, 0.0
            self._last_confidence = 0.0
            return self._machine.update(0.0, features.timestamp)
        prior_time, prior_signature, prior_velocity, _ = self._previous
        elapsed = max(features.timestamp - prior_time, 1e-3)
        motion = abs(signature - prior_signature)
        velocity = motion / elapsed
        acceleration = abs(velocity - prior_velocity) / elapsed
        filtered_motion = self._filter.update(motion)
        confidence = self._confidence(filtered_motion, velocity, acceleration)
        self._last_confidence = confidence
        self._previous = features.timestamp, signature, velocity, filtered_motion
        return self._machine.update(confidence, features.timestamp)

    def _confidence(self, motion: float, velocity: float, acceleration: float) -> float:
        config = self._config
        scores = (min(1.0, motion / config.motion_threshold), min(1.0, velocity / config.velocity_threshold), min(1.0, acceleration / config.acceleration_threshold))
        return sum(score * weight for score, weight in zip(scores, (config.motion_weight, config.velocity_weight, config.acceleration_weight), strict=True))
