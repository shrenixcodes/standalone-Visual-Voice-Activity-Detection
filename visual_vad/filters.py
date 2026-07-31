"""Small stateful filters for scalar mouth-motion signals."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from statistics import median


class FilterStrategy(StrEnum):
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_MOVING_AVERAGE = "exponential_moving_average"
    MEDIAN = "median"


@dataclass(slots=True)
class FilterConfig:
    strategy: FilterStrategy = FilterStrategy.EXPONENTIAL_MOVING_AVERAGE
    window_size: int = 5
    ema_alpha: float = 0.35


class TemporalFilter:
    def __init__(self, config: FilterConfig) -> None:
        if config.window_size < 1 or not 0 < config.ema_alpha <= 1:
            raise ValueError("Invalid temporal filter configuration")
        self._config = config
        self._values: deque[float] = deque(maxlen=config.window_size)
        self._ema: float | None = None

    def update(self, value: float) -> float:
        self._values.append(value)
        if self._config.strategy is FilterStrategy.MOVING_AVERAGE:
            return sum(self._values) / len(self._values)
        if self._config.strategy is FilterStrategy.MEDIAN:
            return float(median(self._values))
        self._ema = value if self._ema is None else self._config.ema_alpha * value + (1 - self._config.ema_alpha) * self._ema
        return self._ema
