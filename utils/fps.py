from collections import deque
from time import monotonic


class FPSCounter:
    def __init__(self, window_seconds: float = 1.0) -> None:
        self._window_seconds = window_seconds
        self._times: deque[float] = deque()

    def update(self) -> float:
        now = monotonic()
        self._times.append(now)
        while self._times and now - self._times[0] > self._window_seconds:
            self._times.popleft()
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed else 0.0
