"""Stage timing helper for pipeline instrumentation."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from time import perf_counter


class TimingTracker:
    """Collects per-stage durations in milliseconds for a single run."""

    def __init__(self) -> None:
        self._active_starts: dict[str, float] = {}
        self._durations_ms: dict[str, float] = {}

    def start(self, stage_name: str) -> None:
        """Starts or restarts a stage timer."""
        self._active_starts[stage_name] = perf_counter()

    def stop(self, stage_name: str) -> float:
        """Stops a stage timer and accumulates elapsed time in milliseconds."""
        started_at = self._active_starts.pop(stage_name, None)
        if started_at is None:
            raise KeyError(f"Stage '{stage_name}' was not started")

        duration_ms = (perf_counter() - started_at) * 1000.0
        self.mark(stage_name, duration_ms)
        return duration_ms

    @contextmanager
    def measure(self, stage_name: str) -> Generator[None, None, None]:
        """Context manager wrapper for start/stop around one block."""
        self.start(stage_name)
        try:
            yield
        finally:
            self.stop(stage_name)

    def mark(self, stage_name: str, duration_ms: float) -> None:
        """Adds externally measured duration in milliseconds.

        Repeated marks for the same stage are summed to keep cumulative stage time.
        """
        if duration_ms < 0:
            raise ValueError(f"Duration must be non-negative, got {duration_ms}")
        self._durations_ms[stage_name] = self._durations_ms.get(stage_name, 0.0) + float(duration_ms)

    def to_dict(self) -> dict[str, float]:
        """Returns a copy of collected stage durations."""
        return dict(self._durations_ms)

    def total_time_ms(self) -> float:
        """Returns total tracked duration across all stages."""
        return sum(self._durations_ms.values())

    def reset(self) -> None:
        """Clears active timers and accumulated durations."""
        self._active_starts.clear()
        self._durations_ms.clear()
