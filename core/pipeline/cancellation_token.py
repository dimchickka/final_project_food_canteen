"""Simple cooperative cancellation primitive for early pipeline stop."""

from __future__ import annotations


class CancellationToken:
    """Stores cancellation state shared between concurrent pipeline branches."""

    def __init__(self) -> None:
        self._cancelled: bool = False
        self._reason: str | None = None

    def cancel(self, reason: str | None = None) -> None:
        """Marks token as cancelled and keeps the first meaningful reason."""
        self._cancelled = True
        if reason and not self._reason:
            self._reason = str(reason)

    def is_cancelled(self) -> bool:
        """Returns True when pipeline should stop as early as possible."""
        return self._cancelled

    def reason(self) -> str | None:
        """Returns stored cancellation reason if available."""
        return self._reason

    def raise_if_cancelled(self) -> None:
        """Raises RuntimeError once token is cancelled to simplify early exits."""
        if self._cancelled:
            raise RuntimeError(self._reason or "Pipeline cancelled")

    def reset(self) -> None:
        """Clears cancellation flag; useful only for tests/reused orchestrators."""
        self._cancelled = False
        self._reason = None
