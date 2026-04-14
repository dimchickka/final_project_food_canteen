"""Validation flow runs Qwen gate asynchronously to support early aborts."""

from __future__ import annotations

from concurrent.futures import Executor, Future
from typing import Any

from core.domain.dto import ValidationResult
from core.domain.enums import ValidationStatus
from core.models.model_registry import ModelRegistry
from core.pipeline.cancellation_token import CancellationToken


class ValidationFlow:
    """Starts/reads validation task and translates invalid status into cancellation."""

    def __init__(self, model_registry: ModelRegistry) -> None:
        self._model_registry = model_registry

    def start_async(self, image: Any, executor: Executor) -> Future[ValidationResult]:
        """Schedules tray validation in provided executor."""
        qwen = self._model_registry.get_qwen()
        return executor.submit(qwen.validate_tray_visibility, image)

    def get_result_non_blocking(self, future: Future[ValidationResult] | None) -> ValidationResult | None:
        """Returns result only when future is completed; otherwise None."""
        if future is None or not future.done():
            return None
        return future.result()

    def ensure_valid_or_cancel(self, result: ValidationResult | None, token: CancellationToken) -> None:
        """Cancels token for blocking invalid validation results."""
        if result is None:
            return
        if result.status == ValidationStatus.INVALID or result.is_blocking:
            reason = result.reason or "Validation failed: tray image is invalid"
            token.cancel(reason)
