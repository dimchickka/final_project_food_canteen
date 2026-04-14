"""Technical trace collector for model/pipeline debug events."""

from __future__ import annotations

import json
import traceback
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from core.domain.entities import PipelineTraceEntry
from core.domain.enums import ProcessingStage


class ModelTraceLogger:
    """Accumulates trace entries and persists them to JSON when needed."""

    def __init__(self) -> None:
        self._entries: list[PipelineTraceEntry] = []

    @staticmethod
    def _normalize_stage(stage: str) -> ProcessingStage:
        """Uses enum when possible; falls back to COMPLETED-compatible safe value only if invalid.

        Because PipelineTraceEntry requires ProcessingStage enum, unknown values are mapped to COMPLETED
        and original stage is preserved in payload by caller helpers.
        """
        try:
            return ProcessingStage(stage)
        except ValueError as exc:
            raise ValueError(f"Unknown processing stage '{stage}'") from exc

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        """Recursively converts rich python objects into JSON-serializable values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return cls._json_safe(value.to_dict())
        if isinstance(value, dict):
            return {str(k): cls._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe(item) for item in value]
        if hasattr(value, "item") and callable(value.item):
            # Handles numpy scalar-like objects.
            return cls._json_safe(value.item())
        return str(value)

    def add_entry(
        self,
        stage: str,
        message: str,
        payload: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> PipelineTraceEntry:
        """Adds one generic trace event."""
        entry = PipelineTraceEntry(
            stage=self._normalize_stage(stage),
            message=message,
            timestamp=datetime.now().isoformat(),
            duration_ms=float(duration_ms) if duration_ms is not None else None,
            payload=self._json_safe(payload) if payload is not None else None,
        )
        self._entries.append(entry)
        return entry

    def add_model_result(
        self,
        stage: str,
        model_name: str,
        chosen_label: str | None = None,
        score: float | None = None,
        raw_response: Any | None = None,
        notes: str | None = None,
        bbox: Any | None = None,
        payload: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> PipelineTraceEntry:
        """Convenience method for standard model-output trace schema."""
        model_payload: dict[str, Any] = {
            "model_name": model_name,
            "chosen_label": chosen_label,
            "score": score,
            "raw_response": raw_response,
            "notes": notes,
            "bbox": bbox,
        }
        if payload:
            model_payload.update(payload)

        return self.add_entry(
            stage=stage,
            message=f"Model result from {model_name}",
            payload=model_payload,
            duration_ms=duration_ms,
        )

    def add_exception(
        self,
        stage: str,
        exc: BaseException,
        payload: dict[str, Any] | None = None,
    ) -> PipelineTraceEntry:
        """Stores structured exception details with traceback for debugging."""
        exception_payload = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }
        if payload:
            exception_payload["context"] = payload

        return self.add_entry(stage=stage, message="Exception raised", payload=exception_payload)

    def entries(self) -> list[PipelineTraceEntry]:
        """Returns copy of current entries to avoid accidental external mutation."""
        return list(self._entries)

    def to_dict(self) -> list[dict[str, Any]]:
        """Converts entries into JSON-safe list of dictionaries."""
        return [self._json_safe(entry.to_dict()) for entry in self._entries]

    def clear(self) -> None:
        """Removes all collected trace entries."""
        self._entries.clear()

    def save(self, output_path: str | Path) -> Path:
        """Writes current trace entries into JSON file."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)
            file.write("\n")
        return destination
