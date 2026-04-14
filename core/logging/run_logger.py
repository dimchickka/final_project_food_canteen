"""Artifact persistence facade for one recognition run."""

from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from collections.abc import Sequence

from core.domain.entities import AnnotatedDetection, PipelineTraceEntry
from core.domain.receipt import Receipt
from core.image_ops.annotation_renderer import AnnotationRenderer
from core.logging.model_trace_logger import ModelTraceLogger
from core.logging.session_paths import SessionPaths
from core.logging.timing import TimingTracker


class RunLogger:
    """Saves all run artifacts: images, timings, trace, receipt, and text summaries."""

    def __init__(self, session_paths: SessionPaths | None = None) -> None:
        self.session_paths = session_paths or SessionPaths()
        self._renderer = AnnotationRenderer()

    def ensure_session(self) -> Path:
        """Ensures on-disk folder structure for the current run exists."""
        return self.session_paths.ensure()

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        """Normalizes nested objects to JSON-friendly primitives recursively."""
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
            return cls._json_safe(value.item())
        return str(value)

    @staticmethod
    def _validate_image(image: np.ndarray, *, context: str) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{context} expects np.ndarray, got {type(image).__name__}")
        if image.ndim not in (2, 3):
            raise ValueError(f"{context} expects 2D/3D image array, got ndim={image.ndim}")
        if image.size == 0:
            raise ValueError(f"{context} received empty image array")

    def save_json(self, path: Path, data: Any) -> Path:
        """Serializes JSON data with UTF-8 and stable indentation."""
        self.ensure_session()
        payload = self._json_safe(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
                file.write("\n")
        except TypeError as exc:
            raise TypeError(f"Failed to serialize JSON for {path}: {exc}") from exc
        return path

    def save_text(self, path: Path, text: str) -> Path:
        """Writes plain text artifact."""
        self.ensure_session()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def save_source_image(self, image: np.ndarray) -> Path:
        """Persists original BGR frame as source.jpg."""
        self._validate_image(image, context="save_source_image")
        self.ensure_session()
        destination = self.session_paths.source_image_path
        ok = cv2.imwrite(str(destination), image)
        if not ok:
            raise RuntimeError(f"Failed to save source image: {destination}")
        return destination

    def save_annotated_result(self, image: np.ndarray, detections: Sequence[AnnotatedDetection]) -> Path:
        """Renders detections via AnnotationRenderer and stores annotated_result.jpg."""
        self._validate_image(image, context="save_annotated_result")
        self.ensure_session()
        return self._renderer.save_rendered(image=image, detections=detections, output_path=self.session_paths.annotated_result_path)

    def save_receipt(self, receipt: Receipt | dict[str, Any] | list[Any]) -> Path:
        """Stores receipt JSON from domain object or already-structured containers."""
        normalized: Any
        if isinstance(receipt, Receipt):
            normalized = receipt.to_dict()
        elif isinstance(receipt, dict):
            normalized = receipt
        elif isinstance(receipt, list):
            normalized_items: list[Any] = []
            for item in receipt:
                if hasattr(item, "to_dict") and callable(item.to_dict):
                    normalized_items.append(item.to_dict())
                elif isinstance(item, dict):
                    normalized_items.append(item)
                else:
                    normalized_items.append(self._json_safe(item))
            normalized = {"items": normalized_items}
        else:
            raise TypeError(f"Unsupported receipt type: {type(receipt).__name__}")

        return self.save_json(self.session_paths.receipt_json_path, normalized)

    def save_summary(self, text: str) -> Path:
        """Stores human-readable summary as summary.txt."""
        return self.save_text(self.session_paths.summary_txt_path, text)

    def save_timings(self, timings: dict[str, float] | TimingTracker) -> Path:
        """Stores stage timings as timings.json."""
        payload = timings.to_dict() if isinstance(timings, TimingTracker) else timings
        return self.save_json(self.session_paths.timings_json_path, payload)

    def save_pipeline_trace(
        self,
        trace: Sequence[PipelineTraceEntry] | ModelTraceLogger | list[dict[str, Any]],
    ) -> Path:
        """Stores technical trace entries as pipeline_trace.json."""
        if isinstance(trace, ModelTraceLogger):
            payload: Any = trace.to_dict()
        else:
            payload = []
            for entry in trace:
                if isinstance(entry, PipelineTraceEntry):
                    payload.append(entry.to_dict())
                elif isinstance(entry, dict):
                    payload.append(entry)
                elif hasattr(entry, "to_dict") and callable(entry.to_dict):
                    payload.append(entry.to_dict())
                else:
                    raise TypeError(f"Unsupported pipeline trace entry type: {type(entry).__name__}")

        return self.save_json(self.session_paths.pipeline_trace_json_path, payload)

    def save_qwen_validation(self, raw_text: str | None, parsed: dict[str, Any] | None = None) -> Path:
        """Stores raw + parsed validation output in readable plain text format."""
        lines: list[str] = [raw_text or ""]
        if parsed is not None:
            lines.extend(["", "PARSED:", json.dumps(self._json_safe(parsed), ensure_ascii=False, indent=2)])
        return self.save_text(self.session_paths.qwen_validation_txt_path, "\n".join(lines))

    def save_detection_crop(self, image: np.ndarray, subdir_name: str, filename: str) -> Path:
        """Saves arbitrary crop under detections/<subdir_name>/filename for debugging."""
        self._validate_image(image, context="save_detection_crop")
        self.ensure_session()
        destination = self.session_paths.detections_dir / subdir_name / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(destination), image)
        if not ok:
            raise RuntimeError(f"Failed to save detection crop: {destination}")
        return destination
