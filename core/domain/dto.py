"""DTO layer for data exchange between pipeline modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.domain.entities import AnnotatedDetection, DetectionBox, PipelineTraceEntry, RecognitionEvidence
from core.domain.enums import DishCategory, FirstHeadClass, SecondHeadClass, ValidationStatus


# Validation gate response from Qwen normalization/parser stage.
@dataclass(slots=True)
class ValidationResult:
    status: ValidationStatus
    reason: str | None = None
    raw_text: str | None = None
    is_blocking: bool = False
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "raw_text": self.raw_text,
            "is_blocking": self.is_blocking,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationResult":
        return cls(
            status=ValidationStatus(data["status"]),
            reason=data.get("reason"),
            raw_text=data.get("raw_text"),
            is_blocking=bool(data.get("is_blocking", False)),
            confidence=(float(data["confidence"]) if data.get("confidence") is not None else None),
        )


# Structured output for sauce box detection branch (Qwen + parser).
@dataclass(slots=True)
class SauceDetectionResult:
    boxes: list[DetectionBox] = field(default_factory=list)
    raw_text: str | None = None
    parse_ok: bool = False
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "boxes": [box.to_dict() for box in self.boxes],
            "raw_text": self.raw_text,
            "parse_ok": self.parse_ok,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SauceDetectionResult":
        return cls(
            boxes=[DetectionBox.from_dict(box_data) for box_data in data.get("boxes", [])],
            raw_text=data.get("raw_text"),
            parse_ok=bool(data.get("parse_ok", False)),
            notes=data.get("notes"),
        )


# Generic classifier response for first/second head components.
# DTO is shared between two heads, so enum union keeps strictness with limited fallback.
@dataclass(slots=True)
class HeadClassificationResult:
    predicted_class: FirstHeadClass | SecondHeadClass | str
    score: float | None = None
    matched_description: str | None = None
    evidence: RecognitionEvidence | None = None

    def to_dict(self) -> dict[str, Any]:
        predicted_value = (
            self.predicted_class.value
            if isinstance(self.predicted_class, (FirstHeadClass, SecondHeadClass))
            else self.predicted_class
        )
        return {
            "predicted_class": predicted_value,
            "score": self.score,
            "matched_description": self.matched_description,
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeadClassificationResult":
        evidence_data = data.get("evidence")
        predicted_raw = str(data["predicted_class"])

        # First/second head outputs can be parsed as their enum classes;
        # raw string fallback keeps DTO resilient for rare extension classes.
        try:
            predicted_class: FirstHeadClass | SecondHeadClass | str = FirstHeadClass(predicted_raw)
        except ValueError:
            try:
                predicted_class = SecondHeadClass(predicted_raw)
            except ValueError:
                predicted_class = predicted_raw

        return cls(
            predicted_class=predicted_class,
            score=(float(data["score"]) if data.get("score") is not None else None),
            matched_description=data.get("matched_description"),
            evidence=RecognitionEvidence.from_dict(evidence_data) if evidence_data is not None else None,
        )


# Final recognized dish unit before conversion to concise receipt items.
@dataclass(slots=True)
class DishRecognitionResult:
    dish_name: str
    category: DishCategory | None = None
    count: int = 1
    bbox: DetectionBox | None = None
    evidences: list[RecognitionEvidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dish_name": self.dish_name,
            "category": self.category.value if self.category is not None else None,
            "count": self.count,
            "bbox": self.bbox.to_dict() if self.bbox is not None else None,
            "evidences": [evidence.to_dict() for evidence in self.evidences],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DishRecognitionResult":
        bbox_data = data.get("bbox")
        category_raw = data.get("category")
        return cls(
            dish_name=str(data["dish_name"]),
            category=DishCategory(category_raw) if category_raw is not None else None,
            count=int(data.get("count", 1)),
            bbox=DetectionBox.from_dict(bbox_data) if bbox_data is not None else None,
            evidences=[RecognitionEvidence.from_dict(item) for item in data.get("evidences", [])],
        )


# Aggregate run result exposed to upper layers (UI/controller/export).
@dataclass(slots=True)
class RecognitionSessionResult:
    success: bool
    aborted: bool = False
    abort_reason: str | None = None
    recognized_items: list[DishRecognitionResult] = field(default_factory=list)
    trace_entries: list[PipelineTraceEntry] = field(default_factory=list)
    annotated_detections: list[AnnotatedDetection] = field(default_factory=list)
    total_time_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "recognized_items": [item.to_dict() for item in self.recognized_items],
            "trace_entries": [entry.to_dict() for entry in self.trace_entries],
            "annotated_detections": [item.to_dict() for item in self.annotated_detections],
            "total_time_ms": self.total_time_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecognitionSessionResult":
        return cls(
            success=bool(data.get("success", False)),
            aborted=bool(data.get("aborted", False)),
            abort_reason=data.get("abort_reason"),
            recognized_items=[DishRecognitionResult.from_dict(item) for item in data.get("recognized_items", [])],
            trace_entries=[PipelineTraceEntry.from_dict(item) for item in data.get("trace_entries", [])],
            annotated_detections=[AnnotatedDetection.from_dict(item) for item in data.get("annotated_detections", [])],
            total_time_ms=(float(data["total_time_ms"]) if data.get("total_time_ms") is not None else None),
        )
