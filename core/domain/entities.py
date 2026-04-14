"""Core domain entities used by the recognition pipeline and storage layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.domain.enums import ClipMatchMode, DishCategory, DishDetectionSource, ProcessingStage


# Universal bounding box container used by detector/classifier outputs.
@dataclass(slots=True)
class DetectionBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float | None = None
    class_name: str | None = None
    label: str | None = None
    source: str | None = None

    @property
    def width(self) -> int:
        """Returns non-negative width for robust downstream rendering/math."""
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        """Returns non-negative height for robust downstream rendering/math."""
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        """Computed rectangle area based on normalized width/height."""
        return self.width * self.height

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def to_dict(self) -> dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "class_name": self.class_name,
            "label": self.label,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectionBox":
        return cls(
            x1=int(data["x1"]),
            y1=int(data["y1"]),
            x2=int(data["x2"]),
            y2=int(data["y2"]),
            confidence=(float(data["confidence"]) if data.get("confidence") is not None else None),
            class_name=data.get("class_name"),
            label=data.get("label"),
            source=data.get("source"),
        )


# Candidate crop unit that links an image fragment to parent detection context.
@dataclass(slots=True)
class CropCandidate:
    crop_id: str
    bbox: DetectionBox
    image_path: str | None = None
    temp_path: str | None = None
    class_name: str | None = None
    parent_crop_id: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop_id": self.crop_id,
            "bbox": self.bbox.to_dict(),
            "image_path": self.image_path,
            "temp_path": self.temp_path,
            "class_name": self.class_name,
            "parent_crop_id": self.parent_crop_id,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CropCandidate":
        return cls(
            crop_id=str(data["crop_id"]),
            bbox=DetectionBox.from_dict(data["bbox"]),
            image_path=data.get("image_path"),
            temp_path=data.get("temp_path"),
            class_name=data.get("class_name"),
            parent_crop_id=data.get("parent_crop_id"),
            notes=data.get("notes"),
        )


# CLIP retrieval output for both phrase-based and photo-based embedding matching.
# Dedicated enum prevents free-form mode strings and keeps the contract strict.
@dataclass(slots=True)
class ClipMatchResult:
    matched_name: str
    matched_category: str
    score: float
    mode: ClipMatchMode
    matched_description: str | None = None
    embedding_source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched_name": self.matched_name,
            "matched_category": self.matched_category,
            "score": self.score,
            "mode": self.mode.value,
            "matched_description": self.matched_description,
            "embedding_source_path": self.embedding_source_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipMatchResult":
        return cls(
            matched_name=str(data["matched_name"]),
            matched_category=str(data["matched_category"]),
            score=float(data["score"]),
            mode=ClipMatchMode(data["mode"]),
            matched_description=data.get("matched_description"),
            embedding_source_path=data.get("embedding_source_path"),
        )


# Unified explanation object to preserve why a label was selected.
@dataclass(slots=True)
class RecognitionEvidence:
    source: DishDetectionSource
    model_name: str | None = None
    score: float | None = None
    chosen_label: str | None = None
    raw_response: str | None = None
    notes: str | None = None
    bbox: DetectionBox | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "model_name": self.model_name,
            "score": self.score,
            "chosen_label": self.chosen_label,
            "raw_response": self.raw_response,
            "notes": self.notes,
            "bbox": self.bbox.to_dict() if self.bbox is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecognitionEvidence":
        bbox_data = data.get("bbox")
        return cls(
            source=DishDetectionSource(data["source"]),
            model_name=data.get("model_name"),
            score=(float(data["score"]) if data.get("score") is not None else None),
            chosen_label=data.get("chosen_label"),
            raw_response=data.get("raw_response"),
            notes=data.get("notes"),
            bbox=DetectionBox.from_dict(bbox_data) if bbox_data is not None else None,
        )


# Menu catalog object (dataset/backoffice) used during matching and management.
@dataclass(slots=True)
class MenuDish:
    dish_id: str
    name: str
    slug: str
    category: DishCategory
    folder_path: str
    crop_image_path: str | None = None
    crop_embedding_path: str | None = None
    qwen_phrase_path: str | None = None
    qwen_phrase_embedding_path: str | None = None
    is_active: bool = True
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dish_id": self.dish_id,
            "name": self.name,
            "slug": self.slug,
            "category": self.category.value,
            "folder_path": self.folder_path,
            "crop_image_path": self.crop_image_path,
            "crop_embedding_path": self.crop_embedding_path,
            "qwen_phrase_path": self.qwen_phrase_path,
            "qwen_phrase_embedding_path": self.qwen_phrase_embedding_path,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MenuDish":
        return cls(
            dish_id=str(data["dish_id"]),
            name=str(data["name"]),
            slug=str(data["slug"]),
            category=DishCategory(data["category"]),
            folder_path=str(data["folder_path"]),
            crop_image_path=data.get("crop_image_path"),
            crop_embedding_path=data.get("crop_embedding_path"),
            qwen_phrase_path=data.get("qwen_phrase_path"),
            qwen_phrase_embedding_path=data.get("qwen_phrase_embedding_path"),
            is_active=bool(data.get("is_active", True)),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


# Time-ordered lifecycle event entry for observability/debugging.
@dataclass(slots=True)
class PipelineTraceEntry:
    stage: ProcessingStage
    message: str
    timestamp: str | None = None
    duration_ms: float | None = None
    payload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineTraceEntry":
        return cls(
            stage=ProcessingStage(data["stage"]),
            message=str(data["message"]),
            timestamp=data.get("timestamp"),
            duration_ms=(float(data["duration_ms"]) if data.get("duration_ms") is not None else None),
            payload=data.get("payload"),
        )


# Render-ready detection record for drawing final overlays and legends.
@dataclass(slots=True)
class AnnotatedDetection:
    bbox: DetectionBox
    display_name: str
    count_hint: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox.to_dict(),
            "display_name": self.display_name,
            "count_hint": self.count_hint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnnotatedDetection":
        return cls(
            bbox=DetectionBox.from_dict(data["bbox"]),
            display_name=str(data["display_name"]),
            count_hint=(int(data["count_hint"]) if data.get("count_hint") is not None else None),
        )
