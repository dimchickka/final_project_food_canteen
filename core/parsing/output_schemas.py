"""Normalized output schema templates for parser and validator modules."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


QWEN_VALIDATION_OUTPUT_SCHEMA: dict[str, Any] = {
    "status": "valid|invalid|unknown",
    "reason": "",
    "confidence": 0.0,
}

QWEN_SAUCE_DETECTION_OUTPUT_SCHEMA: dict[str, Any] = {
    "boxes": [
        {
            "x1": 0,
            "y1": 0,
            "x2": 0,
            "y2": 0,
            "label": "sauce",
        }
    ],
    "notes": "",
}

DISH_PHRASE_OUTPUT_SCHEMA: dict[str, Any] = {
    "dish_name": "",
    "category": "",
    "phrase": "",
}


def make_qwen_validation_output(
    status: str = "unknown",
    reason: str = "",
    confidence: float = 0.0,
) -> dict[str, Any]:
    """Returns a JSON-ready normalized validation structure."""
    return {
        "status": status,
        "reason": reason,
        "confidence": confidence,
    }


def make_sauce_box(
    x1: int = 0,
    y1: int = 0,
    x2: int = 0,
    y2: int = 0,
    label: str = "sauce",
) -> dict[str, Any]:
    """Factory for a single sauce detection box entry."""
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "label": label,
    }


def make_qwen_sauce_detection_output(
    boxes: list[dict[str, Any]] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Returns normalized sauce-detection payload with default schema-compatible shape."""
    return {
        "boxes": boxes if boxes is not None else [make_sauce_box()],
        "notes": notes,
    }


def make_dish_phrase_output(
    dish_name: str = "",
    category: str = "",
    phrase: str = "",
) -> dict[str, Any]:
    """Returns normalized dish phrase generation payload."""
    return {
        "dish_name": dish_name,
        "category": category,
        "phrase": phrase,
    }


def schema_copies() -> dict[str, dict[str, Any]]:
    """Provides isolated copies of all base schemas to avoid accidental mutation."""
    return {
        "qwen_validation": deepcopy(QWEN_VALIDATION_OUTPUT_SCHEMA),
        "qwen_sauce_detection": deepcopy(QWEN_SAUCE_DETECTION_OUTPUT_SCHEMA),
        "dish_phrase": deepcopy(DISH_PHRASE_OUTPUT_SCHEMA),
    }
