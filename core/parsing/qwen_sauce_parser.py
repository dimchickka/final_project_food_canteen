"""Robust parser for Qwen sauce-detection responses."""

from __future__ import annotations

import json
import re
from typing import Any

from core.domain.dto import SauceDetectionResult
from core.domain.entities import DetectionBox

_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


# Remove markdown fences to support responses wrapped in ```json blocks.
def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


# Extract first valid-looking JSON object from mixed plain text output.
def _extract_first_json_object(text: str) -> str | None:
    stripped = _strip_code_fences(text)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    if start < 0:
        return None

    depth = 0
    for idx in range(start, len(stripped)):
        char = stripped[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : idx + 1]

    regex_match = _JSON_OBJECT_RE.search(stripped)
    return regex_match.group(0) if regex_match else None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _load_json_payload(raw_text: str) -> dict[str, Any] | None:
    candidates = [raw_text, _strip_code_fences(raw_text)]
    extracted = _extract_first_json_object(raw_text)
    if extracted:
        candidates.append(extracted)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


# Normalize one raw box entry into typed DetectionBox; return None for bad items.
def _normalize_box(raw_box: Any) -> DetectionBox | None:
    if not isinstance(raw_box, dict):
        return None

    x1 = _safe_int(raw_box.get("x1"))
    y1 = _safe_int(raw_box.get("y1"))
    x2 = _safe_int(raw_box.get("x2"))
    y2 = _safe_int(raw_box.get("y2"))

    if None in (x1, y1, x2, y2):
        return None
    if x2 <= x1 or y2 <= y1:
        return None

    label_raw = raw_box.get("label")
    label = str(label_raw).strip() if label_raw not in (None, "") else "sauce"

    return DetectionBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label)


def parse_qwen_sauce_response(raw_text: str | None) -> SauceDetectionResult:
    """Normalize raw Qwen sauce output into strict DTO for pipeline stage."""
    safe_raw = (raw_text or "").strip()
    if not safe_raw:
        return SauceDetectionResult(
            boxes=[],
            raw_text=raw_text,
            parse_ok=False,
            notes="empty model response",
        )

    payload = _load_json_payload(safe_raw)
    if payload is None:
        return SauceDetectionResult(
            boxes=[],
            raw_text=raw_text,
            parse_ok=False,
            notes="failed to parse sauce JSON",
        )

    raw_boxes = payload.get("boxes")
    if raw_boxes is None:
        raw_boxes = []
    if not isinstance(raw_boxes, list):
        return SauceDetectionResult(
            boxes=[],
            raw_text=raw_text,
            parse_ok=False,
            notes="'boxes' field is not a list",
        )

    boxes: list[DetectionBox] = []
    skipped = 0
    for raw_box in raw_boxes:
        normalized = _normalize_box(raw_box)
        if normalized is None:
            skipped += 1
            continue
        boxes.append(normalized)

    notes = payload.get("notes")
    notes_text = str(notes).strip() if notes is not None else None

    if skipped and notes_text:
        notes_text = f"{notes_text}; skipped={skipped}"
    elif skipped:
        notes_text = f"skipped invalid boxes: {skipped}"

    return SauceDetectionResult(
        boxes=boxes,
        raw_text=raw_text,
        parse_ok=True,
        notes=notes_text,
    )


__all__ = ["parse_qwen_sauce_response"]
