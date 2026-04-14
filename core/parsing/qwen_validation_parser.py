"""Robust parser for Qwen tray-validation responses."""

from __future__ import annotations

import json
import re
from typing import Any

from core.domain.dto import ValidationResult
from core.domain.enums import ValidationStatus

_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


# Remove markdown wrappers so JSON extraction stays stable on LLM deviations.
def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


# Extract first balanced JSON object from arbitrary text.
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

    # Last fallback for malformed balanced scanning.
    regex_match = _JSON_OBJECT_RE.search(stripped)
    return regex_match.group(0) if regex_match else None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_status(value: Any) -> ValidationStatus:
    text = str(value).strip().lower()
    if text == ValidationStatus.VALID.value:
        return ValidationStatus.VALID
    if text == ValidationStatus.INVALID.value:
        return ValidationStatus.INVALID
    return ValidationStatus.UNKNOWN


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


def parse_qwen_validation_response(raw_text: str | None) -> ValidationResult:
    """Normalize raw Qwen validation output into strict domain DTO."""
    safe_raw = (raw_text or "").strip()
    if not safe_raw:
        return ValidationResult(
            status=ValidationStatus.UNKNOWN,
            reason="empty model response",
            raw_text=raw_text,
            is_blocking=False,
            confidence=None,
        )

    payload = _load_json_payload(safe_raw)
    if payload is None:
        return ValidationResult(
            status=ValidationStatus.UNKNOWN,
            reason="failed to parse validation JSON",
            raw_text=raw_text,
            is_blocking=False,
            confidence=None,
        )

    status = _normalize_status(payload.get("status"))
    reason = payload.get("reason")
    reason_text = str(reason).strip() if reason is not None else None
    confidence = _safe_float(payload.get("confidence"))

    return ValidationResult(
        status=status,
        reason=reason_text,
        raw_text=raw_text,
        is_blocking=(status == ValidationStatus.INVALID),
        confidence=confidence,
    )


__all__ = ["parse_qwen_validation_response"]
