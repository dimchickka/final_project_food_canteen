"""Domain enums shared across dish-recognition pipeline modules."""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """String-backed enum base to keep values JSON-friendly by default."""

    def __str__(self) -> str:
        return self.value


# High-level dish taxonomy used in menu metadata and final recognition output.
class DishCategory(StrEnum):
    BEVERAGE = "beverage"
    SOUP = "soup"
    PORTIONED = "portioned"
    GARNISH = "garnish"
    MEAT = "meat"
    SAUCE = "sauce"


# Origin of recognition decision/evidence, used for traceability in logs.
class DishDetectionSource(StrEnum):
    YOLO = "yolo"
    CLIP_TEXT = "clip_text"
    CLIP_PHOTO = "clip_photo"
    QWEN = "qwen"
    FIRST_HEAD = "first_head"
    SECOND_HEAD = "second_head"
    MANUAL = "manual"
    AGGREGATED = "aggregated"


# Primary YOLO_main detector classes on tray image.
class TrayObjectClass(StrEnum):
    CUP = "cup"
    PLATE_FLAT = "plate_flat"
    BOWL_DEEP = "bowl_deep"


# Output classes from first head classifier.
class FirstHeadClass(StrEnum):
    EMPTY_PLATE = "empty_plate"
    SOUP = "soup"
    PORTIONED_DISH = "portioned_dish"
    OTHER_DISH = "other_dish"


# Output classes from second head classifier.
class SecondHeadClass(StrEnum):
    EMPTY_AFTER_EXTRACTION = "empty_after_extraction"
    GARNISH_REMAINING = "garnish_remaining"


# Validation status after Qwen gate check.
class ValidationStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"


# Coarse pipeline milestones used in traces and orchestrator events.
class ProcessingStage(StrEnum):
    IMAGE_ACQUIRED = "image_acquired"
    VALIDATION_STARTED = "validation_started"
    VALIDATION_FINISHED = "validation_finished"
    YOLO_MAIN_FINISHED = "yolo_main_finished"
    BEVERAGE_PROCESSED = "beverage_processed"
    FIRST_HEAD_PROCESSED = "first_head_processed"
    OTHER_DISH_PROCESSED = "other_dish_processed"
    SECOND_HEAD_PROCESSED = "second_head_processed"
    RECEIPT_READY = "receipt_ready"
    COMPLETED = "completed"
    ABORTED = "aborted"


# CLIP matching mode helps distinguish phrase-vs-photo embedding retrieval.
class ClipMatchMode(StrEnum):
    TEXT = "text"
    PHOTO = "photo"
