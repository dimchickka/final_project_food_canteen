"""Project-wide shared constants.

Use this module for stable domain-level constant lists/tuples only.
"""

from typing import Final

SUPPORTED_IMAGE_EXTENSIONS: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

MAIN_YOLO_CLASSES: Final[tuple[str, ...]] = ("cup", "plate_flat", "bowl_deep")
DISH_CATEGORIES: Final[tuple[str, ...]] = ("beverage", "soup", "portioned", "garnish", "meat", "sauce")

FIRST_HEAD_CLASSES: Final[tuple[str, ...]] = (
    "empty_plate",
    "soup",
    "portioned_dish",
    "other_dish",
)
SECOND_HEAD_CLASSES: Final[tuple[str, ...]] = (
    "empty_after_extraction",
    "garnish_remaining",
)

# Common naming helpers used by multiple modules.
UNKNOWN_LABEL: Final[str] = "unknown"
EMPTY_LABEL: Final[str] = "empty"
