"""Main runtime settings.

Keep this file simple: plain typed constants consumed by app modules.
"""

from typing import Final

# General runtime settings.
DEVICE: Final[str] = "cuda"
TEST_RECOGNITION: Final[bool] = True
ENABLE_LOGGING: Final[bool] = True
ADMIN_PASSWORD: Final[str] = "12345"
CAMERA_INDEX: Final[int] = 0

# YOLO_main detection thresholds.
MAIN_YOLO_CONF: Final[float] = 0.30
MAIN_YOLO_IOU: Final[float] = 0.45

# YOLO_meat detection thresholds.
MEAT_YOLO_CONF: Final[float] = 0.25
MEAT_YOLO_IOU: Final[float] = 0.45

# YOLO_meat_sauce detection thresholds.
MEAT_SAUCE_YOLO_CONF: Final[float] = 0.25
MEAT_SAUCE_YOLO_IOU: Final[float] = 0.45

# Bounding-box area filters (in pixels^2).
MIN_MEAT_BOX_AREA: Final[int] = 900
MIN_SAUCE_BOX_AREA: Final[int] = 400

# Qwen generation limits by task.
QWEN_VALIDATION_MAX_NEW_TOKENS: Final[int] = 32
QWEN_SAUCES_MAX_NEW_TOKENS: Final[int] = 64
QWEN_PHRASE_MAX_NEW_TOKENS: Final[int] = 48

# CLIP matching modes by dish category (without hybrid mode).
CLIP_MATCH_MODES: Final[dict[str, str]] = {
    "beverage": "photo",
    "soup": "text",
    "portioned": "text",
    "garnish": "photo",
    "meat": "photo",
    "sauce": "text",
}
