"""UI-related constants only.

This module isolates presentation flags from runtime/model logic.
"""

from typing import Final

WINDOW_TITLE: Final[str] = "Food Canteen Tray Detector"
SHOW_ADMIN_BUTTON_IN_TEST: Final[bool] = True
SECRET_ADMIN_CLICKS: Final[int] = 7
SECRET_CLICK_TIMEOUT_MS: Final[int] = 1500

# Additional practical UI defaults.
DEFAULT_WINDOW_WIDTH: Final[int] = 1280
DEFAULT_WINDOW_HEIGHT: Final[int] = 720
