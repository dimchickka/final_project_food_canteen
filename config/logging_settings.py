"""Logging and artifact persistence settings.

This module stores only logging/output flags and directories.
"""

from pathlib import Path
from typing import Final

from config.model_paths import DATA_DIR

LOG_RUNS_TO_DISK: Final[bool] = True
LOG_ROOT_DIR: Final[Path] = DATA_DIR / "logs"  # Canonical location: data/logs

SAVE_ANNOTATED_RESULT: Final[bool] = True
SAVE_SOURCE_IMAGE: Final[bool] = True
SAVE_TRACE_JSON: Final[bool] = True
SAVE_TIMINGS_JSON: Final[bool] = True
SAVE_RECEIPT_JSON: Final[bool] = True
