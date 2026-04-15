"""Centralized model path configuration for the whole project.

This module contains only path/name constants and does not load any models.
"""

from pathlib import Path

# Repository root: .../final_project_food_canteen
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Shared data directories.
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = DATA_DIR / "models"

# YOLO weights.
YOLO_MAIN_WEIGHTS: Path = MODELS_DIR / "yolo" / "yolo_main.pt"
YOLO_MEAT_WEIGHTS: Path = MODELS_DIR / "yolo" / "yolo_meat.pt"
YOLO_MEAT_SAUCE_WEIGHTS: Path = MODELS_DIR / "yolo" / "yolo_meat_sauce.pt"

# Qwen local model directory / checkpoint path.
QWEN_MODEL_PATH: Path = Path("D:/.ml_cache/Qwen/Qwen3-VL-4B-Instruct")

# CLIP may be local path or model id from provider (e.g., Hugging Face).
CLIP_MODEL_NAME_OR_PATH: str = str(MODELS_DIR / "clip")
