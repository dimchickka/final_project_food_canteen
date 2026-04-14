"""Beverage flow: cup crops -> CLIP match against today's beverage index."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from config.settings import CLIP_MATCH_MODES
from core.domain.dto import DishRecognitionResult
from core.domain.entities import DetectionBox, RecognitionEvidence
from core.domain.enums import ClipMatchMode, DishCategory, DishDetectionSource
from core.image_ops.cropper import Cropper
from core.models.model_registry import ModelRegistry


class BeverageFlow:
    """Recognizes beverages from cup detections without cross-item aggregation."""

    def __init__(self, model_registry: ModelRegistry, today_menu_root: str | Path | None = None) -> None:
        self._model_registry = model_registry
        self._today_menu_root = Path(today_menu_root) if today_menu_root is not None else Path("data/menu/today")

    def process(self, image: np.ndarray, cup_boxes: Sequence[DetectionBox]) -> list[DishRecognitionResult]:
        """Matches each cup crop to beverage index according to configured CLIP mode."""
        if not cup_boxes:
            return []

        index_path = self._today_menu_root / DishCategory.BEVERAGE.value / "indexes" / self._resolve_mode("beverage").value
        if not index_path.exists():
            raise FileNotFoundError(f"Today beverage index not found: {index_path}")

        clip = self._model_registry.get_clip()
        mode = self._resolve_mode("beverage")
        results: list[DishRecognitionResult] = []

        for box in cup_boxes:
            try:
                crop = Cropper.crop_by_box(image, box)
                clip_match = (
                    clip.compare_image_to_text_index(crop, index_path)
                    if mode == ClipMatchMode.TEXT
                    else clip.compare_image_to_photo_index(crop, index_path)
                )
                evidence = RecognitionEvidence(
                    source=DishDetectionSource.CLIP_TEXT if mode == ClipMatchMode.TEXT else DishDetectionSource.CLIP_PHOTO,
                    model_name="clip",
                    score=clip_match.score,
                    chosen_label=clip_match.matched_name,
                    notes=f"category=beverage; mode={mode.value}",
                    bbox=box,
                )
                results.append(
                    DishRecognitionResult(
                        dish_name=clip_match.matched_name,
                        category=DishCategory.BEVERAGE,
                        count=1,
                        bbox=box,
                        evidences=[evidence],
                    )
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    DishRecognitionResult(
                        dish_name="unknown_beverage",
                        category=DishCategory.BEVERAGE,
                        count=1,
                        bbox=box,
                        evidences=[
                            RecognitionEvidence(
                                source=DishDetectionSource.MANUAL,
                                model_name="beverage_flow",
                                chosen_label="unknown_beverage",
                                notes=f"beverage matching failed: {exc}",
                                bbox=box,
                            )
                        ],
                    )
                )

        return results

    @staticmethod
    def _resolve_mode(category: str) -> ClipMatchMode:
        raw_mode = str(CLIP_MATCH_MODES.get(category, "photo")).strip().lower()
        return ClipMatchMode.TEXT if raw_mode == ClipMatchMode.TEXT.value else ClipMatchMode.PHOTO
