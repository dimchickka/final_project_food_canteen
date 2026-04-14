"""First-head flow routes dish boxes into soup/portioned/other branches."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from config.settings import CLIP_MATCH_MODES
from core.domain.dto import DishRecognitionResult, HeadClassificationResult
from core.domain.entities import DetectionBox, RecognitionEvidence
from core.domain.enums import ClipMatchMode, DishCategory, DishDetectionSource, FirstHeadClass
from core.image_ops.cropper import Cropper
from core.models.model_registry import ModelRegistry


class FirstHeadFlow:
    """Classifies plate/bowl crops with first-head index and resolves non-other dishes."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        today_menu_root: str | Path | None = None,
        heads_root: str | Path | None = None,
    ) -> None:
        self._model_registry = model_registry
        self._today_menu_root = Path(today_menu_root) if today_menu_root is not None else Path("data/menu/today")
        self._heads_root = Path(heads_root) if heads_root is not None else Path("data/heads")

    def classify_dish_crops(
        self,
        image: np.ndarray,
        dish_boxes: Sequence[DetectionBox],
    ) -> list[tuple[DetectionBox, HeadClassificationResult]]:
        """Runs first-head CLIP classification for each dish bbox."""
        if not dish_boxes:
            return []

        index_path = self._heads_root / "first_head"
        if not index_path.exists():
            raise FileNotFoundError(f"First head index not found: {index_path}")

        clip = self._model_registry.get_clip()
        result_pairs: list[tuple[DetectionBox, HeadClassificationResult]] = []

        for box in dish_boxes:
            crop = Cropper.crop_by_box(image, box)
            match = clip.compare_image_to_text_index(crop, index_path)
            predicted = self._to_first_head_class(match.matched_name)
            evidence = RecognitionEvidence(
                source=DishDetectionSource.FIRST_HEAD,
                model_name="clip_first_head",
                score=match.score,
                chosen_label=str(predicted.value if isinstance(predicted, FirstHeadClass) else predicted),
                notes="first head class",
                bbox=box,
            )
            result_pairs.append(
                (
                    box,
                    HeadClassificationResult(
                        predicted_class=predicted,
                        score=match.score,
                        matched_description=match.matched_description,
                        evidence=evidence,
                    ),
                )
            )

        return result_pairs

    def resolve_non_other_dishes(
        self,
        image: np.ndarray,
        classified: Sequence[tuple[DetectionBox, HeadClassificationResult]],
    ) -> tuple[list[DishRecognitionResult], list[DetectionBox]]:
        """Resolves soup/portioned items and returns deferred `other_dish` bboxes."""
        clip = self._model_registry.get_clip()
        recognized: list[DishRecognitionResult] = []
        other_dish_boxes: list[DetectionBox] = []

        for box, head_result in classified:
            cls = head_result.predicted_class
            if cls == FirstHeadClass.EMPTY_PLATE:
                continue

            if cls == FirstHeadClass.OTHER_DISH:
                other_dish_boxes.append(box)
                continue

            if cls not in (FirstHeadClass.SOUP, FirstHeadClass.PORTIONED_DISH):
                other_dish_boxes.append(box)
                continue

            category = DishCategory.SOUP if cls == FirstHeadClass.SOUP else DishCategory.PORTIONED
            mode = self._resolve_mode(category.value)
            index_path = self._today_menu_root / category.value / "indexes" / mode.value
            if not index_path.exists():
                continue

            crop = Cropper.crop_by_box(image, box)
            match = (
                clip.compare_image_to_text_index(crop, index_path)
                if mode == ClipMatchMode.TEXT
                else clip.compare_image_to_photo_index(crop, index_path)
            )

            head_ev = head_result.evidence
            clip_ev = RecognitionEvidence(
                source=DishDetectionSource.CLIP_TEXT if mode == ClipMatchMode.TEXT else DishDetectionSource.CLIP_PHOTO,
                model_name="clip",
                score=match.score,
                chosen_label=match.matched_name,
                notes=f"category={category.value}; mode={mode.value}",
                bbox=box,
            )
            evidences = [ev for ev in (head_ev, clip_ev) if ev is not None]
            recognized.append(
                DishRecognitionResult(
                    dish_name=match.matched_name,
                    category=category,
                    count=1,
                    bbox=box,
                    evidences=evidences,
                )
            )

        return recognized, other_dish_boxes

    @staticmethod
    def _resolve_mode(category: str) -> ClipMatchMode:
        raw_mode = str(CLIP_MATCH_MODES.get(category, "photo")).strip().lower()
        return ClipMatchMode.TEXT if raw_mode == ClipMatchMode.TEXT.value else ClipMatchMode.PHOTO

    @staticmethod
    def _to_first_head_class(raw_label: str) -> FirstHeadClass | str:
        normalized = str(raw_label).strip().lower()
        aliases = {
            "portioned": FirstHeadClass.PORTIONED_DISH,
            "portioned_dish": FirstHeadClass.PORTIONED_DISH,
            "other": FirstHeadClass.OTHER_DISH,
            "other_dish": FirstHeadClass.OTHER_DISH,
            "soup": FirstHeadClass.SOUP,
            "empty": FirstHeadClass.EMPTY_PLATE,
            "empty_plate": FirstHeadClass.EMPTY_PLATE,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return FirstHeadClass(normalized)
        except ValueError:
            return normalized
