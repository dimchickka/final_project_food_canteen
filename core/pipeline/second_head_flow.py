"""Second-head flow resolves residual region after meat/sauce extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config.settings import CLIP_MATCH_MODES
from core.domain.dto import DishRecognitionResult, HeadClassificationResult
from core.domain.entities import DetectionBox, RecognitionEvidence
from core.domain.enums import ClipMatchMode, DishCategory, DishDetectionSource, SecondHeadClass
from core.image_ops.cropper import Cropper
from core.models.model_registry import ModelRegistry


class SecondHeadFlow:
    """Classifies residual dish area and optionally resolves garnish via CLIP."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        today_menu_root: str | Path | None = None,
        heads_root: str | Path | None = None,
    ) -> None:
        self._model_registry = model_registry
        self._today_menu_root = Path(today_menu_root) if today_menu_root is not None else Path("data/menu/today")
        self._heads_root = Path(heads_root) if heads_root is not None else Path("data/heads")

    def classify_residual(self, image: np.ndarray, residual_box: DetectionBox) -> HeadClassificationResult | None:
        """Runs second-head classifier over one residual bbox."""
        if residual_box.width <= 0 or residual_box.height <= 0:
            return None

        index_path = self._heads_root / "second_head"
        if not index_path.exists():
            raise FileNotFoundError(f"Second head index not found: {index_path}")

        crop = Cropper.crop_by_box(image, residual_box)
        clip = self._model_registry.get_clip()
        match = clip.compare_image_to_text_index(crop, index_path)
        predicted = self._to_second_head_class(match.matched_name)

        return HeadClassificationResult(
            predicted_class=predicted,
            score=match.score,
            matched_description=match.matched_description,
            evidence=RecognitionEvidence(
                source=DishDetectionSource.SECOND_HEAD,
                model_name="clip_second_head",
                score=match.score,
                chosen_label=str(predicted.value if isinstance(predicted, SecondHeadClass) else predicted),
                notes="second head class",
                bbox=residual_box,
            ),
        )

    def resolve_garnish_if_present(self, image: np.ndarray, residual_box: DetectionBox) -> DishRecognitionResult | None:
        """Returns garnish recognition if second-head predicts garnish_remaining."""
        second_head_result = self.classify_residual(image, residual_box)
        if second_head_result is None:
            return None

        if second_head_result.predicted_class == SecondHeadClass.EMPTY_AFTER_EXTRACTION:
            return None

        if second_head_result.predicted_class != SecondHeadClass.GARNISH_REMAINING:
            return None

        mode = self._resolve_mode(DishCategory.GARNISH.value)
        index_path = self._today_menu_root / DishCategory.GARNISH.value / "indexes" / mode.value
        if not index_path.exists():
            return None

        crop = Cropper.crop_by_box(image, residual_box)
        clip = self._model_registry.get_clip()
        match = (
            clip.compare_image_to_text_index(crop, index_path)
            if mode == ClipMatchMode.TEXT
            else clip.compare_image_to_photo_index(crop, index_path)
        )

        garnish_ev = RecognitionEvidence(
            source=DishDetectionSource.CLIP_TEXT if mode == ClipMatchMode.TEXT else DishDetectionSource.CLIP_PHOTO,
            model_name="clip",
            score=match.score,
            chosen_label=match.matched_name,
            notes=f"category=garnish; mode={mode.value}",
            bbox=residual_box,
        )

        evidences = [ev for ev in (second_head_result.evidence, garnish_ev) if ev is not None]
        return DishRecognitionResult(
            dish_name=match.matched_name,
            category=DishCategory.GARNISH,
            count=1,
            bbox=residual_box,
            evidences=evidences,
        )

    @staticmethod
    def _resolve_mode(category: str) -> ClipMatchMode:
        raw_mode = str(CLIP_MATCH_MODES.get(category, "photo")).strip().lower()
        return ClipMatchMode.TEXT if raw_mode == ClipMatchMode.TEXT.value else ClipMatchMode.PHOTO

    @staticmethod
    def _to_second_head_class(raw_label: str) -> SecondHeadClass | str:
        normalized = str(raw_label).strip().lower()
        aliases = {
            "empty": SecondHeadClass.EMPTY_AFTER_EXTRACTION,
            "empty_after_extraction": SecondHeadClass.EMPTY_AFTER_EXTRACTION,
            "garnish": SecondHeadClass.GARNISH_REMAINING,
            "garnish_remaining": SecondHeadClass.GARNISH_REMAINING,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return SecondHeadClass(normalized)
        except ValueError:
            return normalized
