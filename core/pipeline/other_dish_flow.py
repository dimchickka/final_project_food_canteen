"""Heavy `other_dish` branch with parallel detectors and residual processing."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from config.settings import CLIP_MATCH_MODES, MIN_MEAT_BOX_AREA, MIN_SAUCE_BOX_AREA
from core.domain.dto import DishRecognitionResult, SauceDetectionResult
from core.domain.entities import DetectionBox, RecognitionEvidence
from core.domain.enums import ClipMatchMode, DishCategory, DishDetectionSource
from core.image_ops.cropper import Cropper
from core.image_ops.overlap_cleaner import OverlapCleaner
from core.models.model_registry import ModelRegistry
from core.pipeline.second_head_flow import SecondHeadFlow


class OtherDishFlow:
    """Processes `other_dish`: sauces/meats detection + pragmatic residual garnish pass."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        second_head_flow: SecondHeadFlow,
        today_menu_root: str | Path | None = None,
    ) -> None:
        self._model_registry = model_registry
        self._second_head_flow = second_head_flow
        self._today_menu_root = Path(today_menu_root) if today_menu_root is not None else Path("data/menu/today")

    def process(
        self,
        image: np.ndarray,
        other_dish_boxes: Sequence[DetectionBox],
        executor: Executor | None = None,
    ) -> list[DishRecognitionResult]:
        """Runs branch for every other_dish box and continues on per-box failures."""
        if not other_dish_boxes:
            return []

        own_executor = executor is None
        active_executor = executor or ThreadPoolExecutor(max_workers=3)
        results: list[DishRecognitionResult] = []
        try:
            for dish_box in other_dish_boxes:
                try:
                    results.extend(self._process_single_other_dish(image=image, dish_box=dish_box, executor=active_executor))
                except Exception:  # noqa: BLE001
                    continue
        finally:
            if own_executor and isinstance(active_executor, ThreadPoolExecutor):
                active_executor.shutdown(wait=True)

        return results

    def _process_single_other_dish(
        self,
        image: np.ndarray,
        dish_box: DetectionBox,
        executor: Executor,
    ) -> list[DishRecognitionResult]:
        crop = Cropper.crop_by_box(image, dish_box)
        qwen_future, meat_future, meat_sauce_future = self._start_parallel_detectors(crop=crop, executor=executor)

        qwen_result = qwen_future.result()
        meat_boxes = self._filter_small_boxes(meat_future.result(), min_area=MIN_MEAT_BOX_AREA)
        meat_sauce_boxes = self._filter_small_boxes(meat_sauce_future.result(), min_area=MIN_SAUCE_BOX_AREA)

        local_results: list[DishRecognitionResult] = []
        matched_boxes_for_residual: list[DetectionBox] = []

        sauce_items = self._match_sauces(crop=crop, parent_box=dish_box, sauce_result=qwen_result)
        local_results.extend(sauce_items)
        matched_boxes_for_residual.extend([item.bbox for item in sauce_items if item.bbox is not None])

        meat_items = self._match_meats(crop=crop, parent_box=dish_box, meat_boxes=[*meat_boxes, *meat_sauce_boxes])
        local_results.extend(meat_items)
        matched_boxes_for_residual.extend([item.bbox for item in meat_items if item.bbox is not None])

        residual_box = self._build_residual_box(base_box=dish_box, detected_boxes=matched_boxes_for_residual)
        if residual_box is not None:
            garnish_item = self._second_head_flow.resolve_garnish_if_present(image=image, residual_box=residual_box)
            if garnish_item is not None:
                local_results.append(garnish_item)

        return local_results

    def _start_parallel_detectors(
        self,
        crop: np.ndarray,
        executor: Executor,
    ) -> tuple[Future[SauceDetectionResult], Future[list[DetectionBox]], Future[list[DetectionBox]]]:
        """Starts Qwen sauces + YOLO meat + YOLO meat_sauce concurrently."""
        qwen = self._model_registry.get_qwen()
        yolo_meat = self._model_registry.get_yolo_meat()
        yolo_meat_sauce = self._model_registry.get_yolo_meat_sauce()

        qwen_future = executor.submit(qwen.detect_sauces_on_other_dish, crop)
        meat_future = executor.submit(yolo_meat.detect, crop)
        meat_sauce_future = executor.submit(yolo_meat_sauce.detect, crop)
        return qwen_future, meat_future, meat_sauce_future

    @staticmethod
    def _filter_small_boxes(boxes: Sequence[DetectionBox], min_area: int) -> list[DetectionBox]:
        return [box for box in boxes if box.area >= min_area]

    def _match_sauces(
        self,
        crop: np.ndarray,
        parent_box: DetectionBox,
        sauce_result: SauceDetectionResult,
    ) -> list[DishRecognitionResult]:
        mode = self._resolve_mode(DishCategory.SAUCE.value)
        index_path = self._today_menu_root / DishCategory.SAUCE.value / "indexes" / mode.value
        if not index_path.exists():
            return []

        clip = self._model_registry.get_clip()
        results: list[DishRecognitionResult] = []
        for sauce_box in self._filter_small_boxes(sauce_result.boxes, MIN_SAUCE_BOX_AREA):
            local_crop = Cropper.crop_by_box(crop, sauce_box)
            match = (
                clip.compare_image_to_text_index(local_crop, index_path)
                if mode == ClipMatchMode.TEXT
                else clip.compare_image_to_photo_index(local_crop, index_path)
            )
            global_box = self._to_global_box(parent_box, sauce_box)
            results.append(
                DishRecognitionResult(
                    dish_name=match.matched_name,
                    category=DishCategory.SAUCE,
                    count=1,
                    bbox=global_box,
                    evidences=[
                        RecognitionEvidence(
                            source=DishDetectionSource.QWEN,
                            model_name="qwen",
                            chosen_label=sauce_box.label or "sauce",
                            notes=sauce_result.notes,
                            bbox=global_box,
                        ),
                        RecognitionEvidence(
                            source=DishDetectionSource.CLIP_TEXT
                            if mode == ClipMatchMode.TEXT
                            else DishDetectionSource.CLIP_PHOTO,
                            model_name="clip",
                            score=match.score,
                            chosen_label=match.matched_name,
                            notes=f"category=sauce; mode={mode.value}",
                            bbox=global_box,
                        ),
                    ],
                )
            )
        return results

    def _match_meats(
        self,
        crop: np.ndarray,
        parent_box: DetectionBox,
        meat_boxes: Sequence[DetectionBox],
    ) -> list[DishRecognitionResult]:
        mode = self._resolve_mode(DishCategory.MEAT.value)
        index_path = self._today_menu_root / DishCategory.MEAT.value / "indexes" / mode.value
        if not index_path.exists():
            return []

        clip = self._model_registry.get_clip()
        results: list[DishRecognitionResult] = []
        for meat_box in meat_boxes:
            local_crop = Cropper.crop_by_box(crop, meat_box)
            match = (
                clip.compare_image_to_text_index(local_crop, index_path)
                if mode == ClipMatchMode.TEXT
                else clip.compare_image_to_photo_index(local_crop, index_path)
            )
            global_box = self._to_global_box(parent_box, meat_box)
            results.append(
                DishRecognitionResult(
                    dish_name=match.matched_name,
                    category=DishCategory.MEAT,
                    count=1,
                    bbox=global_box,
                    evidences=[
                        RecognitionEvidence(
                            source=DishDetectionSource.YOLO,
                            model_name="yolo_meat_family",
                            score=meat_box.confidence,
                            chosen_label=meat_box.class_name,
                            bbox=global_box,
                        ),
                        RecognitionEvidence(
                            source=DishDetectionSource.CLIP_TEXT
                            if mode == ClipMatchMode.TEXT
                            else DishDetectionSource.CLIP_PHOTO,
                            model_name="clip",
                            score=match.score,
                            chosen_label=match.matched_name,
                            notes=f"category=meat; mode={mode.value}",
                            bbox=global_box,
                        ),
                    ],
                )
            )
        return results

    def _build_residual_box(self, base_box: DetectionBox, detected_boxes: Sequence[DetectionBox]) -> DetectionBox | None:
        """Pragmatic bbox residual logic: subtract overlaps, no pixel-level segmentation."""
        if base_box.area <= 0:
            return None
        residual = base_box
        for det_box in detected_boxes:
            residual = OverlapCleaner.subtract_overlap_from_box(residual, det_box)
            if residual.area <= 0:
                return None

        if residual.width < OverlapCleaner.MIN_SIDE_PX or residual.height < OverlapCleaner.MIN_SIDE_PX:
            return None
        return residual

    @staticmethod
    def _to_global_box(parent_box: DetectionBox, local_box: DetectionBox) -> DetectionBox:
        return DetectionBox(
            x1=parent_box.x1 + local_box.x1,
            y1=parent_box.y1 + local_box.y1,
            x2=parent_box.x1 + local_box.x2,
            y2=parent_box.y1 + local_box.y2,
            confidence=local_box.confidence,
            class_name=local_box.class_name,
            label=local_box.label,
            source=local_box.source,
        )

    @staticmethod
    def _resolve_mode(category: str) -> ClipMatchMode:
        raw_mode = str(CLIP_MATCH_MODES.get(category, "photo")).strip().lower()
        return ClipMatchMode.TEXT if raw_mode == ClipMatchMode.TEXT.value else ClipMatchMode.PHOTO
