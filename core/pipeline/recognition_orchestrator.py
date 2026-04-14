"""Main orchestrator that coordinates all recognition pipeline flows end-to-end."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from core.domain.dto import DishRecognitionResult, RecognitionSessionResult, ValidationResult
from core.domain.entities import AnnotatedDetection, DetectionBox
from core.domain.enums import ProcessingStage, TrayObjectClass, ValidationStatus
from core.image_ops.overlap_cleaner import OverlapCleaner
from core.logging.model_trace_logger import ModelTraceLogger
from core.logging.run_logger import RunLogger
from core.logging.timing import TimingTracker
from core.models.model_registry import ModelRegistry, get_model_registry
from core.pipeline.beverage_flow import BeverageFlow
from core.pipeline.cancellation_token import CancellationToken
from core.pipeline.first_head_flow import FirstHeadFlow
from core.pipeline.other_dish_flow import OtherDishFlow
from core.pipeline.receipt_flow import ReceiptFlow
from core.pipeline.second_head_flow import SecondHeadFlow
from core.pipeline.validation_flow import ValidationFlow


class RecognitionOrchestrator:
    """Main coordinating layer: starts async validation, runs detectors/flows, persists artifacts."""

    def __init__(
        self,
        model_registry: ModelRegistry | None = None,
        run_logger: RunLogger | None = None,
        today_menu_root: str | Path | None = None,
        heads_root: str | Path | None = None,
        max_workers: int = 4,
    ) -> None:
        self._model_registry = model_registry or get_model_registry()
        self._run_logger = run_logger or RunLogger()
        self._validation_flow = ValidationFlow(self._model_registry)
        self._beverage_flow = BeverageFlow(self._model_registry, today_menu_root=today_menu_root)
        self._first_head_flow = FirstHeadFlow(
            self._model_registry,
            today_menu_root=today_menu_root,
            heads_root=heads_root,
        )
        self._second_head_flow = SecondHeadFlow(
            self._model_registry,
            today_menu_root=today_menu_root,
            heads_root=heads_root,
        )
        self._other_dish_flow = OtherDishFlow(
            self._model_registry,
            second_head_flow=self._second_head_flow,
            today_menu_root=today_menu_root,
        )
        self._receipt_flow = ReceiptFlow()
        self._max_workers = max(2, int(max_workers))

    def recognize(self, image: np.ndarray) -> RecognitionSessionResult:
        token = CancellationToken()
        timings = TimingTracker()
        trace = ModelTraceLogger()
        recognized_items: list[DishRecognitionResult] = []
        annotated: list[AnnotatedDetection] = []
        validation_result: ValidationResult | None = None

        self._run_logger.ensure_session()
        trace.add_entry(ProcessingStage.IMAGE_ACQUIRED.value, "Image acquired for recognition")

        with timings.measure("total"):
            self._run_logger.save_source_image(image)

            try:
                with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                    trace.add_entry(ProcessingStage.VALIDATION_STARTED.value, "Validation started asynchronously")
                    timings.start("validation")
                    validation_future: Future[ValidationResult] = self._validation_flow.start_async(image, executor)

                    timings.start("yolo_main")
                    yolo_main = self._model_registry.get_yolo_main()
                    main_detections = yolo_main.detect(image)
                    yolo_ms = timings.stop("yolo_main")
                    trace.add_entry(
                        ProcessingStage.YOLO_MAIN_FINISHED.value,
                        "YOLO main finished",
                        payload={"detections": len(main_detections)},
                        duration_ms=yolo_ms,
                    )

                    self._update_validation_state(validation_future, token, trace)
                    token.raise_if_cancelled()

                    cups = [box for box in main_detections if box.class_name == TrayObjectClass.CUP.value]
                    dish_boxes = [
                        box
                        for box in main_detections
                        if box.class_name in (TrayObjectClass.PLATE_FLAT.value, TrayObjectClass.BOWL_DEEP.value)
                    ]
                    cleaned_dish_boxes = OverlapCleaner.clean_overlapping_boxes(dish_boxes)
                    trace.add_entry(
                        ProcessingStage.YOLO_MAIN_FINISHED.value,
                        "Main detections split",
                        payload={
                            "cups": len(cups),
                            "dish_boxes_raw": len(dish_boxes),
                            "dish_boxes_cleaned": len(cleaned_dish_boxes),
                        },
                    )

                    timings.start("beverage_flow")
                    beverage_items = self._beverage_flow.process(image=image, cup_boxes=cups)
                    timings.stop("beverage_flow")
                    recognized_items.extend(beverage_items)
                    trace.add_entry(
                        ProcessingStage.BEVERAGE_PROCESSED.value,
                        "Beverage flow completed",
                        payload={"recognized": len(beverage_items)},
                    )

                    self._update_validation_state(validation_future, token, trace)
                    token.raise_if_cancelled()

                    timings.start("first_head_flow")
                    classified = self._first_head_flow.classify_dish_crops(image=image, dish_boxes=cleaned_dish_boxes)
                    first_head_items, other_dish_boxes = self._first_head_flow.resolve_non_other_dishes(
                        image=image,
                        classified=classified,
                    )
                    timings.stop("first_head_flow")
                    recognized_items.extend(first_head_items)
                    trace.add_entry(
                        ProcessingStage.FIRST_HEAD_PROCESSED.value,
                        "First head flow completed",
                        payload={
                            "classified": len(classified),
                            "resolved_non_other": len(first_head_items),
                            "other_dish_boxes": len(other_dish_boxes),
                        },
                    )

                    self._update_validation_state(validation_future, token, trace)
                    token.raise_if_cancelled()

                    timings.start("other_dish_flow")
                    other_items = self._other_dish_flow.process(
                        image=image,
                        other_dish_boxes=other_dish_boxes,
                        executor=executor,
                    )
                    timings.stop("other_dish_flow")
                    recognized_items.extend(other_items)
                    trace.add_entry(
                        ProcessingStage.OTHER_DISH_PROCESSED.value,
                        "Other dish flow completed",
                        payload={"recognized": len(other_items)},
                    )

                    if validation_future.done():
                        try:
                            timings.stop("validation")
                        except KeyError:
                            pass
                    validation_result = validation_future.result()
                    self._validation_flow.ensure_valid_or_cancel(validation_result, token)
                    if token.is_cancelled():
                        token.raise_if_cancelled()

            except RuntimeError as exc:
                abort_reason = str(exc)
                trace.add_entry(ProcessingStage.ABORTED.value, "Pipeline aborted", payload={"reason": abort_reason})
                return self._finalize(
                    success=False,
                    aborted=True,
                    abort_reason=abort_reason,
                    recognized_items=[],
                    annotated_detections=[],
                    timings=timings,
                    trace=trace,
                    validation_result=validation_result,
                    image=image,
                )
            except Exception as exc:  # noqa: BLE001
                trace.add_exception(ProcessingStage.ABORTED.value, exc)
                return self._finalize(
                    success=False,
                    aborted=False,
                    abort_reason=str(exc),
                    recognized_items=[],
                    annotated_detections=[],
                    timings=timings,
                    trace=trace,
                    validation_result=validation_result,
                    image=image,
                )

        timings.start("receipt_flow")
        aggregated_items = self._receipt_flow.aggregate_items(recognized_items)
        receipt = self._receipt_flow.build_receipt(aggregated_items)
        timings.stop("receipt_flow")
        trace.add_entry(
            ProcessingStage.RECEIPT_READY.value,
            "Receipt built",
            payload={"line_items": len(receipt.items)},
        )

        annotated = self._build_annotated_detections(aggregated_items)
        trace.add_entry(
            ProcessingStage.COMPLETED.value,
            "Pipeline completed",
            payload={"recognized_items": len(aggregated_items)},
        )

        return self._finalize(
            success=True,
            aborted=False,
            abort_reason=None,
            recognized_items=aggregated_items,
            annotated_detections=annotated,
            timings=timings,
            trace=trace,
            validation_result=validation_result,
            image=image,
            receipt=receipt,
        )

    def _update_validation_state(
        self,
        validation_future: Future[ValidationResult],
        token: CancellationToken,
        trace: ModelTraceLogger,
    ) -> None:
        result = self._validation_flow.get_result_non_blocking(validation_future)
        if result is None:
            return

        self._validation_flow.ensure_valid_or_cancel(result, token)
        trace.add_entry(
            ProcessingStage.VALIDATION_FINISHED.value,
            "Validation finished",
            payload={
                "status": result.status.value,
                "reason": result.reason,
                "confidence": result.confidence,
            },
        )

    @staticmethod
    def _build_annotated_detections(items: list[DishRecognitionResult]) -> list[AnnotatedDetection]:
        detections: list[AnnotatedDetection] = []
        for item in items:
            if item.bbox is None:
                continue
            detections.append(
                AnnotatedDetection(
                    bbox=item.bbox,
                    display_name=item.dish_name,
                    count_hint=item.count,
                )
            )
        return detections

    def _finalize(
        self,
        *,
        success: bool,
        aborted: bool,
        abort_reason: str | None,
        recognized_items: list[DishRecognitionResult],
        annotated_detections: list[AnnotatedDetection],
        timings: TimingTracker,
        trace: ModelTraceLogger,
        validation_result: ValidationResult | None,
        image: np.ndarray,
        receipt: object | None = None,
    ) -> RecognitionSessionResult:
        timings_payload = timings.to_dict()
        total_time_ms = timings_payload.get("total", timings.total_time_ms())

        if receipt is not None:
            self._run_logger.save_receipt(receipt)

        timings.start("annotated_render/save")
        self._run_logger.save_annotated_result(image=image, detections=annotated_detections)
        timings.stop("annotated_render/save")

        self._run_logger.save_timings(timings)
        self._run_logger.save_pipeline_trace(trace)
        self._run_logger.save_qwen_validation(
            raw_text=validation_result.raw_text if validation_result is not None else None,
            parsed=validation_result.to_dict() if validation_result is not None else {"status": ValidationStatus.UNKNOWN.value},
        )

        return RecognitionSessionResult(
            success=success,
            aborted=aborted,
            abort_reason=abort_reason,
            recognized_items=recognized_items,
            trace_entries=trace.entries(),
            annotated_detections=annotated_detections,
            total_time_ms=total_time_ms,
        )
