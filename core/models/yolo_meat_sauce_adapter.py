"""YOLO adapter for meat_sauce detection on `other_dish` crops."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from config.settings import MEAT_SAUCE_YOLO_CONF, MEAT_SAUCE_YOLO_IOU
from core.domain.entities import CropCandidate, DetectionBox
from core.domain.enums import DishDetectionSource


class YoloMeatSauceAdapter:
    """Runs YOLO_meat_sauce and keeps only `meat_sauce` detections."""

    TARGET_CLASS = "meat_sauce"

    def __init__(
        self,
        weight_path: str | Path | None = None,
        device: str = "cuda",
        conf: float | None = None,
        iou: float | None = None,
        weights_path: str | Path | None = None,
    ) -> None:
        resolved_path = weight_path or weights_path
        if resolved_path is None:
            raise ValueError("YOLO meat_sauce weight path is required.")

        self.weight_path = Path(resolved_path)
        if not self.weight_path.exists():
            raise FileNotFoundError(f"YOLO meat_sauce weights not found: '{self.weight_path}'")

        self.device = device
        self.conf = float(conf if conf is not None else MEAT_SAUCE_YOLO_CONF)
        self.iou = float(iou if iou is not None else MEAT_SAUCE_YOLO_IOU)

        self._model: Any | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load ultralytics YOLO model and provide actionable dependency errors."""
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Ultralytics is not installed. Install dependency 'ultralytics' to use YOLO adapters."
            ) from exc

        try:
            self._model = YOLO(str(self.weight_path))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load YOLO meat_sauce model from '{self.weight_path}': {exc}") from exc

    def _ensure_ready(self) -> Any:
        if self._model is None:
            raise RuntimeError("YOLO meat_sauce model is not loaded.")
        return self._model

    def _normalize_input_image(self, image: Any) -> Any:
        """Accept ndarray/PIL/path-like input for plug-and-play usage from pipeline."""
        if isinstance(image, np.ndarray):
            return image

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Input image path does not exist: '{image_path}'")
            return str(image_path)

        raise ValueError("Unsupported image input type. Expected np.ndarray, PIL.Image.Image, str, or Path.")

    def _run_inference(self, image: Any) -> list[Any]:
        model = self._ensure_ready()
        normalized_image = self._normalize_input_image(image)
        results = model.predict(
            source=normalized_image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        return list(results) if results is not None else []

    @staticmethod
    def _extract_names(result: Any, model: Any) -> dict[int, str]:
        names = getattr(result, "names", None)
        if not isinstance(names, dict):
            names = getattr(model, "names", {})
        return {int(k): str(v) for k, v in (names or {}).items()}

    @staticmethod
    def _to_python_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        if hasattr(value, "tolist"):
            return list(value.tolist())
        return list(value)

    def _result_to_detection_boxes(self, results: list[Any]) -> list[DetectionBox]:
        """Map results to DetectionBox and keep only class `meat_sauce`."""
        detections: list[DetectionBox] = []
        model = self._ensure_ready()

        for result in results:
            boxes_obj = getattr(result, "boxes", None)
            if boxes_obj is None:
                continue

            xyxy_list = self._to_python_list(getattr(boxes_obj, "xyxy", None))
            conf_list = self._to_python_list(getattr(boxes_obj, "conf", None))
            cls_list = self._to_python_list(getattr(boxes_obj, "cls", None))
            names = self._extract_names(result, model)

            for i, coords in enumerate(xyxy_list):
                if len(coords) < 4:
                    continue

                cls_idx = int(cls_list[i]) if i < len(cls_list) else -1
                class_name = names.get(cls_idx, str(cls_idx))
                if class_name != self.TARGET_CLASS:
                    continue

                confidence = float(conf_list[i]) if i < len(conf_list) else None
                detections.append(
                    DetectionBox(
                        x1=int(coords[0]),
                        y1=int(coords[1]),
                        x2=int(coords[2]),
                        y2=int(coords[3]),
                        confidence=confidence,
                        class_name=self.TARGET_CLASS,
                        source=DishDetectionSource.YOLO.value,
                    )
                )

        return detections

    def _result_to_crop_candidates(self, detections: list[DetectionBox]) -> list[CropCandidate]:
        # We intentionally avoid saving crops on disk here to keep adapters stateless.
        return [
            CropCandidate(
                crop_id=f"yolo-meat-sauce-{uuid4().hex}",
                bbox=box,
                class_name=box.class_name,
                parent_crop_id=None,
                image_path=None,
                temp_path=None,
            )
            for box in detections
        ]

    # Warmup runs one tiny dummy forward pass to reduce first-real-frame latency.
    def warmup(self) -> None:
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        _ = self._run_inference(dummy)

    def detect(self, image: Any) -> list[DetectionBox]:
        results = self._run_inference(image)
        return self._result_to_detection_boxes(results)

    def detect_with_crops(self, image: Any) -> list[CropCandidate]:
        detections = self.detect(image)
        return self._result_to_crop_candidates(detections)


__all__ = ["YoloMeatSauceAdapter"]
