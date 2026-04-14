"""Final image annotation renderer used for logs and visual debug artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from core.domain.entities import AnnotatedDetection, DetectionBox


class AnnotationRenderer:
    """Renders detection boxes/labels and can persist resulting annotated image."""

    def __init__(
        self,
        box_color: tuple[int, int, int] = (0, 255, 0),
        text_color: tuple[int, int, int] = (255, 255, 255),
        text_background: tuple[int, int, int] = (0, 120, 0),
        thickness: int = 2,
        font_scale: float = 0.6,
    ) -> None:
        self._box_color = box_color
        self._text_color = text_color
        self._text_background = text_background
        self._thickness = thickness
        self._font_scale = font_scale
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def _normalize_box(box: DetectionBox) -> DetectionBox:
        x1, x2 = sorted((box.x1, box.x2))
        y1, y2 = sorted((box.y1, box.y2))
        return DetectionBox(x1=x1, y1=y1, x2=x2, y2=y2)

    @staticmethod
    def _build_label(det: AnnotatedDetection) -> str:
        if det.count_hint is not None and det.count_hint > 1:
            return f"{det.display_name} x{det.count_hint}"
        return det.display_name

    def render(self, image: np.ndarray, detections: Sequence[AnnotatedDetection]) -> np.ndarray:
        """Draw detections over a copy of input BGR image."""
        if image.ndim < 2:
            raise ValueError(f"Expected image with at least 2 dimensions, got {image.ndim}")

        canvas = image.copy()
        height, width = canvas.shape[:2]

        for detection in detections:
            box = self._normalize_box(detection.bbox)
            x1 = max(0, min(box.x1, width - 1))
            y1 = max(0, min(box.y1, height - 1))
            x2 = max(0, min(box.x2, width - 1))
            y2 = max(0, min(box.y2, height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(canvas, (x1, y1), (x2, y2), self._box_color, self._thickness)

            label = self._build_label(detection)
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                self._font,
                self._font_scale,
                max(1, self._thickness),
            )
            top = max(0, y1 - text_h - baseline - 4)
            bottom = min(height - 1, y1)
            right = min(width - 1, x1 + text_w + 6)

            cv2.rectangle(canvas, (x1, top), (right, bottom), self._text_background, -1)
            text_y = max(text_h + 2, bottom - baseline - 2)
            cv2.putText(
                canvas,
                label,
                (x1 + 3, text_y),
                self._font,
                self._font_scale,
                self._text_color,
                max(1, self._thickness - 1),
                cv2.LINE_AA,
            )

        return canvas

    def save_rendered(
        self,
        image: np.ndarray,
        detections: Sequence[AnnotatedDetection],
        output_path: str | Path,
    ) -> Path:
        """Render detections and write annotated image to disk."""
        destination = Path(output_path)
        if destination.exists() and destination.is_dir():
            raise ValueError(f"Output path points to a directory, expected file: {destination}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        rendered = self.render(image, detections)

        ok = cv2.imwrite(str(destination), rendered)
        if not ok:
            raise RuntimeError(f"Failed to save annotated image to: {destination}")

        return destination
