"""Bounding-box based crop helpers with strict bounds safety."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from core.domain.entities import DetectionBox


class Cropper:
    """Utility class for robust crop extraction from xyxy rectangular boxes."""

    @staticmethod
    def _normalize_xyxy(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
        """Normalize corner order so (x1, y1) is top-left and (x2, y2) is bottom-right."""
        nx1, nx2 = sorted((int(x1), int(x2)))
        ny1, ny2 = sorted((int(y1), int(y2)))
        return nx1, ny1, nx2, ny2

    @classmethod
    def clip_box_to_image(
        cls,
        box: DetectionBox,
        image_shape: tuple[int, ...],
    ) -> DetectionBox:
        """Clip a box to valid image bounds and preserve metadata fields."""
        if len(image_shape) < 2:
            raise ValueError(f"Invalid image shape: {image_shape}")

        height, width = int(image_shape[0]), int(image_shape[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"Image has invalid size: width={width}, height={height}")

        x1, y1, x2, y2 = cls._normalize_xyxy(box.x1, box.y1, box.x2, box.y2)

        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        return DetectionBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=box.confidence,
            class_name=box.class_name,
            label=box.label,
            source=box.source,
        )

    @classmethod
    def crop_by_xyxy(
        cls,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> np.ndarray:
        """Crop image by integer coordinates after normalization and boundary clipping."""
        if image.ndim < 2:
            raise ValueError(f"Expected image with at least 2 dimensions, got {image.ndim}")

        box = DetectionBox(x1=x1, y1=y1, x2=x2, y2=y2)
        clipped = cls.clip_box_to_image(box, image.shape)
        if clipped.width <= 0 or clipped.height <= 0:
            raise ValueError(
                "Crop is empty after clipping box to image bounds: "
                f"({x1}, {y1}, {x2}, {y2})"
            )

        crop = image[clipped.y1 : clipped.y2, clipped.x1 : clipped.x2]
        if crop.size == 0:
            raise ValueError(
                "Crop extraction produced an empty array for box "
                f"({clipped.x1}, {clipped.y1}, {clipped.x2}, {clipped.y2})"
            )

        return crop

    @classmethod
    def crop_by_box(cls, image: np.ndarray, box: DetectionBox) -> np.ndarray:
        """Crop image using DetectionBox container."""
        return cls.crop_by_xyxy(image, box.x1, box.y1, box.x2, box.y2)

    @classmethod
    def crop_candidates_from_boxes(
        cls,
        image: np.ndarray,
        boxes: Sequence[DetectionBox],
    ) -> list[np.ndarray]:
        """Generate crops for all valid boxes and fail on first invalid/empty crop."""
        return [cls.crop_by_box(image, box) for box in boxes]
