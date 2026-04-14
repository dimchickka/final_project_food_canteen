"""Backend bbox-editor utilities for coordinate normalization and crop extraction."""

from __future__ import annotations

import numpy as np

from core.domain.entities import DetectionBox
from core.image_ops.cropper import Cropper


class BBoxEditorLogic:
    """Non-GUI logic for admin bbox editor flows."""

    @staticmethod
    def normalize_box(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        image_shape: tuple[int, ...],
    ) -> DetectionBox:
        """Normalize coordinate order and clip bbox to image bounds."""
        raw_box = DetectionBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))
        clipped = Cropper.clip_box_to_image(raw_box, image_shape)
        if clipped.width <= 0 or clipped.height <= 0:
            raise ValueError(
                "Invalid bbox after normalization/clipping: "
                f"({x1}, {y1}, {x2}, {y2})"
            )
        return clipped

    @classmethod
    def validate_box(cls, box: DetectionBox, image_shape: tuple[int, ...]) -> bool:
        """Validate that box can produce a non-empty crop inside image bounds."""
        try:
            clipped = Cropper.clip_box_to_image(box, image_shape)
        except ValueError:
            return False
        return clipped.width > 0 and clipped.height > 0

    @classmethod
    def extract_crop(cls, image: np.ndarray, box: DetectionBox) -> np.ndarray:
        """Extract crop for an already prepared DetectionBox."""
        if not cls.validate_box(box, image.shape):
            raise ValueError(
                "Cannot extract crop from invalid bbox: "
                f"({box.x1}, {box.y1}, {box.x2}, {box.y2})"
            )
        return Cropper.crop_by_box(image, box)

    @classmethod
    def extract_crop_from_coordinates(
        cls,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> np.ndarray:
        """Normalize editor coordinates and extract corresponding crop."""
        normalized = cls.normalize_box(x1, y1, x2, y2, image.shape)
        return cls.extract_crop(image, normalized)
