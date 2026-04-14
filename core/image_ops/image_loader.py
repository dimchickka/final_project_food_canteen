"""Utilities for loading images from disk as OpenCV BGR arrays."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageLoader:
    """Image loading and file-level validation utility used by input adapters."""

    @staticmethod
    def validate_image_file(path: str | Path) -> bool:
        """Check that a path exists, is a file, and is readable as an image."""
        image_path = Path(path)
        if not image_path.exists() or not image_path.is_file():
            return False

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        return image is not None

    @staticmethod
    def load_image(path: str | Path) -> np.ndarray:
        """Load image from disk and normalize channel layout to BGR."""
        image_path = Path(path)
        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Unable to read image file: {image_path}")

        # Project convention: every ndarray in vision layer is OpenCV-style BGR.
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3:
            channels = image.shape[2]
            if channels == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            elif channels == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif channels != 3:
                raise ValueError(
                    f"Unsupported channel count {channels} for image: {image_path}"
                )
        else:
            raise ValueError(
                f"Unsupported image shape {image.shape} for image: {image_path}"
            )

        return image

    @classmethod
    def ensure_image_loaded(cls, path: str | Path) -> np.ndarray:
        """Semantic alias that raises explicit exceptions on failure."""
        return cls.load_image(path)
