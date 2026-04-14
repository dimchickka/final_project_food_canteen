"""Camera acquisition service for BGR frames used by the vision pipeline."""

from __future__ import annotations

import cv2
import numpy as np


class CameraService:
    """Thin wrapper over OpenCV VideoCapture with explicit lifecycle handling."""

    def __init__(self, camera_index: int = 0) -> None:
        self._camera_index = camera_index
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> "CameraService":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()

    def open(self) -> None:
        """Open camera by index and fail fast if the device is unavailable."""
        if self._capture is not None and self._capture.isOpened():
            return

        capture = cv2.VideoCapture(self._camera_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to open camera with index {self._camera_index}.")

        self._capture = capture

    def is_opened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def read_frame(self) -> np.ndarray:
        """Read single frame in OpenCV-native BGR format."""
        if not self.is_opened():
            raise RuntimeError("Camera is not opened. Call open() before reading frames.")

        assert self._capture is not None  # Narrow optional type after is_opened() check.
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from camera.")

        return frame

    def read_frame_safe(self) -> np.ndarray | None:
        """Best-effort read for preview-like consumers that tolerate missing frames."""
        try:
            return self.read_frame()
        except RuntimeError:
            return None

    def capture_frame(self) -> np.ndarray:
        """Semantic alias for read_frame() used by capture-oriented callers."""
        return self.read_frame()

    def release(self) -> None:
        """Release the underlying camera resource."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
