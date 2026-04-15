"""Camera dialog with lightweight live preview and capture action."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from config.settings import CAMERA_INDEX
from core.image_ops.camera_service import CameraService


class CameraCaptureDialog(QDialog):
    """Reads frames from CameraService and returns captured BGR frame."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Камера")
        self.resize(900, 620)

        self._camera = CameraService(CAMERA_INDEX)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._update_preview)

        self._latest_frame: np.ndarray | None = None
        self._captured_frame: np.ndarray | None = None

        root = QVBoxLayout(self)
        self.preview = QLabel("Открываю камеру...")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background: #111; color: white;")

        actions = QHBoxLayout()
        self.capture_btn = QPushButton("Сделать фото")
        self.cancel_btn = QPushButton("Отмена")

        self.capture_btn.clicked.connect(self._on_capture)
        self.cancel_btn.clicked.connect(self.reject)

        actions.addStretch(1)
        actions.addWidget(self.capture_btn)
        actions.addWidget(self.cancel_btn)

        root.addWidget(self.preview, stretch=1)
        root.addLayout(actions)

        self._start_camera()

    def _start_camera(self) -> None:
        self._camera.open()
        self._timer.start()

    def _update_preview(self) -> None:
        frame = self._camera.read_frame_safe()
        if frame is None:
            return
        self._latest_frame = frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pix)

    def _on_capture(self) -> None:
        if self._latest_frame is None:
            return
        self._captured_frame = self._latest_frame.copy()
        self.accept()

    def captured_frame(self) -> np.ndarray | None:
        return self._captured_frame

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._timer.stop()
        self._camera.release()
        super().closeEvent(event)
