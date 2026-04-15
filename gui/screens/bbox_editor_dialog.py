"""Simple rectangle-selection bbox editor dialog.

This is intentionally a practical rectangle selector, not a full image editor.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from core.domain.entities import DetectionBox
from core.image_ops.bbox_editor_logic import BBoxEditorLogic


class _Canvas(QLabel):
    """Canvas widget for drag-selecting axis-aligned bbox."""

    def __init__(self, pixmap: QPixmap, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setPixmap(pixmap)
        self.setAlignment(Qt.AlignCenter)
        self._start: QPoint | None = None
        self._end: QPoint | None = None

    @property
    def rect_selection(self) -> QRect | None:
        if self._start is None or self._end is None:
            return None
        return QRect(self._start, self._end).normalized()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self._start = event.position().toPoint()
            self._end = self._start
            self.update()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._start is not None:
            self._end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._start is not None:
            self._end = event.position().toPoint()
            self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        rect = self.rect_selection
        if rect is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(Qt.red, 2))
        painter.drawRect(rect)


class BBoxEditorDialog(QDialog):
    """Shows an image and returns selected bbox + crop."""

    def __init__(self, image_bgr: np.ndarray, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("BBox editor")
        self.resize(960, 700)

        self._image_bgr = image_bgr
        self.selected_box: DetectionBox | None = None
        self.selected_crop: np.ndarray | None = None

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        root = QVBoxLayout(self)
        self.canvas = _Canvas(pix)
        self.canvas.setScaledContents(False)
        self.canvas.setMinimumSize(640, 480)

        actions = QHBoxLayout()
        ok_btn = QPushButton("Подтвердить")
        cancel_btn = QPushButton("Отмена")
        ok_btn.clicked.connect(self._on_accept)
        cancel_btn.clicked.connect(self.reject)
        actions.addStretch(1)
        actions.addWidget(ok_btn)
        actions.addWidget(cancel_btn)

        root.addWidget(self.canvas, stretch=1)
        root.addLayout(actions)

    def _on_accept(self) -> None:
        rect = self.canvas.rect_selection
        if rect is None or rect.width() <= 1 or rect.height() <= 1:
            QMessageBox.warning(self, "BBox", "Выделите прямоугольник мышью.")
            return
        try:
            # Map QLabel pixels to image coords (best-effort proportional mapping).
            img_h, img_w = self._image_bgr.shape[:2]
            scale_x = img_w / max(1, self.canvas.width())
            scale_y = img_h / max(1, self.canvas.height())
            x1 = int(rect.left() * scale_x)
            y1 = int(rect.top() * scale_y)
            x2 = int(rect.right() * scale_x)
            y2 = int(rect.bottom() * scale_y)

            box = BBoxEditorLogic.normalize_box(x1, y1, x2, y2, self._image_bgr.shape)
            crop = BBoxEditorLogic.extract_crop(self._image_bgr, box)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "BBox", f"Не удалось применить bbox: {exc}")
            return

        self.selected_box = box
        self.selected_crop = crop
        self.accept()
