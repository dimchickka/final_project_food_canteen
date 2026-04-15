"""Small practical card widget to display one dish in menu lists."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout


class DishCardWidget(QFrame):
    """Renders dish name/category/status and optional thumbnail."""

    def __init__(self, row: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.row = row
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { border: 1px solid #cfcfcf; border-radius: 8px; }")

        root = QHBoxLayout(self)

        self.preview = QLabel()
        self.preview.setFixedSize(72, 72)
        self.preview.setAlignment(Qt.AlignCenter)
        self._set_preview(row.get("crop_image_path"))

        text_col = QVBoxLayout()
        self.name_label = QLabel(str(row.get("name", "—")))
        self.name_label.setStyleSheet("font-weight: 700;")
        self.category_label = QLabel(f"Категория: {row.get('category', '—')}")
        self.status_label = QLabel("Активно" if row.get("is_active", True) else "Неактивно")

        self.toggle_btn = QPushButton("Отключить" if row.get("is_active", True) else "Включить")

        text_col.addWidget(self.name_label)
        text_col.addWidget(self.category_label)
        text_col.addWidget(self.status_label)

        root.addWidget(self.preview)
        root.addLayout(text_col, stretch=1)
        root.addWidget(self.toggle_btn)

    def _set_preview(self, path: str | None) -> None:
        if not path:
            self.preview.setText("—")
            return
        p = Path(path)
        if not p.exists():
            self.preview.setText("—")
            return
        pix = QPixmap(str(p)).scaled(72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pix)
