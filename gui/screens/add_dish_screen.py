"""Admin flow for adding new dish card from photo/camera + bbox + phrase."""

from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config.constants import DISH_CATEGORIES
from core.image_ops.image_loader import ImageLoader
from gui.screens.bbox_editor_dialog import BBoxEditorDialog
from gui.screens.camera_capture_dialog import CameraCaptureDialog


class AddDishScreen(QWidget):
    """Guided admin workflow for manual dish-card creation."""

    back_requested = Signal()
    regenerate_phrase_requested = Signal(object, str, str)
    confirm_dish_requested = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source_image: np.ndarray | None = None
        self._crop_image: np.ndarray | None = None

        root = QVBoxLayout(self)

        form = QFormLayout()
        self.name_input = QLineEdit()
        self.category_combo = QComboBox()
        self.category_combo.addItems(list(DISH_CATEGORIES))
        self.phrase_text = QTextEdit()
        self.phrase_text.setPlaceholderText("Фраза от Qwen...")

        form.addRow("Название блюда:", self.name_input)
        form.addRow("Категория:", self.category_combo)
        form.addRow("Qwen фраза:", self.phrase_text)

        img_actions = QHBoxLayout()
        self.file_btn = QPushButton("Загрузить фото")
        self.camera_btn = QPushButton("Снять фото")
        self.bbox_btn = QPushButton("Открыть bbox editor")
        self.bbox_btn.setEnabled(False)
        img_actions.addWidget(self.file_btn)
        img_actions.addWidget(self.camera_btn)
        img_actions.addWidget(self.bbox_btn)

        phrase_actions = QHBoxLayout()
        self.regen_btn = QPushButton("Регенерировать фразу")
        self.confirm_btn = QPushButton("Подтвердить блюдо")
        phrase_actions.addWidget(self.regen_btn)
        phrase_actions.addWidget(self.confirm_btn)

        self.status = QLabel("Сначала загрузите изображение и выделите bbox.")

        self.back_btn = QPushButton("Назад в админ-панель")

        self.file_btn.clicked.connect(self._load_from_file)
        self.camera_btn.clicked.connect(self._load_from_camera)
        self.bbox_btn.clicked.connect(self._open_bbox_editor)
        self.regen_btn.clicked.connect(self._on_regenerate_phrase)
        self.confirm_btn.clicked.connect(self._on_confirm)
        self.back_btn.clicked.connect(self.back_requested)

        root.addLayout(img_actions)
        root.addLayout(form)
        root.addLayout(phrase_actions)
        root.addWidget(self.status)
        root.addStretch(1)
        root.addWidget(self.back_btn)

    def _load_from_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not path:
            return
        try:
            self._source_image = ImageLoader.ensure_image_loaded(path)
            self._crop_image = None
            self.bbox_btn.setEnabled(True)
            self.status.setText("Изображение загружено. Откройте bbox editor.")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Загрузка изображения", str(exc))

    def _load_from_camera(self) -> None:
        dlg = CameraCaptureDialog(self)
        if dlg.exec() != dlg.Accepted:
            return
        frame = dlg.captured_frame()
        if frame is None:
            QMessageBox.warning(self, "Камера", "Не удалось получить кадр.")
            return
        self._source_image = frame
        self._crop_image = None
        self.bbox_btn.setEnabled(True)
        self.status.setText("Кадр получен. Откройте bbox editor.")

    def _open_bbox_editor(self) -> None:
        if self._source_image is None:
            return
        dlg = BBoxEditorDialog(self._source_image, self)
        if dlg.exec() != dlg.Accepted:
            return
        self._crop_image = dlg.selected_crop
        if self._crop_image is not None:
            self.status.setText("Crop сохранён. Можно регенерировать фразу и подтвердить блюдо.")

    def _on_regenerate_phrase(self) -> None:
        if self._crop_image is None:
            QMessageBox.warning(self, "Фраза", "Сначала выделите bbox/crop.")
            return
        name = self.name_input.text().strip()
        category = self.category_combo.currentText()
        if not name:
            QMessageBox.warning(self, "Фраза", "Введите название блюда перед генерацией.")
            return
        self.regenerate_phrase_requested.emit(self._crop_image, name, category)

    def _on_confirm(self) -> None:
        if self._crop_image is None:
            QMessageBox.warning(self, "Сохранение", "Сначала выделите bbox/crop.")
            return
        name = self.name_input.text().strip()
        phrase = self.phrase_text.toPlainText().strip()
        category = self.category_combo.currentText()
        if not name:
            QMessageBox.warning(self, "Сохранение", "Название блюда обязательно.")
            return

        payload = {
            "name": name,
            "category": category,
            "crop_image": self._crop_image,
            "phrase": phrase if phrase else None,
        }
        self.confirm_dish_requested.emit(payload)

    def set_phrase(self, text: str) -> None:
        self.phrase_text.setPlainText(text)

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def reset_form(self) -> None:
        self._source_image = None
        self._crop_image = None
        self.name_input.clear()
        self.phrase_text.clear()
        self.category_combo.setCurrentIndex(0)
        self.bbox_btn.setEnabled(False)
        self.status.setText("Сначала загрузите изображение и выделите bbox.")
