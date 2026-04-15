"""Admin flow for adding new dish card from photo/camera + bbox + phrase."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QTimer, Signal
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
    QDialog,
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
        self._phrase_generation_in_progress = False
        self._slow_feedback_timer = QTimer(self)
        self._slow_feedback_timer.setSingleShot(True)
        self._slow_feedback_timer.timeout.connect(self._on_phrase_generation_slow)
        self._slow_warning_shown = False

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
            self.phrase_text.clear()
            self.bbox_btn.setEnabled(True)
            self.status.setText("Изображение загружено. Откройте bbox editor.")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Загрузка изображения", str(exc))

    def _load_from_camera(self) -> None:
        # Ошибка инициализации камеры не должна ронять GUI.
        try:
            dlg = CameraCaptureDialog(self)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Камера", str(exc))
            return

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        captured_frame = dlg.captured_frame
        frame = captured_frame() if callable(captured_frame) else captured_frame
        if frame is None:
            QMessageBox.warning(self, "Камера", "Не удалось получить кадр.")
            return

        self._source_image = frame
        self._crop_image = None
        self.phrase_text.clear()
        self.bbox_btn.setEnabled(True)
        self.status.setText("Кадр получен. Откройте bbox editor.")

    def _open_bbox_editor(self) -> None:
        if self._source_image is None:
            QMessageBox.warning(self, "BBox", "Сначала загрузите исходное изображение.")
            return

        dlg = BBoxEditorDialog(self._source_image, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # Сохраняем crop только после явного подтверждения в bbox editor.
        self._crop_image = dlg.selected_crop
        if self._crop_image is None:
            QMessageBox.warning(self, "BBox", "Не удалось получить crop из bbox editor.")
            return

        self.phrase_text.clear()
        self.status.setText("Crop сохранён. Можно регенерировать фразу и подтвердить блюдо.")

    def _validate_for_phrase_regeneration(self) -> tuple[bool, str, str, str]:
        if self._crop_image is None:
            # Deliberately block heavy regeneration until required state is complete.
            return False, "Сначала выделите bbox/crop.", "", ""
        name = self.name_input.text().strip()
        if not name:
            return False, "Сначала заполните название блюда.", "", ""
        category = self.category_combo.currentText().strip()
        if not category:
            return False, "Сначала выберите категорию блюда.", "", ""
        return True, "", name, category

    def _validate_for_confirm(self) -> tuple[bool, str, str, str, str]:
        if self._source_image is None:
            # Deliberately block heavy create action until required state is complete.
            return False, "Сначала загрузите исходное изображение блюда.", "", "", ""
        if self._crop_image is None:
            return False, "Сначала выделите bbox/crop.", "", "", ""

        name = self.name_input.text().strip()
        if not name:
            return False, "Название блюда обязательно.", "", "", ""

        category = self.category_combo.currentText().strip()
        if not category:
            return False, "Категория блюда обязательна.", "", "", ""

        phrase = self.phrase_text.toPlainText().strip()
        if not phrase:
            return (
                False,
                "Фраза пустая. Нажмите «Регенерировать фразу», затем повторите подтверждение.",
                "",
                "",
                "",
            )

        return True, "", name, category, phrase

    def _on_regenerate_phrase(self) -> None:
        is_valid, error_text, name, category = self._validate_for_phrase_regeneration()
        if not is_valid:
            QMessageBox.warning(self, "Фраза", error_text)
            self.status.setText(error_text)
            return

        self.set_phrase_generation_in_progress(True)
        # Initial status is immediately replaced by worker-reported granular stages.
        self.status.setText("Подготавливаю Qwen...")
        self.regenerate_phrase_requested.emit(self._crop_image, name, category)

    def _on_confirm(self) -> None:
        if self._phrase_generation_in_progress:
            self.status.setText("Дождитесь завершения генерации фразы.")
            return
        is_valid, error_text, name, category, phrase = self._validate_for_confirm()
        if not is_valid:
            QMessageBox.warning(self, "Сохранение", error_text)
            self.status.setText(error_text)
            return

        payload = {
            "name": name,
            "category": category,
            "crop_image": self._crop_image,
            "phrase": phrase,
        }
        self.status.setText("Сохраняю блюдо...")
        self.confirm_dish_requested.emit(payload)

    def set_phrase(self, text: str) -> None:
        self.phrase_text.setPlainText(text)
        self.status.setText("Фраза успешно сгенерирована.")

    def set_phrase_generation_in_progress(self, in_progress: bool) -> None:
        # Block repeated regenerate clicks to avoid parallel phrase-generation workers.
        self._phrase_generation_in_progress = in_progress
        self.regen_btn.setEnabled(not in_progress)
        self.confirm_btn.setEnabled(not in_progress)
        if in_progress:
            self._slow_warning_shown = False
            self._slow_feedback_timer.start(12000)
        else:
            self._slow_feedback_timer.stop()

    def is_phrase_generation_in_progress(self) -> bool:
        return self._phrase_generation_in_progress

    def finish_phrase_generation(self, success: bool) -> None:
        self.set_phrase_generation_in_progress(False)
        if success:
            self.status.setText("Фраза успешно сгенерирована.")
            return
        self.status.setText("Не удалось сгенерировать фразу.")

    def _on_phrase_generation_slow(self) -> None:
        if self._phrase_generation_in_progress and not self._slow_warning_shown:
            self._slow_warning_shown = True
            self.status.setText(
                "Подготовка модели занимает больше обычного. "
                "Проверьте путь к модели Qwen и доступную память."
            )

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def reset_form(self) -> None:
        self._source_image = None
        self._crop_image = None
        self.set_phrase_generation_in_progress(False)
        self.name_input.clear()
        self.phrase_text.clear()
        self.category_combo.setCurrentIndex(0)
        self.bbox_btn.setEnabled(False)
        self.status.setText("Сначала загрузите изображение и выделите bbox.")
