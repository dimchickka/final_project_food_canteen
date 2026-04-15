"""Recognition state screen with spinner/status and non-blocking UX."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from gui.widgets.loading_overlay import LoadingOverlay


class RecognitionScreen(QWidget):
    """Simple processing screen used while pipeline worker is running."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignCenter)

        self.title = QLabel("Распознавание подноса")
        self.title.setStyleSheet("font-size: 28px; font-weight: 700;")
        self.status = QLabel("Ожидание запуска...")
        self.status.setStyleSheet("font-size: 16px; color: #666;")

        root.addWidget(self.title, alignment=Qt.AlignCenter)
        root.addSpacing(10)
        root.addWidget(self.status, alignment=Qt.AlignCenter)

        self.overlay = LoadingOverlay(parent=self)

    def start_processing(self, message: str = "Идёт распознавание...") -> None:
        self.status.setText(message)
        self.overlay.show_loading(message)

    def set_status(self, message: str) -> None:
        self.status.setText(message)
        self.overlay.set_message(message)

    def stop_processing(self) -> None:
        self.overlay.hide_loading()

    def show_error(self, text: str) -> None:
        self.status.setText(f"Ошибка: {text}")
        self.overlay.hide_loading()
