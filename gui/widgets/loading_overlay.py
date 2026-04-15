"""Reusable semi-transparent loading overlay for long operations."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget


class LoadingOverlay(QWidget):
    """Overlay widget shown above content while async worker is running."""

    def __init__(self, message: str = "Идёт обработка...", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 90);")

        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignCenter)

        self._label = QLabel(message)
        self._label.setStyleSheet("color: white; font-size: 16px; font-weight: 600;")
        self._spinner = QProgressBar()
        self._spinner.setRange(0, 0)
        self._spinner.setFixedWidth(260)

        root.addWidget(self._label, alignment=Qt.AlignCenter)
        root.addWidget(self._spinner, alignment=Qt.AlignCenter)
        self.hide()

    def set_message(self, message: str) -> None:
        self._label.setText(message)

    def show_loading(self, message: str | None = None) -> None:
        if message:
            self.set_message(message)
        self.setGeometry(self.parentWidget().rect() if self.parentWidget() else self.rect())
        self.show()
        self.raise_()

    def hide_loading(self) -> None:
        self.hide()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        if self.parentWidget() is not None:
            self.setGeometry(self.parentWidget().rect())
        super().resizeEvent(event)
