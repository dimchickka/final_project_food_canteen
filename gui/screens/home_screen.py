"""Home screen with start-recognition CTA and admin entry controls."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from config.settings import TEST_RECOGNITION
from config.ui_settings import SHOW_ADMIN_BUTTON_IN_TEST
from gui.widgets.animated_ready_label import AnimatedReadyLabel
from gui.widgets.secret_admin_click_zone import SecretAdminClickZone


class HomeScreen(QWidget):
    """Landing screen for cashier operator."""

    start_recognition_requested = Signal()
    admin_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addStretch(1)

        self._secret_zone: SecretAdminClickZone | None = None
        if TEST_RECOGNITION and SHOW_ADMIN_BUTTON_IN_TEST:
            self.admin_btn = QPushButton("Панель админа")
            self.admin_btn.clicked.connect(self.admin_requested)
            top.addWidget(self.admin_btn, alignment=Qt.AlignRight)
        else:
            self._secret_zone = SecretAdminClickZone()
            self._secret_zone.secret_triggered.connect(self.admin_requested)
            top.addWidget(self._secret_zone)

        root.addLayout(top)
        root.addStretch(1)

        self.ready_label = AnimatedReadyLabel()
        self.start_button = QPushButton("Начать распознавание")
        self.start_button.setMinimumHeight(120)
        self.start_button.setStyleSheet("font-size: 28px; font-weight: 700;")
        self.start_button.clicked.connect(self.start_recognition_requested)

        self.status = QLabel("")
        self.status.setStyleSheet("font-size: 14px; color: #666;")
        self.status.setAlignment(Qt.AlignHCenter)
        self.status.hide()

        root.addWidget(self.ready_label, alignment=Qt.AlignHCenter)
        root.addSpacing(16)
        root.addWidget(self.start_button, alignment=Qt.AlignHCenter)
        root.addSpacing(12)
        root.addWidget(self.status, alignment=Qt.AlignHCenter)
        root.addStretch(2)

    def set_status(self, message: str) -> None:
        text = message.strip()
        if text:
            self.status.setText(text)
            self.status.show()
        else:
            self.status.clear()
            self.status.hide()
