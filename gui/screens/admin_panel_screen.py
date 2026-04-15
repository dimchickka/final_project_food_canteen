"""Admin dashboard screen with main management entry buttons."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class AdminPanelScreen(QWidget):
    """Entry point to admin workflows (add dish, browse menu, today menu)."""

    add_dish_requested = Signal()
    browse_menu_requested = Signal()
    today_menu_requested = Signal()
    exit_admin_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)

        title = QLabel("Админ-панель")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")

        self.status = QLabel("")
        self.status.setStyleSheet("color: #666;")

        self.add_btn = QPushButton("Добавить блюдо")
        self.menu_btn = QPushButton("Global menu")
        self.today_btn = QPushButton("Today menu")
        self.exit_btn = QPushButton("Выйти из админки")

        self.add_btn.clicked.connect(self.add_dish_requested)
        self.menu_btn.clicked.connect(self.browse_menu_requested)
        self.today_btn.clicked.connect(self.today_menu_requested)
        self.exit_btn.clicked.connect(self.exit_admin_requested)

        for btn in (self.add_btn, self.menu_btn, self.today_btn, self.exit_btn):
            btn.setMinimumHeight(56)

        root.addWidget(title)
        root.addSpacing(8)
        root.addWidget(self.status)
        root.addWidget(self.add_btn)
        root.addWidget(self.menu_btn)
        root.addWidget(self.today_btn)
        root.addStretch(1)
        root.addWidget(self.exit_btn)

    def set_status(self, text: str) -> None:
        self.status.setText(text)
