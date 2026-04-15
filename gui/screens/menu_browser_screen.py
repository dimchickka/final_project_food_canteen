"""Global menu browsing screen with search/filter and activation toggle."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QMessageBox, QPushButton, QVBoxLayout, QWidget

from gui.widgets.searchable_dish_list import SearchableDishList


class MenuBrowserScreen(QWidget):
    """Admin view of global menu entries."""

    back_requested = Signal()
    query_changed = Signal(str, str)
    refresh_requested = Signal()
    toggle_requested = Signal(str, str, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)

        self.list_widget = SearchableDishList(with_toggle=True)
        self.back_btn = QPushButton("Назад в админ-панель")

        self.list_widget.filters_changed.connect(self.query_changed)
        self.list_widget.refresh_requested.connect(self.refresh_requested)
        self.list_widget.toggle_requested.connect(self.toggle_requested)
        self.back_btn.clicked.connect(self.back_requested)

        root.addWidget(self.list_widget, stretch=1)
        root.addWidget(self.back_btn)

    def set_rows(self, rows: list[dict[str, Any]]) -> None:
        self.list_widget.set_rows(rows)

    def notify(self, text: str) -> None:
        QMessageBox.information(self, "Global menu", text)
