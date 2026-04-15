"""Reusable list widget with search + category filter + dish cards."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from config.constants import DISH_CATEGORIES
from gui.widgets.dish_card_widget import DishCardWidget


class SearchableDishList(QWidget):
    """Shared search/filter surface for global and today menu screens."""

    filters_changed = Signal(str, str)
    refresh_requested = Signal()
    toggle_requested = Signal(str, str, bool)

    def __init__(self, parent: QWidget | None = None, with_toggle: bool = False) -> None:
        super().__init__(parent)
        self._with_toggle = with_toggle
        self._cards_layout: QVBoxLayout

        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Поиск по названию...")
        self.category_combo = QComboBox()
        self.category_combo.addItem("all")
        self.category_combo.addItems(list(DISH_CATEGORIES))
        self.refresh_btn = QPushButton("Обновить")

        top.addWidget(QLabel("Поиск:"))
        top.addWidget(self.search_input, stretch=1)
        top.addWidget(QLabel("Категория:"))
        top.addWidget(self.category_combo)
        top.addWidget(self.refresh_btn)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        self._cards_layout = QVBoxLayout(self.container)
        self._cards_layout.addStretch(1)
        self.scroll.setWidget(self.container)

        root.addLayout(top)
        root.addWidget(self.scroll, stretch=1)

        self.search_input.textChanged.connect(self._emit_filters)
        self.category_combo.currentTextChanged.connect(self._emit_filters)
        self.refresh_btn.clicked.connect(self.refresh_requested)

    def _emit_filters(self) -> None:
        self.filters_changed.emit(self.search_input.text().strip(), self.category_combo.currentText())

    def set_rows(self, rows: list[dict[str, Any]]) -> None:
        while self._cards_layout.count() > 1:
            item = self._cards_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        for row in rows:
            card = DishCardWidget(row)
            if self._with_toggle:
                card.toggle_btn.clicked.connect(
                    lambda _=False, r=row: self.toggle_requested.emit(
                        str(r.get("category", "")), str(r.get("slug", "")), bool(r.get("is_active", True))
                    )
                )
            else:
                card.toggle_btn.hide()
            self._cards_layout.insertWidget(self._cards_layout.count() - 1, card)
