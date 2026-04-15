"""Today-menu composition screen by category with dual-list UX."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config.constants import DISH_CATEGORIES


class TodayMenuScreen(QWidget):
    """Lets admin pick active global dishes included in today's menu."""

    back_requested = Signal()
    load_for_category_requested = Signal(str)
    save_requested = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._selected: dict[str, set[str]] = {cat: set() for cat in DISH_CATEGORIES}
        self._current_rows: list[dict[str, Any]] = []

        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.category_combo = QComboBox()
        self.category_combo.addItems(list(DISH_CATEGORIES))
        self.reload_btn = QPushButton("Загрузить категорию")
        top.addWidget(QLabel("Категория:"))
        top.addWidget(self.category_combo)
        top.addWidget(self.reload_btn)
        top.addStretch(1)

        lists = QHBoxLayout()
        self.available = QListWidget()
        self.selected = QListWidget()

        middle = QVBoxLayout()
        self.add_btn = QPushButton(">>")
        self.remove_btn = QPushButton("<<")
        middle.addStretch(1)
        middle.addWidget(self.add_btn)
        middle.addWidget(self.remove_btn)
        middle.addStretch(1)

        lists.addWidget(self.available, stretch=1)
        lists.addLayout(middle)
        lists.addWidget(self.selected, stretch=1)

        bottom = QHBoxLayout()
        self.save_btn = QPushButton("Сохранить today menu")
        self.back_btn = QPushButton("Назад в админ-панель")
        bottom.addWidget(self.save_btn)
        bottom.addStretch(1)
        bottom.addWidget(self.back_btn)

        self.status = QLabel("Выберите категорию и загрузите блюда.")

        root.addLayout(top)
        root.addLayout(lists, stretch=1)
        root.addWidget(self.status)
        root.addLayout(bottom)

        self.reload_btn.clicked.connect(self._reload)
        self.category_combo.currentTextChanged.connect(lambda _: self._reload())
        self.add_btn.clicked.connect(self._add_selected)
        self.remove_btn.clicked.connect(self._remove_selected)
        self.save_btn.clicked.connect(self._save)
        self.back_btn.clicked.connect(self.back_requested)

    def _reload(self) -> None:
        self.load_for_category_requested.emit(self.category_combo.currentText())

    def set_available_rows(self, rows: list[dict[str, Any]]) -> None:
        category = self.category_combo.currentText()
        self._current_rows = rows

        self.available.clear()
        self.selected.clear()

        selected_slugs = self._selected[category]
        for row in rows:
            slug = str(row.get("slug", ""))
            name = str(row.get("name", slug))
            item = QListWidgetItem(f"{name} ({slug})")
            item.setData(256, slug)
            if slug in selected_slugs:
                self.selected.addItem(item)
            else:
                self.available.addItem(item)
        self.status.setText(f"Категория {category}: доступно {self.available.count()}, выбрано {self.selected.count()}.")

    def _add_selected(self) -> None:
        category = self.category_combo.currentText()
        for item in self.available.selectedItems():
            slug = str(item.data(256))
            self._selected[category].add(slug)
        self.set_available_rows(self._current_rows)

    def _remove_selected(self) -> None:
        category = self.category_combo.currentText()
        for item in self.selected.selectedItems():
            slug = str(item.data(256))
            self._selected[category].discard(slug)
        self.set_available_rows(self._current_rows)

    def _save(self) -> None:
        payload = {cat: sorted(list(values)) for cat, values in self._selected.items()}
        self.save_requested.emit(payload)

    def set_today_rows(self, rows: list[dict[str, Any]]) -> None:
        grouped: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            grouped[str(row.get("category", ""))].add(str(row.get("slug", "")))
        for cat in DISH_CATEGORIES:
            self._selected[cat] = grouped.get(cat, set())
        # Today rows may arrive after available rows; force refresh of current category view.
        if self._current_rows:
            self.set_available_rows(self._current_rows)

    def notify(self, text: str) -> None:
        QMessageBox.information(self, "Today menu", text)
