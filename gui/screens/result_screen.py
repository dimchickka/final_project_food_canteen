"""Recognition result screen: dish names/counts + elapsed time."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


class ResultScreen(QWidget):
    """Shows concise receipt-like recognition output."""

    back_home_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)

        self.title = QLabel("Результат распознавания")
        self.title.setStyleSheet("font-size: 24px; font-weight: 700;")

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Блюдо", "Кол-во"])
        self.table.horizontalHeader().setStretchLastSection(True)

        self.time_label = QLabel("Время: —")

        self.back_btn = QPushButton("Вернуться на главный экран")
        self.back_btn.clicked.connect(self.back_home_requested)

        root.addWidget(self.title)
        root.addWidget(self.table, stretch=1)
        root.addWidget(self.time_label, alignment=Qt.AlignRight)
        root.addWidget(self.back_btn)

    def set_result(self, rows: list[tuple[str, int]], total_time_ms: float | None) -> None:
        self.table.setRowCount(len(rows))
        for i, (dish, count) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(dish))
            self.table.setItem(i, 1, QTableWidgetItem(str(count)))
        if total_time_ms is None:
            self.time_label.setText("Время: —")
        else:
            self.time_label.setText(f"Время: {total_time_ms / 1000:.2f} сек")
