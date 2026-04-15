"""Decorative animated label for home-screen readiness message."""

from __future__ import annotations

from PySide6.QtCore import Property, QPropertyAnimation, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class AnimatedReadyLabel(QWidget):
    """Renders a flowing gray gradient text with lightweight animation."""

    def __init__(self, text: str = "Готов, когда ты готов!", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = text
        self._shift = 0.0
        self.setMinimumHeight(80)

        self._anim = QPropertyAnimation(self, b"gradientShift", self)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setDuration(2200)
        self._anim.setLoopCount(-1)
        self._anim.start()

    def get_gradient_shift(self) -> float:
        return self._shift

    def set_gradient_shift(self, value: float) -> None:
        self._shift = float(value)
        self.update()

    gradientShift = Property(float, get_gradient_shift, set_gradient_shift)  # type: ignore[assignment]

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        font = self.font()
        font.setPointSize(26)
        font.setBold(True)
        painter.setFont(font)

        rect = self.rect()
        path = QPainterPath()
        margin = 12
        baseline = rect.center().y() + 10
        path.addText(margin, baseline, font, self._text)

        w = max(1, rect.width())
        offset = int(self._shift * w)
        gradient = QLinearGradient(-w + offset, 0, offset, 0)
        gradient.setColorAt(0.0, QColor(110, 110, 110))
        gradient.setColorAt(0.5, QColor(210, 210, 210))
        gradient.setColorAt(1.0, QColor(110, 110, 110))

        painter.setPen(Qt.NoPen)
        painter.setBrush(gradient)
        painter.drawPath(path)

        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(70, 70, 70, 140), 1))
        painter.drawPath(path)
