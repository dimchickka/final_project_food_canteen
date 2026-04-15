"""Hidden click-zone widget for secret admin entry in production mode."""

from __future__ import annotations

from collections import deque

from PySide6.QtCore import QDateTime, Qt, Signal
from PySide6.QtWidgets import QWidget

from config.ui_settings import SECRET_ADMIN_CLICKS, SECRET_CLICK_TIMEOUT_MS


class SecretAdminClickZone(QWidget):
    """Tracks rapid clicks and emits when SECRET_ADMIN_CLICKS reached in timeout."""

    secret_triggered = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._clicks = deque(maxlen=SECRET_ADMIN_CLICKS)
        self.setFixedHeight(64)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("background: transparent;")

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.LeftButton:
            return
        now = int(QDateTime.currentMSecsSinceEpoch())
        self._clicks.append(now)

        if len(self._clicks) >= SECRET_ADMIN_CLICKS:
            if now - self._clicks[0] <= SECRET_CLICK_TIMEOUT_MS:
                self._clicks.clear()
                self.secret_triggered.emit()
