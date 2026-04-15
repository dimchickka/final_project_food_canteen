"""Application entrypoint for the PySide6 desktop GUI."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from config.ui_settings import WINDOW_TITLE
from gui.main_window import MainWindow


def main() -> int:
    """Create QApplication + main window and start Qt event loop."""
    app = QApplication(sys.argv)
    app.setApplicationName(WINDOW_TITLE)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
