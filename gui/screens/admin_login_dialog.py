"""Admin login dialog with password validation."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QFormLayout, QHBoxLayout, QLineEdit, QMessageBox, QPushButton, QVBoxLayout

from config.settings import ADMIN_PASSWORD


class AdminLoginDialog(QDialog):
    """Asks for admin password and blocks until valid or cancelled."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Вход в админ-панель")

        root = QVBoxLayout(self)
        form = QFormLayout()

        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        form.addRow("Пароль:", self.password_input)

        buttons = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self._try_accept)
        cancel_btn.clicked.connect(self.reject)

        buttons.addStretch(1)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)

        root.addLayout(form)
        root.addLayout(buttons)

    def _try_accept(self) -> None:
        if self.password_input.text() != ADMIN_PASSWORD:
            QMessageBox.warning(self, "Неверный пароль", "Пароль неверный. Попробуйте снова.")
            self.password_input.selectAll()
            self.password_input.setFocus()
            return
        self.accept()
