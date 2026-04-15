"""Main application shell window.

Holds screen stack and delegates heavy business actions to AppController.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QDialog,
)

from config.settings import TEST_RECOGNITION
from config.ui_settings import DEFAULT_WINDOW_HEIGHT, DEFAULT_WINDOW_WIDTH, WINDOW_TITLE
from gui.app_controller import AppController
from gui.screens.add_dish_screen import AddDishScreen
from gui.screens.admin_login_dialog import AdminLoginDialog
from gui.screens.admin_panel_screen import AdminPanelScreen
from gui.screens.camera_capture_dialog import CameraCaptureDialog
from gui.screens.home_screen import HomeScreen
from gui.screens.menu_browser_screen import MenuBrowserScreen
from gui.screens.recognition_screen import RecognitionScreen
from gui.screens.result_screen import ResultScreen
from gui.screens.today_menu_screen import TodayMenuScreen
from gui.workers import aggregate_recognition_rows, normalize_menu_rows


class MainWindow(QMainWindow):
    """Root window with navigation/state transitions across app screens."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

        self.controller = AppController()
        self._build_ui()
        self._connect_controller_signals()

        self.controller.ensure_models_ready_if_needed()

    def _build_ui(self) -> None:
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Main screens.
        self.home_screen = HomeScreen()
        self.recognition_screen = RecognitionScreen()
        self.result_screen = ResultScreen()
        self.admin_panel_screen = AdminPanelScreen()
        self.add_dish_screen = AddDishScreen()
        self.menu_browser_screen = MenuBrowserScreen()
        self.today_menu_screen = TodayMenuScreen()

        for screen in (
            self.home_screen,
            self.recognition_screen,
            self.result_screen,
            self.admin_panel_screen,
            self.add_dish_screen,
            self.menu_browser_screen,
            self.today_menu_screen,
        ):
            self.stack.addWidget(screen)

        # Screen-to-window signal wiring.
        self.home_screen.start_recognition_requested.connect(self._on_start_recognition)
        self.home_screen.admin_requested.connect(self._on_admin_entry)
        self.result_screen.back_home_requested.connect(self._go_home)

        self.admin_panel_screen.add_dish_requested.connect(lambda: self._show(self.add_dish_screen))
        self.admin_panel_screen.browse_menu_requested.connect(self._open_menu_browser)
        self.admin_panel_screen.today_menu_requested.connect(self._open_today_menu)
        self.admin_panel_screen.exit_admin_requested.connect(self._go_home)

        self.add_dish_screen.back_requested.connect(lambda: self._show(self.admin_panel_screen))
        self.add_dish_screen.regenerate_phrase_requested.connect(self._on_regenerate_phrase)
        self.add_dish_screen.confirm_dish_requested.connect(self._on_confirm_dish)

        self.menu_browser_screen.back_requested.connect(lambda: self._show(self.admin_panel_screen))
        self.menu_browser_screen.refresh_requested.connect(self._open_menu_browser)
        self.menu_browser_screen.query_changed.connect(self._on_menu_query_changed)
        self.menu_browser_screen.toggle_requested.connect(self._on_toggle_dish_active)

        self.today_menu_screen.back_requested.connect(lambda: self._show(self.admin_panel_screen))
        self.today_menu_screen.load_for_category_requested.connect(self._load_today_category)
        self.today_menu_screen.save_requested.connect(self._save_today_menu)

        self._show(self.home_screen)

    def _connect_controller_signals(self) -> None:
        # Non-trivial signal path: controller emits status from background threads.
        self.controller.recognition_progress.connect(self.recognition_screen.set_status)
        self.controller.recognition_finished.connect(self._on_recognition_done)
        self.controller.operation_error.connect(self._on_controller_error)
        self.controller.operation_progress.connect(self._on_controller_progress)

    def _show(self, widget) -> None:
        self.stack.setCurrentWidget(widget)

    def _go_home(self) -> None:
        self.recognition_screen.stop_processing()
        self._show(self.home_screen)

    def _on_start_recognition(self) -> None:
        if TEST_RECOGNITION:
            choice, ok = QInputDialog.getItem(
                self,
                "Источник изображения",
                "Выберите источник:",
                ["Камера", "Файл"],
                editable=False,
            )
            if not ok:
                return
            if choice == "Файл":
                self._start_recognition_from_file()
                return
        self._start_recognition_from_camera()

    def _start_recognition_from_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not path:
            return
        try:
            self.recognition_screen.start_processing("Загрузка изображения...")
            self._show(self.recognition_screen)
            self.controller.start_recognition_from_file(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Распознавание", str(exc))
            self._go_home()

    def _start_recognition_from_camera(self) -> None:
        try:
            dialog = CameraCaptureDialog(self)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Камера", str(exc))
            return
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        frame = dialog.captured_frame()
        if frame is None:
            QMessageBox.warning(self, "Камера", "Не удалось получить кадр.")
            return

        self.recognition_screen.start_processing("Распознавание запущено...")
        self._show(self.recognition_screen)
        self.controller.start_recognition_from_camera_frame(frame)

    def _on_recognition_done(self, result) -> None:
        self.recognition_screen.stop_processing()
        if not getattr(result, "success", False):
            reason = getattr(result, "abort_reason", None) or "Неизвестная ошибка"
            QMessageBox.warning(self, "Распознавание", f"Распознавание не завершено: {reason}")
            self._go_home()
            return

        rows = aggregate_recognition_rows(result)
        self.result_screen.set_result(rows, getattr(result, "total_time_ms", None))
        self._show(self.result_screen)

    def _on_admin_entry(self) -> None:
        if TEST_RECOGNITION:
            self._show(self.admin_panel_screen)
            return
        dialog = AdminLoginDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._show(self.admin_panel_screen)

    def _on_regenerate_phrase(self, image, dish_name: str, category: str) -> None:
        self.controller.regenerate_phrase(image, dish_name, category, on_result=self.add_dish_screen.set_phrase)

    def _on_confirm_dish(self, payload: dict) -> None:
        self.controller.create_dish(payload, on_result=self._on_dish_created)

    def _on_dish_created(self, _dish: object) -> None:
        self.add_dish_screen.set_status("Блюдо успешно добавлено.")
        QMessageBox.information(self, "Добавление блюда", "Блюдо сохранено в global menu.")
        self.add_dish_screen.reset_form()

    def _open_menu_browser(self) -> None:
        self._show(self.menu_browser_screen)
        self.controller.load_global_menu(None, on_result=self._set_menu_rows)

    def _on_menu_query_changed(self, query: str, category: str) -> None:
        self.controller.search_global_menu(query=query, category=category, on_result=self._set_menu_rows)

    def _set_menu_rows(self, rows: object) -> None:
        self.menu_browser_screen.set_rows(normalize_menu_rows(rows))

    def _on_toggle_dish_active(self, category: str, slug: str, is_active_now: bool) -> None:
        self.controller.set_dish_active(
            category=category,
            slug_or_name=slug,
            is_active=not is_active_now,
            on_result=lambda _: self._open_menu_browser(),
        )

    def _open_today_menu(self) -> None:
        self._show(self.today_menu_screen)
        category = self.today_menu_screen.category_combo.currentText()
        self._load_today_category(category)
        # Explicitly apply saved today selections after async load so current category is redrawn immediately.
        self.controller.load_today_menu(
            None,
            on_result=lambda rows: self.today_menu_screen.set_today_rows(normalize_menu_rows(rows)),
        )

    def _load_today_category(self, category: str) -> None:
        self.controller.load_global_menu(category, on_result=lambda rows: self.today_menu_screen.set_available_rows(normalize_menu_rows(rows)))

    def _save_today_menu(self, payload: dict) -> None:
        self.controller.set_today_menu(payload, on_result=lambda _: self.today_menu_screen.notify("Today menu сохранено."))

    def _on_controller_progress(self, text: str) -> None:
        if self.stack.currentWidget() is self.recognition_screen:
            self.recognition_screen.set_status(text)
        elif self.stack.currentWidget() is self.add_dish_screen:
            self.add_dish_screen.set_status(text)

    def _on_controller_error(self, text: str) -> None:
        QMessageBox.critical(self, "Ошибка", text)
        if self.stack.currentWidget() is self.recognition_screen:
            self.recognition_screen.show_error(text)
