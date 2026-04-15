"""AppController is the GUI-to-core glue layer.

It coordinates services/workers and exposes simple methods for screens/main window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QObject, Signal

from core.image_ops.image_loader import ImageLoader
from core.menu.menu_index_builder import MenuIndexBuilder
from core.menu.menu_repository import MenuRepository
from core.menu.phrase_regenerator import PhraseRegenerator
from core.menu.today_menu_service import TodayMenuService
from core.models.model_registry import get_model_registry
from core.pipeline.recognition_orchestrator import RecognitionOrchestrator
from gui.workers import (
    ConfirmPhraseWorker,
    CreateDishWorker,
    ModelWarmupWorker,
    PhraseGenerationWorker,
    QueryMenuWorker,
    RecognitionWorker,
    SetTodayMenuWorker,
    UpdateDishWorker,
    WorkerRunner,
    WorkerTask,
)


class AppController(QObject):
    """Central app orchestration facade used by MainWindow and screens."""

    recognition_progress = Signal(str)
    recognition_finished = Signal(object)
    operation_error = Signal(str)
    operation_progress = Signal(str)
    menu_loaded = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._runner = WorkerRunner()
        self._models_ready = False

        self._model_registry = get_model_registry()

        self._global_menu_root = Path("data/menu/global")
        self._today_menu_root = Path("data/menu/today")
        self._heads_root = Path("data/heads")
        self._first_head_descriptions = self._heads_root / "first_head" / "descriptions.txt"
        self._second_head_descriptions = self._heads_root / "second_head" / "descriptions.txt"

        # Heavy CLIP-dependent services are lazy to avoid blocking UI in simple navigation flows.
        self._index_builder: MenuIndexBuilder | None = None
        self._menu_repository: MenuRepository | None = None
        self._today_menu_service: TodayMenuService | None = None
        self._phrase_regenerator: PhraseRegenerator | None = None
        self._orchestrator = RecognitionOrchestrator(
            model_registry=self._model_registry,
            today_menu_root=self._today_menu_root,
        )

    # ---------------------------
    # Preflight / readiness layer
    # ---------------------------
    def get_system_status(self) -> dict[str, Any]:
        """Snapshot of GUI-level readiness checks used before risky user actions."""
        heads_ready, heads_reason = self.check_heads_ready()
        today_ready, today_reason = self.check_today_menu_ready()
        recognition_ready, recognition_reason = self.check_recognition_ready()
        menu_ready, menu_reason = self.check_menu_admin_ready()
        return {
            "heads_ready": heads_ready,
            "heads_reason": heads_reason,
            "today_menu_ready": today_ready,
            "today_menu_reason": today_reason,
            "menu_admin_ready": menu_ready,
            "menu_admin_reason": menu_reason,
            "recognition_ready": recognition_ready,
            "recognition_reason": recognition_reason,
        }

    def check_heads_ready(self) -> tuple[bool, str]:
        """Recognition requires human-prepared class descriptions for both heads."""
        missing: list[str] = []
        for path in (self._first_head_descriptions, self._second_head_descriptions):
            if not path.exists() or not path.is_file():
                missing.append(str(path))

        if missing:
            return False, "Не подготовлены описания голов классификации (descriptions.txt)."
        return True, "Головы классификации подготовлены."

    def check_today_menu_ready(self) -> tuple[bool, str]:
        """Recognition should not start if today menu is absent/empty."""
        if not self._today_menu_root.exists() or not self._today_menu_root.is_dir():
            return False, "Today menu ещё не создан."

        if self._count_today_dishes() == 0:
            return False, "Today menu пуст — сначала добавьте блюда в меню сегодняшнего дня."

        return True, "Today menu готов."

    def check_menu_admin_ready(self) -> tuple[bool, str]:
        """Admin/menu screens must stay available even if menu is currently empty."""
        try:
            self._global_menu_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            return False, f"Не удалось подготовить папку global menu: {exc}"
        return True, "Menu/admin workflow доступен."

    def check_recognition_ready(self) -> tuple[bool, str]:
        """Preflight gate to avoid launching heavy pipeline when prerequisites are missing."""
        today_ready, today_reason = self.check_today_menu_ready()
        if not today_ready:
            return False, f"Распознавание пока недоступно. {today_reason}"

        heads_ready, heads_reason = self.check_heads_ready()
        if not heads_ready:
            return False, f"Распознавание пока недоступно. {heads_reason}"

        return True, "Система готова к распознаванию."

    def _count_today_dishes(self) -> int:
        count = 0
        for category_dir in self._today_menu_root.iterdir() if self._today_menu_root.exists() else []:
            if not category_dir.is_dir():
                continue
            for dish_dir in category_dir.iterdir():
                if dish_dir.is_dir() and (dish_dir / "meta.json").exists():
                    count += 1
        return count

    # -----------------------
    # Lazy heavy dependencies
    # -----------------------
    def _get_index_builder(self) -> MenuIndexBuilder:
        if self._index_builder is None:
            # Delayed CLIP adapter access: keep startup/light UI flows responsive.
            self._index_builder = MenuIndexBuilder(self._model_registry.get_clip())
        return self._index_builder

    def _get_menu_repository(self) -> MenuRepository:
        if self._menu_repository is None:
            # Repository itself is cheap; heavy builder is still lazy and only created on first need.
            self._menu_repository = MenuRepository(index_builder=self._get_index_builder())
        return self._menu_repository

    def _get_today_menu_service(self) -> TodayMenuService:
        if self._today_menu_service is None:
            self._today_menu_service = TodayMenuService(index_builder=self._get_index_builder())
        return self._today_menu_service

    def _get_phrase_regenerator(self) -> PhraseRegenerator:
        if self._phrase_regenerator is None:
            self._phrase_regenerator = PhraseRegenerator(
                menu_root=self._global_menu_root,
                model_registry=self._model_registry,
                index_builder=self._get_index_builder(),
            )
        return self._phrase_regenerator

    def ensure_models_ready_if_needed(self, on_done: Callable[[bool], None] | None = None) -> None:
        if self._models_ready:
            if on_done:
                on_done(True)
            return

        worker = ModelWarmupWorker(warmup_fn=self._warmup_models)
        worker.progress.connect(self.operation_progress)
        worker.error.connect(self.operation_error)
        if on_done:
            worker.result.connect(on_done)
        self._runner.run(worker)

    def _warmup_models(self) -> None:
        self._model_registry.load_all()
        self._model_registry.warmup()
        self._models_ready = True

    def start_recognition_from_file(self, path: str | Path) -> None:
        image = ImageLoader.ensure_image_loaded(path)
        self.start_recognition_from_camera_frame(image)

    def start_recognition_from_camera_frame(self, frame: np.ndarray) -> None:
        worker = RecognitionWorker(self._orchestrator, frame)
        worker.progress.connect(self.recognition_progress)
        worker.error.connect(self.operation_error)
        worker.result.connect(self.recognition_finished)
        self._runner.run(worker)

    def regenerate_phrase(self, image: np.ndarray, dish_name: str, category: str, on_result: Callable[[str], None]) -> None:
        qwen = self._model_registry.get_qwen()

        def _fn(**kwargs: Any) -> str:
            return qwen.generate_short_dish_phrase(**kwargs)

        worker = PhraseGenerationWorker(
            WorkerTask(fn=_fn, kwargs={"image": image, "dish_name": dish_name, "category": category})
        )
        worker.progress.connect(self.operation_progress)
        worker.error.connect(self.operation_error)
        worker.result.connect(on_result)
        self._runner.run(worker)

    def create_dish(self, payload: dict[str, Any], on_result: Callable[[object], None]) -> None:
        try:
            repository = self._get_menu_repository()
        except Exception as exc:  # noqa: BLE001
            self.operation_error.emit(f"Не удалось инициализировать global menu сервис: {exc}")
            return
        worker = CreateDishWorker(repository, payload)
        worker.progress.connect(self.operation_progress)
        worker.error.connect(self.operation_error)
        worker.result.connect(on_result)
        self._runner.run(worker)

    def set_dish_active(
        self,
        category: str,
        slug_or_name: str,
        is_active: bool,
        on_result: Callable[[object], None],
    ) -> None:
        try:
            repository = self._get_menu_repository()
        except Exception as exc:  # noqa: BLE001
            self.operation_error.emit(f"Не удалось инициализировать global menu сервис: {exc}")
            return
        worker = UpdateDishWorker(repository, category, slug_or_name, {"is_active": is_active})
        worker.progress.connect(self.operation_progress)
        worker.error.connect(self.operation_error)
        worker.result.connect(on_result)
        self._runner.run(worker)

    def confirm_phrase(self, dish_dir: str | Path, phrase: str, on_result: Callable[[object], None]) -> None:
        try:
            regenerator = self._get_phrase_regenerator()
        except Exception as exc:  # noqa: BLE001
            self.operation_error.emit(f"Не удалось инициализировать phrase сервис: {exc}")
            return

        worker = ConfirmPhraseWorker(regenerator, dish_dir, phrase)
        worker.progress.connect(self.operation_progress)
        worker.error.connect(self.operation_error)
        worker.result.connect(on_result)
        self._runner.run(worker)

    def load_global_menu(self, category: str | None, on_result: Callable[[object], None]) -> None:
        try:
            repository = self._get_menu_repository()
            if category and category != "all":
                task = WorkerTask(
                    fn=repository.list_by_category,
                    kwargs={"category": category, "include_inactive": True},
                )
            else:
                task = WorkerTask(fn=self._load_all_categories, kwargs={})
        except Exception as exc:  # noqa: BLE001
            self.operation_error.emit(f"Не удалось открыть global menu: {exc}")
            on_result([])
            return

        worker = QueryMenuWorker(task)
        worker.error.connect(self.operation_error)
        worker.result.connect(on_result)
        self._runner.run(worker)

    def search_global_menu(self, query: str, category: str | None, on_result: Callable[[object], None]) -> None:
        kwargs = {"query": query, "include_inactive": True}
        if category and category != "all":
            kwargs["category"] = category

        try:
            repository = self._get_menu_repository()
        except Exception as exc:  # noqa: BLE001
            self.operation_error.emit(f"Не удалось выполнить поиск в global menu: {exc}")
            on_result([])
            return

        worker = QueryMenuWorker(WorkerTask(fn=repository.search_dishes, kwargs=kwargs))
        worker.error.connect(self.operation_error)
        worker.result.connect(on_result)
        self._runner.run(worker)

    def load_today_menu(self, category: str | None, on_result: Callable[[object], None]) -> None:
        kwargs: dict[str, Any] = {}
        if category and category != "all":
            kwargs["category"] = category

        try:
            service = self._get_today_menu_service()
        except Exception as exc:  # noqa: BLE001
            self.operation_error.emit(f"Не удалось открыть today menu: {exc}")
            on_result([])
            return

        worker = QueryMenuWorker(WorkerTask(fn=service.list_today_dishes, kwargs=kwargs))
        worker.error.connect(self.operation_error)
        worker.result.connect(on_result)
        self._runner.run(worker)

    def set_today_menu(self, mapping: dict[str, list[str]], on_result: Callable[[object], None]) -> None:
        try:
            service = self._get_today_menu_service()
        except Exception as exc:  # noqa: BLE001
            self.operation_error.emit(f"Не удалось инициализировать today menu сервис: {exc}")
            return

        worker = SetTodayMenuWorker(service, mapping)
        worker.progress.connect(self.operation_progress)
        worker.error.connect(self.operation_error)
        worker.result.connect(on_result)
        self._runner.run(worker)

    def _load_all_categories(self) -> list[Any]:
        rows: list[Any] = []
        repository = self._get_menu_repository()
        for category in ("beverage", "soup", "portioned", "garnish", "meat", "sauce"):
            rows.extend(repository.list_by_category(category, include_inactive=True))
        return rows
