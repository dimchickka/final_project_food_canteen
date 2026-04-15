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
        self._index_builder = MenuIndexBuilder(self._model_registry.get_clip())
        self._menu_repository = MenuRepository(index_builder=self._index_builder)
        self._today_menu_service = TodayMenuService(index_builder=self._index_builder)
        self._phrase_regenerator = PhraseRegenerator(
            menu_root="data/menu/global",
            model_registry=self._model_registry,
            index_builder=self._index_builder,
        )
        self._orchestrator = RecognitionOrchestrator(
            model_registry=self._model_registry,
            today_menu_root="data/menu/today",
        )

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
        worker = CreateDishWorker(self._menu_repository, payload)
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
        worker = UpdateDishWorker(self._menu_repository, category, slug_or_name, {"is_active": is_active})
        worker.progress.connect(self.operation_progress)
        worker.error.connect(self.operation_error)
        worker.result.connect(on_result)
        self._runner.run(worker)

    def confirm_phrase(self, dish_dir: str | Path, phrase: str, on_result: Callable[[object], None]) -> None:
        worker = ConfirmPhraseWorker(self._phrase_regenerator, dish_dir, phrase)
        worker.progress.connect(self.operation_progress)
        worker.error.connect(self.operation_error)
        worker.result.connect(on_result)
        self._runner.run(worker)

    def load_global_menu(self, category: str | None, on_result: Callable[[object], None]) -> None:
        if category and category != "all":
            task = WorkerTask(fn=self._menu_repository.list_by_category, kwargs={"category": category, "include_inactive": True})
        else:
            task = WorkerTask(fn=self._load_all_categories, kwargs={})
        worker = QueryMenuWorker(task)
        worker.error.connect(self.operation_error)
        worker.result.connect(on_result)
        self._runner.run(worker)

    def search_global_menu(self, query: str, category: str | None, on_result: Callable[[object], None]) -> None:
        kwargs = {"query": query, "include_inactive": True}
        if category and category != "all":
            kwargs["category"] = category
        worker = QueryMenuWorker(WorkerTask(fn=self._menu_repository.search_dishes, kwargs=kwargs))
        worker.error.connect(self.operation_error)
        worker.result.connect(on_result)
        self._runner.run(worker)

    def load_today_menu(self, category: str | None, on_result: Callable[[object], None]) -> None:
        kwargs: dict[str, Any] = {}
        if category and category != "all":
            kwargs["category"] = category
        worker = QueryMenuWorker(WorkerTask(fn=self._today_menu_service.list_today_dishes, kwargs=kwargs))
        worker.error.connect(self.operation_error)
        worker.result.connect(on_result)
        self._runner.run(worker)

    def set_today_menu(self, mapping: dict[str, list[str]], on_result: Callable[[object], None]) -> None:
        worker = SetTodayMenuWorker(self._today_menu_service, mapping)
        worker.progress.connect(self.operation_progress)
        worker.error.connect(self.operation_error)
        worker.result.connect(on_result)
        self._runner.run(worker)

    def _load_all_categories(self) -> list[Any]:
        rows: list[Any] = []
        for category in ("beverage", "soup", "portioned", "garnish", "meat", "sauce"):
            rows.extend(self._menu_repository.list_by_category(category, include_inactive=True))
        return rows
