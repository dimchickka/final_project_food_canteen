"""AppController is the GUI-to-core glue layer.

It coordinates services/workers and exposes simple methods for screens/main window.
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
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
from core.models.qwen_adapter import QwenAdapter
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
    phrase_generation_finished = Signal()
    menu_loaded = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._runner = WorkerRunner()
        self._models_ready = False

        self._model_registry = get_model_registry()
        self._admin_phrase_log_path = Path("data/logs/admin_phrase_generation.log")

        self._global_menu_root = Path("data/menu/global")
        self._today_menu_root = Path("data/menu/today")
        self._heads_root = Path("data/heads")
        self._first_head_descriptions = self._heads_root / "first_head" / "descriptions.txt"
        self._second_head_descriptions = self._heads_root / "second_head" / "descriptions.txt"

        # Heavy model/index path stays isolated from lightweight menu reading/listing path.
        self._index_builder: MenuIndexBuilder | None = None
        self._menu_repository_light: MenuRepository | None = None
        self._menu_repository_heavy: MenuRepository | None = None
        self._today_menu_service_light: TodayMenuService | None = None
        self._today_menu_service_heavy: TodayMenuService | None = None
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
            # CLIP is created only for heavy writes/rebuilds, never for simple read/list flows.
            self._index_builder = MenuIndexBuilder(self._model_registry.get_clip())
        return self._index_builder

    def _get_menu_repository_light(self) -> MenuRepository:
        if self._menu_repository_light is None:
            # Listing/search reads meta.json from filesystem and should not depend on CLIP init.
            self._menu_repository_light = MenuRepository(menu_root=self._global_menu_root, index_builder=None)
        return self._menu_repository_light

    def _require_index_builder(self) -> MenuIndexBuilder:
        """Controller-level centralized heavy service access for deterministic entrypoints."""
        try:
            return self._get_index_builder()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Не удалось инициализировать индексатор меню: {exc}") from exc

    def _require_heavy_menu_repository(self) -> MenuRepository:
        """Controller-level centralized heavy service access for deterministic entrypoints."""
        if self._menu_repository_heavy is None:
            self._menu_repository_heavy = MenuRepository(
                menu_root=self._global_menu_root,
                index_builder=self._require_index_builder(),
            )
        return self._menu_repository_heavy

    def _get_today_menu_service_light(self) -> TodayMenuService:
        if self._today_menu_service_light is None:
            # Today-menu listing is lightweight and must not trigger CLIP load.
            self._today_menu_service_light = TodayMenuService(
                global_menu_root=self._global_menu_root,
                today_menu_root=self._today_menu_root,
                index_builder=None,
            )
        return self._today_menu_service_light

    def _require_today_menu_service_heavy(self) -> TodayMenuService:
        """Controller-level centralized heavy service access for deterministic entrypoints."""
        if self._today_menu_service_heavy is None:
            self._today_menu_service_heavy = TodayMenuService(
                global_menu_root=self._global_menu_root,
                today_menu_root=self._today_menu_root,
                index_builder=self._require_index_builder(),
            )
        return self._today_menu_service_heavy

    def _require_phrase_regenerator(self) -> PhraseRegenerator:
        """Controller-level centralized heavy service access for deterministic entrypoints."""
        if self._phrase_regenerator is None:
            self._phrase_regenerator = PhraseRegenerator(
                menu_root=self._global_menu_root,
                model_registry=self._model_registry,
                index_builder=self._require_index_builder(),
            )
        return self._phrase_regenerator

    def _emit_entrypoint_error(self, message: str) -> None:
        self.operation_error.emit(message)

    def _log_admin_phrase_event(self, message: str) -> None:
        """Dedicated lightweight file log for admin regenerate-phrase diagnostics."""
        ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S%z")
        try:
            self._admin_phrase_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._admin_phrase_log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"[{ts}] {message}\n")
        except Exception:
            # Logging must be best-effort and never block UI workflow.
            return

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
        ready, reason = self.check_recognition_ready()
        if not ready:
            raise RuntimeError(reason)
        try:
            image = ImageLoader.ensure_image_loaded(path)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Не удалось загрузить изображение: {exc}") from exc
        self.start_recognition_from_camera_frame(image)

    def start_recognition_from_camera_frame(self, frame: np.ndarray) -> None:
        ready, reason = self.check_recognition_ready()
        if not ready:
            raise RuntimeError(reason)

        worker = RecognitionWorker(self._orchestrator, frame)
        worker.progress.connect(self.recognition_progress)
        worker.error.connect(self.operation_error)
        worker.result.connect(self.recognition_finished)
        self._runner.run(worker)

    def regenerate_phrase(self, image: np.ndarray, dish_name: str, category: str, on_result: Callable[[str], None]) -> None:
        if image is None:
            self._emit_entrypoint_error("Не удалось сгенерировать фразу: отсутствует crop изображения.")
            return
        dish_name = dish_name.strip()
        category = category.strip()
        if not dish_name:
            self._emit_entrypoint_error("Не удалось сгенерировать фразу: укажите название блюда.")
            return
        if not category:
            self._emit_entrypoint_error("Не удалось сгенерировать фразу: укажите категорию блюда.")
            return

        self._log_admin_phrase_event(
            f"regenerate phrase requested | dish_name={dish_name!r} | category={category!r}"
        )

        # Lazy first Qwen init may be long; we emit explicit phases from worker thread.
        def _fn(progress_callback: Callable[[str], None] | None = None, **kwargs: Any) -> str:
            self._log_admin_phrase_event("worker started")
            # Early hook must be set before get_qwen() so first-load stages are visible in UI.
            QwenAdapter.set_early_admin_diagnostics(progress_callback)
            try:
                if progress_callback:
                    progress_callback("Подготавливаю Qwen...")
                self._log_admin_phrase_event("status: preparing qwen")

                self._log_admin_phrase_event("enter: ModelRegistry.get_qwen()")
                qwen = self._model_registry.get_qwen()
                self._log_admin_phrase_event("exit: ModelRegistry.get_qwen()")

                # Adapter-level stage updates come from inside lazy load and generation.
                if hasattr(qwen, "configure_admin_diagnostics"):
                    qwen.configure_admin_diagnostics(progress_callback=progress_callback)
                    self._log_admin_phrase_event("qwen diagnostics callback configured")

                self._log_admin_phrase_event("status: phrase generation started")
                phrase = qwen.generate_short_dish_phrase(**kwargs)
                self._log_admin_phrase_event("status: phrase generation finished")
                return phrase
            except Exception:
                self._log_admin_phrase_event(f"exception traceback:\n{traceback.format_exc()}")
                raise
            finally:
                # Clear process-local hook to avoid leaking callback between independent runs.
                QwenAdapter.set_early_admin_diagnostics(None)

        worker = PhraseGenerationWorker(
            WorkerTask(fn=_fn, kwargs={"image": image, "dish_name": dish_name, "category": category})
        )
        worker.progress.connect(self.operation_progress)
        worker.error.connect(self.operation_error)
        worker.result.connect(on_result)
        worker.finished.connect(self.phrase_generation_finished)
        self._runner.run(worker)

    def create_dish(self, payload: dict[str, Any], on_result: Callable[[object], None]) -> None:
        name = str(payload.get("name", "")).strip()
        category = str(payload.get("category", "")).strip()
        crop_image = payload.get("crop_image")
        if not name:
            self._emit_entrypoint_error("Не удалось создать блюдо: пустое название.")
            return
        if not category:
            self._emit_entrypoint_error("Не удалось создать блюдо: не выбрана категория.")
            return
        if crop_image is None:
            self._emit_entrypoint_error("Не удалось создать блюдо: отсутствует crop изображения.")
            return

        worker = CreateDishWorker(repository_factory=self._require_heavy_menu_repository, payload=payload)
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
        worker = UpdateDishWorker(
            repository_factory=self._require_heavy_menu_repository,
            category=category,
            slug_or_name=slug_or_name,
            patch={"is_active": is_active},
        )
        worker.progress.connect(self.operation_progress)
        worker.error.connect(self.operation_error)
        worker.result.connect(on_result)
        self._runner.run(worker)

    def confirm_phrase(self, dish_dir: str | Path, phrase: str, on_result: Callable[[object], None]) -> None:
        if not str(dish_dir).strip():
            self._emit_entrypoint_error("Не удалось подтвердить фразу: не задан путь к блюду.")
            return
        phrase = phrase.strip()
        if not phrase:
            self._emit_entrypoint_error("Не удалось подтвердить фразу: пустая фраза.")
            return

        worker = ConfirmPhraseWorker(
            regenerator_factory=self._require_phrase_regenerator,
            dish_dir=dish_dir,
            phrase=phrase,
        )
        worker.progress.connect(self.operation_progress)
        worker.error.connect(self.operation_error)
        worker.result.connect(on_result)
        self._runner.run(worker)

    def load_global_menu(self, category: str | None, on_result: Callable[[object], None]) -> None:
        try:
            repository = self._get_menu_repository_light()
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
            repository = self._get_menu_repository_light()
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
            service = self._get_today_menu_service_light()
        except Exception as exc:  # noqa: BLE001
            self.operation_error.emit(f"Не удалось открыть today menu: {exc}")
            on_result([])
            return

        worker = QueryMenuWorker(WorkerTask(fn=service.list_today_dishes, kwargs=kwargs))
        worker.error.connect(self.operation_error)
        worker.result.connect(on_result)
        self._runner.run(worker)

    def set_today_menu(self, mapping: dict[str, list[str]], on_result: Callable[[object], None]) -> None:
        worker = SetTodayMenuWorker(service_factory=self._require_today_menu_service_heavy, mapping=mapping)
        worker.progress.connect(self.operation_progress)
        worker.error.connect(self.operation_error)
        worker.result.connect(on_result)
        self._runner.run(worker)

    def _load_all_categories(self) -> list[Any]:
        rows: list[Any] = []
        repository = self._get_menu_repository_light()
        for category in ("beverage", "soup", "portioned", "garnish", "meat", "sauce"):
            rows.extend(repository.list_by_category(category, include_inactive=True))
        return rows
