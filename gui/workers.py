"""Qt worker utilities.

Heavy operations are moved off the UI thread via QObject+QThread pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from core.domain.dto import RecognitionSessionResult
from core.domain.entities import MenuDish
from core.menu.menu_repository import MenuRepository
from core.menu.phrase_regenerator import PhraseRegenerator
from core.menu.today_menu_service import TodayMenuService
from core.pipeline.recognition_orchestrator import RecognitionOrchestrator


def _format_worker_error(exc: Exception) -> str:
    """Convert internal exceptions to stable user-displayable worker error text."""
    text = str(exc).strip()
    if text:
        return text
    return f"{exc.__class__.__name__}: операция завершилась с ошибкой"


@dataclass(slots=True)
class WorkerTask:
    """Simple callable wrapper for reusable background tasks."""

    fn: Callable[..., Any]
    kwargs: dict[str, Any]


class BaseWorker(QObject):
    """Base worker with standard lifecycle signals."""

    finished = Signal()
    error = Signal(str)
    progress = Signal(str)


class RecognitionWorker(BaseWorker):
    """Runs recognition orchestrator on a BGR frame in background thread."""

    result = Signal(object)

    def __init__(self, orchestrator: RecognitionOrchestrator, image: np.ndarray) -> None:
        super().__init__()
        self._orchestrator = orchestrator
        self._image = image

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit("Запускаю распознавание...")
            output = self._orchestrator.recognize(self._image)
            self.result.emit(output)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(_format_worker_error(exc))
        finally:
            self.finished.emit()


class ModelWarmupWorker(BaseWorker):
    """Loads and optionally warms up model registry lazily."""

    result = Signal(bool)

    def __init__(self, warmup_fn: Callable[[], None]) -> None:
        super().__init__()
        self._warmup_fn = warmup_fn

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit("Подготавливаю модели...")
            self._warmup_fn()
            self.result.emit(True)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(_format_worker_error(exc))
            self.result.emit(False)
        finally:
            self.finished.emit()


class PhraseGenerationWorker(BaseWorker):
    """Regenerates phrase by direct image+meta call to Qwen adapter."""

    result = Signal(str)

    def __init__(self, task: WorkerTask) -> None:
        super().__init__()
        self._task = task

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit("Генерирую фразу...")
            phrase = self._task.fn(**self._task.kwargs)
            self.result.emit(str(phrase))
        except Exception as exc:  # noqa: BLE001
            self.error.emit(_format_worker_error(exc))
        finally:
            self.finished.emit()


class CreateDishWorker(BaseWorker):
    """Creates a global menu dish (and embeddings/index updates via repository)."""

    result = Signal(object)

    def __init__(self, repository: MenuRepository, payload: dict[str, Any]) -> None:
        super().__init__()
        self._repository = repository
        self._payload = payload

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit("Сохраняю блюдо...")
            dish = self._repository.create_dish(**self._payload)
            self.result.emit(dish)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(_format_worker_error(exc))
        finally:
            self.finished.emit()


class UpdateDishWorker(BaseWorker):
    """Updates one dish card fields in global menu."""

    result = Signal(object)

    def __init__(self, repository: MenuRepository, category: str, slug_or_name: str, patch: dict[str, Any]) -> None:
        super().__init__()
        self._repository = repository
        self._category = category
        self._slug_or_name = slug_or_name
        self._patch = patch

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit("Обновляю карточку блюда...")
            dish = self._repository.update_dish(self._category, self._slug_or_name, **self._patch)
            self.result.emit(dish)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(_format_worker_error(exc))
        finally:
            self.finished.emit()


class ConfirmPhraseWorker(BaseWorker):
    """Persists phrase to dish folder and rebuilds phrase embedding if configured."""

    result = Signal(object)

    def __init__(self, regenerator: PhraseRegenerator, dish_dir: str | Path, phrase: str) -> None:
        super().__init__()
        self._regenerator = regenerator
        self._dish_dir = dish_dir
        self._phrase = phrase

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit("Подтверждаю фразу...")
            saved_path = self._regenerator.confirm_phrase_for_dish(self._dish_dir, self._phrase)
            self.result.emit(str(saved_path))
        except Exception as exc:  # noqa: BLE001
            self.error.emit(_format_worker_error(exc))
        finally:
            self.finished.emit()


class SetTodayMenuWorker(BaseWorker):
    """Saves today dishes per category and rebuilds today indexes."""

    result = Signal(object)

    def __init__(self, service: TodayMenuService, mapping: dict[str, list[str]]) -> None:
        super().__init__()
        self._service = service
        self._mapping = mapping

    @Slot()
    def run(self) -> None:
        try:
            for category, slugs in self._mapping.items():
                self.progress.emit(f"Сохраняю category={category}...")
                self._service.set_today_dishes(category, slugs)
            self.progress.emit("Пересобираю индексы today menu...")
            indexes = self._service.rebuild_today_indexes()
            self.result.emit({"indexes": indexes})
        except Exception as exc:  # noqa: BLE001
            self.error.emit(_format_worker_error(exc))
        finally:
            self.finished.emit()


class QueryMenuWorker(BaseWorker):
    """Background fetch/search for global/today menu cards."""

    result = Signal(object)

    def __init__(self, task: WorkerTask) -> None:
        super().__init__()
        self._task = task

    @Slot()
    def run(self) -> None:
        try:
            rows = self._task.fn(**self._task.kwargs)
            self.result.emit(rows)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(_format_worker_error(exc))
        finally:
            self.finished.emit()


class WorkerRunner(QObject):
    """Helper to start workers and auto-clean QThread objects."""

    def __init__(self) -> None:
        super().__init__()
        self._threads: list[QThread] = []

    def run(self, worker: BaseWorker, started_slot: str = "run") -> QThread:
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(getattr(worker, started_slot))
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._threads.remove(thread) if thread in self._threads else None)
        self._threads.append(thread)
        thread.start()
        return thread


def aggregate_recognition_rows(result: RecognitionSessionResult) -> list[tuple[str, int]]:
    """UI helper: merge same dish names for compact result output."""
    totals: dict[str, int] = {}
    for item in result.recognized_items:
        totals[item.dish_name] = totals.get(item.dish_name, 0) + int(item.count)
    return sorted(totals.items(), key=lambda x: x[0].lower())


def normalize_menu_rows(rows: list[MenuDish]) -> list[dict[str, Any]]:
    """UI-safe dict rows for flexible widgets."""
    result: list[dict[str, Any]] = []
    for dish in rows:
        result.append(
            {
                "dish_id": dish.dish_id,
                "name": dish.name,
                "slug": dish.slug,
                "category": dish.category.value,
                "folder_path": dish.folder_path,
                "crop_image_path": dish.crop_image_path,
                "is_active": dish.is_active,
            }
        )
    return result
