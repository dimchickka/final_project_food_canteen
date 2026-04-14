"""Centralized registry for heavy model adapters.

Why this exists:
- Qwen/CLIP/YOLO models are heavy for GPU memory.
- We must create shared instances once and reuse them across modules.
- Lazy loading keeps startup responsive and avoids blocking GUI unnecessarily.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, cast

from config.model_paths import (
    CLIP_MODEL_NAME_OR_PATH,
    QWEN_MODEL_PATH,
    YOLO_MAIN_WEIGHTS,
    YOLO_MEAT_SAUCE_WEIGHTS,
    YOLO_MEAT_WEIGHTS,
)
from config.settings import DEVICE


class _Warmable(Protocol):
    """Optional adapter contract for warmup support."""

    def warmup(self) -> None:
        """Preload kernels/cache/tensors for faster first inference."""


class _Unloadable(Protocol):
    """Optional adapter contract for explicit resource release."""

    def unload(self) -> None:
        """Release GPU/CPU resources held by the model."""


class ModelRegistry:
    """Thread-safe registry with lazy loading and single-instance semantics."""

    QWEN_KEY = "qwen"
    CLIP_KEY = "clip"
    YOLO_MAIN_KEY = "yolo_main"
    YOLO_MEAT_KEY = "yolo_meat"
    YOLO_MEAT_SAUCE_KEY = "yolo_meat_sauce"

    def __init__(self) -> None:
        # Loaded model instances are stored here and reused on every get_* call.
        self._models: dict[str, Any] = {}
        # Per-model loader callbacks to keep initialization logic explicit.
        self._loaders: dict[str, Callable[[], Any]] = {
            self.QWEN_KEY: self._load_qwen,
            self.CLIP_KEY: self._load_clip,
            self.YOLO_MAIN_KEY: self._load_yolo_main,
            self.YOLO_MEAT_KEY: self._load_yolo_meat,
            self.YOLO_MEAT_SAUCE_KEY: self._load_yolo_meat_sauce,
        }
        # Recursive lock protects against duplicate loads from concurrent callers.
        self._lock = RLock()

    # ----------------------------- Public API -----------------------------

    def load_all(self) -> None:
        """Force-load all known models once."""
        for model_key in self._loaders:
            self._get_or_load(model_key)

    def warmup(self) -> None:
        """Run warmup hook on loaded models when supported.

        Warmup is best-effort by design: if a model has no warmup() method,
        it is skipped silently.
        """
        with self._lock:
            loaded_items = list(self._models.items())

        for model_key, model in loaded_items:
            warmup_method = getattr(model, "warmup", None)
            if callable(warmup_method):
                try:
                    cast(_Warmable, model).warmup()
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(f"Warmup failed for model '{model_key}': {exc}") from exc

    def get_qwen(self) -> Any:
        return self._get_or_load(self.QWEN_KEY)

    def get_clip(self) -> Any:
        return self._get_or_load(self.CLIP_KEY)

    def get_yolo_main(self) -> Any:
        return self._get_or_load(self.YOLO_MAIN_KEY)

    def get_yolo_meat(self) -> Any:
        return self._get_or_load(self.YOLO_MEAT_KEY)

    def get_yolo_meat_sauce(self) -> Any:
        return self._get_or_load(self.YOLO_MEAT_SAUCE_KEY)

    def is_loaded(self, model_key: str) -> bool:
        """Check whether a concrete model is already initialized."""
        with self._lock:
            return model_key in self._models

    def unload_all(self) -> None:
        """Unload all models if adapters expose unload(), then clear cache."""
        with self._lock:
            models_to_unload = list(self._models.values())
            self._models.clear()

        for model in models_to_unload:
            unload_method = getattr(model, "unload", None)
            if callable(unload_method):
                cast(_Unloadable, model).unload()

    # ---------------------------- Internal core ---------------------------

    def _get_or_load(self, model_key: str) -> Any:
        """Return cached model or load exactly once in a thread-safe way."""
        with self._lock:
            if model_key in self._models:
                return self._models[model_key]

            loader = self._loaders.get(model_key)
            if loader is None:
                raise KeyError(f"Unknown model key '{model_key}'. Available keys: {tuple(self._loaders)}")

            model = loader()
            self._models[model_key] = model
            return model

    def _load_qwen(self) -> Any:
        self._ensure_path_exists(QWEN_MODEL_PATH, model_name="Qwen model")
        return self._instantiate_adapter(
            module_name="core.models.qwen_adapter",
            class_candidates=("QwenAdapter", "QwenModelAdapter", "QwenInferAdapter"),
            model_name_for_error="Qwen",
            kwargs={"model_path": QWEN_MODEL_PATH, "device": DEVICE},
        )

    def _load_clip(self) -> Any:
        # CLIP may use model id (not a local path), so existence check is conditional.
        if CLIP_MODEL_NAME_OR_PATH.startswith((".", "/", "\\")):
            self._ensure_path_exists(Path(CLIP_MODEL_NAME_OR_PATH), model_name="CLIP model")

        return self._instantiate_adapter(
            module_name="core.models.clip_adapter",
            class_candidates=("ClipAdapter", "CLIPAdapter", "ClipInferAdapter"),
            model_name_for_error="CLIP",
            kwargs={"model_name_or_path": CLIP_MODEL_NAME_OR_PATH, "device": DEVICE},
        )

    def _load_yolo_main(self) -> Any:
        self._ensure_path_exists(YOLO_MAIN_WEIGHTS, model_name="YOLO main weights")
        return self._instantiate_adapter(
            module_name="core.models.yolo_main_adapter",
            class_candidates=("YoloMainAdapter", "YOLOMainAdapter", "YoloAdapter"),
            model_name_for_error="YOLO main",
            kwargs={"weights_path": YOLO_MAIN_WEIGHTS, "device": DEVICE},
        )

    def _load_yolo_meat(self) -> Any:
        self._ensure_path_exists(YOLO_MEAT_WEIGHTS, model_name="YOLO meat weights")
        return self._instantiate_adapter(
            module_name="core.models.yolo_meat_adapter",
            class_candidates=("YoloMeatAdapter", "YOLOMeatAdapter", "YoloAdapter"),
            model_name_for_error="YOLO meat",
            kwargs={"weights_path": YOLO_MEAT_WEIGHTS, "device": DEVICE},
        )

    def _load_yolo_meat_sauce(self) -> Any:
        self._ensure_path_exists(YOLO_MEAT_SAUCE_WEIGHTS, model_name="YOLO meat_sauce weights")
        return self._instantiate_adapter(
            module_name="core.models.yolo_meat_sauce_adapter",
            class_candidates=("YoloMeatSauceAdapter", "YOLOMeatSauceAdapter", "YoloAdapter"),
            model_name_for_error="YOLO meat_sauce",
            kwargs={"weights_path": YOLO_MEAT_SAUCE_WEIGHTS, "device": DEVICE},
        )

    # --------------------------- Internal helpers -------------------------

    def _instantiate_adapter(
        self,
        *,
        module_name: str,
        class_candidates: tuple[str, ...],
        model_name_for_error: str,
        kwargs: dict[str, Any],
    ) -> Any:
        """Import adapter module and instantiate adapter with graceful fallbacks.

        Adapter modules are imported only when needed so app startup stays cheap.
        If future adapter API changes, candidate class/factory names let us remain
        compatible without touching the registry call sites.
        """
        module = self._import_adapter_module(module_name=module_name, model_name=model_name_for_error)

        for class_name in class_candidates:
            adapter_class = getattr(module, class_name, None)
            if callable(adapter_class):
                return self._call_with_supported_kwargs(adapter_class, kwargs)

        factory = getattr(module, "create_adapter", None)
        if callable(factory):
            return self._call_with_supported_kwargs(factory, kwargs)

        expected = ", ".join(class_candidates) + ", create_adapter"
        raise NotImplementedError(
            f"{model_name_for_error} adapter is not available yet. "
            f"Expected one of: {expected} in module '{module_name}'."
        )

    @staticmethod
    def _import_adapter_module(*, module_name: str, model_name: str) -> Any:
        """Import adapter lazily and provide clear message if module is missing."""
        try:
            return import_module(module_name)
        except ModuleNotFoundError as exc:
            raise NotImplementedError(
                f"{model_name} adapter is not available yet. Expected module: {module_name}"
            ) from exc

    @staticmethod
    def _call_with_supported_kwargs(callable_obj: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
        """Instantiate adapter and tolerate constructors with partial kwargs.

        This keeps the registry forward-compatible with adapter constructor
        signatures while still passing standard args when supported.
        """
        try:
            return callable_obj(**kwargs)
        except TypeError:
            # Fallback for adapters that accept positional model source + device.
            if "weights_path" in kwargs:
                return callable_obj(kwargs["weights_path"], kwargs.get("device"))
            if "model_path" in kwargs:
                return callable_obj(kwargs["model_path"], kwargs.get("device"))
            return callable_obj(kwargs.get("model_name_or_path"), kwargs.get("device"))

    @staticmethod
    def _ensure_path_exists(path: Path, *, model_name: str) -> None:
        """Validate configured local path before attempting expensive model init."""
        if not path.exists():
            raise FileNotFoundError(
                f"{model_name} path does not exist: '{path}'. "
                "Check config/model_paths.py and ensure model files are present."
            )


# Optional shared singleton for modules that want one global registry instance.
_global_registry: ModelRegistry | None = None
_global_registry_lock = RLock()


def get_model_registry() -> ModelRegistry:
    """Return process-wide registry instance initialized on first request."""
    global _global_registry
    with _global_registry_lock:
        if _global_registry is None:
            _global_registry = ModelRegistry()
        return _global_registry
