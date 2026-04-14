"""Regenerate and confirm Qwen dish phrases for menu cards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.menu.menu_index_builder import MenuIndexBuilder
from core.models.model_registry import ModelRegistry


class PhraseRegenerator:
    """Service that generates phrase candidates and persists confirmed phrase files."""

    def __init__(
        self,
        menu_root: str | Path,
        model_registry: ModelRegistry,
        index_builder: MenuIndexBuilder | None = None,
    ) -> None:
        self.menu_root = Path(menu_root)
        self.model_registry = model_registry
        self.index_builder = index_builder

    def generate_phrase_for_dish(self, dish_dir: str | Path) -> str:
        dish_path = self._validate_dish_dir(dish_dir)
        meta = self._load_meta(dish_path)
        crop_path = dish_path / "crop.jpg"
        if not crop_path.exists():
            raise FileNotFoundError(f"crop.jpg is missing for dish: {dish_path}")

        dish_name = str(meta.get("name", "")).strip()
        category = str(meta.get("category", "")).strip()
        if not dish_name:
            raise ValueError(f"Dish name is missing in meta.json: {dish_path}")
        if not category:
            raise ValueError(f"Dish category is missing in meta.json: {dish_path}")

        qwen = self.model_registry.get_qwen()
        return qwen.generate_short_dish_phrase(image=crop_path, dish_name=dish_name, category=category)

    def save_phrase_for_dish(self, dish_dir: str | Path, phrase: str, rebuild_embedding: bool = True) -> Path:
        dish_path = self._validate_dish_dir(dish_dir)
        normalized = phrase.strip()
        if not normalized:
            raise ValueError("phrase must not be empty")

        phrase_path = dish_path / "qwen_phrase.txt"
        phrase_path.write_text(normalized, encoding="utf-8")
        self._update_meta_phrase_paths(dish_path)

        # Rebuild text embedding after phrase confirmation so CLIP text retrieval stays fresh.
        if rebuild_embedding:
            if self.index_builder is None:
                raise RuntimeError("Cannot rebuild phrase embedding: index_builder is not provided")
            self.index_builder.build_dish_text_embedding(dish_path)
            self._update_meta_phrase_paths(dish_path)

        return phrase_path

    def regenerate_phrase_for_dish(self, dish_dir: str | Path, rebuild_embedding: bool = False) -> str:
        phrase = self.generate_phrase_for_dish(dish_dir)
        if rebuild_embedding:
            # Explicitly disallow hidden persistence: UI should call confirm to save text.
            raise ValueError("rebuild_embedding=True requires phrase confirmation via confirm_phrase_for_dish")
        return phrase

    def confirm_phrase_for_dish(self, dish_dir: str | Path, phrase: str, rebuild_embedding: bool = True) -> Path:
        return self.save_phrase_for_dish(
            dish_dir=dish_dir,
            phrase=phrase,
            rebuild_embedding=rebuild_embedding,
        )

    def _validate_dish_dir(self, dish_dir: str | Path) -> Path:
        dish_path = Path(dish_dir)
        if not dish_path.exists() or not dish_path.is_dir():
            raise FileNotFoundError(f"Dish directory does not exist: {dish_path}")
        return dish_path

    def _load_meta(self, dish_dir: Path) -> dict[str, Any]:
        meta_path = dish_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json is missing for dish: {dish_dir}")

        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Broken meta.json in {dish_dir}: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"meta.json must contain JSON object: {meta_path}")
        return payload

    def _update_meta_phrase_paths(self, dish_dir: Path) -> None:
        meta = self._load_meta(dish_dir)
        meta["qwen_phrase_path"] = str(dish_dir / "qwen_phrase.txt")
        meta["qwen_phrase_embedding_path"] = str(dish_dir / "qwen_phrase_embedding.pt")
        meta_path = dish_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
