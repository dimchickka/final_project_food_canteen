"""Utilities for reading and validating a single dish card on disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.domain.entities import MenuDish


class DishCardService:
    """Service layer for loading/saving dish-card files without GUI concerns."""

    def __init__(self, menu_root: str | Path) -> None:
        self.menu_root = Path(menu_root)

    def get_dish_card(self, dish_dir: str | Path) -> MenuDish:
        meta = self.load_meta(dish_dir)
        return MenuDish.from_dict(meta)

    def load_meta(self, dish_dir: str | Path) -> dict[str, Any]:
        dish_path = Path(dish_dir)
        meta_path = dish_path / "meta.json"
        if not dish_path.exists() or not dish_path.is_dir():
            raise FileNotFoundError(f"Dish directory does not exist: {dish_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json is missing for dish: {dish_path}")

        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Broken meta.json in {dish_path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"meta.json must contain a JSON object: {meta_path}")
        return payload

    def save_meta(self, dish_dir: str | Path, meta: dict[str, Any]) -> None:
        dish_path = Path(dish_dir)
        if not dish_path.exists() or not dish_path.is_dir():
            raise FileNotFoundError(f"Dish directory does not exist: {dish_path}")
        if not isinstance(meta, dict):
            raise ValueError("meta must be a dict")

        meta_path = dish_path / "meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def resolve_files(self, dish_dir: str | Path) -> dict[str, Path | None]:
        dish_path = Path(dish_dir)
        if not dish_path.exists() or not dish_path.is_dir():
            raise FileNotFoundError(f"Dish directory does not exist: {dish_path}")

        mapping: dict[str, Path | None] = {
            "dish_dir": dish_path,
            "meta": dish_path / "meta.json",
            "crop": dish_path / "crop.jpg",
            "crop_embedding": dish_path / "crop_embedding.pt",
            "qwen_phrase": dish_path / "qwen_phrase.txt",
            "qwen_phrase_embedding": dish_path / "qwen_phrase_embedding.pt",
        }

        return {key: value if value.exists() else None for key, value in mapping.items()}

    def validate_dish_card(self, dish_dir: str | Path) -> dict[str, bool]:
        files = self.resolve_files(dish_dir)
        return {
            "meta_exists": files["meta"] is not None,
            "crop_exists": files["crop"] is not None,
            "crop_embedding_exists": files["crop_embedding"] is not None,
            "qwen_phrase_exists": files["qwen_phrase"] is not None,
            "qwen_phrase_embedding_exists": files["qwen_phrase_embedding"] is not None,
        }
