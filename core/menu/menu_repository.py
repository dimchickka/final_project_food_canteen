"""Filesystem repository for global menu CRUD and metadata consistency."""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config.constants import DISH_CATEGORIES
from core.domain.entities import MenuDish
from core.domain.enums import DishCategory
from core.menu.menu_index_builder import MenuIndexBuilder


class MenuRepository:
    """Main service that manages dish folders in `data/menu/global` and their meta.json."""

    def __init__(
        self,
        menu_root: str | Path = "data/menu/global",
        index_builder: MenuIndexBuilder | None = None,
    ) -> None:
        self.menu_root = Path(menu_root)
        self.index_builder = index_builder
        self.menu_root.mkdir(parents=True, exist_ok=True)

    def create_dish(
        self,
        name: str,
        category: DishCategory | str,
        crop_image: np.ndarray | Image.Image | str | Path,
        phrase: str | None = None,
        dish_id: str | None = None,
        allow_existing: bool = False,
    ) -> MenuDish:
        category_enum = self._validate_category(category)
        slug = self._slugify_name(name)
        dish_dir = self._resolve_category_dir(category_enum) / slug

        if dish_dir.exists() and not allow_existing:
            raise FileExistsError(f"Dish already exists: {dish_dir}")

        dish_dir.mkdir(parents=True, exist_ok=True)
        crop_path = dish_dir / "crop.jpg"
        self._save_crop_image(crop_image=crop_image, output_path=crop_path)

        phrase_path = dish_dir / "qwen_phrase.txt"
        if phrase is not None and phrase.strip():
            phrase_path.write_text(phrase.strip(), encoding="utf-8")

        now = self._now_iso()
        meta: dict[str, Any] = {
            "dish_id": dish_id or str(uuid.uuid4()),
            "name": name.strip(),
            "slug": slug,
            "category": category_enum.value,
            "folder_path": str(dish_dir),
            "crop_image_path": str(crop_path),
            "crop_embedding_path": str(dish_dir / "crop_embedding.pt"),
            "qwen_phrase_path": str(phrase_path),
            "qwen_phrase_embedding_path": str(dish_dir / "qwen_phrase_embedding.pt"),
            "is_active": True,
            "created_at": now,
            "updated_at": now,
        }

        if self.index_builder is not None:
            # After crop update we rebuild photo embedding for this dish.
            self.index_builder.build_dish_photo_embedding(dish_dir)
            if phrase_path.exists():
                # Text embedding is rebuilt only when phrase exists.
                self.index_builder.build_dish_text_embedding(dish_dir)

        self._write_meta(dish_dir, meta)
        return MenuDish.from_dict(meta)

    def update_dish(
        self,
        category: DishCategory | str,
        slug_or_name: str,
        *,
        name: str | None = None,
        crop_image: np.ndarray | Image.Image | str | Path | None = None,
        phrase: str | None = None,
        is_active: bool | None = None,
        rename_folder: bool = False,
    ) -> MenuDish:
        category_enum = self._validate_category(category)
        dish_dir = self.get_dish_dir(category_enum, slug_or_name)
        meta = self._read_meta(dish_dir)

        if name is not None and name.strip():
            meta["name"] = name.strip()
            if rename_folder:
                new_slug = self._slugify_name(name)
                new_dish_dir = self._resolve_category_dir(category_enum) / new_slug
                if new_dish_dir.exists() and new_dish_dir != dish_dir:
                    raise FileExistsError(f"Cannot rename dish folder, target exists: {new_dish_dir}")
                if new_dish_dir != dish_dir:
                    dish_dir.rename(new_dish_dir)
                    dish_dir = new_dish_dir
                meta["slug"] = new_slug
                meta["folder_path"] = str(dish_dir)

        if crop_image is not None:
            crop_path = dish_dir / "crop.jpg"
            self._save_crop_image(crop_image=crop_image, output_path=crop_path)
            meta["crop_image_path"] = str(crop_path)
            if self.index_builder is not None:
                self.index_builder.build_dish_photo_embedding(dish_dir)

        if phrase is not None:
            phrase_path = dish_dir / "qwen_phrase.txt"
            cleaned = phrase.strip()
            if cleaned:
                phrase_path.write_text(cleaned, encoding="utf-8")
            elif phrase_path.exists():
                phrase_path.unlink()
            meta["qwen_phrase_path"] = str(phrase_path)
            if self.index_builder is not None and phrase_path.exists():
                self.index_builder.build_dish_text_embedding(dish_dir)

        if is_active is not None:
            meta["is_active"] = bool(is_active)

        meta["updated_at"] = self._now_iso()
        self._write_meta(dish_dir, meta)
        return MenuDish.from_dict(meta)

    def disable_dish(self, category: DishCategory | str, slug_or_name: str) -> MenuDish:
        return self.update_dish(category=category, slug_or_name=slug_or_name, is_active=False)

    def search_dishes(
        self,
        query: str,
        category: DishCategory | str | None = None,
        include_inactive: bool = False,
    ) -> list[MenuDish]:
        needle = query.strip().lower()
        candidates: list[MenuDish] = []

        if category is None:
            for value in DISH_CATEGORIES:
                candidates.extend(self.list_by_category(value, include_inactive=include_inactive))
        else:
            candidates = self.list_by_category(category, include_inactive=include_inactive)

        if not needle:
            return candidates
        return [dish for dish in candidates if needle in dish.name.lower() or needle in dish.slug.lower()]

    def list_by_category(self, category: DishCategory | str, include_inactive: bool = False) -> list[MenuDish]:
        category_enum = self._validate_category(category)
        category_dir = self._resolve_category_dir(category_enum)
        if not category_dir.exists():
            return []

        dishes: list[MenuDish] = []
        for dish_dir in sorted(entry for entry in category_dir.iterdir() if entry.is_dir()):
            meta_path = dish_dir / "meta.json"
            if not meta_path.exists():
                continue
            meta = self._read_meta(dish_dir)
            dish = MenuDish.from_dict(meta)
            if include_inactive or dish.is_active:
                dishes.append(dish)
        return dishes

    def get_dish(self, category: DishCategory | str, slug_or_name: str) -> MenuDish:
        dish_dir = self.get_dish_dir(category, slug_or_name)
        return MenuDish.from_dict(self._read_meta(dish_dir))

    def get_dish_dir(self, category: DishCategory | str, slug_or_name: str) -> Path:
        category_enum = self._validate_category(category)
        category_dir = self._resolve_category_dir(category_enum)
        slug_candidate = self._slugify_name(slug_or_name)

        by_slug = category_dir / slug_candidate
        if by_slug.exists() and by_slug.is_dir():
            return by_slug

        for dish_dir in sorted(entry for entry in category_dir.iterdir() if entry.is_dir()):
            meta_path = dish_dir / "meta.json"
            if not meta_path.exists():
                continue
            meta = self._read_meta(dish_dir)
            if meta.get("name", "").strip().lower() == slug_or_name.strip().lower():
                return dish_dir

        raise FileNotFoundError(
            f"Dish not found in category '{category_enum.value}' by slug/name: {slug_or_name}"
        )

    def dish_exists(self, category: DishCategory | str, slug_or_name: str) -> bool:
        try:
            self.get_dish_dir(category, slug_or_name)
            return True
        except FileNotFoundError:
            return False

    # Helper to keep category parsing strict and aligned with DishCategory enum.
    def _validate_category(self, category: DishCategory | str) -> DishCategory:
        if isinstance(category, DishCategory):
            return category
        try:
            normalized = str(category).strip().lower()
            return DishCategory(normalized)
        except ValueError as exc:
            raise ValueError(f"Invalid dish category '{category}'. Allowed: {DISH_CATEGORIES}") from exc

    def _resolve_category_dir(self, category: DishCategory) -> Path:
        category_dir = self.menu_root / category.value
        category_dir.mkdir(parents=True, exist_ok=True)
        return category_dir

    def _read_meta(self, dish_dir: Path) -> dict[str, Any]:
        meta_path = dish_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json is missing in dish folder: {dish_dir}")
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Broken meta.json for dish {dish_dir}: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"meta.json must contain object payload: {meta_path}")
        return payload

    def _write_meta(self, dish_dir: Path, meta: dict[str, Any]) -> None:
        meta_path = dish_dir / "meta.json"
        self._safe_write_json(meta_path, meta)

    # Atomic-like JSON write via temporary file prevents truncated meta files.
    def _safe_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def _slugify_name(self, name: str) -> str:
        value = name.strip().lower()
        value = re.sub(r"\s+", "-", value)
        value = re.sub(r"[^a-z0-9а-яё\-]+", "-", value)
        value = re.sub(r"-+", "-", value).strip("-")
        if not value:
            raise ValueError("Dish name produces empty slug")
        return value

    # Crop can arrive from OpenCV ndarray, PIL image, or existing file path.
    def _save_crop_image(self, crop_image: np.ndarray | Image.Image | str | Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(crop_image, np.ndarray):
            if crop_image.ndim == 2:
                pil = Image.fromarray(crop_image).convert("RGB")
            elif crop_image.ndim == 3 and crop_image.shape[-1] == 3:
                rgb = crop_image[..., ::-1]  # OpenCV BGR -> RGB.
                pil = Image.fromarray(rgb).convert("RGB")
            elif crop_image.ndim == 3 and crop_image.shape[-1] == 4:
                rgba = crop_image[..., [2, 1, 0, 3]]  # OpenCV BGRA -> RGBA.
                pil = Image.fromarray(rgba, mode="RGBA").convert("RGB")
            else:
                raise ValueError("Unsupported crop ndarray shape. Expected HxW, HxWx3 or HxWx4")
            pil.save(output_path, format="JPEG", quality=95)
            return

        if isinstance(crop_image, Image.Image):
            crop_image.convert("RGB").save(output_path, format="JPEG", quality=95)
            return

        if isinstance(crop_image, (str, Path)):
            source = Path(crop_image)
            if not source.exists():
                raise FileNotFoundError(f"crop image source does not exist: {source}")
            with Image.open(source) as img:
                img.convert("RGB").save(output_path, format="JPEG", quality=95)
            return

        raise ValueError(f"Unsupported crop_image type: {type(crop_image)!r}")

    def _now_iso(self) -> str:
        return datetime.now(UTC).replace(microsecond=0).isoformat()
