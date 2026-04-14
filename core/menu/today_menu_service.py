"""Service for managing today's lightweight menu selections from global menu."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from config.constants import DISH_CATEGORIES
from core.domain.entities import MenuDish
from core.domain.enums import DishCategory
from core.menu.menu_index_builder import MenuIndexBuilder


class TodayMenuService:
    """Maintains `data/menu/today` as lightweight links to global dish cards."""

    def __init__(
        self,
        global_menu_root: str | Path = "data/menu/global",
        today_menu_root: str | Path = "data/menu/today",
        index_builder: MenuIndexBuilder | None = None,
    ) -> None:
        self.global_menu_root = Path(global_menu_root)
        self.today_menu_root = Path(today_menu_root)
        self.index_builder = index_builder

        self.global_menu_root.mkdir(parents=True, exist_ok=True)
        self.today_menu_root.mkdir(parents=True, exist_ok=True)

    def clear_today_menu(self) -> None:
        if self.today_menu_root.exists():
            shutil.rmtree(self.today_menu_root)
        self.today_menu_root.mkdir(parents=True, exist_ok=True)

    def set_today_dishes(self, category: DishCategory | str, dish_slugs: Sequence[str]) -> list[Path]:
        category_enum = self._validate_category(category)
        category_dir = self.today_menu_root / category_enum.value
        if category_dir.exists():
            shutil.rmtree(category_dir)
        category_dir.mkdir(parents=True, exist_ok=True)

        created: list[Path] = []
        for dish_slug in dish_slugs:
            created.append(self.add_dish_to_today(category_enum, dish_slug))
        return created

    def add_dish_to_today(self, category: DishCategory | str, dish_slug: str) -> Path:
        category_enum = self._validate_category(category)
        slug = self._normalize_slug(dish_slug)

        global_dish_dir = self.global_menu_root / category_enum.value / slug
        if not global_dish_dir.exists() or not global_dish_dir.is_dir():
            raise FileNotFoundError(f"Global dish does not exist: {global_dish_dir}")

        global_meta = self._load_meta(global_dish_dir)
        if not bool(global_meta.get("is_active", True)):
            raise ValueError(f"Inactive dish cannot be added to today menu: {global_dish_dir}")

        today_dish_dir = self.today_menu_root / category_enum.value / slug
        today_dish_dir.mkdir(parents=True, exist_ok=True)

        # Today layer remains lightweight via link.json, but we also copy builder-critical
        # files so existing MenuIndexBuilder can scan `data/menu/today` directly.
        self._sync_builder_files(global_dish_dir=global_dish_dir, today_dish_dir=today_dish_dir)

        link_payload = {
            "global_dish_dir": str(global_dish_dir),
            "category": category_enum.value,
            "slug": slug,
            "linked_at": self._now_iso(),
        }
        (today_dish_dir / "link.json").write_text(
            json.dumps(link_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        today_meta = {
            "dish_id": global_meta.get("dish_id"),
            "name": global_meta.get("name", slug),
            "slug": slug,
            "category": category_enum.value,
            "folder_path": str(today_dish_dir),
            "crop_image_path": str(today_dish_dir / "crop.jpg"),
            "crop_embedding_path": str(today_dish_dir / "crop_embedding.pt"),
            "qwen_phrase_path": str(today_dish_dir / "qwen_phrase.txt"),
            "qwen_phrase_embedding_path": str(today_dish_dir / "qwen_phrase_embedding.pt"),
            "is_active": bool(global_meta.get("is_active", True)),
            "created_at": global_meta.get("created_at"),
            "updated_at": self._now_iso(),
            "global_dish_dir": str(global_dish_dir),
        }
        (today_dish_dir / "meta.json").write_text(json.dumps(today_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return today_dish_dir

    def remove_dish_from_today(self, category: DishCategory | str, dish_slug: str) -> None:
        category_enum = self._validate_category(category)
        slug = self._normalize_slug(dish_slug)
        today_dish_dir = self.today_menu_root / category_enum.value / slug
        if not today_dish_dir.exists():
            return
        shutil.rmtree(today_dish_dir)

    def list_today_dishes(self, category: DishCategory | str | None = None) -> list[MenuDish]:
        categories = [self._validate_category(category)] if category is not None else [DishCategory(v) for v in DISH_CATEGORIES]
        result: list[MenuDish] = []

        for category_enum in categories:
            category_dir = self.today_menu_root / category_enum.value
            if not category_dir.exists() or not category_dir.is_dir():
                continue

            for dish_dir in sorted(entry for entry in category_dir.iterdir() if entry.is_dir()):
                meta_path = dish_dir / "meta.json"
                if not meta_path.exists():
                    continue
                meta = self._load_json(meta_path)
                if not isinstance(meta, dict):
                    raise ValueError(f"meta.json must contain JSON object: {meta_path}")
                if not bool(meta.get("is_active", True)):
                    continue
                result.append(MenuDish.from_dict(meta))

        return result

    def rebuild_today_indexes(self) -> dict[str, Path]:
        if self.index_builder is None:
            raise RuntimeError("Cannot rebuild today indexes: index_builder is not provided")

        # Builder scans dish directories in today root and expects local embedding files.
        return self.index_builder.rebuild_today_menu_indexes(self.today_menu_root)

    def _validate_category(self, category: DishCategory | str) -> DishCategory:
        if isinstance(category, DishCategory):
            return category
        try:
            return DishCategory(str(category).strip().lower())
        except ValueError as exc:
            raise ValueError(f"Invalid dish category '{category}'. Allowed: {DISH_CATEGORIES}") from exc

    def _normalize_slug(self, dish_slug: str) -> str:
        normalized = dish_slug.strip().lower()
        if not normalized:
            raise ValueError("dish_slug must not be empty")
        return normalized

    def _load_meta(self, dish_dir: Path) -> dict[str, Any]:
        meta_path = dish_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json is missing for dish: {dish_dir}")
        payload = self._load_json(meta_path)
        if not isinstance(payload, dict):
            raise ValueError(f"meta.json must contain JSON object: {meta_path}")
        return payload

    def _load_json(self, path: Path) -> dict[str, Any] | list[Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Broken JSON file {path}: {exc}") from exc

    # Copy only minimal files required by existing index builder contract.
    def _sync_builder_files(self, global_dish_dir: Path, today_dish_dir: Path) -> None:
        required = ("crop_embedding.pt",)
        optional = ("qwen_phrase_embedding.pt", "crop.jpg", "qwen_phrase.txt")

        for filename in required:
            source = global_dish_dir / filename
            if not source.exists():
                raise FileNotFoundError(
                    f"Cannot add dish to today menu without required file '{filename}': {global_dish_dir}"
                )
            shutil.copy2(source, today_dish_dir / filename)

        for filename in optional:
            source = global_dish_dir / filename
            target = today_dish_dir / filename
            if source.exists():
                shutil.copy2(source, target)
            elif target.exists():
                target.unlink()

    def _now_iso(self) -> str:
        return datetime.now(UTC).replace(microsecond=0).isoformat()
