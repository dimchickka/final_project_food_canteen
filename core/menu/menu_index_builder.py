"""Index builder for menu dishes and head classifiers using CLIP embeddings on disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.constants import DISH_CATEGORIES, SUPPORTED_IMAGE_EXTENSIONS
from core.models.clip_adapter import ClipAdapter


class MenuIndexBuilder:
    """Build dish/head text/photo embeddings and persist reproducible CLIP index folders."""

    def __init__(self, clip_adapter: ClipAdapter) -> None:
        self.clip_adapter = clip_adapter

    # Build crop_embedding.pt for a single dish folder from its crop image.
    def build_dish_photo_embedding(self, dish_dir: str | Path) -> Path:
        dish_path = Path(dish_dir)
        if not dish_path.exists() or not dish_path.is_dir():
            raise RuntimeError(f"Dish directory does not exist: {dish_path}")

        image_path = self._resolve_dish_crop_image_path(dish_path)
        embedding = self.clip_adapter.encode_image(image_path).detach().cpu()

        output_path = dish_path / "crop_embedding.pt"
        self._save_tensor(embedding, output_path)
        return output_path

    # Build qwen_phrase_embedding.pt for a single dish folder from qwen_phrase.txt.
    def build_dish_text_embedding(self, dish_dir: str | Path) -> Path:
        dish_path = Path(dish_dir)
        if not dish_path.exists() or not dish_path.is_dir():
            raise RuntimeError(f"Dish directory does not exist: {dish_path}")

        phrase_path = dish_path / "qwen_phrase.txt"
        if not phrase_path.exists():
            raise RuntimeError(f"Dish text phrase file not found: {phrase_path}")

        text = phrase_path.read_text(encoding="utf-8").strip()
        if not text:
            raise RuntimeError(f"Dish text phrase is empty: {phrase_path}")

        embedding = self.clip_adapter.encode_text(text).detach().cpu()
        output_path = dish_path / "qwen_phrase_embedding.pt"
        self._save_tensor(embedding, output_path)
        return output_path

    # Build category-wide photo index (embeddings.pt + metadata.json).
    def build_category_photo_index(self, category_dir: str | Path, output_index_dir: str | Path) -> Path:
        category_path = Path(category_dir)
        output_dir = Path(output_index_dir)

        records, skipped = self._collect_category_records(
            category_dir=category_path,
            embedding_filename="crop_embedding.pt",
            mode="photo",
        )
        if not records:
            raise RuntimeError(
                f"No valid dishes for photo index in {category_path}. Skipped: {self._join_notes(skipped)}"
            )

        return self._save_index(
            embeddings=[record["embedding"] for record in records],
            metadata=[record["metadata"] for record in records],
            output_index_dir=output_dir,
            mode="photo",
            notes=skipped,
        )

    # Build category-wide text index (embeddings.pt + metadata.json).
    def build_category_text_index(self, category_dir: str | Path, output_index_dir: str | Path) -> Path:
        category_path = Path(category_dir)
        output_dir = Path(output_index_dir)

        records, skipped = self._collect_category_records(
            category_dir=category_path,
            embedding_filename="qwen_phrase_embedding.pt",
            mode="text",
        )
        if not records:
            raise RuntimeError(
                f"No valid dishes for text index in {category_path}. Skipped: {self._join_notes(skipped)}"
            )

        return self._save_index(
            embeddings=[record["embedding"] for record in records],
            metadata=[record["metadata"] for record in records],
            output_index_dir=output_dir,
            mode="text",
            notes=skipped,
        )

    # Build first-head text index from descriptions.txt where each line is class || desc1 || desc2 ...
    def build_first_head_index(self, head_dir: str | Path) -> Path:
        return self._build_head_text_index(head_dir=head_dir, index_name="first_head")

    # Build second-head text index from descriptions.txt where each line is class || desc1 || desc2 ...
    def build_second_head_index(self, head_dir: str | Path) -> Path:
        return self._build_head_text_index(head_dir=head_dir, index_name="second_head")

    # Rebuild both text/photo indexes for each today-menu category when possible.
    def rebuild_today_menu_indexes(self, today_menu_root: str | Path) -> dict[str, Path]:
        root = Path(today_menu_root)
        if not root.exists() or not root.is_dir():
            raise RuntimeError(f"Today menu root does not exist: {root}")

        built_paths: dict[str, Path] = {}
        for category in DISH_CATEGORIES:
            category_dir = root / category
            if not category_dir.exists() or not category_dir.is_dir():
                continue

            photo_index_dir = category_dir / "indexes" / "photo"
            text_index_dir = category_dir / "indexes" / "text"

            try:
                photo_path = self.build_category_photo_index(category_dir=category_dir, output_index_dir=photo_index_dir)
                built_paths[f"{category}_photo"] = photo_path
            except RuntimeError:
                pass

            try:
                text_path = self.build_category_text_index(category_dir=category_dir, output_index_dir=text_index_dir)
                built_paths[f"{category}_text"] = text_path
            except RuntimeError:
                pass

        return built_paths

    # Build head index only when missing; return existing path otherwise.
    def build_first_head_index_if_missing(self, head_dir: str | Path) -> Path:
        head_path = Path(head_dir)
        if self._is_existing_index_dir(head_path):
            return head_path
        return self.build_first_head_index(head_path)

    # Build head index only when missing; return existing path otherwise.
    def build_second_head_index_if_missing(self, head_dir: str | Path) -> Path:
        head_path = Path(head_dir)
        if self._is_existing_index_dir(head_path):
            return head_path
        return self.build_second_head_index(head_path)

    # Rebuild all indexes for one category directory.
    def rebuild_category_indexes(self, category_dir: str | Path, indexes_root: str | Path) -> dict[str, Path]:
        category_path = Path(category_dir)
        indexes_path = Path(indexes_root)

        result: dict[str, Path] = {}
        result["photo"] = self.build_category_photo_index(
            category_dir=category_path,
            output_index_dir=indexes_path / "photo",
        )
        result["text"] = self.build_category_text_index(
            category_dir=category_path,
            output_index_dir=indexes_path / "text",
        )
        return result

    # Iterate only dish folders to keep category scanning deterministic.
    def _iter_dish_dirs(self, category_dir: Path) -> list[Path]:
        if not category_dir.exists() or not category_dir.is_dir():
            raise RuntimeError(f"Category directory does not exist: {category_dir}")
        return sorted([entry for entry in category_dir.iterdir() if entry.is_dir()])

    # Read per-dish meta.json with tolerant defaults.
    def _load_meta(self, dish_dir: Path) -> dict[str, Any]:
        meta_path = dish_dir / "meta.json"
        if not meta_path.exists():
            return {}

        with meta_path.open("r", encoding="utf-8") as fp:
            meta = json.load(fp)
        if not isinstance(meta, dict):
            raise RuntimeError(f"Dish metadata must be a JSON object: {meta_path}")
        return meta

    def _is_dish_active(self, meta: dict[str, Any]) -> bool:
        return bool(meta.get("is_active", True))

    def _collect_category_records(
        self,
        category_dir: Path,
        embedding_filename: str,
        mode: str,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        records: list[dict[str, Any]] = []
        skipped_notes: list[str] = []

        category_name = category_dir.name
        for dish_dir in self._iter_dish_dirs(category_dir):
            try:
                meta = self._load_meta(dish_dir)
            except Exception as exc:  # noqa: BLE001
                skipped_notes.append(f"{dish_dir.name}: invalid meta.json ({exc})")
                continue

            if not self._is_dish_active(meta):
                skipped_notes.append(f"{dish_dir.name}: inactive")
                continue

            embedding_path = dish_dir / embedding_filename
            if not embedding_path.exists():
                skipped_notes.append(f"{dish_dir.name}: missing {embedding_filename}")
                continue

            try:
                embedding = self._load_single_embedding(embedding_path)
            except Exception as exc:  # noqa: BLE001
                skipped_notes.append(f"{dish_dir.name}: bad embedding ({exc})")
                continue

            name = str(meta.get("name", dish_dir.name))
            category = str(meta.get("category", category_name))
            description = meta.get("description")

            if description is None and mode == "text":
                phrase_path = dish_dir / "qwen_phrase.txt"
                if phrase_path.exists():
                    description = phrase_path.read_text(encoding="utf-8").strip() or None

            metadata_item = {
                "name": name,
                "category": category,
                "description": description,
                "embedding_source_path": str(embedding_path),
                "dish_dir": str(dish_dir),
            }
            records.append({"embedding": embedding, "metadata": metadata_item})

        return records, skipped_notes

    def _resolve_dish_crop_image_path(self, dish_dir: Path) -> Path:
        explicit = dish_dir / "crop.jpg"
        if explicit.exists():
            return explicit

        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            candidate = dish_dir / f"crop{ext}"
            if candidate.exists():
                return candidate

        available = ", ".join(f"crop{ext}" for ext in SUPPORTED_IMAGE_EXTENSIONS)
        raise RuntimeError(f"Dish crop image not found in {dish_dir}. Expected one of: {available}")

    def _load_single_embedding(self, path: Path):
        self.clip_adapter._ensure_ready()  # noqa: SLF001 - shared runtime guard from adapter.
        torch = self.clip_adapter._torch  # noqa: SLF001 - adapter owns torch lifecycle.
        loaded = torch.load(path, map_location="cpu")

        if not torch.is_tensor(loaded):
            raise RuntimeError(f"Embedding file does not contain torch.Tensor: {path}")

        normalized = self.clip_adapter._normalize_embedding(loaded)  # noqa: SLF001
        if normalized.shape[0] != 1:
            raise RuntimeError(f"Expected single embedding vector in {path}, got shape {tuple(normalized.shape)}")
        return normalized[0].detach().cpu()

    # Persist index as transparent pair: embeddings.pt + metadata.json.
    def _save_index(
        self,
        embeddings: list[Any],
        metadata: list[dict[str, Any]],
        output_index_dir: Path,
        mode: str,
        notes: list[str],
    ) -> Path:
        if not embeddings:
            raise RuntimeError("Cannot save empty index")

        output_index_dir.mkdir(parents=True, exist_ok=True)
        torch = self.clip_adapter._torch  # noqa: SLF001

        stacked = torch.stack([item.detach().float().cpu() for item in embeddings], dim=0)
        normalized = self.clip_adapter._normalize_embedding(stacked).detach().cpu()  # noqa: SLF001

        embeddings_path = output_index_dir / "embeddings.pt"
        metadata_path = output_index_dir / "metadata.json"

        torch.save(normalized, embeddings_path)
        payload = {
            "mode": mode,
            "count": len(metadata),
            "notes": notes,
            "items": metadata,
        }
        # Keep metadata.json human-readable for easy audit/rebuild debugging.
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Additional sidecar with build notes keeps skip diagnostics without changing match contract.
        build_info_path = output_index_dir / "build_info.json"
        build_info_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return output_index_dir

    def _save_tensor(self, embedding: Any, output_path: Path) -> None:
        self.clip_adapter._ensure_ready()  # noqa: SLF001
        torch = self.clip_adapter._torch  # noqa: SLF001

        normalized = self.clip_adapter._normalize_embedding(embedding)  # noqa: SLF001
        if normalized.shape[0] != 1:
            raise RuntimeError(f"Expected one embedding row before saving {output_path}, got {tuple(normalized.shape)}")

        torch.save(normalized[0].detach().cpu(), output_path)

    # Parse descriptions and flatten into row-level metadata/embeddings.
    def _build_head_text_index(self, head_dir: str | Path, index_name: str) -> Path:
        head_path = Path(head_dir)
        if not head_path.exists() or not head_path.is_dir():
            raise RuntimeError(f"{index_name} directory does not exist: {head_path}")

        descriptions_path = head_path / "descriptions.txt"
        if not descriptions_path.exists():
            raise RuntimeError(f"{index_name} descriptions file not found: {descriptions_path}")

        lines = descriptions_path.read_text(encoding="utf-8").splitlines()
        texts: list[str] = []
        metadata: list[dict[str, Any]] = []

        for line_idx, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue

            chunks = [chunk.strip() for chunk in line.split("||") if chunk.strip()]
            if len(chunks) < 2:
                raise RuntimeError(
                    f"Invalid line in {descriptions_path} at {line_idx + 1}. Expected 'class || description...'."
                )

            head_class = chunks[0]
            for item_idx, description in enumerate(chunks[1:]):
                texts.append(description)
                metadata.append(
                    {
                        "name": head_class,
                        "category": index_name,
                        "description": description,
                        "embedding_source_path": str(descriptions_path),
                        "head_class": head_class,
                        "line_index": line_idx,
                        "item_index": item_idx,
                        "source_path": str(descriptions_path),
                    }
                )

        if not texts:
            raise RuntimeError(f"No valid descriptions found in {descriptions_path}")

        embeddings = self.clip_adapter.encode_texts_batch(texts).detach().cpu()
        if embeddings.ndim != 2 or embeddings.shape[0] != len(metadata):
            raise RuntimeError(
                f"Head index embeddings shape mismatch: {tuple(embeddings.shape)} vs metadata {len(metadata)}"
            )

        torch = self.clip_adapter._torch  # noqa: SLF001
        torch.save(embeddings, head_path / "embeddings.pt")
        (head_path / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        return head_path

    def _is_existing_index_dir(self, index_dir: Path) -> bool:
        return (index_dir / "embeddings.pt").exists() and (index_dir / "metadata.json").exists()

    def _join_notes(self, notes: list[str]) -> str:
        return "; ".join(notes) if notes else "none"


__all__ = ["MenuIndexBuilder"]
