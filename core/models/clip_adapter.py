"""CLIP adapter for embedding generation and index-based retrieval in menu/head tasks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from core.domain.entities import ClipMatchResult
from core.domain.enums import ClipMatchMode


class ClipAdapter:
    """Unified CLIP layer: encode image/text and match image embeddings against saved indexes."""

    def __init__(self, model_name_or_path: str, device: str = "cuda") -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._lazy_load_model()

    def warmup(self) -> None:
        """Run tiny encode calls to initialize kernels/caches before first real request."""
        self._ensure_ready()
        dummy = Image.new("RGB", (64, 64), color=(127, 127, 127))
        _ = self.encode_image(dummy)
        _ = self.encode_text("dummy")

    def encode_image(self, image: Any):
        """Encode one image and return a normalized 1D embedding tensor."""
        batch = self.encode_images_batch([image])
        return batch[0]

    def encode_images_batch(self, images: Sequence[Any]):
        """Encode an image batch and return normalized 2D embeddings [N, D]."""
        self._ensure_ready()
        if not images:
            raise ValueError("images batch is empty")

        pil_images = [self._normalize_image_input(image) for image in images]
        model_inputs = self._processor(images=pil_images, return_tensors="pt", padding=True)
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        with self._torch.no_grad():
            embeddings = self._model.get_image_features(**model_inputs)

        return self._normalize_embedding(embeddings)

    def encode_text(self, text: str):
        """Encode one text and return a normalized 1D embedding tensor."""
        batch = self.encode_texts_batch([text])
        return batch[0]

    def encode_texts_batch(self, texts: Sequence[str]):
        """Encode a text batch and return normalized 2D embeddings [N, D]."""
        self._ensure_ready()
        if not texts:
            raise ValueError("texts batch is empty")

        prepared_texts = [self._normalize_text_input(text) for text in texts]
        model_inputs = self._processor(text=prepared_texts, return_tensors="pt", padding=True, truncation=True)
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        with self._torch.no_grad():
            embeddings = self._model.get_text_features(**model_inputs)

        return self._normalize_embedding(embeddings)

    def compare_image_to_text_index(self, image: Any, index_path: str | Path) -> ClipMatchResult:
        """Return best match in a text index for the given image."""
        image_embedding = self.encode_image(image)
        index_embeddings, metadata = self._load_index(index_path)
        best_idx, best_score = self._best_match(image_embedding=image_embedding, index_embeddings=index_embeddings)
        return self._build_match_result(metadata_item=metadata[best_idx], score=best_score, mode=ClipMatchMode.TEXT)

    def compare_image_to_photo_index(self, image: Any, index_path: str | Path) -> ClipMatchResult:
        """Return best match in a photo index for the given image."""
        image_embedding = self.encode_image(image)
        index_embeddings, metadata = self._load_index(index_path)
        best_idx, best_score = self._best_match(image_embedding=image_embedding, index_embeddings=index_embeddings)
        return self._build_match_result(metadata_item=metadata[best_idx], score=best_score, mode=ClipMatchMode.PHOTO)

    def _lazy_load_model(self) -> None:
        """Lazy dependency/model loading so importing this module stays lightweight."""
        if self._model is not None and self._processor is not None and self._torch is not None:
            return

        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("ClipAdapter requires 'torch' package, but it is not installed.")
        if importlib.util.find_spec("transformers") is None:
            raise RuntimeError("ClipAdapter requires 'transformers' package, but it is not installed.")

        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self._processor = CLIPProcessor.from_pretrained(self.model_name_or_path)
        self._model = CLIPModel.from_pretrained(self.model_name_or_path)
        self._model.to(self.device)
        self._model.eval()

    def _ensure_ready(self) -> None:
        if self._model is None or self._processor is None or self._torch is None:
            raise RuntimeError("CLIP model is not ready. Check dependencies and model path.")

    # Normalize image inputs from OpenCV/PIL/path to deterministic RGB PIL.
    def _normalize_image_input(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return Image.fromarray(image).convert("RGB")
            if image.ndim == 3 and image.shape[-1] == 1:
                return Image.fromarray(image[:, :, 0]).convert("RGB")
            if image.ndim == 3 and image.shape[-1] == 3:
                rgb = image[..., ::-1]  # OpenCV BGR -> RGB.
                return Image.fromarray(rgb).convert("RGB")
            if image.ndim == 3 and image.shape[-1] == 4:
                rgba = image[..., [2, 1, 0, 3]]  # OpenCV BGRA -> RGBA.
                return Image.fromarray(rgba, mode="RGBA").convert("RGB")
            raise ValueError("Unsupported numpy image shape for CLIP. Expected HxW, HxWx1, HxWx3 or HxWx4.")

        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise RuntimeError(f"Image path does not exist: {image_path}")
            return Image.open(image_path).convert("RGB")

        raise ValueError(f"Unsupported image type for CLIP input: {type(image)!r}")

    def _normalize_text_input(self, text: str) -> str:
        normalized = str(text).strip()
        if not normalized:
            raise ValueError("Text input for CLIP cannot be empty")
        return normalized

    def _normalize_embedding(self, embedding):
        tensor = embedding.detach().float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise RuntimeError(f"Invalid embedding shape: {tuple(tensor.shape)}")

        norms = tensor.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        normalized = tensor / norms
        return normalized

    def _best_match(self, image_embedding, index_embeddings) -> tuple[int, float]:
        if index_embeddings.ndim != 2:
            raise RuntimeError(f"Index embeddings must be 2D [N, D], got {tuple(index_embeddings.shape)}")
        if image_embedding.ndim != 1:
            raise RuntimeError(f"Image embedding must be 1D [D], got {tuple(image_embedding.shape)}")
        if index_embeddings.shape[1] != image_embedding.shape[0]:
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"image={image_embedding.shape[0]}, index={index_embeddings.shape[1]}"
            )

        scores = index_embeddings @ image_embedding
        best_idx = int(self._torch.argmax(scores).item())
        best_score = float(scores[best_idx].item())
        return best_idx, best_score

    def _resolve_index_files(self, index_path: str | Path) -> tuple[Path, Path]:
        root = Path(index_path)
        if not root.exists():
            raise RuntimeError(f"Index path does not exist: {root}")
        if not root.is_dir():
            raise RuntimeError(
                f"Index path must be a directory with embeddings.pt and metadata.json, got file: {root}"
            )

        embeddings_path = root / "embeddings.pt"
        metadata_path = root / "metadata.json"

        if not embeddings_path.exists():
            raise RuntimeError(f"Index embeddings file not found: {embeddings_path}")
        if not metadata_path.exists():
            raise RuntimeError(f"Index metadata file not found: {metadata_path}")

        return embeddings_path, metadata_path

    def _load_index(self, index_path: str | Path) -> tuple[Any, list[dict[str, Any]]]:
        """Load normalized index vectors and row-aligned metadata."""
        self._ensure_ready()
        embeddings_path, metadata_path = self._resolve_index_files(index_path)

        loaded = self._torch.load(embeddings_path, map_location="cpu")
        embeddings = self._normalize_embedding(loaded)

        with metadata_path.open("r", encoding="utf-8") as fp:
            metadata = json.load(fp)

        if not isinstance(metadata, list):
            raise RuntimeError(f"metadata.json must contain a JSON array: {metadata_path}")
        if embeddings.shape[0] != len(metadata):
            raise RuntimeError(
                "Index size mismatch: "
                f"embeddings rows={embeddings.shape[0]}, metadata rows={len(metadata)} in {index_path}"
            )

        return embeddings.to(self.device), metadata

    def _build_match_result(self, metadata_item: dict[str, Any], score: float, mode: ClipMatchMode) -> ClipMatchResult:
        return ClipMatchResult(
            matched_name=str(metadata_item.get("name", "unknown")),
            matched_category=str(metadata_item.get("category", "unknown")),
            score=float(score),
            mode=mode,
            matched_description=metadata_item.get("description"),
            embedding_source_path=metadata_item.get("embedding_source_path"),
        )


__all__ = ["ClipAdapter"]
