"""Unified adapter for all Qwen-VL tasks used by the tray-recognition pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config.settings import (
    QWEN_PHRASE_MAX_NEW_TOKENS,
    QWEN_SAUCES_MAX_NEW_TOKENS,
    QWEN_VALIDATION_MAX_NEW_TOKENS,
)
from core.domain.dto import SauceDetectionResult, ValidationResult
from core.parsing.qwen_sauce_parser import parse_qwen_sauce_response
from core.parsing.qwen_validation_parser import parse_qwen_validation_response


class QwenAdapter:
    """Thin integration layer around Qwen model + task-specific prompts/parsers."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        validation_max_new_tokens: int = QWEN_VALIDATION_MAX_NEW_TOKENS,
        sauce_max_new_tokens: int = QWEN_SAUCES_MAX_NEW_TOKENS,
        phrase_max_new_tokens: int = QWEN_PHRASE_MAX_NEW_TOKENS,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.validation_max_new_tokens = int(validation_max_new_tokens)
        self.sauce_max_new_tokens = int(sauce_max_new_tokens)
        self.phrase_max_new_tokens = int(phrase_max_new_tokens)

        self._model: Any | None = None
        self._processor: Any | None = None

        self._lazy_load_model()

    def _lazy_load_model(self) -> None:
        """Load processor/model once; keep imports lazy for lightweight module import."""
        if self._model is not None and self._processor is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Qwen dependencies are not available. Install 'torch' and 'transformers'."
            ) from exc

        torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self._model.to(self.device)
        self._model.eval()

    # Safe minimal warmup to initialize CUDA kernels / model caches.
    def warmup(self) -> None:
        self._ensure_ready()
        dummy = Image.new("RGB", (64, 64), color=(127, 127, 127))
        _ = self._run_generation(
            image=dummy,
            prompt_text='Return JSON only: {"status":"unknown"}',
            max_new_tokens=8,
        )

    # Scenario 1: validate tray visibility/quality before costly downstream processing.
    def validate_tray_visibility(self, image: Any) -> ValidationResult:
        prompt = self._build_validation_prompt()
        raw_text = self._run_generation(
            image=image,
            prompt_text=prompt,
            max_new_tokens=self.validation_max_new_tokens,
        )
        return parse_qwen_validation_response(raw_text)

    # Scenario 2: detect sauce regions on crop classified as "other_dish".
    def detect_sauces_on_other_dish(self, image: Any) -> SauceDetectionResult:
        prompt = self._build_sauce_prompt()
        raw_text = self._run_generation(
            image=image,
            prompt_text=prompt,
            max_new_tokens=self.sauce_max_new_tokens,
        )
        return parse_qwen_sauce_response(raw_text)

    # Scenario 3: produce a short visual phrase for CLIP text embedding pipeline.
    def generate_short_dish_phrase(self, image: Any, dish_name: str, category: str) -> str:
        prompt = self._build_phrase_prompt(dish_name=dish_name, category=category)
        raw_text = self._run_generation(
            image=image,
            prompt_text=prompt,
            max_new_tokens=self.phrase_max_new_tokens,
        )
        return self._normalize_phrase(raw_text=raw_text, dish_name=dish_name)

    def _ensure_ready(self) -> None:
        if self._model is None or self._processor is None:
            raise RuntimeError("Qwen model is not loaded. Check model path and dependencies.")

    def _normalize_image(self, image: Any) -> Image.Image:
        """Accept ndarray/PIL/path-like inputs and normalize to RGB PIL.Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return Image.fromarray(image).convert("RGB")
            if image.ndim == 3 and image.shape[-1] in (3, 4):
                mode = "RGBA" if image.shape[-1] == 4 else "RGB"
                return Image.fromarray(image, mode=mode).convert("RGB")
            raise ValueError("Unsupported numpy image shape. Expected HxW, HxWx3 or HxWx4.")

        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise RuntimeError(f"Image path does not exist: {image_path}")
            return Image.open(image_path).convert("RGB")

        raise ValueError("Unsupported image input type for QwenAdapter.")

    def _run_generation(self, image: Any, prompt_text: str, max_new_tokens: int) -> str:
        """Core Qwen call shared by all task-specific scenarios."""
        self._ensure_ready()

        pil_image = self._normalize_image(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        prompt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self._processor(text=[prompt], images=[pil_image], padding=True, return_tensors="pt")
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        generated_ids = self._model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_length = model_inputs["input_ids"].shape[-1]
        completion_ids = generated_ids[:, prompt_length:]
        decoded = self._processor.batch_decode(completion_ids, skip_special_tokens=True)

        return decoded[0].strip() if decoded else ""

    def _build_validation_prompt(self) -> str:
        return (
            "Assess whether this is a usable tray photo for automated dish recognition. "
            "Return JSON only with keys: status, reason, confidence. "
            "status must be one of: valid, invalid, unknown. "
            "Set invalid only for serious blocking issues (hand occlusion, tray mostly cut, severe blur)."
        )

    def _build_sauce_prompt(self) -> str:
        return (
            "Find sauce regions on this single dish image. "
            "Return JSON only with keys: boxes, notes. "
            "boxes must be a list of objects with x1,y1,x2,y2,label. "
            "Coordinates are integer pixel positions. "
            "If no sauce is visible, return {\"boxes\": [], \"notes\": \"no sauce\"}."
        )

    def _build_phrase_prompt(self, dish_name: str, category: str) -> str:
        return (
            "Generate one short visual Russian phrase for this dish image. "
            f"Dish name: {dish_name}. Category: {category}. "
            "Rules: 2-6 words, concrete appearance cues (color/texture/form), no long sentence, "
            "do not repeat dish name verbatim. Output only phrase text."
        )

    def _normalize_phrase(self, raw_text: str, dish_name: str) -> str:
        text = (raw_text or "").strip()
        if not text:
            return "блюдо на тарелке"

        stripped = text.strip("` \n\t\"")

        # If model still returns JSON, recover phrase from common keys.
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                for key in ("phrase", "text", "description"):
                    value = parsed.get(key)
                    if isinstance(value, str) and value.strip():
                        stripped = value.strip()
                        break
        except json.JSONDecodeError:
            pass

        # Keep phrase compact for downstream CLIP text embedding.
        one_line = " ".join(stripped.split())
        if not one_line:
            return "блюдо на тарелке"

        dish_name_l = dish_name.strip().lower()
        if one_line.lower() == dish_name_l:
            return "аппетитное блюдо на подносе"

        words = one_line.split(" ")
        return " ".join(words[:8])


__all__ = ["QwenAdapter"]
