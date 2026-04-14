"""Receipt flow aggregates recognized dish items into customer-facing receipt."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

from core.domain.dto import DishRecognitionResult
from core.domain.entities import RecognitionEvidence
from core.domain.enums import DishDetectionSource
from core.domain.receipt import Receipt


class ReceiptFlow:
    """Converts raw recognized items to merged dish counts and final Receipt."""

    def build_receipt(self, items: Sequence[DishRecognitionResult]) -> Receipt:
        """Builds receipt with only dish names and counts."""
        receipt = Receipt()
        for item in items:
            receipt.add_or_increment(item.dish_name, item.count)
        receipt.merge_same_items()
        return receipt

    def aggregate_items(self, items: Sequence[DishRecognitionResult]) -> list[DishRecognitionResult]:
        """Merges equal dish names while keeping first-seen order."""
        merged: OrderedDict[str, DishRecognitionResult] = OrderedDict()

        for item in items:
            if item.dish_name not in merged:
                merged[item.dish_name] = DishRecognitionResult(
                    dish_name=item.dish_name,
                    category=item.category,
                    count=max(1, int(item.count)),
                    bbox=item.bbox,
                    evidences=list(item.evidences),
                )
                continue

            existing = merged[item.dish_name]
            existing.count += max(1, int(item.count))
            if existing.category is None and item.category is not None:
                existing.category = item.category
            existing.evidences.extend(item.evidences)

        # Add deterministic aggregation evidence for traceability.
        for aggregated in merged.values():
            aggregated.evidences.append(
                RecognitionEvidence(
                    source=DishDetectionSource.AGGREGATED,
                    model_name="receipt_flow",
                    chosen_label=aggregated.dish_name,
                    notes=f"aggregated_count={aggregated.count}",
                    bbox=aggregated.bbox,
                )
            )

        return list(merged.values())
