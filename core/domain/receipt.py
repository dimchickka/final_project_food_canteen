"""Receipt domain contracts: final customer-facing dish names and counts only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Atomic line item in the final receipt output.
@dataclass(slots=True)
class ReceiptItem:
    dish_name: str
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"dish_name": self.dish_name, "count": self.count}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReceiptItem":
        return cls(dish_name=str(data["dish_name"]), count=int(data.get("count", 1)))


# Final receipt aggregate; intentionally keeps only business-required minimal fields.
@dataclass(slots=True)
class Receipt:
    items: list[ReceiptItem] = field(default_factory=list)

    def add_or_increment(self, dish_name: str, amount: int = 1) -> None:
        """Increment existing line item or append a new one if dish is not present."""
        if amount <= 0:
            return

        for item in self.items:
            if item.dish_name == dish_name:
                item.count += amount
                return

        self.items.append(ReceiptItem(dish_name=dish_name, count=amount))

    def merge_same_items(self) -> None:
        """Normalizes duplicates while preserving first-seen order for stable UI output."""
        merged: dict[str, int] = {}
        order: list[str] = []

        for item in self.items:
            if item.dish_name not in merged:
                order.append(item.dish_name)
                merged[item.dish_name] = 0
            merged[item.dish_name] += item.count

        self.items = [ReceiptItem(dish_name=name, count=merged[name]) for name in order]

    def to_dict(self) -> dict[str, Any]:
        return {"items": [item.to_dict() for item in self.items]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Receipt":
        return cls(items=[ReceiptItem.from_dict(item) for item in data.get("items", [])])
