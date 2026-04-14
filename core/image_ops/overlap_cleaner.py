"""Pragmatic bbox overlap cleaning utilities for tray plate regions."""

from __future__ import annotations

from collections.abc import Sequence

from core.domain.entities import DetectionBox


class OverlapCleaner:
    """Heuristic overlap reducer, not a geometrically perfect segmentation cutout."""

    MIN_SIDE_PX = 8

    @staticmethod
    def _normalize(box: DetectionBox) -> DetectionBox:
        x1, x2 = sorted((box.x1, box.x2))
        y1, y2 = sorted((box.y1, box.y2))
        return DetectionBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=box.confidence,
            class_name=box.class_name,
            label=box.label,
            source=box.source,
        )

    @classmethod
    def compute_iou(cls, box_a: DetectionBox, box_b: DetectionBox) -> float:
        """Compute IoU in [0, 1] for two axis-aligned boxes."""
        a = cls._normalize(box_a)
        b = cls._normalize(box_b)
        inter = cls.intersect_boxes(a, b)
        if inter is None:
            return 0.0

        inter_area = inter.area
        if inter_area <= 0:
            return 0.0

        union = a.area + b.area - inter_area
        if union <= 0:
            return 0.0

        return inter_area / union

    @classmethod
    def intersect_boxes(cls, box_a: DetectionBox, box_b: DetectionBox) -> DetectionBox | None:
        """Return intersection rectangle if present, otherwise None."""
        a = cls._normalize(box_a)
        b = cls._normalize(box_b)

        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return DetectionBox(x1=x1, y1=y1, x2=x2, y2=y2)

    @classmethod
    def subtract_overlap_from_box(
        cls,
        base_box: DetectionBox,
        overlap_box: DetectionBox,
    ) -> DetectionBox:
        """Shrink base box by cutting the largest valid side outside overlap."""
        base = cls._normalize(base_box)
        overlap = cls.intersect_boxes(base, overlap_box)
        if overlap is None:
            return base

        # Candidate rectangles: left/right/top/bottom leftovers after overlap removal.
        candidates = [
            DetectionBox(base.x1, base.y1, overlap.x1, base.y2),
            DetectionBox(overlap.x2, base.y1, base.x2, base.y2),
            DetectionBox(base.x1, base.y1, base.x2, overlap.y1),
            DetectionBox(base.x1, overlap.y2, base.x2, base.y2),
        ]

        valid = [
            candidate
            for candidate in candidates
            if candidate.width >= cls.MIN_SIDE_PX and candidate.height >= cls.MIN_SIDE_PX
        ]
        if not valid:
            return DetectionBox(0, 0, 0, 0, confidence=base.confidence, class_name=base.class_name)

        best = max(valid, key=lambda candidate: candidate.area)
        best.confidence = base.confidence
        best.class_name = base.class_name
        best.label = base.label
        best.source = base.source
        return best

    @classmethod
    def clean_overlapping_boxes(cls, boxes: Sequence[DetectionBox]) -> list[DetectionBox]:
        """Resolve overlaps by shrinking lower-priority boxes and dropping tiny residues."""
        normalized = [cls._normalize(box) for box in boxes]

        # Higher confidence first; area is a secondary tiebreaker.
        def sort_key(box: DetectionBox) -> tuple[float, int]:
            return (box.confidence if box.confidence is not None else 0.0, box.area)

        ordered = sorted(normalized, key=sort_key, reverse=True)
        cleaned: list[DetectionBox] = []

        for candidate in ordered:
            current = candidate
            for strong_box in cleaned:
                if cls.intersect_boxes(current, strong_box) is None:
                    continue
                current = cls.subtract_overlap_from_box(current, strong_box)
                if current.width < cls.MIN_SIDE_PX or current.height < cls.MIN_SIDE_PX:
                    break

            if current.width >= cls.MIN_SIDE_PX and current.height >= cls.MIN_SIDE_PX:
                cleaned.append(current)

        return cleaned
