"""
box_merging.py

Bounding-box decoding and merging for the EAST text-detector pipeline.
Pulled out of the main script so it can be understood, tested, and reused
independently of the OpenCV/Tesseract I/O happening around it.

Covers what was previously: decode(), calc_sim(), merge_boxes(),
compare_vertices_with_box_shows_overlap(), is_overlap(), and merge_algo().
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in (xmin, ymin, xmax, ymax) form."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def corner_distance(self, other: "BoundingBox") -> float:
        """Cheap "how close are these boxes" heuristic (not true geometric
        distance) — the smallest gap between any pair of x-edges plus the
        smallest gap between any pair of y-edges. Mirrors the original
        `calc_sim`, just renamed to say what it actually measures.
        """
        x_dist = min(
            abs(self.xmin - other.xmin), abs(self.xmin - other.xmax),
            abs(self.xmax - other.xmin), abs(self.xmax - other.xmax),
        )
        y_dist = min(
            abs(self.ymin - other.ymin), abs(self.ymin - other.ymax),
            abs(self.ymax - other.ymin), abs(self.ymax - other.ymax),
        )
        return x_dist + y_dist

    def overlaps(self, other: "BoundingBox") -> bool:
        """True if the two boxes intersect at all.

        The original checked this by testing whether any *corner* of one
        box fell inside the other. That misses cases where two boxes cross
        in a "+" shape without either containing a corner of the other —
        e.g. a tall thin box and a wide short box crossing in the middle.
        A standard axis-aligned bounding-box (AABB) test handles every
        case in four comparisons and needs no vertex list at all.
        """
        return not (
            self.xmax < other.xmin
            or other.xmax < self.xmin
            or self.ymax < other.ymin
            or other.ymax < self.ymin
        )

    def merge(self, other: "BoundingBox") -> "BoundingBox":
        """Smallest box that contains both boxes."""
        return BoundingBox(
            xmin=min(self.xmin, other.xmin),
            ymin=min(self.ymin, other.ymin),
            xmax=max(self.xmax, other.xmax),
            ymax=max(self.ymax, other.ymax),
        )


def merge_close_boxes(boxes: List[BoundingBox], dist_limit: float) -> List[BoundingBox]:
    """Repeatedly merge boxes that overlap or sit within `dist_limit` of
    each other, until no more merges are possible.

    Equivalent to the original `merge_algo`, but doesn't mutate a list
    while iterating over it. The original ran
    `for j in bboxes: for k in bboxes: ... bboxes.append(...); bboxes.remove(...)`
    — appending/removing from a list you're actively looping over is a
    classic source of skipped or duplicated elements in Python, because
    indices shift out from under the iterator mid-loop. It happened to
    work here since the function returns immediately after the first merge
    each call, but it's fragile and easy to break with a small edit later.
    This version builds a fresh list on each merge instead.
    """
    remaining = list(boxes)
    merged_any = True
    while merged_any:
        merged_any = False
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                a, b = remaining[i], remaining[j]
                if a.corner_distance(b) < dist_limit or a.overlaps(b):
                    new_box = a.merge(b)
                    remaining = [
                        box for k, box in enumerate(remaining) if k not in (i, j)
                    ] + [new_box]
                    merged_any = True
                    break
            if merged_any:
                break
    return remaining


@dataclass(frozen=True)
class Detection:
    """A single rotated-box text detection from the EAST model."""

    center: Point
    width: float
    height: float
    angle_degrees: float
    confidence: float


def decode(scores, geometry, score_thresh: float) -> List[Detection]:
    """Decode raw EAST model output into rotated-box detections above
    `score_thresh`.

    `scores` has shape (1, 1, H, W); `geometry` has shape (1, 5, H, W),
    where the 5 channels are the standard EAST layout: distance to the
    top/right/bottom/left edge of the text box, then rotation angle.
    Same math as the original `decode()` — renamed variables and added
    shape validation that raises a clear error instead of a bare assert.
    """
    if scores.ndim != 4 or geometry.ndim != 4:
        raise ValueError("scores and geometry must both be 4-D arrays")
    if scores.shape[:2] != (1, 1) or geometry.shape[:2] != (1, 5):
        raise ValueError(
            f"Unexpected EAST output shapes: scores={scores.shape}, "
            f"geometry={geometry.shape}"
        )
    if scores.shape[2:] != geometry.shape[2:]:
        raise ValueError("scores and geometry spatial dimensions must match")

    height, width = scores.shape[2], scores.shape[3]
    detections: List[Detection] = []

    for y in range(height):
        scores_row = scores[0][0][y]
        dist_top = geometry[0][0][y]
        dist_right = geometry[0][1][y]
        dist_bottom = geometry[0][2][y]
        dist_left = geometry[0][3][y]
        angles_row = geometry[0][4][y]

        for x in range(width):
            confidence = float(scores_row[x])
            if confidence < score_thresh:
                continue

            offset_x, offset_y = x * 4.0, y * 4.0
            angle = angles_row[x]
            cos_a, sin_a = math.cos(angle), math.sin(angle)

            box_h = dist_top[x] + dist_bottom[x]
            box_w = dist_right[x] + dist_left[x]

            offset = (
                offset_x + cos_a * dist_right[x] + sin_a * dist_bottom[x],
                offset_y - sin_a * dist_right[x] + cos_a * dist_bottom[x],
            )
            p1 = (-sin_a * box_h + offset[0], -cos_a * box_h + offset[1])
            p3 = (-cos_a * box_w + offset[0], sin_a * box_w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))

            detections.append(
                Detection(
                    center=center,
                    width=box_w,
                    height=box_h,
                    angle_degrees=-angle * 180.0 / math.pi,
                    confidence=confidence,
                )
            )
    return detections


# --------------------------------------------------------------------------
# Demo / sanity checks. Run this file directly (`python box_merging.py`) to
# see it exercise the merge logic on a small made-up example.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Two boxes far apart (should NOT merge) and two that are close/overlapping
    # (should merge into one).
    far_apart = [
        BoundingBox(0, 0, 10, 10),
        BoundingBox(500, 500, 520, 520),
    ]
    result = merge_close_boxes(far_apart, dist_limit=40)
    assert len(result) == 2, "Far-apart boxes should not merge"
    print("OK: distant boxes stay separate ->", result)

    close_together = [
        BoundingBox(0, 0, 10, 10),
        BoundingBox(15, 2, 25, 12),   # within dist_limit of the first
        BoundingBox(500, 500, 520, 520),  # unrelated, stays separate
    ]
    result = merge_close_boxes(close_together, dist_limit=40)
    assert len(result) == 2, "Close boxes should merge into one"
    print("OK: nearby boxes merge, distant one stays ->", result)

    # Cross-shaped overlap that the old corner-in-box test would have missed:
    tall_thin = BoundingBox(45, 0, 55, 100)
    wide_short = BoundingBox(0, 45, 100, 55)
    assert tall_thin.overlaps(wide_short), "Cross-shaped overlap should be detected"
    print("OK: cross-shaped overlap correctly detected")

    print("\nAll checks passed.")
