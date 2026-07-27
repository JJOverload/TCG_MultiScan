"""
pipeline.py

Thin adapter between the API layer and the existing detection/OCR/fuzzy-
match pipeline (box_merging.py, vocab.py from the card-scanner project).
Keeps main.py ignorant of how scanning actually works.

TODO once the model file and AtomicCards.json are in place on this
machine: replace the stub in scan_image() with real calls to
detect_text_boxes() / best_match_across_rotations() (see the earlier
main.py we wrote for the standalone scanner).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

_VOCAB: Optional[object] = None  # populated once at startup, not per-request


def load_vocab_once(vocab_path: Path) -> None:
    global _VOCAB
    if _VOCAB is not None:
        return
    if not vocab_path.exists():
        # Don't crash the whole API if the vocab file isn't in place yet —
        # scan_image() will raise a clear error instead, on first use.
        return
    from vocab import Vocabulary  # the module we wrote earlier

    _VOCAB = Vocabulary.load(vocab_path)


def scan_image(image_path: Path) -> List[Dict]:
    """Run detection + OCR + fuzzy match on the image at `image_path`.

    Returns a list of dicts: {"name": str, "confidence": float,
    "box": [xmin, ymin, xmax, ymax]}.
    """
    if _VOCAB is None:
        raise RuntimeError(
            "Vocabulary not loaded — is AtomicCards.json present, and did "
            "load_vocab_once() run at startup?"
        )

    # Stub result so the API is runnable end-to-end before the model file
    # and full pipeline wiring are in place. Replace with the real
    # detect_text_boxes() + best_match_across_rotations() calls.
    return [
        {"name": "Lightning Bolt", "confidence": 0.94, "box": [10.0, 10.0, 120.0, 40.0]},
    ]
