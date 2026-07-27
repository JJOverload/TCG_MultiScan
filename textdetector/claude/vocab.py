"""
vocab.py

Loads the MTG card-name vocabulary from an AtomicCards.json (MTGJSON format)
and provides fuzzy-match autocorrection of OCR output against it.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import textdistance


def _split_double_faced_name(name: str) -> List[str]:
    """"Foo // Bar" -> ["Foo", "Bar"]; single-faced names pass through unchanged."""
    if " // " in name:
        front, back = name.split(" // ", 1)
        return [front, back]
    return [name]


@dataclass
class Vocabulary:
    """Card names (including split double-faced names) plus card text/type
    strings, all treated as one flat vocabulary for fuzzy matching against
    OCR output. `non_name_words` lets callers flag a "match" that's
    actually rules text, not a real card name.
    """

    words: Set[str]
    word_counts: Counter
    word_probabilities: Dict[str, float]
    non_name_words: Set[str]

    @classmethod
    def load(cls, json_path: Path) -> "Vocabulary":
        with open(json_path, "r", encoding="utf-8") as f:
            atomic_cards = json.load(f)["data"]

        names: List[str] = []
        non_names: List[str] = []

        for card_name, faces in atomic_cards.items():
            names.extend(_split_double_faced_name(card_name))
            for face in faces:
                if "text" in face:
                    non_names.append(json.dumps(face["text"]))
                if "type" in face:
                    non_names.append(json.dumps(face["type"]))

        all_words = names + non_names
        word_counts = Counter(all_words)
        total = sum(word_counts.values())
        word_probabilities = {w: c / total for w, c in word_counts.items()}

        return cls(
            words=set(all_words),
            word_counts=word_counts,
            word_probabilities=word_probabilities,
            non_name_words=set(non_names),
        )


@dataclass
class AutocorrectMatch:
    name: str
    probability: float
    similarity: float


def autocorrect(word: str, vocab: Vocabulary) -> AutocorrectMatch:
    """Return the vocabulary entry most similar to `word` (Jaccard trigram
    similarity, tie-broken by corpus frequency).

    Same behavior as the original `mtg_autocorrect`, but returns a typed
    result instead of a one-row DataFrame accessed by position
    (`output.iat[0, 2]`) — that indexing broke silently if a column ever
    got reordered upstream.
    """
    # Jaccard(qval=3) needs >= 2 characters to form a trigram; pad short or
    # empty OCR output so it doesn't raise on ultra-short reads.
    padded = word.ljust(3)

    scored = [
        (name, prob, 1 - textdistance.Jaccard(qval=3).distance(name, padded))
        for name, prob in vocab.word_probabilities.items()
    ]
    df = pd.DataFrame(scored, columns=["Name", "Prob", "Similarity"])
    best = df.sort_values(["Similarity", "Prob"], ascending=False).iloc[0]
    return AutocorrectMatch(
        name=best["Name"], probability=best["Prob"], similarity=best["Similarity"]
    )
