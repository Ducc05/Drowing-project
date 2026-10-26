"""Display classifier outputs without confusing them with detector confidence."""
from __future__ import annotations

from typing import Any

SCORE_KEYS = ("drowning", "swimming", "out_of_water")


def class_scores_text(scores: Any) -> str:
    if scores is None or len(scores) != len(SCORE_KEYS):
        return ""
    return " | ".join(f"{label}={float(value):.3f}" for label, value in zip(SCORE_KEYS, scores))
