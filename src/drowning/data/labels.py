"""Conservative, auditable source label normalization and mapping."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

CANONICAL_LABELS = {"drowning", "swimming", "out_of_water"}


def normalize_label(value: str) -> str:
    """Normalize only typography and separators, never semantic similarity."""
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    return re.sub(r"[\s_-]+", " ", normalized)


@dataclass(frozen=True)
class LabelDecision:
    original_label: str
    normalized_label: str
    action: str
    canonical_label: str | None
    reason: str | None = None


def resolve_label(label: str, mapping: dict[str, dict[str, object]]) -> LabelDecision:
    normalized_mapping = {normalize_label(key): value for key, value in mapping.items()}
    rule = normalized_mapping.get(normalize_label(label))
    if rule is None:
        return LabelDecision(label, normalize_label(label), "review", None, "unmapped_label")
    action = str(rule.get("action", ""))
    if action == "map":
        target = rule.get("target")
        if target not in CANONICAL_LABELS:
            raise ValueError(f"Invalid canonical mapping target: {target!r}")
        return LabelDecision(label, normalize_label(label), action, str(target))
    if action in {"ignore", "context", "review"}:
        return LabelDecision(label, normalize_label(label), action, None, str(rule.get("reason", "")) or None)
    raise ValueError(f"Unsupported mapping action for {label!r}: {action!r}")
