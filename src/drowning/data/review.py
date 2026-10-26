"""Human review records for label mappings and source-group verification.

A review is a small JSON file under ``reviews/`` that names the reviewer and is bound to the exact
mapping (by hash) or exact group assignment (by hash) that was reviewed. Changing the mapping
invalidates the record, so approval can never silently carry over to different semantics.
Nothing here approves anything by itself: ``record_*`` functions require an explicit reviewer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from drowning.data.labels import CANONICAL_LABELS, normalize_label

# Names whose meaning differs between datasets; they may be mapped only with a written definition.
AMBIGUOUS_SOURCE_LABELS = {
    "normal", "not drowning", "safe", "person", "human", "person in water", "standing",
    "struggling", "distress", "potential drowning", "diving", "floating", "head up", "head down",
}
MIN_SAMPLES_PER_MAPPED_LABEL = 10


class ReviewError(ValueError):
    pass


def default_reviews_dir() -> Path:
    override = os.environ.get("DROWNING_REVIEWS_DIR", "").strip()
    return Path(override) if override else Path(__file__).resolve().parents[3] / "reviews"


def mapping_fingerprint(source: dict[str, Any]) -> str:
    payload = json.dumps(source.get("mapping", {}), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-") or "unnamed"


def review_path(source: dict[str, Any], kind: str, reviews_dir: Path | None = None) -> Path:
    if kind not in {"mapping", "grouping"}:
        raise ValueError("unknown review kind")
    directory = reviews_dir or default_reviews_dir()
    return directory / f"{_slug(source['source_id'])}@{_slug(source['source_version'])}.{kind}-review.json"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def load_mapping_review(source: dict[str, Any], reviews_dir: Path | None = None) -> dict[str, Any] | None:
    """Return the stored approval only if it was made for the current mapping."""
    record = _read(review_path(source, "mapping", reviews_dir))
    if (record is None or record.get("status") != "approved"
            or record.get("mapping_hash") != mapping_fingerprint(source)):
        return None
    return record


def groups_fingerprint(by_path: dict[str, str]) -> str:
    payload = json.dumps(sorted(by_path.items()), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_grouping_review(source: dict[str, Any], reviews_dir: Path | None = None) -> dict[str, Any] | None:
    record = _read(review_path(source, "grouping", reviews_dir))
    by_path = record.get("by_path") if record else None
    if (record is None or record.get("status") != "verified" or not isinstance(by_path, dict)
            or record.get("groups_hash") != groups_fingerprint(by_path)):
        return None
    return record


def apply_reviews(source: dict[str, Any], reviews_dir: Path | None = None) -> dict[str, Any]:
    """Return a copy of ``source`` with valid stored reviews applied (YAML flags are untouched on disk)."""
    result = copy.deepcopy(source)
    mapping = load_mapping_review(result, reviews_dir)
    if mapping is not None:
        result["mapping_review"] = {"status": "approved", "sample_review_required": False,
                                    "reviewer": mapping["reviewer"], "reviewed_at": mapping["reviewed_at"],
                                    "source": "reviews_directory"}
    grouping = load_grouping_review(result, reviews_dir)
    if grouping is not None:
        result["_reviewed_groups"] = grouping["by_path"]
        result["grouping_review"] = {"status": "verified", "reviewer": grouping["reviewer"],
                                     "reviewed_at": grouping["reviewed_at"], "method": grouping.get("method")}
    return result


def is_mapping_approved(source: dict[str, Any]) -> bool:
    """True when the (already review-overlaid) descriptor carries an approved, fully reviewed mapping."""
    review = source.get("mapping_review", {})
    return isinstance(review, dict) and review.get("status") == "approved" and not review.get("sample_review_required", True)


def mapping_review_requirements(source: dict[str, Any]) -> list[dict[str, Any]]:
    """What must be confirmed before approval: one entry per source label with a ``map`` action."""
    requirements = []
    for label, rule in source.get("mapping", {}).items():
        if not isinstance(rule, dict) or rule.get("action") != "map":
            continue
        ambiguous = normalize_label(label) in AMBIGUOUS_SOURCE_LABELS
        requirements.append({"source_label": label, "canonical_label": rule.get("target"),
                             "ambiguous_name": ambiguous, "definition_note_required": ambiguous,
                             "min_samples_viewed": MIN_SAMPLES_PER_MAPPED_LABEL})
    return requirements


def _write_new(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def record_mapping_review(source: dict[str, Any], *, reviewer: str, confirmed_labels: list[str],
                          samples_viewed: dict[str, int], notes: dict[str, str] | None = None,
                          reviews_dir: Path | None = None, now: datetime | None = None,
                          reviewer_kind: str = "human", delegated_by: str | None = None,
                          evidence_sha256: str | None = None, rules_version: str | None = None) -> dict[str, Any]:
    """Store an approval for the current mapping after checking every mapped label was confirmed."""
    reviewer = (reviewer or "").strip()
    if not reviewer:
        raise ReviewError("reviewer name is required")
    notes = notes or {}
    confirmed = {normalize_label(label) for label in confirmed_labels}
    viewed = {normalize_label(label): count for label, count in samples_viewed.items()}
    problems: list[str] = []
    for requirement in mapping_review_requirements(source):
        key = normalize_label(requirement["source_label"])
        if key not in confirmed:
            problems.append(f"{requirement['source_label']!r}: mapping not confirmed")
        if viewed.get(key, 0) < MIN_SAMPLES_PER_MAPPED_LABEL:
            problems.append(f"{requirement['source_label']!r}: at least {MIN_SAMPLES_PER_MAPPED_LABEL} sample crops must be viewed")
        if requirement["definition_note_required"] and len(notes.get(requirement["source_label"], "").strip()) < 20:
            problems.append(f"{requirement['source_label']!r}: ambiguous source name needs a written definition (20+ characters)")
    if not source.get("mapping"):
        problems.append("source has no mapping to approve")
    if problems:
        raise ReviewError("; ".join(problems))
    record = {
        "schema_version": 1, "status": "approved", "source_id": source["source_id"],
        "source_version": str(source["source_version"]), "mapping_hash": mapping_fingerprint(source),
        "mapping": {label: dict(rule) for label, rule in source["mapping"].items()},
        "reviewer": reviewer, "reviewed_at": (now or datetime.now(timezone.utc)).isoformat(),
        "confirmed_labels": sorted(confirmed_labels), "samples_viewed": dict(samples_viewed), "notes": notes,
        "reviewer_kind": reviewer_kind, "delegated_by": delegated_by, "evidence_sha256": evidence_sha256,
        "rules_version": rules_version,
    }
    _write_new(review_path(source, "mapping", reviews_dir), record)
    return record


def record_grouping_review(source: dict[str, Any], *, reviewer: str, by_path: dict[str, str], method: str,
                           evidence: dict[str, Any], note: str, reviews_dir: Path | None = None,
                           now: datetime | None = None, reviewer_kind: str = "human",
                           delegated_by: str | None = None, evidence_sha256: str | None = None,
                           rules_version: str | None = None) -> dict[str, Any]:
    """Store reviewer-verified source groups (path -> group id)."""
    reviewer = (reviewer or "").strip()
    if not reviewer:
        raise ReviewError("reviewer name is required")
    if len(note.strip()) < 20:
        raise ReviewError("a written basis for treating these groups as source-disjoint is required (20+ characters)")
    if not by_path or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in by_path.items()):
        raise ReviewError("group assignment must map every media path to a nonempty group id")
    record = {
        "schema_version": 1, "status": "verified", "source_id": source["source_id"],
        "source_version": str(source["source_version"]), "method": method, "reviewer": reviewer,
        "reviewed_at": (now or datetime.now(timezone.utc)).isoformat(), "note": note.strip(),
        "evidence": evidence, "groups_hash": groups_fingerprint(by_path), "by_path": by_path,
        "reviewer_kind": reviewer_kind, "delegated_by": delegated_by, "evidence_sha256": evidence_sha256,
        "rules_version": rules_version,
    }
    _write_new(review_path(source, "grouping", reviews_dir), record)
    return record


def review_status(source: dict[str, Any], reviews_dir: Path | None = None) -> dict[str, Any]:
    stored = _read(review_path(source, "mapping", reviews_dir))
    current = mapping_fingerprint(source)
    grouping = load_grouping_review(source, reviews_dir)
    return {
        "mapping_hash": current,
        "mapping_review": ("approved" if load_mapping_review(source, reviews_dir) is not None else
                           "stale" if stored and stored.get("mapping_hash") != current else "pending"),
        "mapping_reviewer": (stored or {}).get("reviewer") if load_mapping_review(source, reviews_dir) else None,
        "mapping_reviewer_kind": (stored or {}).get("reviewer_kind", "human") if load_mapping_review(source, reviews_dir) else None,
        "grouping_reviewer": grouping.get("reviewer") if grouping else None,
        "grouping_reviewer_kind": grouping.get("reviewer_kind", "human") if grouping else None,
        "grouping_review": "verified" if grouping is not None else "pending",
        "grouping_method": grouping.get("method") if grouping else None,
        "requirements": mapping_review_requirements(source),
        "canonical_labels": sorted(CANONICAL_LABELS),
    }
