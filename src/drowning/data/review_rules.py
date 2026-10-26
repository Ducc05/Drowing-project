"""Review policy: automatic rules computed by the machine and thresholds applied to an AI reviewer's findings.

The policy lives in ``configs/review_policy.yaml``. Nothing here looks at images: automatic rules read
numbers from the quality/audit reports, and AI findings are numbers an AI reported after viewing the
evidence images. The verdict a person or AI submits is never trusted over the thresholds: a reviewer
who says "approve" with numbers below the thresholds is recorded as rejected.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from drowning.data.labels import normalize_label
from drowning.data.review import AMBIGUOUS_SOURCE_LABELS, mapping_fingerprint

POLICY_PATH = Path(__file__).resolve().parents[3] / "configs" / "review_policy.yaml"
VERDICT_SCHEMA_VERSION = 1
DECISIONS = {"approve", "reject", "uncertain"}


class VerdictError(ValueError):
    pass


def load_policy(path: Path | None = None) -> dict[str, Any]:
    policy = yaml.safe_load((path or POLICY_PATH).read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or policy.get("version") != 1:
        raise ValueError("unsupported review policy")
    return policy


def policy_version(policy: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _rule(rule_id: str, status: str, value: Any, threshold: Any, detail: str) -> dict[str, Any]:
    return {"id": rule_id, "status": status, "value": value, "threshold": threshold, "detail": detail}


def evaluate_auto_rules(source: dict[str, Any], quality: dict[str, Any], audit: dict[str, Any],
                        policy: dict[str, Any]) -> dict[str, Any]:
    """Objective rules. ``outcome`` is ``blocked`` when any fail-severity rule fails, else ``pass``."""
    rules_cfg = policy["auto_rules"]
    results: list[dict[str, Any]] = []

    def graded(rule_id: str, ok: bool, value: Any, threshold: Any, detail: str) -> None:
        severity = rules_cfg[rule_id]["severity"]
        results.append(_rule(rule_id, "pass" if ok else ("fail" if severity == "fail" else "warn"), value, threshold, detail))

    observations = max(1, quality["counts"]["observations"])
    labels = quality["labels"]
    mapped = [label for label in labels if label["mapping_action"] == "map"]
    graded("audit_valid", bool(audit.get("valid")), len(audit.get("issues", [])), 0, "structural audit of the converted annotations")
    # Conversion refuses any malformed row, so a converted source has none by construction.
    graded("no_malformed_annotations", True, 0, 0, "conversion is strict: a malformed row stops it")
    unresolved = [label["original_label"] for label in labels if label["mapping_action"] == "review"]
    graded("all_labels_resolved", not unresolved, unresolved, [], "every source label must be mapped, ignored or marked context")
    conflict_fraction = len(audit.get("same_box_label_conflicts", [])) / observations
    graded("same_box_conflict_fraction", conflict_fraction <= rules_cfg["same_box_conflict_fraction"]["max"],
           round(conflict_fraction, 5), rules_cfg["same_box_conflict_fraction"]["max"], "identical boxes carrying different labels")
    smallest = min((label["count"] for label in mapped), default=0)
    graded("min_boxes_per_mapped_label", smallest >= rules_cfg["min_boxes_per_mapped_label"]["min"], smallest,
           rules_cfg["min_boxes_per_mapped_label"]["min"], "rarest mapped label")
    total_mapped = max(1, sum(label["count"] for label in mapped))
    share = smallest / total_mapped if mapped else 0.0
    graded("min_class_share", share >= rules_cfg["min_class_share"]["min"], round(share, 4),
           rules_cfg["min_class_share"]["min"], "share of the rarest class among mapped boxes")
    worst_tiny = max((label["tiny_box_fraction"] for label in mapped), default=0.0)
    graded("tiny_box_fraction", worst_tiny <= rules_cfg["tiny_box_fraction"]["max"], worst_tiny,
           rules_cfg["tiny_box_fraction"]["max"], "boxes under 16 px or 0.05% of the frame")
    ambiguous = []
    for name, rule in (source.get("mapping") or {}).items():
        if isinstance(rule, dict) and rule.get("action") == "map" and normalize_label(name) in AMBIGUOUS_SOURCE_LABELS:
            if len(str(rule.get("definition", "")).strip()) < 20:
                ambiguous.append(name)
    graded("ambiguous_label_names", not ambiguous, ambiguous, [],
           "ambiguous names (normal, person, ...) need a written `definition` in the descriptor's mapping rule")
    near = quality.get("near_duplicates", {})
    has_images = any(label for label in labels) and quality["counts"]["media"] > 0 and "video_groups" not in quality
    graded("near_duplicate_checked", bool(near.get("checked")) or not has_images, bool(near.get("checked")), True,
           "near-duplicate analysis of image media")
    largest = near.get("largest_cluster", 1) / max(1, quality["counts"]["media"]) if near.get("checked") else 0.0
    cluster_cfg = rules_cfg["largest_cluster_fraction"]
    status = "pass" if largest <= cluster_cfg["max"] else ("fail" if largest > cluster_cfg["fail_above"] else "warn")
    results.append(_rule("largest_cluster_fraction", status, round(largest, 4), cluster_cfg["max"],
                         "one huge cluster suggests chained, unrelated images; the largest-cluster sheet must be checked"))
    failed = [r["id"] for r in results if r["status"] == "fail"]
    return {"policy_version": policy_version(policy), "mapping_hash": mapping_fingerprint(source), "rules": results,
            "outcome": "blocked" if failed else "pass", "failed": failed,
            "warnings": [r["id"] for r in results if r["status"] == "warn"]}


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise VerdictError(f"{name} must be a number between 0 and 1")
    return float(value)


def evaluate_ai_verdict(verdict: dict[str, Any], *, source: dict[str, Any], evidence_sha256: str,
                        policy: dict[str, Any], has_group_evidence: bool) -> dict[str, Any]:
    """Apply the thresholds to what an AI reported. Returns decisions for the mapping and the grouping."""
    if not isinstance(verdict, dict) or verdict.get("schema_version") != VERDICT_SCHEMA_VERSION:
        raise VerdictError("unsupported verdict schema_version")
    reviewer = verdict.get("reviewer") or {}
    if reviewer.get("kind") != "ai" or not str(reviewer.get("name", "")).strip():
        raise VerdictError("reviewer must be {kind: ai, name: ...}")
    if verdict.get("mapping_hash") != mapping_fingerprint(source):
        raise VerdictError("verdict was written for a different mapping")
    if policy["delegation"].get("require_evidence_hash") and verdict.get("evidence_sha256") != evidence_sha256:
        raise VerdictError("verdict does not match the current evidence bundle")
    if verdict.get("viewed_images") is not True:
        raise VerdictError("an AI reviewer must confirm it looked at the evidence images (viewed_images: true)")
    ai = policy["ai_rules"]
    problems: dict[str, list[str]] = {"mapping": [], "grouping": []}

    mapping = verdict.get("mapping") or {}
    per_label = mapping.get("labels") or {}
    required = [name for name, rule in (source.get("mapping") or {}).items() if isinstance(rule, dict) and rule.get("action") == "map"]
    normalized = {normalize_label(k): v for k, v in per_label.items()}
    for name in required:
        entry = normalized.get(normalize_label(name))
        if not isinstance(entry, dict):
            problems["mapping"].append(f"{name}: no finding reported")
            continue
        viewed = entry.get("samples_viewed")
        if not isinstance(viewed, int) or viewed < ai["mapping"]["min_samples_viewed_per_label"]:
            problems["mapping"].append(f"{name}: viewed {viewed} samples, need {ai['mapping']['min_samples_viewed_per_label']}")
        visible = _fraction(entry.get("visible_person_fraction"), f"{name}.visible_person_fraction")
        consistent = _fraction(entry.get("label_consistent_fraction"), f"{name}.label_consistent_fraction")
        if visible < ai["mapping"]["min_visible_person_fraction"]:
            problems["mapping"].append(f"{name}: a person is clearly visible in only {visible:.0%} of samples (need {ai['mapping']['min_visible_person_fraction']:.0%})")
        if consistent < ai["mapping"]["min_label_consistent_fraction"]:
            problems["mapping"].append(f"{name}: the label fits only {consistent:.0%} of samples (need {ai['mapping']['min_label_consistent_fraction']:.0%})")

    grouping = verdict.get("grouping") or {}
    if has_group_evidence:
        pairs = grouping.get("cross_split_pairs_viewed")
        if not isinstance(pairs, int) or pairs < ai["grouping"]["min_cross_split_pairs_viewed"]:
            problems["grouping"].append(f"viewed {pairs} cross-split clusters, need {ai['grouping']['min_cross_split_pairs_viewed']}")
        same = _fraction(grouping.get("same_scene_fraction"), "grouping.same_scene_fraction")
        largest = _fraction(grouping.get("largest_cluster_same_scene_fraction"), "grouping.largest_cluster_same_scene_fraction")
        if same < ai["grouping"]["min_same_scene_fraction"]:
            problems["grouping"].append(f"only {same:.0%} of cross-split clusters are the same photo/scene (need {ai['grouping']['min_same_scene_fraction']:.0%})")
        if largest < ai["grouping"]["min_largest_cluster_same_scene_fraction"]:
            problems["grouping"].append(f"only {largest:.0%} of the largest cluster looks like one scene (need {ai['grouping']['min_largest_cluster_same_scene_fraction']:.0%})")

    decisions = verdict.get("decision") or {}
    result: dict[str, Any] = {"reviewer": reviewer, "problems": problems}
    for part in ("mapping", "grouping"):
        stated = decisions.get(part)
        if stated not in DECISIONS:
            raise VerdictError(f"decision.{part} must be one of {sorted(DECISIONS)}")
        if part == "grouping" and not has_group_evidence:
            result[part] = "not_applicable"
        elif stated == "uncertain":
            result[part] = "uncertain"
        elif stated == "reject" or problems[part]:
            result[part] = "reject"
        else:
            result[part] = "approve"
    return result
