"""Source review workflow: convert -> review sheets -> quality report, run in the background."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from drowning.data.adapters import convert_to_jsonl
from drowning.data.audit import audit_jsonl
from drowning.data.quality import build_quality_report
from drowning.data.registry import SourceConfigError, load_source, resolve_source_root, validate_source
from drowning.data.review import (groups_fingerprint, is_mapping_approved, load_grouping_review, mapping_fingerprint,
                                  record_grouping_review, record_mapping_review, review_status)
from drowning.data.review_rules import (VERDICT_SCHEMA_VERSION, VerdictError, evaluate_ai_verdict, evaluate_auto_rules,
                                        load_policy, policy_version)
from drowning.data.samples import build_group_sheets, build_review_sheets


class ReviewConflict(RuntimeError):
    pass


class SourceReviewService:
    def __init__(self, sources_dir: Path, *, reviews_dir: Path, work_dir: Path, policy_path: Path | None = None):
        self.sources_dir = sources_dir
        self.reviews_dir = reviews_dir
        self.work_dir = work_dir
        self.policy_path = policy_path
        self._lock = threading.Lock()
        self._tasks: dict[str, dict[str, Any]] = {}

    # -- lookup ----------------------------------------------------------------------------
    def source(self, source_id: str) -> dict[str, Any]:
        for path in sorted(self.sources_dir.glob("*.yaml")):
            candidate = load_source(path, self.reviews_dir)
            if candidate.get("source_id") == source_id:
                return candidate
        raise KeyError(source_id)

    def _paths(self, source: dict[str, Any]) -> dict[str, Path]:
        label = f"{source['source_id']}@{source['source_version']}"
        mapping8 = mapping_fingerprint(source)[:8]
        grouping = load_grouping_review(source, self.reviews_dir)
        groups8 = groups_fingerprint(grouping["by_path"])[:8] if grouping else "nogroups"
        base = self.work_dir / "review" / label / mapping8
        return {"canonical": self.work_dir / "derived" / f"{label}.{mapping8}.{groups8}.jsonl",
                "sheets": base / "sheets", "quality": base / "quality-nogroups"}

    # -- state -----------------------------------------------------------------------------
    def status(self, source_id: str) -> dict[str, Any]:
        source = self.source(source_id)
        paths = self._paths(source)
        sheets, quality = paths["sheets"] / "samples.json", paths["quality"] / "quality.json"
        manifest = json.loads(sheets.read_text(encoding="utf-8")) if sheets.is_file() else None
        report = json.loads(quality.read_text(encoding="utf-8")) if quality.is_file() else None
        return {
            "source_id": source_id, "source_version": str(source["source_version"]),
            **review_status(source, self.reviews_dir),
            "task": self._tasks.get(source_id, {"state": "idle"}),
            "sheets": ([{k: sheet[k] for k in ("key", "kind", "original_label", "mapping_action", "canonical_label",
                                               "sample_count", "total_for_label")} for sheet in manifest["sheets"]]
                       if manifest else None),
            "quality": self._with_live_readiness(source, report), "proposed_groups_available": (paths["quality"] / "proposed-groups.json").is_file(),
            "group_sheets": ([{k: sheet[k] for k in ("key", "kind", "rows", "clusters_in_export")}
                              for sheet in json.loads((paths["quality"] / "group-sheets" / "groups.json").read_text(encoding="utf-8"))["sheets"]]
                             if (paths["quality"] / "group-sheets" / "groups.json").is_file() else None),
            "auto_rules": self.auto_rules(source_id) if report is not None and (paths["quality"] / "audit.json").is_file() else None,
        }

    def _with_live_readiness(self, source: dict[str, Any], report: dict[str, Any] | None) -> dict[str, Any] | None:
        if report is None:
            return None
        readiness = dict(report["image_training_readiness"])
        mapping_ok = is_mapping_approved(source)
        grouping_ok = load_grouping_review(source, self.reviews_dir) is not None or readiness["source_groups_verified"]
        blockers = [b for b in readiness["blockers"] if b not in {"mapping_review_not_approved", "source_groups_not_verified"}]
        if not mapping_ok:
            blockers.append("mapping_review_not_approved")
        if not grouping_ok:
            blockers.append("source_groups_not_verified")
        readiness.update(mapping_review_approved=mapping_ok, source_groups_verified=grouping_ok, blockers=blockers,
                         ready_for_split_and_crop_building=mapping_ok and grouping_ok)
        return {**report, "image_training_readiness": readiness}

    def sheet_file(self, source_id: str, key: str) -> Path:
        source = self.source(source_id)
        paths = self._paths(source)
        for manifest_path in (paths["sheets"] / "samples.json", paths["quality"] / "group-sheets" / "groups.json"):
            if not manifest_path.is_file():
                continue
            files = {sheet["key"]: sheet["file"] for sheet in json.loads(manifest_path.read_text(encoding="utf-8"))["sheets"]}
            if key in files:
                return manifest_path.parent / files[key]
        raise FileNotFoundError(key)

    # -- background preparation ------------------------------------------------------------
    def prepare(self, source_id: str, *, background: bool = True) -> dict[str, Any]:
        source = self.source(source_id)
        errors = validate_source(source)
        if errors:
            raise SourceConfigError("Invalid source descriptor: " + "; ".join(errors))
        root = resolve_source_root(source)
        if root is None or not root.is_dir():
            raise SourceConfigError(f"Configure {source['paths']['root_env']} in .env before preparing review material")
        with self._lock:
            if self._tasks.get(source_id, {}).get("state") == "running":
                raise ReviewConflict("Review preparation is already running for this source")
            self._tasks[source_id] = {"state": "running", "step": "starting", "error": None}
        if background:
            threading.Thread(target=self._run, args=(source, root), name=f"review-{source_id}", daemon=True).start()
        else:
            self._run(source, root)
        return self._tasks[source_id]

    def _set(self, source_id: str, **values: Any) -> None:
        with self._lock:
            self._tasks[source_id] = {**self._tasks.get(source_id, {}), **values}

    def _run(self, source: dict[str, Any], root: Path) -> None:
        source_id, paths = source["source_id"], self._paths(source)
        try:
            needed = [paths["sheets"] / "samples.json", paths["quality"] / "quality.json", paths["quality"] / "audit.json",
                      paths["quality"] / "group-sheets" / "groups.json"]
            has_proposal = (paths["quality"] / "proposed-groups.json").is_file()
            if not paths["canonical"].is_file() and (not all(f.is_file() for f in needed[:3]) or not has_proposal):
                self._set(source_id, step="converting annotations")
                convert_to_jsonl(source, root, paths["canonical"])
            if not (paths["sheets"] / "samples.json").is_file():
                self._set(source_id, step="building review sheets")
                build_review_sheets(source, paths["canonical"], root, paths["sheets"])
            if not (paths["quality"] / "quality.json").is_file():
                self._set(source_id, step="building quality report and near-duplicate groups")
                report, proposed = build_quality_report(source, paths["canonical"], root, reviews_dir=self.reviews_dir)
                paths["quality"].mkdir(parents=True)
                (paths["quality"] / "quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                if proposed:
                    (paths["quality"] / "proposed-groups.json").write_text(json.dumps({
                        "method": "near_duplicate_clusters_user_reviewed", "evidence": report["near_duplicates"],
                        "by_path": proposed}, ensure_ascii=False), encoding="utf-8")
            if not (paths["quality"] / "audit.json").is_file():
                self._set(source_id, step="auditing converted annotations")
                (paths["quality"] / "audit.json").write_text(json.dumps(audit_jsonl([paths["canonical"]]), ensure_ascii=False), encoding="utf-8")
            proposal = paths["quality"] / "proposed-groups.json"
            if proposal.is_file() and not (paths["quality"] / "group-sheets" / "groups.json").is_file():
                self._set(source_id, step="building group evidence sheets")
                try:
                    build_group_sheets(source, paths["canonical"], root, json.loads(proposal.read_text(encoding="utf-8"))["by_path"],
                                       paths["quality"] / "group-sheets")
                except SourceConfigError:
                    pass  # no multi-image clusters: nothing to show, grouping is then trivially safe
            self._set(source_id, state="done", step="complete")
        except Exception as exc:  # surfaced to the UI; details stay in the exception type only
            self._set(source_id, state="failed", step="failed", error=f"{type(exc).__name__}: {exc}"[:400])

    # -- human decisions -------------------------------------------------------------------
    def approve_mapping(self, source_id: str, *, reviewer: str, confirmed_labels: list[str],
                        samples_viewed: dict[str, int], notes: dict[str, str]) -> dict[str, Any]:
        source = self.source(source_id)
        if not (self._paths(source)["sheets"] / "samples.json").is_file():
            raise ReviewConflict("Generate the review sheets for the current mapping before approving it")
        record = record_mapping_review(source, reviewer=reviewer, confirmed_labels=confirmed_labels,
                                       samples_viewed=samples_viewed, notes=notes, reviews_dir=self.reviews_dir)
        return {"status": record["status"], "reviewer": record["reviewer"], "reviewed_at": record["reviewed_at"]}

    def approve_grouping(self, source_id: str, *, reviewer: str, note: str) -> dict[str, Any]:
        source = self.source(source_id)
        proposal = self._paths(source)["quality"] / "proposed-groups.json"
        if not proposal.is_file():
            raise ReviewConflict("Run review preparation to produce proposed groups first")
        data = json.loads(proposal.read_text(encoding="utf-8"))
        record = record_grouping_review(source, reviewer=reviewer, by_path=data["by_path"], method=data["method"],
                                        evidence=data.get("evidence", {}), note=note, reviews_dir=self.reviews_dir)
        return {"status": record["status"], "reviewer": record["reviewer"], "reviewed_at": record["reviewed_at"],
                "group_count": len(set(record["by_path"].values())), "media_count": len(record["by_path"]),
                "source_video_disjointness_verified": False}


    # -- automatic rules, evidence bundle and AI verdicts ------------------------------------
    def auto_rules(self, source_id: str) -> dict[str, Any]:
        source = self.source(source_id)
        paths = self._paths(source)
        quality_file, audit_file = paths["quality"] / "quality.json", paths["quality"] / "audit.json"
        if not quality_file.is_file() or not audit_file.is_file():
            raise FileNotFoundError("quality/audit")
        return evaluate_auto_rules(source, json.loads(quality_file.read_text(encoding="utf-8")),
                                   json.loads(audit_file.read_text(encoding="utf-8")), load_policy(self.policy_path))

    def evidence(self, source_id: str) -> dict[str, Any]:
        """Copy everything a reviewer must look at into one folder and fix its hash, so a verdict is tied to it."""
        source = self.source(source_id)
        paths = self._paths(source)
        policy = load_policy(self.policy_path)
        rules = self.auto_rules(source_id)
        files: dict[str, Path] = {"quality.json": paths["quality"] / "quality.json"}
        for sheet in json.loads((paths["sheets"] / "samples.json").read_text(encoding="utf-8"))["sheets"]:
            files[f"mapping-sheets/{sheet['file']}"] = paths["sheets"] / sheet["file"]
        files["mapping-sheets/samples.json"] = paths["sheets"] / "samples.json"
        group_manifest = paths["quality"] / "group-sheets" / "groups.json"
        if group_manifest.is_file():
            for sheet in json.loads(group_manifest.read_text(encoding="utf-8"))["sheets"]:
                files[f"group-sheets/{sheet['file']}"] = group_manifest.parent / sheet["file"]
            files["group-sheets/groups.json"] = group_manifest
        digest = hashlib.sha256()
        for name in sorted(files):
            digest.update(f"{name}:{hashlib.sha256(files[name].read_bytes()).hexdigest()}\n".encode())
        digest.update(json.dumps(rules, sort_keys=True).encode())
        evidence_sha = digest.hexdigest()
        target = paths["quality"].parent / f"evidence-{evidence_sha[:10]}"
        if not target.is_dir():
            temp = target.with_name(target.name + ".tmp")
            shutil.rmtree(temp, ignore_errors=True)
            for name, source_path in files.items():
                (temp / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, temp / name)
            (temp / "auto-rules.json").write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
            (temp / "verdict-template.json").write_text(json.dumps(self._template(source, evidence_sha, bool(group_manifest.is_file())),
                                                                   ensure_ascii=False, indent=2), encoding="utf-8")
            (temp / "README_AI_REVIEW.md").write_text(self._readme(source, policy, rules, bool(group_manifest.is_file())), encoding="utf-8")
            temp.rename(target)
        return {"source_id": source_id, "evidence_dir": str(target), "evidence_sha256": evidence_sha,
                "auto_rules_outcome": rules["outcome"], "failed": rules["failed"], "warnings": rules["warnings"],
                "has_group_evidence": group_manifest.is_file(),
                "next": f"An AI reviewer reads README_AI_REVIEW.md in that folder, looks at the images, fills verdict-template.json "
                        f"and it is submitted with: drownctl review submit {source_id} <verdict.json>"}

    def _template(self, source: dict[str, Any], evidence_sha: str, has_groups: bool) -> dict[str, Any]:
        labels = {name: {"samples_viewed": 0, "visible_person_fraction": None, "label_consistent_fraction": None, "notes": ""}
                  for name, rule in source["mapping"].items() if isinstance(rule, dict) and rule.get("action") == "map"}
        return {"schema_version": VERDICT_SCHEMA_VERSION, "source_id": source["source_id"],
                "mapping_hash": mapping_fingerprint(source), "evidence_sha256": evidence_sha,
                "reviewer": {"kind": "ai", "name": "<model name and version>", "delegated_by": "user"},
                "viewed_images": False, "mapping": {"labels": labels},
                "grouping": ({"cross_split_pairs_viewed": 0, "same_scene_fraction": None,
                              "largest_cluster_same_scene_fraction": None, "notes": ""} if has_groups else {}),
                "decision": {"mapping": "uncertain", "grouping": "uncertain" if has_groups else "uncertain"},
                "limitations": "<what you could not see or judge>"}

    def _readme(self, source: dict[str, Any], policy: dict[str, Any], rules: dict[str, Any], has_groups: bool) -> str:
        ai = policy["ai_rules"]
        failed = ", ".join(rules["failed"]) or "none"
        return f"""# AI review of source `{source['source_id']}` v{source['source_version']}

You are asked to review a labeled dataset before it may be used to train a drowning detector. The owner does not
review by hand: your findings, checked against fixed thresholds, decide. Be accurate rather than agreeable. A wrong
approval trains a wrong model; a rejection only excludes this source.

## Rules you must follow
1. You must actually LOOK at the images listed below. If you cannot view images, set `viewed_images: false` and every
   decision to `uncertain`. Never infer from file names or statistics alone.
2. Report fractions from what you saw, not what you expect. Do not round up. Note sample numbers you found wrong.
3. If you are unsure, answer `uncertain`; the owner will be asked. `uncertain` is not a failure.
4. Do not edit any file in this folder. Write your verdict to a new JSON file based on `verdict-template.json`.

## What the machine already checked (`auto-rules.json`)
Outcome: **{rules['outcome']}** (failed: {failed}). If any rule failed the source cannot be approved no matter what you find.

## Part 1: label meaning (`mapping-sheets/`)
Each source label has two sheets of crops: `*--random.jpg` ({ai['mapping']['min_samples_viewed_per_label']} random crops, numbered)
and `*--smallest.jpg` (the smallest boxes). For each source label that maps to a canonical class, look at the RANDOM sheet and report:
- `samples_viewed`: how many crops you examined (at least {ai['mapping']['min_samples_viewed_per_label']}).
- `visible_person_fraction`: share of crops where a person is clearly visible (not blur, texture, or an empty region).
- `label_consistent_fraction`: share of crops whose content fits the meaning of the canonical class:
  `drowning` = person showing signs of drowning; `swimming` = person behaving normally IN the water (swimming, treading water, standing in shallow water); `out_of_water` = person outside the water.
Thresholds to pass: visible >= {ai['mapping']['min_visible_person_fraction']}, consistent >= {ai['mapping']['min_label_consistent_fraction']} for EVERY label.
The `*--smallest` sheets are for information: note in `notes` whether tiny boxes are usable.

## Part 2: grouping (`group-sheets/`){'' if has_groups else ' (not applicable: no multi-image clusters)'}
Images judged near-duplicates are placed in one group so they are never split across train/validation/test.
- `groups--cross-split.jpg`: each ROW is one cluster that the original export had spread over several splits (border colour = original split). Question: are the images in a row the same photo or scene (rotated, recoloured, cropped)?
- `groups--same-split.jpg`: ordinary clusters, same question.
- `groups--largest.jpg`: a sample of the largest cluster. Question: do they look like one scene, or unrelated pool images chained together?
Report `cross_split_pairs_viewed` (rows examined, at least {ai['grouping']['min_cross_split_pairs_viewed']}), `same_scene_fraction`
(rows that are really the same photo/scene), `largest_cluster_same_scene_fraction`.
Thresholds: same_scene >= {ai['grouping']['min_same_scene_fraction']}, largest-cluster >= {ai['grouping']['min_largest_cluster_same_scene_fraction']}.
Approving the grouping only says near-duplicates are kept together. It does NOT prove the groups come from different source videos.

## Decisions
`decision.mapping` and `decision.grouping` are each `approve`, `reject` or `uncertain`. The system compares your numbers with the
thresholds: an `approve` with numbers below a threshold is recorded as a rejection.

Set `reviewer.name` to your model name/version, `viewed_images` to true only if you viewed them, and keep `evidence_sha256` unchanged.
"""

    def submit_verdict(self, source_id: str, verdict: dict[str, Any]) -> dict[str, Any]:
        source = self.source(source_id)
        policy = load_policy(self.policy_path)
        current = self.evidence(source_id)
        try:
            judged = evaluate_ai_verdict(verdict, source=source, evidence_sha256=current["evidence_sha256"], policy=policy,
                                         has_group_evidence=current["has_group_evidence"])
        except VerdictError as exc:
            raise ReviewConflict(f"Verdict refused: {exc}") from exc
        rules = self.auto_rules(source_id)
        blocked = rules["outcome"] == "blocked"
        reviewer = f"ai:{verdict['reviewer']['name']}"
        stamp = datetime.now(timezone.utc)
        recorded: dict[str, Any] = {"mapping": "not_recorded", "grouping": "not_recorded"}
        if not policy["delegation"]["allow_ai_approval"]:
            raise ReviewConflict("The review policy does not allow AI approvals")
        if judged["mapping"] == "approve" and not blocked:
            labels = verdict["mapping"]["labels"]
            record_mapping_review(source, reviewer=reviewer, reviewer_kind="ai", delegated_by=policy["delegation"]["delegated_by"],
                                  evidence_sha256=current["evidence_sha256"], rules_version=policy_version(policy),
                                  confirmed_labels=[n for n, r in source["mapping"].items() if isinstance(r, dict) and r.get("action") == "map"],
                                  samples_viewed={k: int(v["samples_viewed"]) for k, v in labels.items()},
                                  notes={k: str(v.get("notes", "")) for k, v in labels.items()}, reviews_dir=self.reviews_dir)
            recorded["mapping"] = "approved"
        elif judged["mapping"] == "approve":
            judged["mapping"] = "reject"
            judged["problems"]["mapping"].append("automatic rules failed: " + ", ".join(rules["failed"]))
        if judged["grouping"] in {"approve", "not_applicable"} and (self._paths(source)["quality"] / "proposed-groups.json").is_file():
            if judged["grouping"] == "approve" and not blocked:
                data = json.loads((self._paths(source)["quality"] / "proposed-groups.json").read_text(encoding="utf-8"))
                grouping = verdict.get("grouping", {})
                note = (str(grouping.get("notes", "")).strip() or
                        f"AI review: {grouping.get('same_scene_fraction'):.0%} of cross-split clusters are the same photo/scene.")
                record_grouping_review(source, reviewer=reviewer, reviewer_kind="ai", delegated_by=policy["delegation"]["delegated_by"],
                                       evidence_sha256=current["evidence_sha256"], rules_version=policy_version(policy),
                                       by_path=data["by_path"], method=data["method"], evidence=data.get("evidence", {}),
                                       note=note + " (near-duplicate groups only; source-video disjointness not verified)",
                                       reviews_dir=self.reviews_dir)
                recorded["grouping"] = "verified"
        outcome = {"source_id": source_id, "decisions": {"mapping": judged["mapping"], "grouping": judged["grouping"]},
                   "problems": judged["problems"], "recorded": recorded, "auto_rules_outcome": rules["outcome"],
                   "ask_user": [part for part in ("mapping", "grouping") if judged[part] == "uncertain"],
                   "reviewer": reviewer, "rules_version": policy_version(policy), "evidence_sha256": current["evidence_sha256"]}
        audit_dir = self.reviews_dir / "verdicts"
        audit_dir.mkdir(parents=True, exist_ok=True)
        name = f"{source_id}@{source['source_version']}.{stamp.strftime('%Y%m%dT%H%M%SZ')}.ai.json"
        (audit_dir / name).write_text(json.dumps({"verdict": verdict, "outcome": outcome, "submitted_at": stamp.isoformat()},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
        return outcome
