"""Command line entry point for local configuration and dataset inspection."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from uuid import UUID

import yaml

from drowning.config import Settings, database_url_from, load_env_file
from drowning.data.labels import resolve_label
from drowning.data.adapters import convert_to_jsonl, scan_source
from drowning.data.audit import audit_jsonl
from drowning.data.crops import build_crops
from drowning.data.registry import SourceConfigError, inspect_source, load_source, read_class_names, resolve_source_root, validate_source
from drowning.data.windows import build_temporal_windows
from drowning.data.grouping import SplitBlocked, build_group_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drownctl", description="Audited local tools for drowning detection datasets")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Path to local .env file")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("config-status", help="Show which integrations are configured without exposing values")
    doctor = commands.add_parser("doctor", help="Check what is configured/missing (never prints values) and how to fix it")
    doctor.add_argument("--online", action="store_true", help="Also test PostgreSQL, Drive folder and Kaggle datasets")
    doctor.add_argument("--json", action="store_true", help="Machine-readable output")
    review = commands.add_parser("review", help="Rule-based review: automatic rules, evidence bundle for an AI reviewer, verdict submission")
    review_actions = review.add_subparsers(dest="review_action", required=True)
    for name, text in (("auto", "Run the automatic rules on the prepared source"),
                       ("evidence", "Build the evidence bundle an AI reviewer must look at"),
                       ("submit", "Submit an AI reviewer's verdict (checked against the policy thresholds)")):
        item = review_actions.add_parser(name, help=text)
        item.add_argument("source_id")
        if name == "submit":
            item.add_argument("verdict", type=Path)
    detector = commands.add_parser("detector", help="Download the default person detector (explicit, hash-verified)")
    detector_actions = detector.add_subparsers(dest="detector_action", required=True)
    fetch = detector_actions.add_parser("fetch", help="Download yolo11n.pt and set PERSON_DETECTOR_WEIGHTS if it is empty")
    fetch.add_argument("--dir", type=Path, default=Path("local-cache/models"))
    auth = commands.add_parser("auth", help="One-time provider sign-in helpers")
    auth_actions = auth.add_subparsers(dest="auth_action", required=True)
    auth_drive = auth_actions.add_parser("drive", help="Open the browser once and store the Google Drive token file")
    auth_drive.add_argument("--force", action="store_true", help="Replace an existing token file")
    serve = commands.add_parser("serve", help="Run the local HTTP API")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true",
                       help="Development: restart the API automatically when Python files change (stops any live camera session)")
    db = commands.add_parser("db", help="Check PostgreSQL or apply Alembic migrations")
    db_actions = db.add_subparsers(dest="db_action", required=True)
    db_actions.add_parser("check", help="Check PostgreSQL connectivity without exposing the DSN")
    upgrade = db_actions.add_parser("upgrade", help="Apply migrations to the configured PostgreSQL database")
    upgrade.add_argument("--revision", default="head")
    jobs = commands.add_parser("jobs", help="Inspect/dispatch durable jobs")
    job_actions = jobs.add_subparsers(dest="jobs_action", required=True)
    job_status = job_actions.add_parser("status", help="Read a stored job and event timeline")
    job_status.add_argument("job_id", type=UUID)
    dispatch = job_actions.add_parser("dispatch", help="Explicitly dispatch a queued Kaggle job")
    dispatch.add_argument("job_id", type=UUID)
    dispatch.add_argument("--kernel-dir", type=Path, required=True)
    dispatch.add_argument("--timeout", type=int, default=3600)
    reconcile = job_actions.add_parser("reconcile", help="Check a remote-unconfirmed Kaggle job by exact version")
    reconcile.add_argument("job_id", type=UUID)
    collect = job_actions.add_parser("collect", help="Collect and verify the exact completed Kaggle candidate output")
    collect.add_argument("job_id", type=UUID)
    sync_retry = job_actions.add_parser("sync", help="Retry Drive synchronization without rerunning compute")
    sync_retry.add_argument("job_id", type=UUID)
    model = commands.add_parser("model", help="Synchronize the approved registry model without training")
    model_actions = model.add_subparsers(dest="model_action", required=True)
    model_sync = model_actions.add_parser("sync")
    model_sync.add_argument("--approved", action="store_true", required=True)
    test_commands = commands.add_parser("test", help="Run local image/video inference with an explicit compatible model and detector")
    test_actions = test_commands.add_subparsers(dest="test_action", required=True)
    image_test = test_actions.add_parser("image", help="Run one-image inference")
    image_test.add_argument("source", type=Path)
    image_test.add_argument("--model-bundle", type=Path, required=True)
    image_test.add_argument("--detector-weights", type=Path, required=True)
    image_test.add_argument("--device", default="cpu")
    video_test = test_actions.add_parser("video", help="Run offline video inference")
    video_test.add_argument("source", type=Path)
    video_test.add_argument("--model-bundle", type=Path, required=True)
    video_test.add_argument("--detector-weights", type=Path, required=True)
    video_test.add_argument("--output", type=Path, required=True)
    video_test.add_argument("--device", default="cpu")
    source = commands.add_parser("source", help="Inspect dataset source descriptors")
    actions = source.add_subparsers(dest="source_action", required=True)
    validate = actions.add_parser("validate", help="Validate a source YAML descriptor")
    validate.add_argument("config", type=Path)
    scan = actions.add_parser("scan", help="Inventory labels and split counts without changing source files")
    scan.add_argument("config", type=Path)
    inspect = actions.add_parser("inspect", help="Report source root and mapping status")
    inspect.add_argument("config", type=Path)
    inspect.add_argument("--read-class-list", action="store_true", help="Read declared class names from configured metadata")
    map_label = actions.add_parser("map-label", help="Resolve one source label using exact normalized mapping")
    map_label.add_argument("config", type=Path)
    map_label.add_argument("label")
    convert = actions.add_parser("convert", help="Convert supported image annotations to canonical JSONL")
    convert.add_argument("config", type=Path)
    convert.add_argument("--output", type=Path, required=True, help="Derived JSONL path outside the raw source root")
    audit = actions.add_parser("audit", help="Validate canonical JSONL records and report exact duplicate conflicts")
    audit.add_argument("files", nargs="+", type=Path)
    audit.add_argument("--output", type=Path, help="Write the report to a new file instead of stdout")
    crops = actions.add_parser("build-crops", help="Build deterministic image crops from canonical JSONL")
    crops.add_argument("config", type=Path)
    crops.add_argument("canonical_jsonl", type=Path)
    crops.add_argument("--output", type=Path, required=True)
    crops.add_argument("--preview-only", action="store_true", help="Allow crops from unapproved mappings; none are training-eligible")
    crops.add_argument("--limit", type=int, help="Required positive crop count limit for preview-only mode")
    crops.add_argument("--split-manifest", type=Path, help="Verified group split manifest required for training-eligible crops")
    crops.add_argument("--context", type=float, default=0.15)
    crops.add_argument("--size", type=int, default=224)
    windows = actions.add_parser("build-windows", help="Build causal windows only from reviewed video/person/time annotations")
    windows.add_argument("canonical_jsonl", type=Path)
    windows.add_argument("--recipe", type=Path, default=Path("configs/recipes/temporal.yaml"))
    windows.add_argument("--split-manifest", type=Path, help="Verified group split manifest required for training eligibility")
    windows.add_argument("--output", type=Path, help="New JSONL manifest path; required when windows are ready")
    splits = actions.add_parser("build-splits", help="Assign deterministic splits without breaking verified source groups")
    splits.add_argument("canonical_jsonl", type=Path)
    splits.add_argument("--output", type=Path, required=True)
    splits.add_argument("--seed", type=int, default=20261026)
    splits.add_argument("--holdout-group", action="append", default=[])
    sheets = actions.add_parser("review-samples", help="Write contact sheets of real crops per source label for mapping review")
    sheets.add_argument("config", type=Path)
    sheets.add_argument("canonical_jsonl", type=Path)
    sheets.add_argument("--output", type=Path, required=True, help="New directory for the sheets")
    sheets.add_argument("--per-label", type=int, default=24)
    quality = actions.add_parser("quality", help="Write a quality report and proposed near-duplicate groups")
    quality.add_argument("config", type=Path)
    quality.add_argument("canonical_jsonl", type=Path)
    quality.add_argument("--output", type=Path, required=True, help="New directory for quality.json and proposed-groups.json")
    quality.add_argument("--near-duplicate-distance", type=int, default=2)
    quality.add_argument("--skip-pixels", action="store_true", help="Skip image hashing (no near-duplicate analysis)")
    approve_map = actions.add_parser("approve-mapping", help="Record a named reviewer's approval of this exact mapping")
    approve_map.add_argument("config", type=Path)
    approve_map.add_argument("--reviewer", required=True)
    approve_map.add_argument("--confirm-all-mapped", action="store_true", help="Confirm every label whose action is map")
    approve_map.add_argument("--viewed", action="append", default=[], metavar="LABEL=COUNT", help="Sample crops viewed for a label")
    approve_map.add_argument("--note", action="append", default=[], metavar="LABEL=TEXT", help="Written definition for a label")
    approve_group = actions.add_parser("approve-grouping", help="Record reviewer-verified source groups from a proposal")
    approve_group.add_argument("config", type=Path)
    approve_group.add_argument("--proposal", type=Path, required=True, help="proposed-groups.json from `source quality`")
    approve_group.add_argument("--reviewer", required=True)
    approve_group.add_argument("--note", required=True, help="Why these groups can be treated as source-disjoint")
    return parser


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _doctor_command(args: argparse.Namespace) -> int:
    from drowning.doctor import run_doctor, summarize

    def db_check(url: str) -> bool:
        from drowning.db.session import check_db, create_db_engine
        engine = create_db_engine(url)
        try:
            return check_db(engine)
        finally:
            engine.dispose()

    def drive_check() -> str | None:
        from drowning.integrations.drive import build_drive_service, check_folder_access
        try:
            service = build_drive_service(Path(os.environ.get("GOOGLE_OAUTH_TOKEN_FILE", "./secrets/google-token.json")))
        except Exception as exc:
            return f"Không khởi tạo được Drive API ({type(exc).__name__})"
        return check_folder_access(service, os.environ["GOOGLE_DRIVE_FOLDER_ID"])

    def kaggle_check(dataset: str) -> str | None:
        from drowning.integrations.kaggle import KaggleCLIClient, KaggleError
        try:
            KaggleCLIClient().dataset_status(dataset)
        except (KaggleError, ValueError) as exc:
            return f"Dataset {dataset}: {str(exc)[:160]}"
        return None

    checks = run_doctor(os.environ, project_root=Path(__file__).resolve().parents[2], online=args.online,
                        db_check=db_check, drive_check=drive_check, kaggle_check=kaggle_check)
    report = summarize(checks)
    if args.json:
        _json(report)
    else:
        group = None
        for check in checks:
            if check.group != group:
                group = check.group
                print(f"\n== {group}")
            mark = "[ OK ]" if check.ok else "[  - ]" if check.ok is None else "[THIẾU]" if check.required else "[CHƯA]"
            hint = f"\n        → {check.hint}" if check.ok is False and check.hint else ""
            print(f"{mark} {check.message}{hint}")
        print(f"\nThiếu bắt buộc: {', '.join(report['required_missing']) or 'không'}")
        print(f"Sẵn sàng chạy Kaggle: {'có' if report['ready_for_kaggle_runs'] else 'chưa'}")
    return 0 if not report["required_missing"] else 3


def _worker_id() -> str:
    from drowning.installation import resolve_worker_id
    return resolve_worker_id(os.environ.get("WORKER_ID", "").strip(),
                             os.environ.get("INPUT_CACHE_DIR", "./local-cache/inputs"))


def _key_values(items: list[str], what: str) -> dict[str, str]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{what} must look like LABEL=VALUE: {item!r}")
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _review_command(args: argparse.Namespace, source: dict) -> int:
    from drowning.data import quality as quality_module
    from drowning.data.review import (record_grouping_review, record_mapping_review, review_status)
    from drowning.data.samples import build_review_sheets

    reviews_dir = Path(args.config).resolve().parents[2] / "reviews"
    if args.source_action == "approve-mapping":
        if not args.confirm_all_mapped:
            raise ValueError("pass --confirm-all-mapped after you have looked at the sample crops for every mapped label")
        viewed = {key: int(value) for key, value in _key_values(args.viewed, "--viewed").items()}
        mapped = [label for label, rule in source["mapping"].items() if rule.get("action") == "map"]
        record = record_mapping_review(source, reviewer=args.reviewer, confirmed_labels=mapped, samples_viewed=viewed,
                                       notes=_key_values(args.note, "--note"), reviews_dir=reviews_dir)
        _json({"status": record["status"], "reviewer": record["reviewer"], "mapping_hash": record["mapping_hash"],
               "review_file_directory": str(reviews_dir), "state": review_status(source, reviews_dir)["mapping_review"]})
        return 0
    if args.source_action == "approve-grouping":
        proposal = json.loads(args.proposal.read_text(encoding="utf-8"))
        record = record_grouping_review(source, reviewer=args.reviewer, by_path=proposal["by_path"],
                                        method=proposal.get("method", "near_duplicate_clusters_user_reviewed"),
                                        evidence=proposal.get("evidence", {}), note=args.note, reviews_dir=reviews_dir)
        _json({"status": record["status"], "reviewer": record["reviewer"], "group_count": len(set(record["by_path"].values())),
               "media_count": len(record["by_path"]), "source_video_disjointness_verified": False,
               "note": "groups are reviewer-verified; the source video of each image is still unknown"})
        return 0
    root = resolve_source_root(source)
    if root is None or not root.is_dir():
        raise SourceConfigError(f"Source root is unset or missing; configure {source['paths']['root_env']} in .env")
    if args.source_action == "review-samples":
        manifest = build_review_sheets(source, args.canonical_jsonl, root, args.output.resolve(), per_label=args.per_label)
        _json({"output": str(args.output.resolve()), "sheet_count": len(manifest["sheets"]),
               "sheets": [{k: sheet[k] for k in ("key", "original_label", "mapping_action", "canonical_label", "sample_count")}
                          for sheet in manifest["sheets"]]})
        return 0
    output = args.output.resolve()
    if output.exists():
        raise SourceConfigError(f"Refusing to overwrite existing output: {output}")
    report, proposed = quality_module.build_quality_report(
        source, args.canonical_jsonl, None if args.skip_pixels else root,
        near_duplicate_distance=args.near_duplicate_distance, reviews_dir=reviews_dir)
    output.mkdir(parents=True)
    (output / "quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if proposed:
        (output / "proposed-groups.json").write_text(json.dumps({
            "method": "near_duplicate_clusters_user_reviewed", "evidence": report["near_duplicates"],
            "by_path": proposed}, ensure_ascii=False, indent=1), encoding="utf-8")
    _json({"output": str(output), "counts": report["counts"], "near_duplicates": {
        k: v for k, v in report["near_duplicates"].items() if k != "cross_split_examples"},
        "readiness": report["image_training_readiness"]})
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        load_env_file(args.env_file)
        if args.command == "config-status":
            _json(Settings.from_env().status())
            return 0
        if args.command == "doctor":
            return _doctor_command(args)
        if args.command == "review":
            from drowning.api.review import ReviewConflict, SourceReviewService
            project = Path(__file__).resolve().parents[2]
            settings = Settings.from_env()
            service = SourceReviewService(project / "configs" / "datasets", reviews_dir=project / "reviews",
                                          work_dir=Path(settings.input_cache_dir).expanduser().resolve().parent)
            try:
                if args.review_action == "auto":
                    _json(service.auto_rules(args.source_id))
                elif args.review_action == "evidence":
                    _json(service.evidence(args.source_id))
                else:
                    outcome = service.submit_verdict(args.source_id, json.loads(args.verdict.read_text(encoding="utf-8")))
                    _json(outcome)
                    return 3 if outcome["ask_user"] else 0
            except (ReviewConflict, KeyError, FileNotFoundError) as exc:
                raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
            return 0
        if args.command == "detector":
            from drowning.detector_fetch import DETECTOR, DetectorFetchError, fetch_detector, set_env_value
            try:
                result = fetch_detector(args.dir.resolve())
            except (DetectorFetchError, OSError) as exc:
                raise RuntimeError(str(exc)) from exc
            changed = set_env_value(args.env_file, "PERSON_DETECTOR_WEIGHTS", result["path"])
            _json({**result, "model": DETECTOR["name"], "note": DETECTOR["note"], "env_updated": changed,
                   "env_message": "PERSON_DETECTOR_WEIGHTS set" if changed else "PERSON_DETECTOR_WEIGHTS already had a value; left unchanged"})
            return 0
        if args.command == "auth":
            from drowning.integrations.drive import DriveError, authorize_drive
            try:
                _json(authorize_drive(Path(os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET_FILE", "./secrets/google-oauth-client.json")),
                                      Path(os.environ.get("GOOGLE_OAUTH_TOKEN_FILE", "./secrets/google-token.json")),
                                      force=args.force))
            except DriveError as exc:
                raise RuntimeError(str(exc)) from exc
            return 0
        if args.command == "serve":
            import uvicorn
            from drowning.api.app import create_app

            settings = Settings.from_env()
            host, port = args.host or settings.api_host, args.port or settings.api_port
            if args.reload:
                # uvicorn needs an import string to restart; the app factory reads .env itself.
                uvicorn.run("drowning.api.app:create_app", factory=True, host=host, port=port, reload=True,
                            reload_dirs=[str(Path(__file__).resolve().parents[1])], log_config=None)
                return 0
            uvicorn.run(create_app(settings=settings), host=host, port=port, log_config=None)
            return 0
        if args.command == "model":
            from sqlalchemy.orm import Session
            from drowning.api.model_sync import sync_approved_model
            from drowning.db.session import create_db_engine, check_db
            from drowning.integrations.drive import GoogleDriveArtifactAdapter, build_drive_service
            settings = Settings.from_env()
            engine = create_db_engine(settings.database_url)
            try:
                if not check_db(engine):
                    raise ValueError("PostgreSQL migration is not ready")
                drive = GoogleDriveArtifactAdapter(build_drive_service(Path(settings.google_oauth_token_file)))
                with Session(engine, expire_on_commit=False) as session:
                    _json(sync_approved_model(session, settings, drive))
            finally:
                engine.dispose()
            return 0
        if args.command == "db":
            database_url = database_url_from(os.environ)
            if not database_url:
                raise ValueError("DATABASE_URL is not configured; fill the local .env before DB operations")
            if not database_url.startswith(("postgresql://", "postgresql+psycopg://")):
                raise ValueError("DB operations require PostgreSQL; SQLite is used only by schema smoke tests")
            if args.db_action == "check":
                from drowning.db.session import check_db, create_db_engine

                engine = create_db_engine(database_url)
                try:
                    ready = check_db(engine)
                    _json({"database_reachable": ready, "database_type": "postgresql",
                           "schema_revision": "20261026_01" if ready else None})
                finally:
                    engine.dispose()
                return 0 if ready else 3
            from alembic import command
            from alembic.config import Config

            alembic_config = Config(str(Path("alembic.ini").resolve()))
            command.upgrade(alembic_config, args.revision)
            _json({"migration": "applied", "revision": args.revision, "database_type": "postgresql"})
            return 0
        if args.command == "jobs":
            database_url = database_url_from(os.environ)
            if not database_url:
                raise ValueError("DATABASE_URL is not configured; fill the local .env before job operations")
            if not database_url.startswith(("postgresql://", "postgresql+psycopg://")):
                raise ValueError("Job dispatch requires PostgreSQL; SQLite is used only for local schema tests")
            from sqlalchemy.orm import sessionmaker
            from drowning.db.session import create_db_engine
            from drowning.db.models import Job, JobEvent

            engine = create_db_engine(database_url)
            try:
                factory = sessionmaker(bind=engine, expire_on_commit=False)
                if args.jobs_action == "status":
                    with factory() as session:
                        job = session.get(Job, args.job_id)
                        if job is None:
                            raise ValueError(f"Unknown job {args.job_id}")
                        events = session.query(JobEvent).filter(JobEvent.job_id == args.job_id).order_by(JobEvent.sequence).all()
                        _json({"job_id": str(job.id), "job_type": job.job_type, "state": job.state,
                               "phase": job.phase, "execution_token": job.execution_token,
                               "remote_ref": job.remote_ref, "remote_version": job.remote_version,
                               "remote_status": job.remote_status,
                               "events": [{"sequence": event.sequence, "event_type": event.event_type,
                                           "from_state": event.from_state, "to_state": event.to_state,
                                           "created_at": event.created_at.isoformat()} for event in events]})
                    return 0
                factory = sessionmaker(bind=engine, expire_on_commit=False)
                from drowning.integrations.kaggle import KaggleCLIClient

                if args.jobs_action in {"collect", "sync"}:
                    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "")
                    if not folder_id:
                        raise ValueError("GOOGLE_DRIVE_FOLDER_ID is not configured in .env")
                    from drowning.integrations.drive import GoogleDriveArtifactAdapter, build_drive_service
                    from drowning.orchestration.candidate_sync import CandidateArtifactSync

                    token_file = Path(os.environ.get("GOOGLE_OAUTH_TOKEN_FILE", "./secrets/google-token.json"))
                    drive = GoogleDriveArtifactAdapter(build_drive_service(token_file))
                    syncer = CandidateArtifactSync(factory, KaggleCLIClient(), drive,
                                                    drive_folder_id=folder_id,
                                                    cache_root=Path(os.environ.get("OUTPUT_CACHE_DIR", "./local-cache/results")),
                                                    worker_id=_worker_id())
                    result = syncer.collect(args.job_id) if args.jobs_action == "collect" else syncer.retry_sync(args.job_id)
                    _json(result)
                    return 0

                from drowning.orchestration.kaggle_dispatch import KaggleJobDispatcher
                dispatcher = KaggleJobDispatcher(factory, KaggleCLIClient(),
                                                  worker_id=_worker_id())
                if args.jobs_action == "dispatch":
                    receipt = dispatcher.dispatch(args.job_id, args.kernel_dir, timeout_seconds=args.timeout)
                else:
                    receipt = dispatcher.reconcile(args.job_id)
                _json({"job_id": receipt.job_id, "state": receipt.state, "kernel_ref": receipt.kernel_ref,
                       "kernel_version": receipt.kernel_version, "fencing_token": receipt.fencing_token,
                       "detail": receipt.detail})
            finally:
                engine.dispose()
            return 0
        if args.command == "test":
            from PIL import Image
            from drowning.inference.detector import UltralyticsPersonDetector
            from drowning.inference.pipeline import PersonStateInference
            from drowning.ml.bundle import load_bundle

            model, manifest = load_bundle(args.model_bundle, require_temporal=True, map_location=args.device)
            detector = UltralyticsPersonDetector(args.detector_weights)
            pipeline = PersonStateInference(model, manifest, detector,
                                            model_version=str(manifest.get("display_version", manifest.get("model_family", "selected-model"))),
                                            device=args.device)
            if args.test_action == "image":
                with Image.open(args.source) as image:
                    _json(pipeline.infer_image(image))
                return 0
            _json(pipeline.infer_video(args.source, args.output))
            return 0
        if args.source_action == "audit":
            report = audit_jsonl(args.files)
            if args.output:
                target = args.output.resolve()
                if target.exists():
                    raise SourceConfigError(f"Refusing to overwrite existing report: {target}")
                try:
                    target.relative_to(Path("D:/DrowningDemo").resolve())
                    protected_demo_output = True
                except ValueError:
                    protected_demo_output = False
                if protected_demo_output:
                    raise SourceConfigError("D:\\DrowningDemo is protected and cannot be an audit output")
                if any(part.casefold() == "sources" for part in target.parts):
                    raise SourceConfigError("A sources/ directory is protected and cannot be an audit output")
                if any(target == input_path.resolve() for input_path in args.files):
                    raise SourceConfigError("Audit output cannot overwrite an input file")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps({"report": str(target), "valid": report["valid"],
                                  "cross_split_duplicate_group_count": report["cross_split_duplicate_group_count"],
                                  "issue_count": len(report["issues"])}, ensure_ascii=False, indent=2))
                return 0 if report["valid"] else 2
            _json(report)
            return 0 if report["valid"] else 2
        if args.source_action == "build-windows":
            recipe = yaml.safe_load(args.recipe.read_text(encoding="utf-8"))
            sampling = recipe.get("sampling", {}) if isinstance(recipe, dict) else {}
            split_assignments: dict[str, dict[str, str]] = {}
            if args.split_manifest:
                for line_number, line in enumerate(args.split_manifest.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise SourceConfigError(f"Invalid split manifest at line {line_number}") from exc
                    if item.get("media_id"):
                        split_assignments[str(item["media_id"])] = {
                            "split": str(item.get("split", "")),
                            "global_group_id": str(item.get("global_group_id", "")),
                        }
            report = build_temporal_windows(
                args.canonical_jsonl,
                sequence_length=int(sampling.get("sequence_length", 16)),
                sample_period_ms=int(sampling.get("sample_period_ms", 250)),
                max_track_gap_ms=int(sampling.get("max_track_gap_ms", 1000)),
                min_valid_ratio=float(sampling.get("min_valid_ratio", 0.75)),
                split_assignments=split_assignments,
            )
            if report["status"] == "blocked":
                _json({key: value for key, value in report.items() if key != "windows"})
                return 3
            if args.output is None:
                raise SourceConfigError("--output is required when temporal windows are ready")
            output = args.output.resolve()
            if output.exists():
                raise SourceConfigError(f"Refusing to overwrite existing output: {output}")
            try:
                output.relative_to(Path("D:/DrowningDemo").resolve())
                protected_demo_output = True
            except ValueError:
                protected_demo_output = False
            if protected_demo_output or any(part.casefold() == "sources" for part in output.parts):
                raise SourceConfigError("Protected D:\\DrowningDemo or sources/ output path")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("".join(json.dumps(window, ensure_ascii=False, sort_keys=True) + "\n" for window in report["windows"]), encoding="utf-8")
            _json({"status": report["status"], "window_count": report["window_count"], "output": str(output),
                   "blockers": report["blockers"], "source_video_disjointness_verified": False})
            return 0
        if args.source_action == "build-splits":
            media_records = []
            for line_number, line in enumerate(args.canonical_jsonl.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SourceConfigError(f"Invalid canonical JSONL at line {line_number}") from exc
                if record.get("entity") == "media":
                    media_records.append(record)
            try:
                report = build_group_split(media_records, seed=args.seed, holdout_group_ids=set(args.holdout_group))
            except SplitBlocked as exc:
                _json({"status": "blocked", "reason": str(exc), "split_manifest_written": False,
                       "source_video_disjointness_verified": False})
                return 3
            output = args.output.resolve()
            if output.exists():
                raise SourceConfigError(f"Refusing to overwrite existing split manifest: {output}")
            try:
                output.relative_to(Path("D:/DrowningDemo").resolve())
                protected_demo_output = True
            except ValueError:
                protected_demo_output = False
            if protected_demo_output or any(part.casefold() == "sources" for part in output.parts):
                raise SourceConfigError("Protected D:\\DrowningDemo or sources/ output path")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in report["manifest"]), encoding="utf-8")
            _json({key: value for key, value in report.items() if key != "manifest"} | {"output": str(output)})
            return 0
        source = load_source(args.config)
        if args.source_action == "validate":
            errors = validate_source(source)
            _json({"valid_config": not errors, "errors": errors})
            return 0 if not errors else 2
        if args.source_action == "scan":
            errors = validate_source(source)
            if errors:
                raise SourceConfigError("Invalid source descriptor: " + "; ".join(errors))
            root = resolve_source_root(source)
            if root is None or not root.is_dir():
                root_key = source.get("paths", {}).get("root_env", "<unknown>")
                raise SourceConfigError(f"Source root is unset or missing; configure {root_key} in .env")
            _json(scan_source(source, root))
            return 0
        if args.source_action == "build-crops":
            errors = validate_source(source)
            if errors:
                raise SourceConfigError("Invalid source descriptor: " + "; ".join(errors))
            root = resolve_source_root(source)
            if root is None or not root.is_dir():
                root_key = source.get("paths", {}).get("root_env", "<unknown>")
                raise SourceConfigError(f"Source root is unset or missing; configure {root_key} in .env")
            _json(build_crops(source, args.canonical_jsonl, root, args.output,
                              preview_only=args.preview_only, limit=args.limit,
                              split_manifest=args.split_manifest,
                              context_fraction=args.context, size=args.size))
            return 0
        if args.source_action == "inspect":
            report = inspect_source(source)
            if args.read_class_list and report["valid_config"]:
                root = resolve_source_root(source)
                if root is None or not root.is_dir():
                    report["class_list_error"] = "source root is not configured or does not exist"
                else:
                    try:
                        report["source_classes"] = [
                            {"source_class_id": class_id, "source_label": name,
                             "decision": resolve_label(name, source["mapping"]).__dict__}
                            for class_id, name in read_class_names(source, root)
                        ]
                    except (SourceConfigError, OSError, ValueError) as exc:
                        report["class_list_error"] = str(exc)
            _json(report)
            return 0 if report["valid_config"] else 2
        if args.source_action == "convert":
            errors = validate_source(source)
            if errors:
                raise SourceConfigError("Invalid source descriptor: " + "; ".join(errors))
            root = resolve_source_root(source)
            if root is None or not root.is_dir():
                root_key = source.get("paths", {}).get("root_env", "<unknown>")
                raise SourceConfigError(f"Source root is unset or missing; configure {root_key} in .env")
            _json(convert_to_jsonl(source, root, args.output))
            return 0
        if args.source_action in {"review-samples", "quality", "approve-mapping", "approve-grouping"}:
            return _review_command(args, source)
        decision = resolve_label(args.label, source.get("mapping", {}))
        _json(decision.__dict__)
        return 0
    except (SourceConfigError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
