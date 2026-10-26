# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DrowningProject is the implementation of the spec in `D:\AI\codex-cli-handoff` (read-only specification: `IMPLEMENTATION_RULES.md`, `DATASET_CONTRACT.md`, `TEST_AND_ACCEPTANCE.md`, etc.). It is a multi-source dataset registry + Kaggle-trained temporal drowning classifier + FastAPI/PostgreSQL orchestration + local React web UI. Specs and docs are largely in Vietnamese. `AGENTS.md` and `IMPLEMENTATION_STATUS.md` here are authoritative; update `IMPLEMENTATION_STATUS.md` at each milestone and keep "code written", "tests run", "real-data/cloud checks" and "user acceptance" clearly separate.

## Hard rules (from AGENTS.md and the handoff rules)

- Never edit, move or delete `sources/` (raw datasets) or `D:\DrowningDemo`; never modify `D:\AI\codex-cli-handoff`. Converters write to separate derived directories.
- Real training runs **only on Kaggle**. Do not add a local CPU fallback; local training config is rejected. Local tests use small synthetic fixtures and must never be described as trained-model results. Don't claim "90%/best/no false alarms" without a measurement.
- Fixed ontology v1 label order: `0 drowning`, `1 swimming`, `2 out_of_water`. Source label mapping must come from the source's real class list. Unmapped/ambiguous labels default to **review** and are blocked from supervised training; no substring/fuzzy auto-mapping (do not auto-map `normal`, `person`, `floating`, etc.).
- Split by source video/session/scene before cropping/windowing/augmenting; dedup across datasets; contradictory-label groups are dropped and reported. Loose images never become fake temporal clips.
- Secrets live only in ignored `.env` (placeholders in `.env.example`). Never log or put DB URLs, tokens, Drive IDs into code, notebooks, logs or reports. Kaggle notebooks stay private and contain no DB/Drive credentials.
- PostgreSQL is the source of truth for registry/jobs/the single global compute lock; Drive only stores artifacts. Only one compute job (Train or Test) system-wide. Publish artifacts/models only after full upload + checksum, with owner/token fencing on worker writes. Don't blindly retry Kaggle pushes (reconcile first); an unconfirmed remote stays "remote-unconfirmed" and blocks new jobs.
- A candidate model never becomes approved automatically; a normal Train request returns the existing approved model rather than retraining.

## Commands (PowerShell, from `D:\AI\DrowningProject`)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[api,dev]"     # extras: ml, api, drive, vision, detector, dev

.\.venv\Scripts\python scripts\run_regression.py           # full suite (unittest discovery) -> reports\regression-results.json
.\.venv\Scripts\python -m pytest tests\test_registry.py    # single file (pytest also configured)
.\.venv\Scripts\python -m pytest tests\test_registry.py -k name   # single test

.\.venv\Scripts\python -m drowning.cli doctor [--online]         # what is configured/missing and how to fix it; never prints values
.\.venv\Scripts\python -m drowning.cli config-status       # reports configured/unset only, never values
.\.venv\Scripts\python -m drowning.cli auth drive           # one-time Google Drive sign-in
.\.venv\Scripts\python -m drowning.cli source review-samples|quality|approve-mapping|approve-grouping ...   # review workflow (also in the web UI)
.\.venv\Scripts\python -m drowning.cli db check
.\.venv\Scripts\python -m drowning.cli db upgrade          # Alembic; needs a PostgreSQL DATABASE_URL
.\.venv\Scripts\python -m drowning.cli serve               # FastAPI; serves web\dist if built
.\.venv\Scripts\python -m drowning.cli model sync --approved
.\.venv\Scripts\python scripts\review_main_flows.py        # regenerates reports\main-flow-review.html from real checks (no training, no cloud)
.\.venv\Scripts\python scripts\provision_local_postgres.py # only creates its own container when DB URLs are blank
```

`drownctl` is the installed entry point for `drowning.cli`. Local dev PostgreSQL is the Docker container `drowning-local-pg-20261026` on loopback port 55439 (`docker start drowning-local-pg-20261026`). The real-PostgreSQL tests (`test_postgres_integration.py`) need that DB configured in `.env`.

Docs: `docs/SETUP.md` (env vars, PostgreSQL/Drive/Kaggle setup), `docs/DATASET_ONBOARDING.md` (adding labeled datasets). Web pages: Tổng quan, Dữ liệu (review), Train, Đánh giá, Camera.

Web (`web/`, React 19 + Vite + TypeScript): `npm ci`, `npm run dev` (proxies `/api`, `/health`, `/ready` to the API at 127.0.0.1:5173 default), `npm run build` (`tsc -b && vite build`), `npm run lint` (oxlint). There is no JS test runner.

## Architecture

Code is in `src/drowning/`:

- `data/` — dataset registry and ingestion. `adapters.py` (YOLO/COCO/VOC/CVAT) scan sources into a canonical observation; `labels.py`/`registry.py` hold per-source mapping with map/ignore/review/context actions; `audit.py`, `crops.py`, `grouping.py` (deterministic group split), `windows.py` (same-person, timestamp-ordered temporal windows). Configs are in `configs/` (`ontology.yaml`, `datasets/`, `recipes/`); examples of the contract are in the handoff `examples/`.
- `ml/` — model family and data for training: `model.py` (image + temporal branches), `bundle.py` (model bundle identity/schema/class-order/hash verification), `preprocessing.py`, `datasets.py` (manifest-gated image/temporal datasets), `training.py`, notebook builder/materializer inputs. Checkpoint reuse must match schema, ontology, class order, preprocessing, temporal contract and hash.
- `inference/` — `detector.py` (person detector interface; needs externally supplied `PERSON_DETECTOR_WEIGHTS`), `tracking.py`, `buffer.py` (causal per-track buffer), `pipeline.py`. Every valid person detection keeps its box/ID; low availability is expressed as confidence, not by dropping the person. Image-branch output during warm-up must not be reported as temporal.
- `db/` + `migrations/` + `alembic.ini` — SQLAlchemy models/repositories (`repositories.py` owns the compute-slot lock, idempotency, approval pointer and fencing); Alembic migrations (`scripts/generate_initial_migration.py` generated the initial one).
- `domain/jobs.py` — job states and modes (`prepare_only`, `remote_smoke`, `train_candidate`; smoke never produces a ready/approved model).
- `integrations/` — `drive.py` (Google Drive, OAuth) and `kaggle.py` (CLI wrapper using argument arrays + timeouts, never shell strings).
- `orchestration/` — `kaggle_dispatch.py` (private notebook push/poll with reconciliation) and `candidate_sync.py` (verified candidate download/publish).
- `orchestration/` also holds `kaggle_run.py` (prepare_only/remote_smoke run inside the notebook), `kaggle_train.py` (training inputs + two-stage training, CPU-testable), `run_collector.py` (collects prepare/smoke output) and `supervisor.py` (background recovery/dispatch for this worker's jobs). The notebook only calls these.
- `data/` also holds `review.py` (human approvals under `reviews/`, bound to mapping/group hashes), `samples.py` (review sheets) and `quality.py` (quality report, near-duplicate groups). Approvals are never inferred.
- `api/` — FastAPI app (`app.py`), media upload/result handling (`media.py`), inference jobs (`inference.py`), approved-model local cache sync (`model_sync.py`; atomic UUID-version directories, earlier versions kept, jobs pin their model version at start), `test_jobs.py`. Video output is VP8/WebM for browser playback.
- `config.py` loads `.env` (process environment takes precedence); `logging_utils.py` redacts secrets.

Flow: dataset sources → adapters/mapping review → canonical manifests + grouped splits → private Kaggle notebook (prepare / remote smoke / train candidate) → candidate artifacts to Drive + metadata to PostgreSQL → user reviews real video and approves → `model sync --approved` verifies size/SHA-256/bundle identity and installs the local cache → local web/CLI inference on images/video/camera.

## Test notes

Tests mix `unittest` (the regression runner uses discovery) and pytest-compatible files. Image/video API tests inject fake inference and fake Drive/Kaggle; they prove plumbing, not model quality. No approved real weights, temporal annotated data, Kaggle GPU run or live Drive/Kaggle check exists yet (see `IMPLEMENTATION_STATUS.md`); don't report those as done. Windows paths with non-ASCII characters (`C:\Users\khách`) and 8.3 names have caused regressions in the source scanner, so keep path handling Unicode-safe.
