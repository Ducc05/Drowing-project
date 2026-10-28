# DrowningProject

Implementation workspace for the specification in `D:\AI\codex-cli-handoff`. This is a separate project tree; the handoff folder and `D:\DrowningDemo` are not modified.

## Setup

Requires Python 3.11+. From this directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -e .
```

Optional CPU-only model contract checks in the local environment (this does not train):

```powershell
.\.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple "torch==2.14.1+cpu" "torchvision==0.29.1+cpu"
```

The Kaggle runtime uses the same model code with its GPU-enabled PyTorch runtime. Do not use the CPU-only local environment for real training.

The local `.env` is created with blank placeholders on first setup. Fill its DB, Drive, Kaggle, and dataset-root values as they become available; do not overwrite an existing `.env` when setting up again.

Fill DB, Drive, Kaggle and dataset-root values in `.env` when available. `drownctl config-status` reports only whether integrations are configured and never prints their values. Training executor is pinned to Kaggle; local training is disabled.

## Local API and web UI

Latest consolidated review: [main-flow-review.html](reports/main-flow-review.html). It separates actual local PostgreSQL/HTTP/source checks, synthetic inference/fake cloud tests, fixed defects, and pending model/cloud acceptance.

Install the API/PostgreSQL adapter dependencies in the project virtual environment:

```powershell
.\.venv\Scripts\python -m pip install -e ".[api]"
.\.venv\Scripts\python -m drowning.cli db check
.\.venv\Scripts\python -m drowning.cli db upgrade
.\.venv\Scripts\python -m drowning.cli serve
```

`db upgrade` requires a PostgreSQL `DATABASE_URL`. Start the web dev server separately in another terminal:

```powershell
cd web
npm ci
npm run dev
```

Open the address printed by Vite (default `http://127.0.0.1:5173`). The Vite dev server proxies `/api`, `/health` and `/ready` to the local API. For same-origin serving, run `npm run build` first and then `drownctl serve`; FastAPI serves `web/dist` when present.

Job creation stores an idempotent database record only. It reports `dispatch: not_started` while no worker dispatcher is configured; the UI does not imply that a Kaggle run started.

On this workstation the dedicated local PostgreSQL container is already provisioned on loopback port 55439, and `.env` contains its development and isolated test database URLs. Restart it with `docker start drowning-local-pg-20261026`; stop it with `docker stop drowning-local-pg-20261026` (data volume retained).

The image/video form requires an approved registry model, its verified local cache and explicitly supplied detector weights. Review the candidate, then click **Đồng bộ model đã duyệt** or run:

```powershell
.\.venv\Scripts\python -m drowning.cli model sync --approved
```

Synchronization downloads the exact approved Drive archive, checks size/SHA-256/bundle identity, and installs a UUID cache directory atomically. It preserves earlier versions and never trains. Configure `APPROVED_MODEL_BUNDLE_DIR`, Drive OAuth, and `PERSON_DETECTOR_WEIGHTS` locally. No trained model is currently available.

Alternatively, set `DETECTION_ENGINE=h20saver` and `H20SAVER_WEIGHTS` to use the external [H20Saver](https://github.com/esonhjz/H20Saver) YOLO11x weights for recognition (person detection and the drowning/swimming/out-of-water label in one model) without a registry model; tracking, alerts and the web UI stay in this project. See `docs/SETUP.md` (section "Engine nhận diện H20Saver").

Upload endpoints `POST /api/v1/test/image` and `/video` accept multipart `file` and an `Idempotency-Key` header. Retry the same file/key after slot contention or an uncertain response. They return a durable job ID and local annotated media/detections links. When Drive is absent or fails, state is `sync_pending`; local results remain usable. `POST /api/v1/jobs/{id}/sync-results` only retries result upload. Offline video output is VP8/WebM; browser playback remains unverified. Camera is not implemented.

After configuring PostgreSQL, Kaggle CLI authentication and `KAGGLE_KERNEL_REF`, materialize a private kernel folder and explicitly dispatch a queued job:

```powershell
python -m notebooks.materialize_kernel --mode prepare_only
drownctl jobs status <job-id>
drownctl jobs dispatch <job-id> --kernel-dir .\notebooks\kernels\drowning-temporal
drownctl jobs reconcile <job-id>
drownctl jobs collect <job-id>
drownctl jobs sync <job-id>
```

The dispatcher holds the singleton slot after ambiguous pushes and never retries them automatically. Kaggle output collection requires the exact kernel version recorded by reconciliation. Candidate output is validated and uploaded to Drive before the DB registers it; a Drive retry never reruns compute.

## Current commands

```powershell
drownctl config-status
drownctl source validate ..\codex-cli-handoff\examples\source_university_yolo.yaml
drownctl source scan configs\datasets\alanoud_swimming_drowning_v4.yaml
drownctl source inspect ..\codex-cli-handoff\examples\source_university_yolo.yaml --read-class-list
drownctl source map-label ..\codex-cli-handoff\examples\source_university_yolo.yaml "not drowning"
drownctl source convert path\to\source.yaml --output .\derived\source-v1.jsonl
drownctl source audit .\derived\source-v1.jsonl other-source.jsonl
drownctl source build-crops configs\datasets\alanoud_swimming_drowning_v4.yaml local-cache\derived\alanoud-v4.jsonl --output local-cache\derived\review-crops --preview-only --limit 30
drownctl source build-windows local-cache\derived\video.jsonl --recipe configs\recipes\temporal.yaml --split-manifest local-cache\derived\split-manifest.jsonl
drownctl source build-splits local-cache\derived\canonical.jsonl --output local-cache\derived\split-manifest.jsonl
```

`source inspect` validates a descriptor and reports whether its configured root exists. `--read-class-list` reads only declared YOLO/COCO category metadata; it does not download, convert, or modify datasets. An unmapped label defaults to `review`.

`source convert` supports YOLO, COCO and Pascal VOC image annotations plus CVAT video tracks. It writes canonical `media` and `observation` JSONL records to a new output outside the source root, refuses overwrites and protected output locations, and retains unmapped labels for review. CVAT videos require explicit timestamps or explicitly reviewed CFR timing; track identity is not trusted unless the source config says it was reviewed.

`source audit` validates canonical record structure and media references, summarizes label/mapping counts, reports exact content duplicates and same-box label conflicts, and never automatically quarantines records or claims source-video-disjoint splits.

`source build-windows` only uses reviewed annotations with verified person IDs, timestamps and a verified split manifest. `source build-splits` only assigns groups with verified global source/session IDs and joins exact content duplicates first. Both commands stop with a blocked status when metadata is missing.

`source build-crops` requires approved source mapping and a verified group split manifest to make training-eligible crops. Use `--preview-only --limit N` to create a bounded review set while mapping is pending; those outputs are marked `training_eligible=false`.

Local inference requires an explicit compatible temporal model bundle and user-provided person-detector weights. For example:

```powershell
drownctl test video .\input.mp4 --model-bundle .\local-cache\models\candidate-v1 --detector-weights .\local-cache\detectors\person.pt --output .\local-cache\results\input-annotated.mp4
```

The video pipeline labels tracker IDs as provisional and records whether temporal warm-up completed. It does not treat scores as calibrated probabilities. No suitable trained bundle or detector weights are present in this workspace, so no real inference result has been generated.

The local `.env` points `ALANOUD_V4_DATA_ROOT` at the read-only export found at `D:\DrowningKaggleCheck\data`. Its source descriptor is `configs/datasets/alanoud_swimming_drowning_v4.yaml`. Conversion output and audit report are kept under ignored `local-cache/derived/`; both are image-dataset artifacts and cannot support temporal training by themselves.

Run local contract tests with `python -m unittest discover -s tests -v`.

The one-off helper scripts (regression runner, main-flow review, local PostgreSQL provisioning, initial-migration generator) are not versioned; they live in the git-ignored `tmp/scripts/` folder on the maintainer's machine.

Build the private Kaggle notebook locally with `python notebooks/build_notebook.py`. It defaults to `prepare_only`; use `--mode remote_smoke` or `--mode train_candidate` to explicitly build another mode. Candidate training requires approved image crops plus reviewed causal video windows and a verified split manifest. Review `notebooks/kernel-metadata.template.json` and fill the Kaggle owner/input references only when those are known. The notebook has not been submitted to Kaggle.
