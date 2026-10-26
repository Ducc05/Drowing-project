# Implementation status

Last updated: 2026-10-26

This status separates code written, checks actually run, and work not verified. It does not claim cloud, model, or user acceptance based on local tests.

## Decimal classifier scores — 2026-10-26

- **Implemented:** the three classifier scores (`drowning`, `swimming`, `out_of_water`) are displayed individually on the 0–1 scale with three decimal places in image results, video playback and the Camera table. YOLO person confidence is displayed separately. Browser overlays show the winning label score as a decimal; newly rendered image/video/camera captions include all three class scores. Existing detections JSON remains unchanged and existing immutable media files are not regenerated.
- **Threshold:** the low 0.02 detector threshold was used only for the earlier diagnostic probe. The live detector default remains 0.25. Classifier scores are model outputs, not calibrated probabilities, and a large classifier score does not validate a weak person detection.
- **Checks:** 38 focused image/video/Camera/evaluation regressions passed; `npm run build` passed and `npm run lint` passed with the same four existing warnings. A synthetic 416px preview confirmed all three decimal scores fit on the rendered image. Backend was reloaded only after the slot and queued/active lists were empty; live health/ready checks passed and no job was created. Live browser inspection remains blocked by the saved localhost origin permission.

## Test ảnh không bắt được người — 2026-10-26

- **Điều tra:** job `ae025bca` dùng candidate `kaggle-5` và hoàn tất với 0 detections. Đúng ảnh người dùng gửi, chạy detector cấu hình person ID 0/confidence 0.25, cũng không có box. Mô phỏng đúng đường màu BGR trên ảnh đó cho box người confidence 0.0293; ngưỡng mặc định loại box này. Trên cùng crop, classifier candidate trả lớp drowning ở điểm 0.999, nhưng candidate không được train bằng `dataset_test_1`, nên điểm đó không xác nhận ảnh được học/đúng nhãn.
- **Nguyên nhân code đã sửa:** adapter trước đây truyền mảng RGB vào Ultralytics, trong khi predictor coi mảng NumPy là BGR rồi tự đổi sang RGB. `UltralyticsPersonDetector` nay đổi RGB→BGR trước khi gọi model. Không hạ ngưỡng detector mặc định vì chưa đánh giá báo nhầm ở ngưỡng thấp.
- **Bằng chứng về training:** registry báo `kaggle-5` có `included_sources=[alanoud_swimming_drowning]`; job 75129 đã gắn duy nhất `vuduc792005/alanoud-pool`, có descriptor Alanoud và `continue_from_approved=false`. Preflight hiện đã thấy descriptor `dataset_test_1`, nhưng đánh dấu stale; mapping vẫn chờ duyệt và lớp 0 trên ảnh chưa được xác nhận.
- **Kiểm tra:** `tests.test_detector` — 1 test đạt; chạy detector thực trên ảnh ở ngưỡng mặc định và ngưỡng chẩn đoán xác nhận confidence 0.0293. Backend được restart khi slot rảnh, `/health` alive, `/test/readiness` ready, compute slot rảnh; không tạo thêm job train/test và không gọi Kaggle/Drive.
- **Giới hạn:** BGR fix đã nạp vào backend, nhưng ảnh vẫn bị loại ở ngưỡng mặc định 0.25. Cần xác minh mapping, gắn dataset đúng vào lượt train, rồi đánh giá detector người trên dữ liệu nước có cả cảnh dương và âm trước khi chọn ngưỡng/huấn luyện detector riêng. Không dùng điểm classifier cao trên crop này làm kết luận an toàn.

## Frontend layout and readability checkpoint — 2026-10-26

- **Implemented:** the app now keeps the sidebar visible while the main content scrolls; the top bar stays at the top of the content pane. Mobile navigation shows every destination as a labeled icon instead of hiding inactive routes. The layout supports 320px screens without forcing the document wider, tables and long model text stay within their panels, form groups have consistent spacing, small dashboard labels are more readable, and keyboard focus has a visible outline. Route changes return the content pane to its top. Overview opens the original video for overlay-mode results, and failed video requests retain their idempotency key for safe retry. Evaluation now accepts dropped or selected images and videos, previews image detections, lists both media types in history, and keeps video progress/cancellation specific to video jobs. Decorative workspace/profile controls no longer appear clickable. Existing page actions and route IDs are preserved.
- **Checked:** static source audit covered App, Data, Train, Evaluation, Camera, RunReport, API helpers and shared styles. `npm run build` passed. `npm run lint` passed with the same four existing `set-state-in-effect` warnings.
- **Visual review:** the browser origin gate reports a saved permission setting blocks observing `http://localhost:8000`; visual acceptance remains unverified.

## Continue training from the single approved model — 2026-10-26

- **Implemented:** Train offers a fresh model or continuation from the current approved model only. Continuation requires a verified local cache, a compatible model kind/preprocessing/architecture, and an explicit nonempty dataset selection. The job snapshots the approved model UUID and manifest/weights hashes and stages its weights separately. Dispatch creates a per-job private Kaggle checkpoint dataset and waits for readiness before pushing the notebook; weights are not embedded in notebook source. Later approvals cannot change an already queued job. Kaggle checks and loads those weights before training with new epoch/optimizer state. Selected source IDs are applied during input assembly. The new candidate records parent UUID/checksum in its manifest/metrics and does not replace the approved pointer automatically. Replayed requests retain the original job/parent; a reused key with different options is rejected. No schema migration is required.
- **Technical verification:** final `python -m unittest discover -s tests` passed **276/276 tests**; an additional 7 focused continuation tests passed after the final dataset transport changes. Checks include exact parent weight equality, private dataset staging/ready status, notebook checksum validation, dataset filtering, missing/incompatible parents, idempotent parent pinning, failed-upload slot release without kernel push, and existing image/temporal paths. Tests use synthetic fixtures and fake Kaggle providers. `npm run build` passed; `npm run lint` passed with the same four existing warnings.
- **Operation/limits:** no actual Kaggle dispatch/training, real model weight changes, or cloud credential transfer was performed. Official installed CLI help confirms dataset creation is private unless `--public` is provided; that flag is never sent. Live private dataset upload/readiness/kernel attachment is still unverified. A failed/ambiguous dataset upload cannot start a kernel and releases compute, but may leave a private dataset to clean up. This is weight continuation, not full optimizer/epoch resume. New dataset registration still uses source descriptors/configuration. The UI checks preflight capability flags and blocks unsupported backend versions. Live browser acceptance remains unverified; the previous browser localhost origin denial was not retried.
- **Runtime verified after the user's restart request:** the shared compute slot and active/queued job lists were empty before restart. Backend PID 54308 was replaced by PID 22532 on loopback port 8000. Live `/health` returned `alive`, `/ready` confirmed PostgreSQL, and `/api/v1/train/preflight` returned both continuation/source-selection capability flags true. `kaggle-3` remained approved, image-tracked, cached locally, and inference-ready. No job was submitted by this restart.

## Approved model overview checkpoint — 2026-10-26

- **Implemented:** Overview recognizes approved `temporal` and `image_tracked` models using `model_kind`, displays the version/type, shows local cache status, and provides synchronization, Evaluation, and Camera actions. The existing image/video form and candidate review actions remain available. Navigating between pages refreshes the overview data so a newly approved model appears on return. Evaluation labels use `model_kind` and only temporal models show the temporal validation panel. Backend code was not changed in this checkpoint.
- **Tested:** all four `tests.test_model_sync` tests passed, including a new image-tracked candidate approval/list/sync/cache test using a temporary database and fake Drive. Full Python regression passed **268/268 tests** with `DATABASE_URL` and `DATABASE_URL_UNPOOLED` empty (106.245 s, including dedicated PostgreSQL checks). `npm run build` passed and rebuilt `web/dist`; `npm run lint` passed with the same four existing warnings.
- **Live UI limit:** the backend was already listening on port 8000 and was not restarted. A separate browser tab for `http://localhost:8000/#overview` was blocked because the browser origin permission was declined; live display and link-click acceptance remain unverified. No live model review, inference job, camera start, or cloud action was triggered.

## Video evaluation speed and cancellation checkpoint — 2026-10-26

- **Implemented:** video inference defaults to `frame_stride=3` (configurable 1–10); skipped frames avoid color conversion/detector/classifier/tracking. Progress is stored periodically in the job payload and exposed by `GET /api/v1/jobs`. Browser-playable MP4/WebM inputs use a copied `source.*` plus detections JSON and a browser canvas overlay; other inputs fall back to VP8/WebM while keeping original duration. The evaluation page selects the correct result, interpolates boxes, shows model scores, polls progress, and can stop a running inference. The local media-job service is now an app-scoped singleton so the stop flag is shared by both requests. Camera inference remains on its existing path; camera captions now include the displayed label score.
- **Tested:** `python -m unittest discover -s tests` — **267 tests passed** (including PostgreSQL concurrency checks). Focused checks cover stride indices/counts, source-copy overlay, unsupported AVI fallback, progress serialization, cancellation cleanup/state/compute-slot release, result downloads, and evaluation behavior. `npm run build` passed; `npm run lint` passed with four existing `set-state-in-effect` warnings. The lint warning locations are `App.tsx` and pre-existing state-loading effects in `EvaluationPage.tsx`.
- **Real local inference measured:** ZoneA (237 frames, HEVC) processed 79 frames at stride 3, fell back to VP8 output, and took about **118 s** versus the previously reported 111 s. The 93 s AV1 video (2,795 frames) processed 932 frames, used `render=overlay`, copied the source byte-for-byte, and took about **292 s (4 min 52 s)** versus the previously reported 20–40 min. Times are approximate, derived from result-directory/file creation timestamps (Windows timestamp resolution); the benchmark completed inference but its Python process later exited with PostgreSQL `IdleInTransactionSessionTimeout` while closing the read-only model session. Another long-running Python process was also consuming CPU during measurement, so timings may include machine-load effects. No inference outputs were written to the database and no cloud calls were made. These are single local runs, not a benchmark distribution.
- **Not visually verified:** browser playback and canvas alignment/interpolation were not reviewed in a live browser. No backend was listening on port 8000 during the check; the shared compute slot was empty, and the stop/release behavior was verified through the API integration test instead. The HEVC result shows that fallback encoding can erase the speed benefit for non-browser-playable inputs; the AV1 overlay path showed the expected large reduction.
- **Artifacts:** direct benchmark outputs are under ignored `local-cache/results/media-tests/bench-*`. No `.pending` files remained. No edits were made to `sources/` or `D:\DrowningDemo`.

## Checkpoint 2026-10-26 (later): user flows that need no credentials

Direction from the user: training data comes from labeled datasets they find; they do not label for training. They only review label meaning, source groups, and model output on their own videos. 107 passing tests were never treated as model acceptance.

- **Implemented and tested (212/212 regression, `reports/regression-results.json`; no real training, no cloud call):**
  - Mapping review + quality report: contact sheets of real crops per source label, per-label box/split statistics, exact and near-duplicate (dHash) detection, proposed groups; human approvals stored under `reviews/` bound to the mapping/group hash (changing the mapping voids approval). CLI `source review-samples|quality|approve-mapping|approve-grouping`, API `/api/v1/sources/{id}/review/*`, UI page **Dữ liệu**.
  - Sources: CVAT now accepts many videos per source (`tasks:`); new `reviewed_jsonl` adapter (image + video, explicit time and person IDs, nothing inferred). Fixed an 8.3-path bug in CVAT relative paths.
  - Tracker: `ByteTracker` (Kalman + Hungarian + two confidence tiers) replaces the greedy IoU tracker as default; identities stay provisional.
  - Camera Start/Stop: `inference/camera.py`, `api/camera.py`, `/api/v1/camera/*`, UI page **Camera** (latest-frame capture, MJPEG preview, fps/latency/dropped stats, confirmed stop, shared compute slot, no local files accepted as a source).
  - Background recovery: `orchestration/supervisor.py` started with the API; recovers dead local jobs, dispatches queued Kaggle jobs, reconciles, collects, retries sync with backoff; only touches jobs owned by its own worker ID. Per-installation worker ID by default.
  - Kaggle side: `kaggle_run.py` (prepare_only with per-source scan/convert/audit/sheets/quality/temporal counts, remote_smoke), `run_collector.py`, `kaggle_train.py` (verified-group split across sources, crops, windows, coverage gates, two-stage training with train-only class weights, validation-only evaluation, no CPU fallback, clean "blocked" report instead of a crashed kernel). The generated notebook is now thin and was executed end-to-end locally from its embedded source for prepare and smoke.
  - Train/Evaluation UI: preflight gates, explicit new-version flow, job timeline, candidate sync, test with an explicitly chosen candidate, per-segment review marks stored server-side, confusion summary. `drownctl doctor [--online]`, `drownctl auth drive`, `docs/SETUP.md`, `docs/DATASET_ONBOARDING.md`.
- **Real-data checks run (read-only):** Alanoud v4 through the API: convert + sheets + quality completed on 9,997 images. Findings: out-of-water is 3.3% of boxes; 512 near-duplicate clusters (2,932 images) cross the export's own train/valid/test splits at 2-bit dHash distance, so the export's splits must not be used for evaluation; many `out of water` and `Drowning` crops are blur/texture or look mislabeled. The UI pages rendered with this data in headless Edge. **No mapping or grouping has been approved**: that needs the user.
- **Not verified (still open):** any live Kaggle push/status/output (CLI absent; status-line format and per-version output are assumptions), Drive OAuth/upload, shared remote PostgreSQL, person-detector weights and real-video inference, camera with a physical device, a real GPU run, tracker quality on real video, browser interaction beyond static rendering. No labeled temporal video dataset exists yet, so `train_candidate` is gated and untested outside CPU contract tests.
- **Next for the user:** fill `.env` per `docs/SETUP.md` (`drownctl doctor` lists exactly what is missing), review Alanoud sheets in the Dữ liệu page, find a labeled video dataset (CVAT or `reviewed_jsonl`), then run prepare_only -> review -> remote_smoke -> train_candidate from the Train page.

## Latest main-flow review — 2026-10-26

- **Implemented:** durable image/video upload -> claim shared slot -> resolve pinned approved model -> annotated media/JSON -> local download -> Drive sync/retry. Candidate review -> approved Drive archive -> verified atomic UUID cache is wired through API/CLI/UI. Worker transitions, remote receipts and candidate/result publication reject stale owner/token.
- **Tested:** final regression **107/107 passed, 0 failed, 0 errors, 0 skipped** (`reports/regression-results.json`), including **six actual PostgreSQL tests**, several with two OS processes. Fake Drive receipts cannot publish unverified/stale candidates. EXIF rotation, VP8/WebM codec/decoding, failed-job replay, busy-slot resume and output retry are covered with synthetic files/fake inference.
- **Real environment verified:** local PostgreSQL 17.10 is migrated; real TCP HTTP review passed 15 checks for health/readiness, production UI/assets, source scan, job persistence/idempotency/events and no-model rejection. The post-fix HTTP scan of 9,997 images / 18,506 boxes completed in 5.681 seconds in the first successful review; source path/size/mtime inventory stayed unchanged. Latest repeated HTTP timings are in the consolidated report JSON.
- **Build checks:** TypeScript/Vite build, UI lint, Python compile, locally generated Kaggle notebook and PostgreSQL readiness passed. Temporal and verified-split preflight correctly return `blocked`/exit 3. Notebook build is not a Kaggle run.
- **Report:** `reports/main-flow-review.html` consolidates the flow matrix, observed bugs and exact changed files, before/after results, startup/retry instructions, HTTP evidence, all test names and acceptance gaps. It was generated by a local helper script (no longer versioned) from actual checks; it never trains or calls cloud. One review-owned prepare job is stored/cancelled without dispatch and retains its events.
- **Not verified / incomplete:** real approved model/detector weights, temporal annotated data + source grouping/mapping review, Kaggle GPU training, live Drive/Kaggle, browser visual/playback acceptance, live camera, background recovery/polling and full training/evaluation/review UI. Model metrics/user acceptance and M6/M7 remain pending. Local DB provisioning resolves the DB configuration blocker; missing external credentials remain in `.env`.
- **Preservation:** no real local training, no edits to `sources/` or `D:\DrowningDemo`, no secrets in code/notebook/report. A Windows Unicode/8.3 path regression introduced during scan optimization was detected and corrected; its intermediate failed evidence is retained separately and is not counted as final success.
- **Final report check:** final HTTP scan took 5.990 seconds, all 15 HTTP checks passed, and all seven build/preflight checks returned their expected exit codes. HTML structure and 23 local/anchor links were verified; actual DB URLs/password values were absent from HTML/JSON. See `reports/report-verification.json`. Browser rendering remains unverified.
- **Runtime handoff:** the review-owned HTTP process was stopped after checks. A later attempt to leave a background API server on port 8000 was rejected by automatic approval review with `blocked by policy` and no detailed reason; no persistent API server is claimed. PostgreSQL remains available. Start the API manually using the README/report command.

## Milestone summary

### Review checkpoint 2026-10-26 (work in progress)

- **Implemented:** multipart image/video upload, pinned approved-model resolution, durable local inference jobs, local result downloads, Drive sync retry without recompute, and fencing checks before worker state/artifact publication.
- **Tested:** 24 focused repository/dispatcher/candidate/media HTTP tests passed after the fencing changes. Image/video tests use explicitly injected fake inference and fake Drive; no model quality or live cloud claim.
- **Real PostgreSQL tested:** dedicated Docker PostgreSQL 17.10 on loopback port 55439; local `.env` populated without exposing credentials. Alembic head applied to `drowning_dev`; five isolated-schema integration checks passed, including two-process slot/idempotency/approval contention and expired-lease retention. Shared/cloud PostgreSQL remains unverified.
- **Fixed during review:** concurrent first model approvals failed with `IntegrityError`; dialect upsert plus row locking now serializes the registry pointer. Docker bootstrap now waits for the final TCP server rather than the temporary initialization socket. UUID hex normalization, multipart headers, idempotent result replay, and action/API status handling were corrected.
- **Pending:** consolidated regression/report, approved-cache synchronization, real approved weights/detector, live Drive/Kaggle, temporal annotated data and browser/user acceptance. No local training was run.

### Approved cache checkpoint 2026-10-26

- **Implemented:** `POST /api/v1/models/sync-approved`, `drownctl model sync --approved`, UI sync button, exact registry SHA-256/size and bundle identity verification, atomic install into UUID version directories. Existing versions are retained and queued jobs resolve their pinned version.
- **Tested:** three new HTTP tests passed: candidate review -> download real synthetic bundle/archive through fake Drive -> local fake inference -> PNG download; cached second request avoids download; interrupted/corrupt download leaves old cache intact and permits retry; unapproved candidate installation refused.
- **Not verified:** actual Drive OAuth/download, trained weights and browser UI operation. Random-weight archive fixtures are technical tests only. API video output was changed to VP8/WebM for browser-compatible encoding; regression is pending at this checkpoint.

### Real scan review checkpoint 2026-10-26

- **Found in real HTTP review:** full source scan timed out at 60 seconds. CLI profiling measured 42.922 seconds, with label-file opens and repeated path checks dominating.
- **Implemented:** per-directory image index, one resolved source root, bounded eight-thread label reads (stable source ordering), uppercase image extensions and preserved ambiguous-image/path guards. Scan now validates numeric/finite/positive/in-bounds YOLO boxes instead of only field/class counts.
- **Measured:** same 9,997-image / 18,506-box dataset scanned in 10.242 seconds in the post-change CLI profile. These are individual local observations, not a repeatable performance benchmark. HTTP rerun and regression for the final changes are pending at this checkpoint.

| Milestone | Status | Evidence |
|---|---|---|
| M0 — Survey and scaffold | Complete for the local foundation | Standalone Python CLI/config scaffold, setup instructions, environment template, ignore rules, logging redaction, and 11 passing local tests. No service integration is claimed. |
| M1 — Dataset registry and conversion | In progress | YOLO/COCO/VOC/CVAT scan/conversion, review-only crops, canonical audit, temporal windows and deterministic group split code are implemented. Real mapping review and global source grouping remain pending; full quality reporting remains outstanding. |
| M2 — Temporal model and inference core | In progress | Model, bundle, track buffer, detector interface, provisional tracker, local image/video inference plumbing and manifest-gated image/temporal datasets are implemented. Tests are synthetic; no approved weights or real-video inference result exist. |
| M3 — Kaggle notebook and full data checks | In progress | Private notebook builder/materializer supports prepare, remote smoke and gated Stage A/B candidate code. Local bundle/cell checks pass; no Kaggle upload/run. |
| M4 — PostgreSQL, Drive, and Kaggle orchestration | In progress | Dedicated local PostgreSQL 17.10 is migrated and six real integration tests passed, including two-process concurrency. Fenced worker writes, private dispatch, exact-output candidate sync/retry and approval are implemented. Live Drive/Kaggle and background recovery remain unverified. |
| M5 — Local web and CLI workflows | In progress | API/UI image/video uploads, local result rendering/download, sync retry and approved-model cache sync are implemented and fixture-tested. Production build passes. No approved real weights; camera, full train/evaluation UI and browser acceptance remain pending. |
| M6 — Kaggle training and evaluation | Pending prerequisites | No real training has been run locally or remotely. |
| M7 — Acceptance and handoff | Pending | Requires real labeled data, configured services, a Kaggle run, real-video evaluation, and user review. |

## M0 — Survey and scaffold

### Implemented

- Surveyed `D:\AI\codex-cli-handoff`: it contains specification/examples, not an application or Git checkout. Created this separate implementation tree at `D:\AI\DrowningProject` as the handoff directs; left the handoff folder untouched. Added repository-specific preservation, cloud, label, and status rules in `AGENTS.md`.
- Python 3.11 package and `drownctl` CLI entry point.
- `.env` loader with process-environment precedence; `.env.example` has placeholders only.
- Safe `config-status` output reports configured/not-configured flags without printing connection strings or IDs.
- Kaggle is the only accepted training executor. Local training configuration is rejected.
- Logging redaction helper for common inline database passwords, tokens, API keys, and client secrets.
- Windows setup/run instructions and `.gitignore` rules for credentials, caches, model/video/archive files.

### Tested

- `python -m unittest discover -s tests -v` — 11 tests passed on Python 3.11.9 with PyYAML 6.0.3.
- `python -m drowning.cli --env-file missing.env config-status` — exited 0; reported DB/Drive/Kaggle unconfigured, Kaggle private, local training disabled, and printed no secret values.
- After creating local `.env`, `python -m drowning.cli config-status` — exited 0 and again confirmed DB/Drive/Kaggle are unconfigured and local training is disabled.
- Tested rejection of `ALLOW_LOCAL_TRAIN=true` and non-Kaggle executor settings.

### Not verified

- Editable installation in the project `.venv` succeeded; installation on a second fresh machine remains unverified.
- Local PostgreSQL is configured and tested. Drive/Kaggle credentials are absent; their live connections remain unverified.
- No Kaggle notebook was published or executed. No training was run.

## M1 — Dataset registry and conversion

### Implemented

- Version-1 source YAML structural validation for supported source format/modality values and explicit mapping actions.
- Exact label mapping after Unicode NFKC, casefold, trim, and space/underscore/hyphen normalization. Unknown labels resolve to `review`; semantic/fuzzy mapping is not used.
- Read-only class-list readers for YOLO-style YAML names and COCO JSON categories, preserving declared IDs including non-contiguous COCO IDs.
- YOLO detection converter resolves matching sibling images, decodes oriented image sizes, converts normalized `cxcywh` boxes to pixel `xyxy`, and rejects segmentation/pose rows, unknown IDs, non-finite values, and out-of-image boxes.
- COCO detection converter resolves image/category IDs (including non-contiguous category IDs), converts pixel `xywh` boxes to `xyxy`, retains crowd/ignore flags, and rejects unknown references or invalid boxes.
- Pascal VOC converter uses the 1-based inclusive coordinate convention, checks annotation/image dimensions, and keeps truncated/difficult flags.
- CVAT video-track XML converter preserves track/frame IDs, reviewed identity status, state attributes, occlusion/outside markers and either explicit PTS timestamps or explicitly verified CFR timing. Outside markers break temporal windows.
- Canonical JSONL output includes separate media and observation records, content SHA-256, source labels/IDs, mapping decision/version, and explicit unknown identity/grouping. Configurable output refuses overwrite, raw-root writes, any `sources/` path, and `D:\DrowningDemo`.
- `drownctl source convert CONFIG --output DERIVED.jsonl` uses the root path configured by `root_env` in `.env`.
- Canonical JSONL audit validates schema version, IDs, references, mapping values and bbox coordinates; summarizes per-source class/action counts; reports exact-content duplicates and same-box label conflicts for review. It does not automatically quarantine or claim split independence.
- `drownctl source audit FILE.jsonl [OTHER.jsonl ...]` runs the read-only audit and returns a nonzero exit code for structural/data issues.
- `drownctl source build-crops CONFIG CANONICAL.jsonl --output DIR` builds deterministic context-padded square crops and a manifest. Unapproved mappings are blocked for training crops; `--preview-only --limit N` creates review crops marked ineligible for training.
- `drownctl source build-windows CANONICAL.jsonl --recipe configs/recipes/temporal.yaml` builds causal windows only from reviewed observations with verified person IDs and timestamps. Image-only inputs return blocked and create no temporal sample.
- `drownctl source build-splits CANONICAL.jsonl --output split-manifest.jsonl` requires verified source/session groups, merges exact-content duplicates before assignment, and preserves user holdouts.
- Source-root status and class-list path traversal protection.
- CLI commands include `source validate`, `scan`, `inspect [--read-class-list]`, `convert`, `audit`, `build-crops`, `build-windows`, `build-splits`, and `map-label`.

### Tested

- At the M1 checkpoint, `python -m unittest discover -s tests -v` — 22 tests passed, including YOLO/COCO fixture conversion, class ID preservation, XYWH conversion, unknown-label review, crowd/ignore flags, malformed annotation rejection, output protection, CLI scan/conversion, and canonical audit conflict/error reporting.
- `python -m compileall -q src tests` — passed.
- A local `.venv` was created and editable package installation succeeded. PyTorch 2.14.1+cpu and TorchVision 0.29.1+cpu are installed there only for CPU architecture checks; `torch.cuda.is_available()` is false.
- Local tests also cover valid/invalid mappings, normalized duplicate labels, unknown and ambiguous labels, YOLO names order, non-contiguous COCO category IDs, and class-list path traversal.
- Both illustrative descriptors `source_university_yolo.yaml` and `source_extra_coco.yaml` passed config validation. They remain examples, not evidence of source data availability or reviewed label semantics.
- `not drowning` against the University example returned `review` with `canonical_label: null`.
- Real read-only inventory of `D:\DrowningKaggleCheck\data`: 9,997 image files and 9,997 YOLO label files; 18,506 bbox rows; zero malformed rows and zero missing image pairs. Class counts by source ID: 0=10,699, 1=7,195, 2=612. Export splits: train=6,014, valid=1,977, test=2,006. Dataset README says 9,999 images, so the checked-out export has two fewer than its README claim.
- Real conversion command: `python -m drowning.cli source convert configs/datasets/alanoud_swimming_drowning_v4.yaml --output local-cache/derived/alanoud-v4.jsonl`; produced 9,997 media + 18,506 observations (28,503 records), mapping counts match source class IDs. Output is ignored local derived data; the raw export was read-only.
- Real audit command: `python -m drowning.cli source audit local-cache/derived/alanoud-v4.jsonl --output local-cache/derived/alanoud-v4-audit.json`; valid JSONL, zero structural issues, 99 exact duplicate-content groups, zero same-box label conflicts, and zero exact duplicates crossing the export's train/valid/test labels.
- Real crop-preview command produced 30 crops stratified 10 per mapped class under `local-cache/derived/alanoud-v4-crop-preview/`; all 30 require review and zero are training-eligible. No crops were written to raw data.
- Real temporal preflight command `python -m drowning.cli source build-windows local-cache/derived/alanoud-v4.jsonl` returned `blocked`: zero temporal videos, tracks and windows. It created no temporal data.
- Real split preflight returned `blocked` for all 9,997 media because global video/session groups are unknown; no split manifest was written.
- Pascal VOC and CVAT track conversion passed synthetic fixtures. The CVAT fixture exercised explicit timestamps, reviewed track identity, and an outside marker; no real video source was available.

### Not implemented or verified

- Source mapping remains `pending_sample_review`; class-name equivalence has not been visually/semantically reviewed and data are not approved for training. The JSONL audit is a partial validator, not a complete versioned schema implementation. There is no mapping review artifact, full image/video quality report, or real verified source-video grouping/split manifest.
- Reviewed-JSONL parsing, classification-folder policy, safe archive inventory, complete quality reports and cross-source source-identity resolution are not implemented. The temporal window and group-split builders are synthetic-fixture-tested only; no real temporal/source-group data are available.
- This real source is image-only; there are no ordered frames/person IDs/state-time annotations from it, so no temporal samples were created. Zero exact duplicate hashes crossing the export split is not evidence that source videos/scenes are disjoint. No image/video data were modified.
- This milestone is not complete.

## M2 — Temporal model and inference core

### Implemented

- Shared MobileNetV3-small image encoder, three-class image head, and unidirectional masked GRUCell temporal head. Pretrained weights are opt-in and are not downloaded implicitly.
- Frame validity masks leave recurrent state unchanged for missing frames; the final frame must be a valid current anchor.
- Stage A/B training-epoch functions, temporal/image loss functions, auxiliary image loss for Stage B, and accuracy/macro-F1/per-class-recall/confusion metrics. These functions have not been used to train this project’s model.
- Image and video-window datasets require training-eligible manifests, split membership, matching verified source groups, label review and crop checksums; the temporal dataset decodes only the assigned frame indices from the same verified person track.
- Model bundles record architecture, class/ontology order, preprocessing, temporal-trained flag, completion kind, file size and checksum. Synthetic-smoke bundles can be loaded for tests but cannot pass candidate validation or be marked approved by bundle save.
- Per-track causal frame buffer resets on non-monotonic timestamps, large gaps, scene changes, or model-version changes. Tracker IDs are provisional; this is not persistent verified identity.
- Shared RGB crop/tensor preprocessing, injected person-detector interface, optional explicit-weight Ultralytics detector (no implicit download), greedy IoU tracking, image predictions, and annotated offline video/detections JSON output are implemented. Tracker IDs remain provisional.
- Baseline image, temporal and evaluation recipes are configurable and marked as unbenchmarked starting values.

### Tested

- Current local `.venv`: Python 3.11.9, PyTorch 2.14.1+cpu, TorchVision 0.29.1+cpu; CUDA unavailable.
- Final regression evidence is recorded by the regression runner (a local helper script, no longer versioned) in `reports/regression-results.json`; earlier 85-test checkpoint is superseded. Checks cover model/data contracts, HTTP uploads, version cache, fencing, actual local PostgreSQL contention, and fake-provider cloud recovery. Synthetic inference tests do not establish model quality.
- `python -m compileall -q src tests` — passed.
- Synthetic model checks used random tensors only; no dataset examples were used and no optimizer/training epoch was run.

### Not implemented or verified

- No approved person-detector weights, production multi-object tracker, webcam/URL camera support, or measured CPU throughput yet. The IoU tracker is a basic provisional tracker.
- Image/video plumbing tests use a fake classifier/detector and a temporary synthetic four-frame clip; they do not establish model accuracy or real-video behavior. The real image dataset was not used for inference.
- No pretrained weights were downloaded, no trained model exists, and no model quality claim can be made.
- Stage A/B training code has not been run on Kaggle or any dataset. Temporal training cannot be prepared from the current image-only dataset.

## M3 — Kaggle notebook and full data checks

### Implemented

- `notebooks/build_notebook.py` embeds the current Python package/config source into a generated notebook bundle; local paths, model weights, dataset files, and `.env` are excluded.
- Generated `notebooks/Drowning_Temporal_Kaggle.ipynb` defaults to `prepare_only`; `remote_smoke` is separate; `train_candidate` includes Stage A/B GPU training and validation, but gates execution on audited inputs, all-class train/validation coverage and verified split groups.
- `notebooks/kernel-metadata.template.json` uses a placeholder owner and sets `is_private=true`, GPU enabled, empty dataset inputs.

### Tested

- `python notebooks/build_notebook.py` generated the notebook.
- All four code cells parsed/compiled locally; notebook builder tests check the embedded archive contains model/training code but no `.env`, weights, local cache, or `D:\DrowningKaggleCheck` path. `--mode` selects prepare, synthetic smoke, or gated candidate mode.
- Template JSON parsed and reports private/GPU settings.
- Current real-data temporal preflight is blocked with zero temporal videos/tracks/windows.

### Not run or verified

- No Kaggle credentials/kernel reference are configured. Notebook was not pushed, published, or run remotely; privacy and GPU execution have only been checked in the local template, not in Kaggle.
- `train_candidate` code path and all notebook cells were locally syntax-checked but not executed. It blocks against the current image-only source because there are no reviewed person/time video annotations or verified source groups. No GPU model candidate exists.

## M4 — PostgreSQL, Drive, and Kaggle orchestration

### Implemented

- SQLAlchemy schema for dataset sources/snapshots, recipes, model versions/approved pointer, workers, jobs/events, artifacts/uploads, evaluations/reviews, and one singleton execution slot.
- Alembic initial migration seeds the singleton slot. `drownctl db check` and `drownctl db upgrade` require PostgreSQL and the local `.env` DSN.
- Job repository uses a unique idempotency key, explicit state transitions, row locking, fencing token increment, and refuses slot release without a terminal state plus confirmed compute stop.
- Kaggle CLI wrapper passes argument arrays, requires `is_private=true`, validates kernel/code paths, collects only exact-version outputs, and marks push timeouts remote-unconfirmed without retrying.
- Kernel materializer creates a mode-matched private kernel directory without pushing; it includes only the notebook and Kaggle metadata. `drownctl jobs dispatch/reconcile/status` operates durable jobs manually; ambiguous pushes keep the fencing slot and are never retried automatically.
- Google Drive adapter supports resumable upload, artifact-key reconciliation, SHA-256/MD5 and size verification, safe download to a temporary file then rename, and blocks overwrite/protected destinations.
- Candidate collector downloads only the exact completed kernel version, rejects unsafe/extra ZIP files, validates model bundle/checksums/fingerprints and metrics, uploads a single bundle artifact, then creates a DB candidate (never approved). Drive failure becomes `sync_pending`; `drownctl jobs sync` retries from the local archive without compute.

### Tested

- SQLite migration upgrade/downgrade remains schema smoke only. Dedicated Docker PostgreSQL 17.10 now also passes real Alembic head/singleton and transaction checks in isolated test schemas.
- SQLite repository tests pass for idempotency, state order, global-slot exclusivity, fencing, terminal confirmation and stale-owner rejection.
- Kaggle/Drive adapter tests use injected fake CLI/provider objects for command construction, private-kernel guard, exact version, checksum and timeout paths. No network or cloud credentials were used.
- Dispatcher/materializer tests check run-mode matching, private-only kernel contents, train-candidate preflight gates, fencing-slot retention through remote-unconfirmed/running/collecting states, and no retry after timeout.
- Candidate sync tests use a locally generated unapproved candidate fixture and fake Drive/Kaggle providers; they verify exact-version lookup, zip validation, candidate registration, confirmed slot release and sync retry.
- Candidate collection tests cover exact-version output verification, Drive artifact registration, candidate-only status, compute-slot release after completed remote compute, and sync retry without a second compute dispatch.
- Database readiness tests require `alembic_version=20261026_01`; an unversioned or stale schema reports not-ready.

### Not run or implemented

- Real local PostgreSQL is configured on loopback port 55439; six tests verify migration, two-process slot competition, two-process idempotency, simultaneous approvals, stale-token rejection and expired-lease retention. Shared/cloud PostgreSQL and multiple physical machines remain unverified.
- No Drive OAuth token/folder ID or Kaggle credentials/kernel owner are configured; live adapters have not been connected.
- Background polling, remote receipt recovery/takeover after process crash and Drive folder provisioning/permission checks remain outstanding. Dispatch/reconcile/collect/sync are explicit CLI operations. Candidate approve/reject is gated on a ready verified Drive artifact.

## M5 — Local API and web UI

### Implemented and tested

- FastAPI `/health`, `/ready`, config status, source list/scan, job create/list/status/events, model list, and OpenAPI docs.
- Multipart image/video upload uses durable pinned-model inference jobs and the shared compute slot. Results remain downloadable locally when Drive fails; sync retry does not rerun inference. API video output is VP8/WebM, with real file codec/decoding checks using synthetic clips.
- Approved model sync is available through API, CLI and UI: verified archive download, bundle/registry identity validation, UUID version cache and atomic installation preserve previous versions. Tests use random-weight bundles and fake Drive, not trained candidates.
- Candidate model review route records reviewer/comment and swaps the approved pointer only after temporal-candidate and verified Drive-artifact checks pass.
- React/TypeScript/Vite UI shows service configuration, source mapping/review, read-only scan results, model versions, jobs and clear DB/Kaggle readiness. Job creation is disabled when PostgreSQL is unavailable.
- `drownctl serve` runs the local API; Vite proxies API routes in development, and FastAPI serves the built `web/dist` UI.
- Current local API scan of Alanoud v4 returned 9,997 images, 18,506 annotations and zero malformed rows; source files were not modified.
- `npm run build` — Vite 8.3.5 and TypeScript build passed.
- API tests pass for degraded config status and idempotent job HTTP flow against a temporary SQLite schema. They do not establish live DB behavior.
- FastAPI served the production React page and its `/assets` bundle. A real API scan of Alanoud v4 returned 9,997 images and 18,506 annotations. Browser visual review was blocked when the browser origin permission gate declined `http://127.0.0.1:8000`; no alternate browser route was attempted. API/static/build checks passed directly.

### Not implemented or verified

- API job creation persists `queued` and explicitly returns `dispatch: not_started`; no worker is dispatched.
- Live camera Start/Stop, SSE, full train/new-version/evaluation and mapping-review UI, background recovery and browser visual acceptance remain outstanding. Image/video rendering markup/build and API files are tested; actual browser playback remains unverified.

## Current configuration blockers

- Local ignored `.env` has the Alanoud read-only root, default private kernel directory, local `DATABASE_URL` and dedicated `TEST_DATABASE_URL`. Credentials were generated locally and are not included in reports/source. Docker container `drowning-local-pg-20261026` retains volume `drowning-local-pgdata-20261026`.
- Google Drive folder/OAuth, Kaggle kernel reference/authentication and detector weights remain unset. The configured approved cache has no real model manifest; the development registry contains no approved real model. Temporal metadata/grouping and mapping review are still missing. These block real training/inference/cloud acceptance, not local implementation or fixture tests.
- These values belong in the local ignored `.env` or their provider-specific local auth files. Nothing is embedded in notebook or source code.
- Temporal video/person/time annotations and real-video acceptance data have not been verified in this workspace.

## Checkpoint 2026-10-26 (late): image mode, optional review, shared setup

- **Tests run:** full regression 262/262 passed (the regression runner (a local helper script, no longer versioned), shared Neon DB URLs blanked so no test touches it). The result file `reports/regression-results.json` is regenerated on each run and is no longer versioned.
- **Real cloud checks:** `prepare_only` on Kaggle T4 completed (9,997 images, 18,506 boxes, 0 malformed); `remote_smoke` completed (Tesla T4, CUDA, gradients received; synthetic input only, creates no model). A `train_candidate` job in image mode (ImageNet init, 10 image epochs, Alanoud only) was submitted; its result is not recorded here until collected and evaluated.
- **Code written:** image-only training branch (video frames first, then images; video frames sampled per person every 500 ms, label changes always kept, at most 300 per video), `image_tracked` bundle kind with per-track score aggregation at inference, optional (advisory) data review with automatic rules and AI verdicts, UTF-8-safe CLI output. The web UI trains the image branch only; the temporal branch remains reachable through the API.
- **Not verified:** accuracy of any model (none evaluated yet); aggregation parameters are unbenchmarked defaults; no labeled video source exists, so the video-frame phase has been tested only on synthetic fixtures; setup by a second person with their own Kaggle/Google accounts has not been tried.
- **Repository:** git root moved to `DrowningProject`; `codex-cli-handoff` is outside the repository.
