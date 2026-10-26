"""FastAPI local service for health, dataset inspection and durable job metadata."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import UUID

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from drowning.config import Settings, load_env_file
from drowning.data.adapters import scan_source
from drowning.api.camera import CameraBusy, CameraNotRunning, CameraService
from drowning.api.evaluation import MarkError, ReviewMarks, model_evaluation
from drowning.api.review import ReviewConflict, SourceReviewService
from drowning.api.train import MODES, TrainBlocked, TrainService
from drowning.data.review import ReviewError
from drowning.installation import installation_id
from drowning.orchestration.supervisor import JobSupervisor
from drowning.api.inference import InferenceUnavailable, get_approved_model
from drowning.inference.camera import (STREAM_MEDIA_TYPE, CameraConnectError, CameraSourceError,
                                        CameraStopUnconfirmed)
from drowning.api.media import MAX_MEDIA_UPLOAD_BYTES
from drowning.api.test_jobs import (LocalInferenceCancelled, LocalInferenceFailed, LocalMediaTestJobs,
                                    MediaRequestConflict, MediaSlotBusy)
from drowning.data.registry import SourceConfigError, inspect_source, load_source, resolve_source_root, validate_source
from drowning.db.models import Artifact, Job, JobEvent, ModelVersion
from drowning.db.repositories import (StaleExecutionOwner, ModelReviewError,
                                      create_job, review_model_version)
from drowning.db.session import check_db, create_db_engine
from drowning.domain.jobs import JobState
from drowning.installation import resolve_worker_id


class CreateJobRequest(BaseModel):
    job_type: Literal["train", "evaluation", "inference", "prepare"]
    idempotency_key: str = Field(min_length=8, max_length=200)
    payload: dict[str, Any] = Field(default_factory=dict)


class ApproveMappingRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=160)
    confirmed_labels: list[str]
    samples_viewed: dict[str, int]
    notes: dict[str, str] = Field(default_factory=dict)


class ApproveGroupingRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=160)
    note: str = Field(min_length=1, max_length=4000)


class TrainSubmitRequest(BaseModel):
    mode: Literal["prepare_only", "remote_smoke", "train_candidate"]
    new_version: bool = False
    reason: str = Field(default="", max_length=2000)
    idempotency_key: str = Field(min_length=8, max_length=200)
    pretrained: bool = False
    image_epochs: int = Field(default=5, ge=1, le=50)
    temporal_epochs: int = Field(default=5, ge=1, le=50)
    video_epochs: int = Field(default=3, ge=1, le=50)
    frame_stride_ms: int = Field(default=500, ge=100, le=5000)
    max_frames_per_video: int = Field(default=300, ge=10, le=5000)
    branch: Literal["temporal", "image_only"] = "image_only"


class ReviewMarkRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=160)
    track_id: str = Field(min_length=1, max_length=80)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    verdict: Literal["correct", "wrong"]
    expected_label: Literal["drowning", "swimming", "out_of_water"] | None = None
    note: str = Field(default="", max_length=1000)


class CameraStartRequest(BaseModel):
    source: str = Field(min_length=1, max_length=2000, description="Webcam index (0-15) or http/https/rtsp/rtmp URL")
    idempotency_key: str = Field(min_length=8, max_length=200)


class CameraProbeRequest(BaseModel):
    source: str = Field(min_length=1, max_length=2000)


class ReviewModelRequest(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer_id: str = Field(min_length=1, max_length=160)
    comment: str = Field(default="", max_length=4000)


def _public_source(source: dict[str, Any], env: dict[str, str] | None = None) -> dict[str, Any]:
    report = inspect_source(source, env)
    return {
        "source_id": report["source_id"],
        "source_version": report["source_version"],
        "format": report["format"],
        "modality": report["modality"],
        "valid_config": report["valid_config"],
        "errors": report["errors"],
        "root_env": report["root_env"],
        "root_configured": report["root_configured"],
        "root_exists": report["root_exists"],
        "mapping_review_status": source.get("mapping_review", {}).get("status", "unknown"),
        "mapping_actions": report["mapping"],
    }


def create_app(*, settings: Settings | None = None, engine: Engine | None = None,
               config_dir: Path | None = None, env_file: Path | None = None,
               inference_factory: Callable[[ModelVersion, dict[str, Any], Path], Any] | None = None,
               drive_factory: Callable[[], Any] | None = None,
               camera_capture_factory: Callable[[Any], Any] | None = None) -> FastAPI:
    env_path = env_file or (Path.cwd() / ".env")
    load_env_file(env_path)
    app_settings = settings or Settings.from_env()
    app_settings = replace(app_settings, worker_id=resolve_worker_id(app_settings.worker_id, app_settings.input_cache_dir))
    db_engine = engine
    if db_engine is None and app_settings.database_url:
        try:
            db_engine = create_db_engine(app_settings.database_url)
        except ValueError:
            db_engine = None
    session_factory = sessionmaker(bind=db_engine, expire_on_commit=False) if db_engine is not None else None
    sources_dir = config_dir or (Path(__file__).resolve().parents[3] / "configs" / "datasets")
    web_dir = sources_dir.parent.parent / "web"
    web_dist = web_dir / "dist"

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        supervisor = application.state.supervisor
        if supervisor is not None:
            supervisor.start()
        yield
        if supervisor is not None:
            supervisor.stop()
        if db_engine is not None and engine is None:
            db_engine.dispose()

    app = FastAPI(title="DrowningProject local API", version="0.1.0", lifespan=lifespan)
    app.state.settings = app_settings
    app.state.engine = db_engine
    app.state.session_factory = session_factory
    app.state.sources_dir = sources_dir
    app.state.web_dist = web_dist
    app.state.inference_factory = inference_factory
    app.state.drive_factory = drive_factory
    app.state.supervisor = None
    app.state.reviews = SourceReviewService(
        sources_dir, reviews_dir=sources_dir.parent.parent / "reviews",
        work_dir=Path(app_settings.input_cache_dir).expanduser().resolve().parent)
    app.state.train = (TrainService(session_factory, app_settings, project_root=sources_dir.parent.parent,
                                    reviews=app.state.reviews) if session_factory is not None else None)
    app.state.camera = (CameraService(session_factory, app_settings, inference_factory=inference_factory,
                                      capture_factory=camera_capture_factory)
                        if session_factory is not None else None)
    app.state.media_jobs = (LocalMediaTestJobs(session_factory, app_settings,
                                               inference_factory=inference_factory,
                                               drive_factory=drive_factory)
                            if session_factory is not None else None)

    if session_factory is not None and app_settings.supervisor_enabled:
        def default_drive():
            from drowning.integrations.drive import GoogleDriveArtifactAdapter, build_drive_service
            return GoogleDriveArtifactAdapter(build_drive_service(Path(app_settings.google_oauth_token_file)))

        def default_kaggle():
            from drowning.integrations.kaggle import KaggleCLIClient
            return KaggleCLIClient()

        app.state.supervisor = JobSupervisor(
            session_factory, worker_id=app_settings.worker_id,
            installation=installation_id(Path(app_settings.input_cache_dir).expanduser().resolve().parent),
            kaggle_factory=default_kaggle, drive_factory=drive_factory or default_drive,
            drive_folder_id=app_settings.drive_folder_id, cache_root=Path(app_settings.output_cache_dir).expanduser(),
            interval_seconds=app_settings.supervisor_interval_seconds,
            is_tracked_locally=lambda job_id: app.state.camera is not None and app.state.camera.owns(job_id))

    if (web_dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=web_dist / "assets"), name="web-assets")

    def get_session(request: Request):
        factory = request.app.state.session_factory
        if factory is None:
            raise HTTPException(status_code=503, detail="PostgreSQL is not configured")
        try:
            if not check_db(request.app.state.engine):
                raise HTTPException(status_code=503, detail="PostgreSQL schema migration is not at the current head")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=503, detail="PostgreSQL readiness check failed")
        with factory() as session:
            yield session

    def get_media_jobs(request: Request, session: Session = Depends(get_session)):
        service = request.app.state.media_jobs
        if service is None:
            raise HTTPException(status_code=503, detail="PostgreSQL is not configured")
        return service

    @app.get("/", response_class=HTMLResponse)
    def index():
        page = web_dist / "index.html"
        if page.is_file():
            return HTMLResponse(page.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>DrowningProject API is running</h1><p>Build the local UI with <code>cd web &amp;&amp; npm run build</code>, or run <code>npm run dev</code> on port 5173.</p><p><a href='/docs'>API documentation</a></p>")

    @app.get("/health")
    def health():
        return {"status": "alive", "service": "drowning-project"}

    @app.get("/ready")
    def ready(request: Request):
        engine_ref = request.app.state.engine
        if engine_ref is None:
            raise HTTPException(status_code=503, detail={"ready": False, "reason": "DATABASE_URL is not configured"})
        try:
            if not check_db(engine_ref):
                raise HTTPException(status_code=503, detail={"ready": False, "reason": "database schema is not at the expected Alembic revision"})
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=503, detail={"ready": False, "reason": "PostgreSQL check failed"})
        return {"ready": True, "database": "reachable"}

    @app.get("/api/v1/config-status")
    def config_status(request: Request):
        status = request.app.state.settings.status()
        status["database_reachable"] = None
        if request.app.state.engine is not None:
            try:
                status["database_reachable"] = check_db(request.app.state.engine)
            except Exception:
                status["database_reachable"] = False
        return status

    @app.get("/api/v1/sources")
    def list_sources(request: Request):
        result = []
        for path in sorted(request.app.state.sources_dir.glob("*.yaml")):
            try:
                source = load_source(path)
                result.append(_public_source(source))
            except SourceConfigError as exc:
                result.append({"config_file": path.name, "valid_config": False, "errors": [str(exc)]})
        return {"sources": result}

    @app.post("/api/v1/sources/{source_id}/scan")
    def scan_dataset(source_id: str, request: Request):
        source = _find_source(request.app.state.sources_dir, source_id)
        root = resolve_source_root(source)
        if root is None or not root.is_dir():
            raise HTTPException(status_code=409, detail=f"Configure {source['paths']['root_env']} in .env")
        errors = validate_source(source)
        if errors:
            raise HTTPException(status_code=422, detail={"errors": errors})
        try:
            return scan_source(source, root)
        except SourceConfigError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    def review_call(action: Callable[[], Any]):
        try:
            return action()
        except KeyError:
            raise HTTPException(status_code=404, detail="Dataset source not found")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Review material has not been generated for the current mapping")
        except (ReviewConflict, SourceConfigError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ReviewError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/api/v1/sources/{source_id}/review")
    def source_review(source_id: str, request: Request):
        return review_call(lambda: request.app.state.reviews.status(source_id))

    @app.post("/api/v1/sources/{source_id}/review/prepare")
    def source_review_prepare(source_id: str, request: Request):
        return review_call(lambda: request.app.state.reviews.prepare(source_id))

    @app.get("/api/v1/sources/{source_id}/review/sheets/{key}")
    def source_review_sheet(source_id: str, key: str, request: Request):
        return review_call(lambda: FileResponse(request.app.state.reviews.sheet_file(source_id, key),
                                                media_type="image/jpeg"))

    @app.get("/api/v1/sources/{source_id}/review/rules")
    def source_review_rules(source_id: str, request: Request):
        return review_call(lambda: request.app.state.reviews.auto_rules(source_id))

    @app.post("/api/v1/sources/{source_id}/review/evidence")
    def source_review_evidence(source_id: str, request: Request):
        return review_call(lambda: request.app.state.reviews.evidence(source_id))

    @app.post("/api/v1/sources/{source_id}/review/verdict")
    def source_review_verdict(source_id: str, body: dict[str, Any], request: Request):
        return review_call(lambda: request.app.state.reviews.submit_verdict(source_id, body))

    @app.post("/api/v1/sources/{source_id}/review/approve-mapping")
    def source_approve_mapping(source_id: str, body: ApproveMappingRequest, request: Request):
        return review_call(lambda: request.app.state.reviews.approve_mapping(
            source_id, reviewer=body.reviewer, confirmed_labels=body.confirmed_labels,
            samples_viewed=body.samples_viewed, notes=body.notes))

    @app.post("/api/v1/sources/{source_id}/review/approve-grouping")
    def source_approve_grouping(source_id: str, body: ApproveGroupingRequest, request: Request):
        return review_call(lambda: request.app.state.reviews.approve_grouping(
            source_id, reviewer=body.reviewer, note=body.note))

    @app.get("/api/v1/test/readiness")
    def test_readiness(request: Request, session: Session = Depends(get_session)):
        try:
            model, _, _ = get_approved_model(session, request.app.state.settings)
            if request.app.state.inference_factory is None:
                weights = request.app.state.settings.person_detector_weights_path
                if not weights or not Path(weights).is_file():
                    raise InferenceUnavailable("PERSON_DETECTOR_WEIGHTS is missing")
            return {"ready_for_request": True, "model_version_id": str(model.id),
                    "model_version": model.display_version, "max_upload_bytes": MAX_MEDIA_UPLOAD_BYTES,
                    "model_loaded": False}
        except InferenceUnavailable as exc:
            return {"ready_for_request": False, "reason": str(exc),
                    "max_upload_bytes": MAX_MEDIA_UPLOAD_BYTES, "model_loaded": False}

    def execute_media(upload: UploadFile, modality: str, key: str, service: LocalMediaTestJobs,
                      model_version_id: UUID | None = None, frame_stride: int = 3):
        try:
            return service.submit(upload, modality, key, model_version_id, frame_stride=frame_stride)
        except (InferenceUnavailable, StaleExecutionOwner) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except MediaSlotBusy as exc:
            raise HTTPException(status_code=409, detail={"reason": str(exc), "job_id": str(exc.job_id), "state": "queued"})
        except MediaRequestConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except LocalInferenceFailed as exc:
            raise HTTPException(status_code=500, detail={"reason": str(exc), "job_id": str(exc.job_id)})
        except LocalInferenceCancelled as exc:
            raise HTTPException(status_code=409, detail={"reason": str(exc), "job_id": str(exc.job_id),
                                                        "state": "cancelled"})
        finally:
            upload.file.close()

    @app.post("/api/v1/test/image")
    def test_image(file: UploadFile = File(...),
                   idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=8, max_length=200),
                   model_version_id: UUID | None = None,
                   service: LocalMediaTestJobs = Depends(get_media_jobs)):
        return execute_media(file, "image", idempotency_key, service, model_version_id)

    @app.post("/api/v1/test/video")
    def test_video(file: UploadFile = File(...),
                   idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=8, max_length=200),
                   model_version_id: UUID | None = None,
                   frame_stride: int = Query(3, ge=1, le=10),
                   service: LocalMediaTestJobs = Depends(get_media_jobs)):
        return execute_media(file, "video", idempotency_key, service, model_version_id, frame_stride)

    def get_camera(request: Request, session: Session = Depends(get_session)) -> CameraService:
        service = request.app.state.camera
        if service is None:
            raise HTTPException(status_code=503, detail="PostgreSQL is not configured")
        return service

    def camera_call(action: Callable[[], Any]):
        try:
            return action()
        except CameraSourceError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except CameraBusy as exc:
            raise HTTPException(status_code=409, detail={"reason": str(exc), "job_id": str(exc.job_id) if exc.job_id else None})
        except (InferenceUnavailable, CameraNotRunning, StaleExecutionOwner) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except CameraConnectError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        except CameraStopUnconfirmed as exc:
            raise HTTPException(status_code=503, detail={"reason": str(exc), "compute_stop_confirmed": False})

    def get_train(request: Request, session: Session = Depends(get_session)) -> TrainService:
        return request.app.state.train

    def train_call(action: Callable[[], Any]):
        try:
            return action()
        except TrainBlocked as exc:
            raise HTTPException(status_code=409, detail={"reasons": exc.reasons})
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")

    @app.get("/api/v1/train/preflight")
    def train_preflight(service: TrainService = Depends(get_train)):
        return service.preflight()

    @app.post("/api/v1/train/submit")
    def train_submit(body: TrainSubmitRequest, service: TrainService = Depends(get_train)):
        return train_call(lambda: service.submit(
            body.mode, new_version=body.new_version, reason=body.reason, idempotency_key=body.idempotency_key,
            options={"pretrained": body.pretrained, "image_epochs": body.image_epochs, "temporal_epochs": body.temporal_epochs,
                     "branch": body.branch, "video_epochs": body.video_epochs, "frame_stride_ms": body.frame_stride_ms,
                     "max_frames_per_video": body.max_frames_per_video}))

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: UUID, service: TrainService = Depends(get_train)):
        return train_call(lambda: service.cancel(job_id))

    @app.get("/api/v1/jobs/{job_id}/run-report")
    def job_run_report(job_id: UUID, session: Session = Depends(get_session)):
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        summary = (job.payload or {}).get("run_summary")
        if summary is None:
            raise HTTPException(status_code=404, detail="This job has no collected run report")
        report_dir = (job.payload or {}).get("run_report_dir")
        report_file = Path(report_dir) / "run-report.json" if report_dir else None
        detail = (json.loads(report_file.read_text(encoding="utf-8"))
                  if report_file is not None and report_file.is_file() else None)
        sources_detail: dict[str, Any] = {}
        if detail is not None:
            for entry in detail.get("sources", []):
                if entry.get("status") != "prepared":
                    continue
                folder = Path(report_dir) / "per-source" / f"{Path(str(entry['source_id'])).name}@{Path(str(entry['source_version'])).name}"
                quality_file = folder / "quality.json"
                if quality_file.is_file():
                    sources_detail[entry["source_id"]] = {"quality": json.loads(quality_file.read_text(encoding="utf-8"))}
        return {"job_id": str(job.id), "summary": summary, "report": detail, "sources_detail": sources_detail,
                "stored_on_this_installation": detail is not None,
                "drive_file_id": (job.payload or {}).get("drive_file_id")}

    @app.post("/api/v1/supervisor/tick")
    def supervisor_tick(request: Request, force: bool = True):
        supervisor = request.app.state.supervisor
        if supervisor is None:
            raise HTTPException(status_code=409, detail="The supervisor is disabled or the database is not configured")
        return {"actions": supervisor.tick(force=force), "status": supervisor.status()}

    @app.get("/api/v1/supervisor/status")
    def supervisor_status(request: Request):
        supervisor = request.app.state.supervisor
        return supervisor.status() if supervisor is not None else {"running": False, "reason": "disabled or no database"}

    @app.get("/api/v1/camera/status")
    def camera_status(service: CameraService = Depends(get_camera)):
        return camera_call(service.status)

    @app.post("/api/v1/camera/probe")
    def camera_probe(body: CameraProbeRequest, service: CameraService = Depends(get_camera)):
        return camera_call(lambda: service.probe(body.source))

    @app.post("/api/v1/camera/start")
    def camera_start(body: CameraStartRequest, service: CameraService = Depends(get_camera)):
        return camera_call(lambda: service.start(body.source, body.idempotency_key))

    @app.post("/api/v1/camera/stop")
    def camera_stop(service: CameraService = Depends(get_camera)):
        return camera_call(service.stop)

    @app.get("/api/v1/camera/stream")
    def camera_stream(service: CameraService = Depends(get_camera)):
        return StreamingResponse(camera_call(service.stream), media_type=STREAM_MEDIA_TYPE)

    @app.get("/api/v1/jobs")
    def list_jobs(session: Session = Depends(get_session)):
        jobs = session.scalars(select(Job).order_by(Job.created_at.desc()).limit(100)).all()
        return {"jobs": [_job_dict(job) for job in jobs]}

    @app.post("/api/v1/jobs")
    def submit_job(body: CreateJobRequest, session: Session = Depends(get_session)):
        job, created = create_job(session, job_type=body.job_type, idempotency_key=body.idempotency_key,
                                  payload=body.payload)
        return {"job": _job_dict(job), "created": created, "dispatch": "not_started"}

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: UUID, session: Session = Depends(get_session)):
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job": _job_dict(job)}

    @app.get("/api/v1/jobs/{job_id}/events")
    def get_job_events(job_id: UUID, session: Session = Depends(get_session)):
        if session.get(Job, job_id) is None:
            raise HTTPException(status_code=404, detail="Job not found")
        events = session.scalars(select(JobEvent).where(JobEvent.job_id == job_id).order_by(JobEvent.sequence)).all()
        return {"events": [{"sequence": event.sequence, "from_state": event.from_state,
                            "to_state": event.to_state, "event_type": event.event_type,
                            "details": event.details, "created_at": event.created_at.isoformat()} for event in events]}

    @app.get("/api/v1/jobs/{job_id}/results")
    def local_results(job_id: UUID, service: LocalMediaTestJobs = Depends(get_media_jobs)):
        try:
            return service.result(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")

    @app.get("/api/v1/jobs/{job_id}/results/{role}")
    def local_result_file(job_id: UUID, role: str, service: LocalMediaTestJobs = Depends(get_media_jobs)):
        try:
            path = service.result_file(job_id, role)
            return FileResponse(path)
        except (KeyError, FileNotFoundError):
            raise HTTPException(status_code=404, detail="Local result file not found on this installation")
        except MediaRequestConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/api/v1/jobs/{job_id}/sync-results")
    def sync_local_results(job_id: UUID, service: LocalMediaTestJobs = Depends(get_media_jobs)):
        try:
            return service.retry_sync(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")
        except MediaRequestConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Result synchronization remains pending ({type(exc).__name__})")

    @app.post("/api/v1/test/jobs/{job_id}/stop")
    def stop_local_test(job_id: UUID, service: LocalMediaTestJobs = Depends(get_media_jobs)):
        try:
            return service.stop(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")
        except MediaRequestConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    def checkpoint_info(session: Session, model: ModelVersion) -> dict[str, Any] | None:
        artifact = session.get(Artifact, model.artifact_id) if model.artifact_id else None
        if artifact is None:
            return None
        provenance = (model.metrics or {}).get("provenance", {})
        return {"storage": artifact.storage_provider, "drive_file_id": artifact.remote_file_id,
                "drive_url": (f"https://drive.google.com/file/d/{artifact.remote_file_id}/view"
                              if artifact.remote_file_id else None),
                "archive_sha256": artifact.sha256, "size_bytes": artifact.size_bytes, "status": artifact.status,
                "weights_sha256": provenance.get("weights_sha256"),
                "note": "Download goes through the app, which checks the SHA-256 against this registry row."}

    @app.get("/api/v1/models/{model_version_id}/checkpoint")
    def model_checkpoint(model_version_id: UUID, session: Session = Depends(get_session)):
        model = session.get(ModelVersion, model_version_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Model version not found")
        info = checkpoint_info(session, model)
        return {"model_version_id": str(model.id), "display_version": model.display_version, "status": model.status,
                "checkpoint": info, "provenance": (model.metrics or {}).get("provenance")}

    @app.get("/api/v1/models")
    def list_models(request: Request, session: Session = Depends(get_session)):
        models = session.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(100)).all()
        cache = Path(request.app.state.settings.approved_model_bundle_dir).expanduser() if request.app.state.settings.approved_model_bundle_dir else None
        return {"models": [{"id": str(model.id), "model_family": model.model_family,
                            "display_version": model.display_version, "status": model.status,
                            "temporal_trained": model.temporal_trained, "model_kind": model.model_kind, "completion_kind": model.completion_kind,
                            "dataset_fingerprint": model.dataset_fingerprint, "metrics": model.metrics or {},
                            "checkpoint": checkpoint_info(session, model),
                            "cached_locally": bool(cache and (cache / str(model.id) / "manifest.json").is_file()),
                            "created_at": model.created_at.isoformat()} for model in models]}

    @app.post("/api/v1/models/{model_version_id}/sync")
    def sync_model(model_version_id: UUID, request: Request, session: Session = Depends(get_session)):
        from drowning.api.model_sync import sync_model_version
        try:
            if request.app.state.drive_factory is not None:
                drive = request.app.state.drive_factory()
            else:
                from drowning.integrations.drive import GoogleDriveArtifactAdapter, build_drive_service
                drive = GoogleDriveArtifactAdapter(build_drive_service(Path(app_settings.google_oauth_token_file)))
            return sync_model_version(session, app_settings, drive, model_version_id)
        except InferenceUnavailable as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model sync failed ({type(exc).__name__}); existing cache is preserved")

    @app.get("/api/v1/models/{model_version_id}/evaluation")
    def evaluation(model_version_id: UUID, request: Request, session: Session = Depends(get_session)):
        try:
            return model_evaluation(request.app.state.session_factory, model_version_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Model version not found")

    def marks_call(action: Callable[[], Any]):
        try:
            return action()
        except KeyError:
            raise HTTPException(status_code=404, detail="Job or mark not found")
        except MarkError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/api/v1/jobs/{job_id}/review-marks")
    def list_review_marks(job_id: UUID, request: Request, session: Session = Depends(get_session)):
        return marks_call(lambda: ReviewMarks(request.app.state.session_factory).list(job_id))

    @app.post("/api/v1/jobs/{job_id}/review-marks")
    def add_review_mark(job_id: UUID, body: ReviewMarkRequest, request: Request,
                        service: LocalMediaTestJobs = Depends(get_media_jobs)):
        return marks_call(lambda: ReviewMarks(request.app.state.session_factory).add(
            job_id, service.result_file, reviewer=body.reviewer, track_id=body.track_id, start_ms=body.start_ms,
            end_ms=body.end_ms, verdict=body.verdict, expected_label=body.expected_label, note=body.note))

    @app.delete("/api/v1/jobs/{job_id}/review-marks/{mark_id}")
    def delete_review_mark(job_id: UUID, mark_id: str, request: Request, session: Session = Depends(get_session)):
        return marks_call(lambda: ReviewMarks(request.app.state.session_factory).delete(job_id, mark_id))

    @app.post("/api/v1/models/sync-approved")
    def sync_approved(request: Request, session: Session = Depends(get_session)):
        from drowning.api.model_sync import sync_approved_model
        try:
            if request.app.state.drive_factory is not None:
                drive = request.app.state.drive_factory()
            else:
                from drowning.integrations.drive import GoogleDriveArtifactAdapter, build_drive_service
                drive = GoogleDriveArtifactAdapter(build_drive_service(Path(app_settings.google_oauth_token_file)))
            return sync_approved_model(session, app_settings, drive)
        except InferenceUnavailable as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Approved model sync failed ({type(exc).__name__}); existing cache is preserved")

    @app.post("/api/v1/models/{model_version_id}/review")
    def review_model(model_version_id: UUID, body: ReviewModelRequest, session: Session = Depends(get_session)):
        try:
            model = review_model_version(session, model_version_id, decision=body.decision,
                                         reviewer_id=body.reviewer_id, comment=body.comment)
        except KeyError:
            raise HTTPException(status_code=404, detail="Model version not found")
        except ModelReviewError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return {"id": str(model.id), "status": model.status, "decision": body.decision,
                "model_family": model.model_family}

    return app


def _find_source(directory: Path, source_id: str) -> dict[str, Any]:
    for path in directory.glob("*.yaml"):
        source = load_source(path)
        if source.get("source_id") == source_id:
            return source
    raise HTTPException(status_code=404, detail="Dataset source not found")


def _job_dict(job: Job) -> dict[str, Any]:
    payload = job.payload or {}
    return {"id": str(job.id), "job_type": job.job_type, "state": job.state,
            "mode": payload.get("mode") or payload.get("modality"), "executor": payload.get("executor"),
            "progress": payload.get("progress"),
            "frame_stride": payload.get("frame_stride"),
            "model_version_id": payload.get("model_version_id"), "evaluation": bool(payload.get("evaluation")),
            "phase": job.phase, "idempotency_key": job.idempotency_key,
            "owner_id": job.owner_id, "execution_token": job.execution_token,
            "remote_ref": job.remote_ref, "remote_version": job.remote_version,
            "remote_status": job.remote_status, "error_code": job.error_code,
            "error_summary": job.error_summary, "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat()}
