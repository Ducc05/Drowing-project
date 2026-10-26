"""Environment loading and safe, non-secret configuration status."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


def load_env_file(path: Path, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Load simple KEY=VALUE lines; process environment always takes precedence."""
    target = os.environ if environ is None else environ
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env line {line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"Invalid .env key on line {line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        loaded[key] = value
        if key not in target:
            target[key] = value
    return loaded


_DATASET_REF = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+$")


def _dataset_sources(raw: str) -> tuple[str, ...]:
    refs = tuple(item.strip() for item in raw.split(",") if item.strip())
    for ref in refs:
        if not _DATASET_REF.fullmatch(ref):
            raise ValueError("KAGGLE_DATASET_SOURCES must be comma-separated owner/dataset-slug references")
    return refs


def _positive_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 5:
        raise ValueError(f"{name} must be at least 5")
    return value


def database_url_from(env) -> str:
    """The app's PostgreSQL URL. A direct (unpooled) URL wins when both exist: Neon's pooled URL sits behind
    PgBouncer, which does not suit migrations or session-level behavior, and Neon provides both."""
    return (env.get("DATABASE_URL_UNPOOLED", "").strip() or env.get("DATABASE_URL", "").strip())


@dataclass(frozen=True)
class Settings:
    app_env: str
    api_host: str
    api_port: int
    database_url: str
    drive_folder_id: str
    kaggle_kernel_ref: str
    kaggle_private: bool
    train_executor: str
    allow_local_train: bool
    dataset_root_configured: bool
    model_bundle_configured: bool
    model_bundle_manifest_present: bool
    detector_weights_configured: bool
    approved_model_bundle_dir: str
    person_detector_weights_path: str
    detector_class_id: int
    model_device: str
    model_family: str
    worker_id: str
    input_cache_dir: str
    output_cache_dir: str
    google_oauth_token_file: str
    kaggle_kernel_dir: str = "./notebooks/kernels"
    kaggle_dataset_sources: tuple[str, ...] = ()
    supervisor_enabled: bool = True
    supervisor_interval_seconds: int = 30

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        try:
            port = int(env.get("API_PORT", "8000"))
        except ValueError as exc:
            raise ValueError("API_PORT must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ValueError("API_PORT must be between 1 and 65535")
        private = env.get("KAGGLE_NOTEBOOK_PRIVATE", "true").strip().lower()
        local_train = env.get("ALLOW_LOCAL_TRAIN", "false").strip().lower()
        if private not in {"true", "false"}:
            raise ValueError("KAGGLE_NOTEBOOK_PRIVATE must be true or false")
        if local_train not in {"true", "false"}:
            raise ValueError("ALLOW_LOCAL_TRAIN must be true or false")
        executor = env.get("TRAIN_DEFAULT_EXECUTOR", "kaggle").strip().lower()
        if executor != "kaggle":
            raise ValueError("TRAIN_DEFAULT_EXECUTOR must be kaggle")
        if local_train == "true":
            raise ValueError("Local training is disabled by this implementation")
        bundle_dir_value = env.get("APPROVED_MODEL_BUNDLE_DIR", "").strip()
        bundle_dir = Path(bundle_dir_value).expanduser() if bundle_dir_value else None
        try:
            detector_class_id = int(env.get("PERSON_DETECTOR_CLASS_ID", "0"))
        except ValueError as exc:
            raise ValueError("PERSON_DETECTOR_CLASS_ID must be an integer") from exc
        device = env.get("MODEL_DEVICE", "cpu").strip().lower()
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError("MODEL_DEVICE must be cpu or a CUDA device")
        return cls(
            app_env=env.get("APP_ENV", "development"),
            api_host=env.get("API_HOST", "127.0.0.1"),
            api_port=port,
            database_url=database_url_from(env),
            drive_folder_id=env.get("GOOGLE_DRIVE_FOLDER_ID", ""),
            kaggle_kernel_ref=env.get("KAGGLE_KERNEL_REF", ""),
            kaggle_private=private == "true",
            train_executor=executor,
            allow_local_train=False,
            dataset_root_configured=bool(env.get("ALANOUD_V4_DATA_ROOT", "").strip()),
            model_bundle_configured=bundle_dir is not None,
            model_bundle_manifest_present=bool(bundle_dir and (bundle_dir / "manifest.json").is_file()),
            detector_weights_configured=bool(env.get("PERSON_DETECTOR_WEIGHTS", "").strip()),
            approved_model_bundle_dir=str(bundle_dir) if bundle_dir else "",
            person_detector_weights_path=env.get("PERSON_DETECTOR_WEIGHTS", "").strip(),
            detector_class_id=detector_class_id,
            model_device=device,
            model_family=env.get("MODEL_FAMILY", "drowning-person-temporal-v1").strip(),
            worker_id=env.get("WORKER_ID", "local-worker").strip() or "local-worker",
            input_cache_dir=env.get("INPUT_CACHE_DIR", "./local-cache/inputs").strip(),
            output_cache_dir=env.get("OUTPUT_CACHE_DIR", "./local-cache/results").strip(),
            google_oauth_token_file=env.get("GOOGLE_OAUTH_TOKEN_FILE", "./secrets/google-token.json").strip(),
            kaggle_kernel_dir=env.get("KAGGLE_KERNEL_DIR", "./notebooks/kernels").strip() or "./notebooks/kernels",
            kaggle_dataset_sources=_dataset_sources(env.get("KAGGLE_DATASET_SOURCES", "")),
            supervisor_enabled=env.get("SUPERVISOR_ENABLED", "true").strip().lower() != "false",
            supervisor_interval_seconds=_positive_int(env.get("SUPERVISOR_INTERVAL_SECONDS", "30"), "SUPERVISOR_INTERVAL_SECONDS"),
        )

    def status(self) -> dict[str, object]:
        """Expose configured flags only; never include connection strings or IDs."""
        return {
            "app_env": self.app_env,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "database_configured": bool(self.database_url),
            "drive_configured": bool(self.drive_folder_id),
            "kaggle_kernel_configured": bool(self.kaggle_kernel_ref),
            "kaggle_private": self.kaggle_private,
            "train_executor": self.train_executor,
            "local_training_enabled": False,
            "dataset_root_configured": self.dataset_root_configured,
            "model_bundle_configured": self.model_bundle_configured,
            "model_bundle_manifest_present": self.model_bundle_manifest_present,
            "person_detector_configured": self.detector_weights_configured,
            "kaggle_datasets_configured": bool(self.kaggle_dataset_sources),
            "supervisor_enabled": self.supervisor_enabled,
        }
