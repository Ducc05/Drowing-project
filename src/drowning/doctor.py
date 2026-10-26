"""`drownctl doctor`: what is configured, what is missing, and how to fix it. Never prints values."""

from __future__ import annotations

import importlib.util
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from drowning.config import database_url_from
from drowning.data.registry import SourceConfigError, load_source

_KERNEL_REF = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")
_DATASET_REF = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+$")
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass
class Check:
    id: str
    group: str
    ok: bool | None  # None = informational / not applicable
    required: bool
    message: str
    hint: str = ""


def _present(environ: Mapping[str, str], key: str) -> bool:
    return bool(environ.get(key, "").strip())


def run_doctor(environ: Mapping[str, str], *, project_root: Path, online: bool = False,
               which: Callable[[str], str | None] | None = None, home: Path | None = None,
               db_check: Callable[[str], bool] | None = None,
               drive_check: Callable[[], str | None] | None = None,
               kaggle_check: Callable[[str], str | None] | None = None) -> list[Check]:
    home = home or Path.home()
    if which is None:
        from drowning.integrations.kaggle import find_kaggle
        which = lambda _name: find_kaggle()
    checks: list[Check] = []

    def add(id_: str, group: str, ok: bool | None, required: bool, message: str, hint: str = "") -> None:
        checks.append(Check(id_, group, ok, required, message, hint))

    # -- PostgreSQL ------------------------------------------------------------------------
    url = database_url_from(environ)
    add("database_url", "PostgreSQL", bool(url), True, "DATABASE_URL đã điền" if url else "DATABASE_URL chưa điền",
        "Dạng: postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require (xem docs/SETUP.md mục 1)")
    if url:
        try:
            from sqlalchemy.engine import make_url
            parsed = make_url(url)
            is_pg = parsed.drivername.startswith("postgresql")
            add("database_is_postgresql", "PostgreSQL", is_pg, True,
                "Kiểu CSDL là PostgreSQL" if is_pg else "DATABASE_URL không phải PostgreSQL",
                "Registry và khóa compute chung bắt buộc dùng PostgreSQL, không dùng SQLite.")
            remote = (parsed.host or "").lower() not in _LOCAL_HOSTS
            has_ssl = "sslmode" in (parsed.query or {})
            add("database_ssl", "PostgreSQL", (has_ssl or not remote), False,
                ("CSDL ở máy khác và đã đặt sslmode" if remote and has_ssl else "CSDL chạy cục bộ" if not remote
                 else "CSDL ở máy khác nhưng chưa đặt sslmode"), "Thêm ?sslmode=require vào cuối DATABASE_URL.")
            test_url = environ.get("TEST_DATABASE_URL", "").strip()
            same = bool(test_url) and make_url(test_url).database == parsed.database and make_url(test_url).host == parsed.host
            add("test_database_separate", "PostgreSQL", bool(test_url) and not same, False,
                "TEST_DATABASE_URL tách riêng khỏi DATABASE_URL" if test_url and not same else
                "TEST_DATABASE_URL trùng hoặc chưa điền",
                "Test PostgreSQL thật chỉ chạy trên database riêng, không bao giờ trên database dùng chung.")
        except Exception:
            add("database_url_valid", "PostgreSQL", False, True, "DATABASE_URL không đọc được", "Kiểm tra cú pháp URL.")
        if online and db_check is not None:
            try:
                ready = db_check(url)
            except Exception as exc:
                ready = False
                add("database_online", "PostgreSQL", False, True, f"Không kết nối được ({type(exc).__name__})",
                    "Kiểm tra host/cổng/tường lửa/mật khẩu/sslmode.")
            else:
                add("database_online", "PostgreSQL", ready, True,
                    "Kết nối được và schema đã migrate" if ready else "Kết nối được nhưng schema chưa ở bản mới nhất",
                    "Chạy: drownctl db upgrade")
    add("worker_id", "PostgreSQL", None, False,
        "WORKER_ID tự sinh theo máy" if not _present(environ, "WORKER_ID") else "WORKER_ID đặt thủ công",
        "Mỗi máy phải có WORKER_ID khác nhau; để trống thì hệ thống tự sinh theo máy.")

    # -- Google Drive ----------------------------------------------------------------------
    add("drive_folder", "Google Drive", _present(environ, "GOOGLE_DRIVE_FOLDER_ID"), False,
        "GOOGLE_DRIVE_FOLDER_ID đã điền" if _present(environ, "GOOGLE_DRIVE_FOLDER_ID") else "GOOGLE_DRIVE_FOLDER_ID chưa điền",
        "ID thư mục chung trên Drive: phần cuối của URL drive.google.com/drive/folders/<ID>.")
    secret = Path(environ.get("GOOGLE_OAUTH_CLIENT_SECRET_FILE", "./secrets/google-oauth-client.json")).expanduser()
    token = Path(environ.get("GOOGLE_OAUTH_TOKEN_FILE", "./secrets/google-token.json")).expanduser()
    add("drive_client_secret", "Google Drive", secret.is_file() or token.is_file(), False,
        "Có file OAuth client hoặc token" if secret.is_file() or token.is_file() else "Chưa có file OAuth client",
        "Tạo OAuth client loại Desktop trong Google Cloud Console và lưu vào GOOGLE_OAUTH_CLIENT_SECRET_FILE.")
    add("drive_token", "Google Drive", token.is_file(), False,
        "Có token đăng nhập Drive" if token.is_file() else "Chưa có token Drive",
        "Chạy: drownctl auth drive (mở trình duyệt để cấp quyền một lần).")
    for module, label in (("googleapiclient", "google-api-python-client"), ("google_auth_oauthlib", "google-auth-oauthlib")):
        add(f"pkg_{module}", "Google Drive", importlib.util.find_spec(module) is not None, False,
            f"Đã cài {label}" if importlib.util.find_spec(module) else f"Chưa cài {label}",
            'Chạy: pip install -e ".[drive]"')
    if online and drive_check is not None and _present(environ, "GOOGLE_DRIVE_FOLDER_ID") and token.is_file():
        problem = drive_check()
        add("drive_online", "Google Drive", problem is None, False,
            "Truy cập được thư mục Drive" if problem is None else problem, "Kiểm tra quyền chia sẻ thư mục cho tài khoản đã đăng nhập.")

    # -- Kaggle ----------------------------------------------------------------------------
    cli = bool(which("kaggle"))
    add("kaggle_cli", "Kaggle", cli, True, "Đã cài Kaggle CLI" if cli else "Chưa cài Kaggle CLI", "Chạy: pip install kaggle")
    credentials = ((_present(environ, "KAGGLE_USERNAME") and _present(environ, "KAGGLE_KEY")) or _present(environ, "KAGGLE_API_TOKEN")
                   or any(path.is_file() for path in (home / ".kaggle" / "kaggle.json", home / ".config" / "kaggle" / "kaggle.json",
                                                      home / ".kaggle" / "access_token")))
    add("kaggle_credentials", "Kaggle", credentials, True, "Có thông tin đăng nhập Kaggle" if credentials else "Chưa có thông tin đăng nhập Kaggle",
        "Kaggle → Settings → API → tạo token; dán token vào file %USERPROFILE%\.kaggle\access_token (hoặc kaggle.json nếu là kiểu cũ). Không đưa vào dự án.")
    ref = environ.get("KAGGLE_KERNEL_REF", "").strip()
    add("kaggle_kernel_ref", "Kaggle", bool(_KERNEL_REF.fullmatch(ref)), True,
        "KAGGLE_KERNEL_REF hợp lệ" if _KERNEL_REF.fullmatch(ref) else "KAGGLE_KERNEL_REF chưa điền hoặc sai dạng",
        "Dạng owner/slug, ví dụ ten-kaggle/drowning-temporal (notebook sẽ được tạo ở chế độ private).")
    sources = [s.strip() for s in environ.get("KAGGLE_DATASET_SOURCES", "").split(",") if s.strip()]
    sources_ok = bool(sources) and all(_DATASET_REF.fullmatch(s) for s in sources)
    add("kaggle_datasets", "Kaggle", sources_ok, True,
        f"{len(sources)} dataset được gắn vào notebook" if sources_ok else "KAGGLE_DATASET_SOURCES chưa điền hoặc sai dạng",
        "Tải dataset lên Kaggle (private) rồi liệt kê owner/slug, cách nhau bằng dấu phẩy.")
    private = environ.get("KAGGLE_NOTEBOOK_PRIVATE", "true").strip().lower() == "true"
    add("kaggle_private", "Kaggle", private, True, "Notebook ở chế độ private" if private else "KAGGLE_NOTEBOOK_PRIVATE phải là true",
        "Hệ thống từ chối push notebook công khai.")
    if online and kaggle_check is not None and cli and credentials:
        for dataset in sources:
            problem = kaggle_check(dataset)
            add(f"kaggle_dataset_online:{dataset}", "Kaggle", problem is None, False,
                f"Truy cập được dataset {dataset}" if problem is None else problem, "Kiểm tra tên dataset và quyền truy cập.")

    # -- dữ liệu & model ---------------------------------------------------------------------
    datasets_dir = project_root / "configs" / "datasets"
    for path in sorted(datasets_dir.glob("*.yaml")) if datasets_dir.is_dir() else []:
        try:
            source = load_source(path)
        except SourceConfigError:
            add(f"source:{path.stem}", "Dữ liệu", False, False, f"{path.name} không đọc được")
            continue
        key = source.get("paths", {}).get("root_env", "")
        root = environ.get(key, "").strip()
        ok = bool(root) and Path(root).expanduser().is_dir()
        add(f"source_root:{source.get('source_id')}", "Dữ liệu", ok, False,
            f"{key} trỏ tới thư mục tồn tại" if ok else f"{key} chưa điền hoặc thư mục không tồn tại",
            "Chỉ dùng để duyệt/convert tại máy; dữ liệu gốc luôn ở chế độ chỉ đọc.")
    weights = environ.get("PERSON_DETECTOR_WEIGHTS", "").strip()
    weights_ok = bool(weights) and Path(weights).expanduser().is_file()
    add("detector_weights", "Model", weights_ok, False, "Có file trọng số detector người" if weights_ok else "PERSON_DETECTOR_WEIGHTS chưa điền hoặc không có file",
        "Cần một file trọng số detector (ví dụ YOLO .pt) có lớp 'person'; hệ thống không tự tải.")
    add("pkg_ultralytics", "Model", importlib.util.find_spec("ultralytics") is not None, False,
        "Đã cài ultralytics" if importlib.util.find_spec("ultralytics") else "Chưa cài ultralytics", 'Chạy: pip install -e ".[detector]"')
    for module, label in (("torch", "PyTorch"), ("cv2", "OpenCV")):
        add(f"pkg_{module}", "Model", importlib.util.find_spec(module) is not None, True,
            f"Đã cài {label}" if importlib.util.find_spec(module) else f"Chưa cài {label}",
            'Chạy: pip install -e ".[ml,vision]"')
    if environ.get("ALLOW_LOCAL_TRAIN", "false").strip().lower() == "true":
        add("local_train_disabled", "An toàn", False, True, "ALLOW_LOCAL_TRAIN=true bị từ chối", "Train thật chỉ chạy trên Kaggle.")
    return checks


def summarize(checks: list[Check]) -> dict[str, Any]:
    missing_required = [c.id for c in checks if c.required and c.ok is False]
    return {"ready_for_kaggle_runs": not [c for c in checks if c.required and c.ok is False and c.group in {"Kaggle", "PostgreSQL"}],
            "required_missing": missing_required, "optional_missing": [c.id for c in checks if not c.required and c.ok is False],
            "checks": [asdict(c) for c in checks]}
