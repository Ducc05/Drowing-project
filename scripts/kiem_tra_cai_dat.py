"""Kiem tra mot may da cai dat xong chua. Khong in gia tri bi mat, khong tao job, khong chay Kaggle.

Chay tu thu muc goc du an:  .\\.venv\\Scripts\\python scripts\\kiem_tra_cai_dat.py
Cac buoc: (1) file va bien .env, (2) ket noi PostgreSQL/Drive/Kaggle, (3) test nhanh khong can mang,
(4) khoi dong API tam thoi o cong 8011 (tien trinh nen TAT) va goi thu cac API chinh.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8011
results: list[tuple[str, bool | None, str]] = []


def record(name: str, ok: bool | None, detail: str = "", hint: str = "") -> None:
    detail = detail or ("" if ok else hint)
    results.append((name, ok, detail))
    mark = {True: " OK ", False: "LOI ", None: "LUU Y"}[ok]
    print(f"[{mark}] {name}" + (f" - {detail}" if detail else ""))


def read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    path = ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def run(args: list[str], *, env: dict[str, str] | None = None, timeout: int = 300) -> tuple[int, str]:
    merged = {**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})}
    done = subprocess.run([sys.executable, *args], cwd=ROOT, env=merged, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    return done.returncode, (done.stdout + done.stderr).strip()


def get(path: str) -> tuple[int, object]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=60) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except (urllib.error.URLError, OSError, ValueError):
        return 0, None


def step_files() -> None:
    print("\n== Buoc 1. File va bien cau hinh ==")
    env = read_env()
    record("Co file .env", bool(env), hint="chay: copy .env.example .env roi dien (xem HUONG_DAN_BAN.md muc 4)")
    if not env:
        return
    database = env.get("DATABASE_URL_UNPOOLED") or env.get("DATABASE_URL")
    record("Co chuoi PostgreSQL (DATABASE_URL_UNPOOLED)", bool(database))
    record("Co GOOGLE_DRIVE_FOLDER_ID", bool(env.get("GOOGLE_DRIVE_FOLDER_ID")))
    kernel = env.get("KAGGLE_KERNEL_REF", "")
    kernel_ok = "/" in kernel and "<" not in kernel and ">" not in kernel
    record("KAGGLE_KERNEL_REF la <ten-kaggle-cua-ban>/drowning-temporal", kernel_ok,
           hint="thay phan <...> bang ten nguoi dung Kaggle cua ban")
    record("Co KAGGLE_DATASET_SOURCES", bool(env.get("KAGGLE_DATASET_SOURCES")))
    record("Co google-oauth-client.json", (ROOT / "secrets" / "google-oauth-client.json").is_file() or
           (ROOT / "secrets" / "google-token.json").is_file(), hint="dat vao thu muc secrets/")
    record("Da dang nhap Drive (secrets/google-token.json)", (ROOT / "secrets" / "google-token.json").is_file(),
           hint="chay: python -m drowning.cli auth drive")
    home = Path.home() / ".kaggle"
    record("Co token Kaggle", (home / "access_token").is_file() or (home / "kaggle.json").is_file() or
           bool(os.environ.get("KAGGLE_API_TOKEN")), hint="dat file access_token vao %USERPROFILE%\\.kaggle")
    detector = env.get("PERSON_DETECTOR_WEIGHTS", "")
    record("Co file detector nguoi", bool(detector) and (ROOT / detector).is_file() or Path(detector).is_file(),
           hint="chay: python -m drowning.cli detector fetch")
    record("Da build giao dien (web/dist)", (ROOT / "web" / "dist" / "index.html").is_file(),
           hint="chay: cd web; npm ci; npm run build")


def step_connections() -> None:
    print("\n== Buoc 2. Ket noi that (chi doc, khong tao job) ==")
    code, out = run(["-m", "drowning.cli", "db", "check"])
    record("PostgreSQL ket noi duoc va schema dung phien ban", code == 0, "" if code == 0 else out[-300:])
    code, out = run(["-m", "drowning.cli", "doctor", "--online"])
    record("doctor --online khong con muc THIEU bat buoc", code == 0 and "Thiếu bắt buộc: không" in out,
           "" if code == 0 else "xem chi tiet o duoi")
    if not (code == 0 and "Thiếu bắt buộc: không" in out):
        print(out)


def step_tests() -> None:
    print("\n== Buoc 3. Test nhanh (khong dung mang, khong cham database that) ==")
    code, out = run(["-m", "unittest", "tests.test_config", "tests.test_doctor", "tests.test_aggregation", "tests.test_tracking"],
                    env={"DATABASE_URL": "", "DATABASE_URL_UNPOOLED": ""}, timeout=600)
    tail = out.splitlines()[-1] if out else ""
    record("Test don vi co ban", code == 0, tail if code == 0 else out[-400:])


def step_api() -> None:
    print(f"\n== Buoc 4. Khoi dong API tam thoi o cong {PORT} (tien trinh nen TAT) ==")
    env = {**os.environ, "SUPERVISOR_ENABLED": "false", "PYTHONIOENCODING": "utf-8"}
    process = subprocess.Popen([sys.executable, "-m", "drowning.cli", "serve", "--port", str(PORT)], cwd=ROOT, env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        alive = False
        for _ in range(40):
            if process.poll() is not None:
                break
            if get("/health")[0] == 200:
                alive = True
                break
            time.sleep(1)
        record("API khoi dong va tra loi /health", alive, "" if alive else "serve khong chay duoc: thu chay tay de xem loi")
        if not alive:
            return
        status, body = get("/ready")
        record("API bao san sang (/ready)", status == 200, "" if status == 200 else f"ma {status}")
        status, body = get("/api/v1/train/preflight")
        if status == 200 and isinstance(body, dict):
            modes = body.get("modes", {})
            summary = ", ".join(f"{m}={'duoc' if v.get('allowed') else 'bi khoa'}" for m, v in modes.items())
            record("Trang Train doc duoc trang thai chung", True, summary)
        else:
            record("Trang Train doc duoc trang thai chung", False, f"ma {status}")
        status, body = get("/api/v1/models")
        count = len(body.get("models", [])) if status == 200 and isinstance(body, dict) else -1
        record("Doc duoc danh sach model chung", status == 200, f"{count} model" if status == 200 else f"ma {status}")
        status, body = get("/api/v1/jobs")
        count = len(body.get("jobs", [])) if status == 200 and isinstance(body, dict) else -1
        record("Doc duoc danh sach job chung", status == 200, f"{count} job" if status == 200 else f"ma {status}")
        page = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=30).read(2000) if alive else b""
        record("Giao dien web duoc phuc vu o /", b"<div id=\"root\"" in page or b"<!doctype html>" in page.lower(),
               "" if page else "chua build web/dist")
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    print(f"Kiem tra cai dat tai {ROOT}")
    for step in (step_files, step_connections, step_tests, step_api):
        try:
            step()
        except Exception as exc:  # a failed check must never hide the other checks
            record(step.__name__, False, f"{type(exc).__name__}: {str(exc)[:200]}")
    failed = [name for name, ok, _ in results if ok is False]
    print("\n== Tong ket ==")
    print(f"{sum(1 for _, ok, _ in results if ok)} dat, {len(failed)} loi, {sum(1 for _, ok, _ in results if ok is None)} luu y")
    for name in failed:
        print(f"  - CAN SUA: {name}")
    if not failed:
        print("Moi thu san sang. Chay: python -m drowning.cli serve   roi mo http://127.0.0.1:8000")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
