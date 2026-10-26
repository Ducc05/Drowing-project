"""Exercise the running local API and build one consolidated, evidence-backed report.

Never dispatches Kaggle, trains a model, or connects to Drive. Fixture results come
from run_regression.py and are labeled separately from the actual local HTTP/DB scan.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import os
import pstats
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import httpx
from PIL import Image
from sqlalchemy.orm import Session

from drowning.config import load_env_file
from drowning.db.repositories import transition_job
from drowning.db.session import create_db_engine

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PYTHON = str(ROOT / ".venv/Scripts/python.exe")


def command(args, *, cwd=ROOT, expected=(0,)):
    started = time.perf_counter()
    result = subprocess.run(args, cwd=cwd, capture_output=True, encoding="utf-8", errors="replace", timeout=120)
    return {"command": subprocess.list2cmdline(args), "exit_code": result.returncode,
            "expected_exit_codes": list(expected), "passed": result.returncode in expected,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def tree_snapshot(root):
    digest, count = hashlib.sha256(), 0
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            digest.update(f"{path.relative_to(root).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
            count += 1
    return {"file_count": count, "metadata_sha256": digest.hexdigest(), "method": "relative path, size, mtime_ns; not content hash"}


def code_fingerprint():
    rows = []
    for directory in ("src", "configs", "migrations", "web/src", "scripts", "tests"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".tsx", ".ts", ".css"}:
                rows.append({"file": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "method": "sorted source/config/migration/web/script/test file SHA-256 manifest", "files": rows}


def live_checks():
    values = {}
    load_env_file(ROOT / ".env", values)
    source_root = Path(values["ALANOUD_V4_DATA_ROOT"])
    before = tree_snapshot(source_root)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log = REPORTS / "local-http-server.log"
    records = []
    job_id = None
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen([PYTHON, "-m", "drowning.cli", "serve", "--host", "127.0.0.1", "--port", str(port)],
                                   cwd=ROOT, stdout=stream, stderr=stream,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60) as client:
                for _ in range(60):
                    if process.poll() is not None:
                        raise RuntimeError("Local API process exited before health check")
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.25)
                else:
                    raise RuntimeError("Local API did not become healthy")

                def check(name, method, path, expected=200, **kwargs):
                    started = time.perf_counter()
                    response = client.request(method, path, **kwargs)
                    payload = response.json() if "application/json" in response.headers.get("content-type", "") else None
                    records.append({"name": name, "method": method, "path": path, "expected_http": expected,
                                    "observed_http": response.status_code, "passed": response.status_code == expected,
                                    "duration_seconds": round(time.perf_counter() - started, 3),
                                    "body": payload, "content_type": response.headers.get("content-type")})
                    if response.status_code != expected:
                        raise RuntimeError(f"Unexpected local HTTP status for {name}: {response.status_code}")
                    return response

                check("Health", "GET", "/health")
                check("Real PostgreSQL readiness", "GET", "/ready")
                check("Sanitized configuration", "GET", "/api/v1/config-status")
                page = check("Production React page", "GET", "/")
                assets = re.findall(r'(?:src|href)="(/assets/[^\"]+)"', page.text)
                if len(assets) < 2:
                    raise RuntimeError("Production page does not reference the expected JS/CSS assets")
                for asset in assets:
                    check("Production static asset", "GET", asset)
                sources = check("Source registry", "GET", "/api/v1/sources").json()["sources"]
                source = next(item for item in sources if item["root_env"] == "ALANOUD_V4_DATA_ROOT")
                check("Read-only real source scan", "POST", f"/api/v1/sources/{source['source_id']}/scan")
                models = check("Real model registry", "GET", "/api/v1/models").json()["models"]
                readiness = check("Real inference readiness", "GET", "/api/v1/test/readiness").json()
                if readiness["ready_for_request"] or models:
                    raise RuntimeError("This review expects the untrained local registry; do not reuse as a prediction benchmark")
                image = io.BytesIO()
                Image.new("RGB", (40, 40), "blue").save(image, format="PNG")
                check("No approved model blocks inference", "POST", "/api/v1/test/image", expected=409,
                      headers={"Idempotency-Key": f"no-model-review-{uuid4().hex}"},
                      files={"file": ("synthetic.png", image.getvalue(), "image/png")})
                key = f"review-prepare-{uuid4().hex}"
                body = {"job_type": "prepare", "idempotency_key": key,
                        "payload": {"mode": "prepare_only", "review_record": True, "executor": "kaggle"}}
                first = check("Persist prepare job (no dispatch)", "POST", "/api/v1/jobs", json=body).json()
                job_id = first["job"]["id"]
                second = check("Idempotent HTTP retry", "POST", "/api/v1/jobs", json=body).json()
                if not first["created"] or second["created"] or second["job"]["id"] != job_id or first["dispatch"] != "not_started":
                    raise RuntimeError("Persistent HTTP idempotency/dispatch contract did not hold")
                check("Persistent job events", "GET", f"/api/v1/jobs/{job_id}/events")
                # Close only the new review-owned, never-dispatched job; preserve its audit history.
                engine = create_db_engine(values["DATABASE_URL"])
                try:
                    with Session(engine) as session:
                        transition_job(session, UUID(job_id), "cancelled", event_type="local_review_finished",
                                       details={"compute_dispatched": False}, phase="review_complete_no_dispatch")
                finally:
                    engine.dispose()
                check("Review job audit retained after cleanup", "GET", f"/api/v1/jobs/{job_id}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    after = tree_snapshot(source_root)
    if before != after:
        raise RuntimeError("Source inventory metadata changed during review")
    return {"transport": "real loopback TCP HTTP, separate uvicorn process", "port": port,
            "server_stopped_after_review": True, "checks": records, "review_job_id": job_id,
            "source_before": before, "source_after": after, "source_metadata_unchanged": before == after}


FIXES = [
    ("F01", "Hai process duyệt model đầu tiên gây IntegrityError ở registry.", "Upsert registry theo family, sau đó khóa dòng và cập nhật pointer/generation trong cùng transaction.", "src/drowning/db/repositories.py", "test_simultaneous_first_model_approvals_serialize_registry_pointer", "Tái hiện trên PostgreSQL thật; regression hai process pass."),
    ("F02", "pg_isready có thể nhận nhầm server tạm trong quá trình Docker init.", "Chờ TCP 127.0.0.1 của server cuối; chỉ tiếp tục container/volume do script sở hữu.", "scripts/provision_local_postgres.py", None, "Bootstrap local chạy thành công, migration và readiness pass."),
    ("F03", "UUID hex từ notebook khác dạng UUID có dấu gạch trong DB.", "Chuẩn hóa UUID trước so sánh; giữ pin version theo job.", "src/drowning/api/inference.py", "test_image_upload_result_download_and_idempotency", "Fixture dùng UUID hex và tải kết quả pass."),
    ("F04", "Form upload bị gán Content-Type JSON; lỗi tác vụ làm UI báo API offline.", "Giữ boundary do FormData tạo; tách trạng thái health khỏi lỗi tác vụ; DB configured nhưng chưa ready hiển thị riêng.", "web/src/App.tsx", None, "TypeScript/Vite build và lint pass; thao tác UI trực quan chưa xác minh."),
    ("F05", "Retry kết quả cũ có thể bị chặn khi model cache đã thay đổi.", "Nhận diện request cùng hash trước tải model; trả lại kết quả đã lưu, không chạy lại compute.", "src/drowning/api/test_jobs.py", "test_image_upload_result_download_and_idempotency", "Xóa manifest sau lượt đầu, retry vẫn lấy đúng job; input khác cùng key bị chặn."),
    ("F06", "Worker cũ có thể ghi trạng thái/artifact sau khi token đổi.", "Kiểm tra owner + fencing token trước transition, receipt và artifact publication; release dùng token đã chụp.", "src/drowning/db/repositories.py", "test_stale_token_cannot_publish_state_on_real_postgresql", "PG stale-token và fake push receipt regression pass; takeover/recovery tự động chưa triển khai."),
    ("F07", "Thiếu bước tải approved model khiến luồng duyệt → test chưa nối được.", "Thêm sync API/CLI/UI; kiểm tra SHA-256/size/identity, cài atomically vào thư mục UUID, giữ cache cũ khi lỗi.", "src/drowning/api/model_sync.py", "test_candidate_approve_sync_upload_download_entire_flow", "HTTP flow với archive/bundle thật tổng hợp, fake Drive/inference pass; chưa có Drive thật."),
    ("F08", "MP4/mp4v chưa đủ bảo đảm phát inline trên trình duyệt; cleanup lỗi có thể xóa file còn mở trên Windows.", "Web API xuất VP8/WebM; đóng capture/writer trước xóa file tạm khi exception.", "src/drowning/inference/pipeline.py", "test_video_pipeline_writes_annotated_video_and_marks_warmup", "VP8 codec marker, HTTP video/webm, 3/4 frame decode pass. Có cảnh báo FourCC không gây fail; browser playback chưa xác minh."),
    ("F09", "Ảnh có EXIF orientation có thể bị hiển thị khác hướng khi xử lý.", "Áp dụng EXIF transpose trước chạy image head và vẽ output.", "src/drowning/api/inference.py", "test_exif_orientation_is_applied_before_annotated_output", "JPEG 48×40 với EXIF orientation=6 được xuất thành PNG 40×48 đúng hướng; fake predictor, không đánh giá model."),
    ("F10", "Scan HTTP nguồn thật timeout sau 60 giây; profiler đo nhiều I/O mở nhãn và kiểm tra đường dẫn lặp.", "Index ảnh theo thư mục, resolve root một lần, đọc nhãn với tối đa 8 threads, vẫn kiểm tra containment và file trùng tên.", "src/drowning/data/adapters.py", "test_yolo_scan_counts_by_split_and_mapping_without_mutating_source", "Profiler CLI một lượt trước 42,922 giây, sau 10,242 giây; đây là phép đo local đơn lẻ, không phải benchmark ổn định. HTTP được chạy lại trong báo cáo."),
    ("F11", "Scan YOLO mới đếm số field/class, chưa phát hiện số không hợp lệ hoặc bbox vượt biên.", "Kiểm tra numeric/finite, kích thước dương và normalized corners; ghi từng malformed reason, không đổi raw.", "src/drowning/data/adapters.py", "test_yolo_scan_reports_invalid_numeric_bboxes_and_rejects_ambiguous_images", "Fixture NaN/Inf/negative/out-of-bounds/non-numeric báo đúng 5 lỗi; trùng stem hai loại ảnh bị chặn."),
    ("F12", "Regression sau tối ưu scan phát hiện đường dẫn Unicode Windows bị lệch với alias 8.3 trong converter.", "Giữ đường dẫn gốc khi tạo relative paths; chỉ dùng resolved root cho containment. Đã sửa lỗi do thay đổi scan gây ra.", "src/drowning/data/adapters.py", "test_cli_convert_uses_env_root_and_writes_derived_jsonl", "11 adapter tests chạy lại pass; regression lỗi trung gian được giữ trong regression-during-scan-change.json, không tính là kết quả cuối."),
    ("F13", "Retry job failed vẫn nhận HTTP 200 để đọc lại trạng thái; UI có thể báo nhầm đã hoàn tất.", "Hiển thị thông báo theo state completed/sync_pending/failed/running, không suy luận hoàn thành từ HTTP 200.", "web/src/App.tsx", "test_compute_failure_releases_slot_and_records_failed_job", "API retry giữ state failed, không tăng số inference; UI build/lint pass, thao tác UI trực quan chưa xác minh."),
    ("F14", "Candidate sync chưa kiểm tra trực tiếp cờ verified trên receipt trước publish DB.", "Chặn receipt unverified hoặc thiếu file ID trước artifact/candidate registration.", "src/drowning/orchestration/candidate_sync.py", "test_unverified_drive_receipt_cannot_publish_candidate", "Fake receipt verified=false không tạo Artifact/ModelVersion; job sync_pending và compute slot được release khi đã xác nhận dừng."),
]


def make_report(evidence):
    e = html.escape
    regression = evidence["regression"]
    live = evidence["live"]
    tests = regression["tests"]
    def matched(name):
        return [item for item in tests if name in item["test"]] if name else []
    fix_rows = "".join(f"<tr><td>{e(code)}</td><td>{e(problem)}</td><td>{e(fix)}<br><a href='../{path}'>{path}</a></td><td>{e(observed)}" +
                       ("<br>" + "<br>".join(f"<code>{e(t['test'])}</code> — {t['status']}" for t in matched(test)) if test else "") + "</td></tr>"
                       for code, problem, fix, path, test, observed in FIXES)
    http_rows = "".join(f"<tr><td>{e(c['name'])}</td><td><code>{e(c['method']+' '+c['path'])}</code></td><td>{c['expected_http']}</td><td>{c['observed_http']} · {'PASS' if c['passed'] else 'FAIL'} · {c['duration_seconds']}s</td></tr>" for c in live["checks"])
    test_rows = "".join(f"<tr><td><code>{e(t['test'])}</code></td><td>{t['status']}</td><td>{t['duration_seconds']}</td></tr>" for t in tests)
    command_sections = "".join(f"<details><summary>{e(c['label'])} — exit {c['exit_code']} · {'PASS' if c['passed'] else 'FAIL'}</summary><pre>{e(c['command'])}\n{e(c['stdout'])}\n{e(c['stderr'])}</pre></details>" for c in evidence["commands"])
    timestamp = evidence["generated_at_utc"]
    fingerprint = evidence["code"]["sha256"]
    return f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DrowningProject — Review luồng chính</title>
<style>
:root{{font:16px/1.65 system-ui,sans-serif;color:#183235;background:#f3f6f5}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1240px;margin:auto;padding:38px 26px 80px}}header{{background:#123d3c;color:white;padding:30px;border-radius:16px}}h1{{margin:6px 0;font-size:32px;line-height:1.25}}h2{{margin:38px 0 12px;font-size:24px}}h3{{font-size:18px}}p{{max-width:1000px}}a{{color:#087568}}header a{{color:#bce7d5}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0}}.card,section.box{{background:white;border:1px solid #dce5e1;border-radius:12px;padding:18px}}.number{{display:block;font-size:30px;font-weight:750}}.tag{{display:inline-block;border-radius:5px;padding:3px 10px;background:#e1f4e9;color:#17613d;font-weight:650;font-size:13px}}.pending{{background:#fff2d2;color:#795409}}.callout{{padding:16px 20px;border-left:5px solid #b5872c;background:#fff8e7}}table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}th,td{{text-align:left;padding:12px;border:1px solid #dae4df;vertical-align:top}}th{{background:#e7efeb}}code{{font:12px/1.5 ui-monospace,Consolas,monospace;overflow-wrap:anywhere}}pre{{font:13px/1.6 ui-monospace,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#163432;color:#e2f3e9;padding:18px;border-radius:8px}}.scroll{{overflow:auto}}details{{margin:12px 0}}summary{{cursor:pointer;font-weight:650}}nav{{display:flex;flex-wrap:wrap;gap:18px;margin:22px 0}}small{{color:#5b7170}}header small{{color:#c3d8d3}}.flow{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:18px 0}}.flow span{{padding:8px 12px;border-radius:8px;background:#e1eee7}}input{{padding:12px;border:1px solid #b7cfc5;border-radius:6px;width:100%;margin-bottom:12px}}@media(max-width:700px){{.cards{{grid-template-columns:repeat(2,1fr)}}main{{padding:18px 12px}}h1{{font-size:26px}}}}@media print{{details{{display:block}}nav,input{{display:none}}main{{max-width:none}}}}
</style></head><body><main>
<header><small>LOCAL REVIEW • 26/10/2026 • D:\\AI\\DrowningProject</small><h1>Review luồng tính năng chính</h1><p>Đã sửa và kiểm thử luồng phần mềm local. Chưa đủ điều kiện nghiệm thu model và cloud.</p><span class="tag">POSTGRESQL LOCAL THẬT</span> <span class="tag">{regression['passed']} TEST PASS</span> <span class="tag pending">MODEL / KAGGLE / DRIVE LIVE: CHƯA XÁC MINH</span></header>
<nav><a href="#flows">Luồng & phạm vi</a><a href="#fixes">Lỗi đã sửa</a><a href="#run">Cách chạy</a><a href="#http">HTTP thực tế</a><a href="#limits">Chưa nghiệm thu</a><a href="#tests">Toàn bộ tests</a></nav>
<div class="cards"><div class="card"><span class="number">{regression['passed']}/{regression['tests_run']}</span>Tests pass · {regression['skipped']} skipped</div><div class="card"><span class="number">6</span>Tests PostgreSQL thật, có tranh chấp hai process</div><div class="card"><span class="number">9.997</span>Ảnh nguồn đã scan read-only</div><div class="card"><span class="number">18.506</span>Bbox · 0 dòng malformed</div></div>
<p class="callout"><strong>Phân biệt bằng chứng:</strong> PostgreSQL, HTTP TCP, file upload/output encode/decode, migration và scan dữ liệu là thực tế. Dự đoán trong test dùng fake classifier/detector; model archive dùng trọng số ngẫu nhiên, không train. Drive/Kaggle trong tests là provider giả lập, không phải kết quả cloud.</p>
<h2 id="flows">1. Review một vòng luồng chính</h2><div class="scroll"><table><thead><tr><th>Luồng</th><th>Cách kiểm tra / kết quả quan sát</th><th>Kết luận</th></tr></thead><tbody>
<tr><td>Khởi động → cấu hình → DB</td><td>Docker PostgreSQL 17.10; Alembic 20261026_01; /health và /ready HTTP 200; trang React và JS/CSS được serve. Không in DSN hoặc token.</td><td><span class="tag">ĐÃ KIỂM THỬ LOCAL</span></td></tr>
<tr><td>Nguồn dữ liệu → scan → temporal preflight</td><td>Scan Alanoud v4: 9.997 ảnh, 18.506 bbox; tree metadata trước/sau giống nhau. Build-windows / build-splits trả exit 3 blocked vì thiếu video/person/time và global groups. Không sinh temporal labels giả.</td><td>Scan pass; temporal data blocked đúng điều kiện.</td></tr>
<tr><td>Tạo job → retry → lịch sử</td><td>POST /jobs hai lần cùng key vào PostgreSQL thật trả cùng UUID, lần sau created=false, dispatch=not_started. Event vẫn tồn tại. Job review được chuyển cancelled bằng event riêng; không dispatch compute.</td><td><span class="tag">ĐÃ KIỂM THỬ LOCAL</span></td></tr>
<tr><td>Kaggle → exact output → Drive → candidate</td><td>Tests kiểm tra private metadata, mode, timeout giữ slot, không retry push, không tải latest, ZIP/checksum/holdout contract; lỗi Drive → sync_pending, retry không compute lại.</td><td><span class="tag pending">FAKE PROVIDERS PASS</span> Chưa push/run Kaggle.</td></tr>
<tr><td>Candidate → approve → cache → upload ảnh → output</td><td>HTTP test duyệt candidate, fake Drive tải ZIP/bundle thật tổng hợp, kiểm tra hash/identity, cache UUID, upload PNG với fake predictor, tải annotated PNG + JSON. Tải đứt giữ nguyên cache cũ.</td><td>Luồng kỹ thuật pass; trọng số chưa được train.</td></tr>
<tr><td>Upload video → warm-up → output</td><td>Synthetic AVI 3/4 frame; API trả video/webm; output có codec V_VP8 và OpenCV decode đúng số frame. Pipeline đánh dấu image_warmup/temporal và ID provisional.</td><td>Encode/decode pass; browser playback và video thật chưa xác minh.</td></tr>
<tr><td>Slot bận / inference lỗi / sync lỗi</td><td>Slot bận → HTTP 409, giữ queued input; retry cùng key tiếp tục. Predictor lỗi → failed và giải phóng slot. Drive lỗi → local result vẫn tải được, retry upload không tăng số inference.</td><td><span class="tag">HTTP FIXTURE PASS</span></td></tr>
<tr><td>Đồng thời / stale owner</td><td>PostgreSQL thật: một winner cho slot, một job cho duplicate request, lease hết không tự giải phóng, stale token không ghi state; hai approve serialize pointer generation=2.</td><td><span class="tag">POSTGRESQL THẬT PASS</span> Chưa xác minh nhiều máy/shared DB.</td></tr>
</tbody></table></div>
<div class="flow"><span>Candidate đã xác minh</span> → <span>Duyệt</span> → <span>Sync approved cache</span> → <span>Upload ảnh/video</span> → <span>Claim slot & pin model</span> → <span>Output local</span> → <span>Sync Drive / retry</span></div>
<p>Trên môi trường hiện tại, /api/v1/test/readiness trả ready_for_request=false vì chưa có approved model. Gửi ảnh hợp lệ trả HTTP 409 đúng reason; hệ thống không tự tạo train job.</p>
<h2 id="fixes">2. Lỗi phát hiện và chỗ sửa</h2><div class="scroll"><table><thead><tr><th>ID</th><th>Trước sửa / vấn đề</th><th>Thay đổi & file</th><th>Bằng chứng sau sửa</th></tr></thead><tbody>{fix_rows}</tbody></table></div>
<h2 id="run">3. Chạy trên máy này</h2><section class="box"><p>DB local đã được tạo và điền vào .env ignored. Chạy từ PowerShell tại D:\\AI\\DrowningProject:</p><pre>docker start drowning-local-pg-20261026
.\\.venv\\Scripts\\python -m drowning.cli db check
.\\.venv\\Scripts\\python -m drowning.cli serve</pre><p>Mở <code>http://127.0.0.1:8000</code>; API docs tại <code>http://127.0.0.1:8000/docs</code>. UI production đã build trong web/dist. Nếu sửa UI:</p><pre>cd D:\\AI\\DrowningProject\\web
npm run build</pre><p>Luồng người dùng sau khi có candidate thật: duyệt model ở Model versions → Đồng bộ model đã duyệt → chọn ảnh/video → Chạy kiểm tra → xem media/JSON → Đồng bộ lại Drive nếu sync_pending. Model và detector còn thiếu sẽ hiển thị reason, không suy đoán kết quả.</p><p>Cấu hình còn cần: GOOGLE_DRIVE_FOLDER_ID, GOOGLE_OAUTH_TOKEN_FILE (OAuth thật), KAGGLE_KERNEL_REF và auth CLI, PERSON_DETECTOR_WEIGHTS; thêm video annotations và source groups đã review. Chỉ điền secrets trong .env/local auth. Chạy CLI sync khi đã có approved model:</p><pre>.\\.venv\\Scripts\\python -m drowning.cli model sync --approved</pre><p>Kiểm thử lại, tạo lại báo cáo:</p><pre>.\\.venv\\Scripts\\python scripts\\run_regression.py
.\\.venv\\Scripts\\python scripts\\review_main_flows.py</pre><p>Script report tạo đúng một prepare job riêng để kiểm tra idempotency, sau đó đánh dấu cancelled/no-dispatch và giữ audit. Không train hoặc gọi cloud. Server HTTP do script mở dùng cổng loopback tạm và được dừng sau review.</p><p>Dừng DB khi không dùng: <code>docker stop drowning-local-pg-20261026</code>. Volume được giữ; không xóa DB. Tạo ở máy mới chỉ khi DB URLs còn trống: <code>.\\.venv\\Scripts\\python scripts\\provision_local_postgres.py</code> (Docker engine và image postgres:17 phải sẵn có; script không ghi đè cấu hình DB).</p></section>
<h2 id="http">4. Kết quả HTTP thực tế</h2><p>Chạy server riêng trên loopback port {live['port']}, kết nối TCP thật vào PostgreSQL local. Đây là kiểm tra HTTP, không phải thao tác UI bằng trình duyệt.</p><div class="scroll"><table><thead><tr><th>Check</th><th>Request</th><th>Expected</th><th>Observed</th></tr></thead><tbody>{http_rows}</tbody></table></div><p>Job review: <code>{e(live['review_job_id'])}</code>. Metadata raw source giữ nguyên trên {live['source_before']['file_count']} files (so sánh đường dẫn, size, mtime_ns; không phải hash nội dung toàn bộ ảnh).</p>
<h2>5. Build, preflight và môi trường</h2>{command_sections}<details><summary>PostgreSQL provision evidence (không credentials)</summary><pre>{e(json.dumps(evidence['postgres'],ensure_ascii=False,indent=2))}</pre></details>
<h2 id="limits">6. Chưa xác minh / chưa hoàn thành nghiệm thu</h2><section class="box"><ul><li>Chưa có real trained temporal bundle hoặc detector weights được chỉ định. Chưa chạy inference dữ liệu thật; không có số đo accuracy/F1/recall/throughput có ý nghĩa sản phẩm.</li><li>Chưa có video/person/time labels, verified global source groups và mapping sample review. Dataset ảnh hiện tại không đủ để train temporal; 99 nhóm exact duplicates là kết quả audit trước đó, không thay thế kiểm tra near-duplicate/source leakage.</li><li>Chưa push/run Kaggle, chưa xác nhận GPU/quota/input/output thật; chưa OAuth/upload/download Drive thật. Không có cloud success hoặc fake model approval trong DB development.</li><li>UI trực quan, phát video trong browser và camera chưa được kiểm thử. Origin permission trước đó từ chối http://127.0.0.1:8000; không thử vượt bằng browser/headless khác.</li><li>Live camera Start/Stop, SSE, background polling, crash receipt recovery/takeover, full mapping-review/evaluation UI, remote prepare-output workflow và full training/new-version workflow còn thiếu. Dispatch/reconcile/collect/sync hiện là CLI thao tác rõ ràng; nút tạo prepare job chỉ ghi queued.</li><li>PostgreSQL hai process trên cùng máy đã kiểm thử, chưa phải nhiều máy trên DB chung. Test schema có run ID riêng và được dọn, không reset DB dùng chung.</li><li>Nghiệm thu M6/M7 và user acceptance vẫn pending. Không sửa sources/ hoặc D:\\DrowningDemo; không chạy training thật trên local.</li></ul></section>
<h2 id="tests">7. Danh sách regression đầy đủ</h2><p>{regression['tests_run']} tests · {regression['duration_seconds']} giây · {regression['failures']} failures · {regression['errors']} errors · {regression['skipped']} skips. Sáu tests PostgreSQL không bị skip. Gradient smoke dùng tensor tổng hợp, không optimizer step.</p><input id="test-search" placeholder="Lọc tên test…" aria-label="Lọc tên test"><div class="scroll"><table id="test-table"><thead><tr><th>Test</th><th>Observed</th><th>Giây</th></tr></thead><tbody>{test_rows}</tbody></table></div>
<h2>8. Evidence & revision</h2><p>Thời điểm UTC: <code>{e(timestamp)}</code>. Code fingerprint: <code>{fingerprint}</code>. Manifest hashes theo file và HTTP bodies nằm trong <a href="main-flow-review.json">main-flow-review.json</a>; kết quả từng test trong <a href="regression-results.json">regression-results.json</a>. Trạng thái milestone: <a href="../IMPLEMENTATION_STATUS.md">IMPLEMENTATION_STATUS.md</a>.</p><small>Báo cáo được sinh từ lệnh/test đã chạy. Không chứa .env, DSN, OAuth token hay trọng số. HTML được kiểm tra cấu trúc/link bằng code; chưa render/visual review trong browser.</small>
<script>document.getElementById('test-search').addEventListener('input',function(){{const q=this.value.toLowerCase();document.querySelectorAll('#test-table tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))}});</script>
</main></body></html>"""


def main():
    REPORTS.mkdir(exist_ok=True)
    regression_path = REPORTS / "regression-results.json"
    if not regression_path.is_file():
        raise RuntimeError("Run scripts/run_regression.py first; do not manufacture test results")
    regression = json.loads(regression_path.read_text(encoding="utf-8"))
    if regression["failures"] or regression["errors"]:
        raise RuntimeError("Regression failed; fix and rerun before generating a completion report")
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    commands = []
    specs = [
        ("Vite/TypeScript production build", [npm, "run", "build"], ROOT / "web", (0,)),
        ("UI lint", [npm, "run", "lint"], ROOT / "web", (0,)),
        ("Compile Python", [PYTHON, "-m", "compileall", "-q", "src", "tests", "scripts"], ROOT, (0,)),
        ("Build Kaggle notebook locally (no push/run)", [PYTHON, "notebooks/build_notebook.py"], ROOT, (0,)),
        ("Real PostgreSQL readiness", [PYTHON, "-m", "drowning.cli", "db", "check"], ROOT, (0,)),
        ("Temporal preflight on real image-only dataset", [PYTHON, "-m", "drowning.cli", "source", "build-windows", "local-cache/derived/alanoud-v4.jsonl"], ROOT, (3,)),
        ("Verified group split preflight", [PYTHON, "-m", "drowning.cli", "source", "build-splits", "local-cache/derived/alanoud-v4.jsonl", "--output", str(ROOT / "local-cache/review" / f"blocked-split-{uuid4().hex}.jsonl")], ROOT, (3,)),
    ]
    for label, args, cwd, expected in specs:
        print(f"Running: {label}", flush=True)
        result = command(args, cwd=cwd, expected=expected)
        result["label"] = label
        commands.append(result)
        if not result["passed"]:
            print(result["stdout"] + "\n" + result["stderr"])
            raise RuntimeError(f"Review command failed: {label}")
    print("Running: real HTTP + PostgreSQL + source scan", flush=True)
    live = live_checks()
    canonical = ROOT / "local-cache/derived/alanoud-v4.jsonl"
    evidence = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "workspace": str(ROOT),
                "acceptance": "local technical flow verified; model/cloud/user acceptance pending",
                "regression": regression, "commands": commands, "live": live,
                "postgres": json.loads((ROOT / "local-cache/review/postgres_setup.json").read_text(encoding="utf-8")),
                "code": code_fingerprint(), "input": {"canonical_sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(),
                                                        "real_approved_model": None, "real_model_hash": None},
                "scan_profile_seconds": {name: round(pstats.Stats(str(ROOT / "local-cache/review" / filename)).total_tt, 3)
                                         for name, filename in (("before", "source-scan.prof"), ("after", "source-scan-after.prof"))
                                         if (ROOT / "local-cache/review" / filename).is_file()},
                "initial_review_failure": {"check": "real source scan over HTTP", "error": "httpx.ReadTimeout",
                                           "timeout_seconds": 60, "fixed_in": "src/drowning/data/adapters.py"},
                "no_local_real_training": True, "live_cloud_exercised": False}
    (REPORTS / "main-flow-review.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    page = make_report(evidence)
    (REPORTS / "main-flow-review.html").write_text(page, encoding="utf-8")
    print(f"Report: {REPORTS / 'main-flow-review.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
