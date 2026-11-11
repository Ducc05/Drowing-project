# Công tắc "Tự chia vùng", chỉnh số ô và hiện fps video — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cho phép bật/tắt lớp lọc theo ô lưới và chỉnh số cột × hàng ngay trên web theo từng lượt chạy (test video, camera), và hiện tốc độ model gán nhãn video (khung/giây).

**Architecture:** Đi theo đúng đường của `confidence`: query/body param -> `resolve_*` trong `api/inference.py` -> lưu trong payload job -> truyền vào `load_approved_pipeline(cell_grid=...)` -> `load_h20saver_pipeline`. fps lấy từ vòng `progress` sẵn có trong `inference/pipeline.py`. Web thêm một hook `useCellGrid` và một component `CellGridControl` dùng chung cho 3 chỗ (Tổng quan, Đánh giá, Camera).

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy, unittest/pytest; React 19 + Vite + TypeScript (không có test runner JS; kiểm bằng `npm run build` và `npm run lint`).

**Spec:** `docs/superpowers/specs/2026-11-11-cell-gate-toggle-and-fps-design.md`

## Global Constraints

- Chạy lệnh từ thư mục gốc dự án; Python: `.\.venv\Scripts\python`. Toàn bộ suite: `.\.venv\Scripts\python -m unittest discover -s tests` (hiện 417 test đạt, không được giảm).
- Giá trị tham số lưới: chuỗi `"<cột>x<hàng>"` (mỗi cạnh 1–32, `MAX_CELLS_PER_AXIS` trong `inference/cells.py`) hoặc `"off"`. Dùng `parse_cell_grid` để phân tích, không viết parser thứ hai. Không gửi tham số = giữ `settings.h20saver_cell_grid` (mặc định 6x4).
- Chỉ lớp lọc theo ô được bật/tắt; lớp lọc chuyển động (`H20SAVER_MOTION_*`) giữ nguyên.
- fps là tốc độ đo trên máy chạy server; không ghi nó như kết quả chất lượng model. Camera trực tiếp không hiện fps.
- Không commit trừ khi người dùng yêu cầu. Cây làm việc đang có thay đổi không liên quan ở `src/drowning/api/train.py` — không đụng, không `git add` nó.
- Theo `CLAUDE.md`: không in/log secrets; sau khi xong cập nhật `IMPLEMENTATION_STATUS.md`, tách rõ "code đã viết" / "test đã chạy" / "kiểm tra dữ liệu thật" / "người dùng nghiệm thu". Không khẳng định hiệu quả mô hình.
- Tiếng Việt cho mọi chữ hiển thị trên UI.

## Review Focus

- Chuỗi lưới sai (`"0x4"`, `"33x4"`, `"abc"`, `"6x"`): video -> 409 và không tạo job; camera -> 422 và không tạo job.
- Chuỗi có khoảng trắng/hoa thường (`" 6X4 "`) phải được chuẩn hoá thành `"6x4"` trước khi lưu và so sánh `Idempotency-Key`.
- Engine không phải `h20saver` bỏ qua `cell_grid`: readiness báo `cell_grid_supported=false` và UI ẩn điều khiển (không để công tắc không có tác dụng).
- `elapsed_s == 0` hoặc chưa xử lý khung nào: `fps` là `None`, không chia cho 0, UI hiện "—".
- Video bị dừng/huỷ giữa chừng: không có `processing_fps` trong summary, nhưng `progress.fps` cuối cùng vẫn còn để trình xem trực tiếp hiển thị.
- Tham số `cell_grid` gửi kèm test ảnh: bỏ qua, không lưu vào payload.

---

### Task 1: Tham số lưới đi từ API tới pipeline

**Files:**
- Modify: `src/drowning/inference/cells.py` (thêm `format_cell_grid`)
- Modify: `src/drowning/api/inference.py` (`resolve_cell_grid`, `load_approved_pipeline`)
- Modify: `src/drowning/api/test_jobs.py` (`submit`, `result`)
- Modify: `src/drowning/api/camera.py` (`start`)
- Modify: `src/drowning/api/app.py` (`CameraStartRequest`, `/test/video`, `execute_media`, `test_readiness`, `_job_dict`)
- Test: `tests/test_cell_gate_settings.py` (mới)

**Interfaces:**
- Produces:
  - `format_cell_grid(grid: tuple[int, int] | None) -> str` trong `drowning.inference.cells` ("6x4" hoặc "off").
  - `resolve_cell_grid(settings: Settings, requested: str | None) -> tuple[int, int] | None` trong `drowning.api.inference`: `None` -> `settings.h20saver_cell_grid`; chuỗi -> `parse_cell_grid`; sai -> `ValueError`.
  - `load_approved_pipeline(..., cell_grid=USE_SETTINGS)` với `USE_SETTINGS` là sentinel module-level (vì `None` nghĩa là "tắt"); giá trị khác sentinel thay cho `settings.h20saver_cell_grid` khi gọi `load_h20saver_pipeline`.
  - `LocalMediaTestJobs.submit(..., cell_grid: str | None = None)`; payload job lưu `"cell_grid": "<chuẩn hoá>"` chỉ khi `modality == "video"`; `result()` và `_job_dict` trả thêm khoá `cell_grid`.
  - `CameraService.start(..., cell_grid: str | None = None)`; payload camera lưu `"cell_grid"`.
  - `GET /api/v1/test/readiness` và `GET /api/v1/camera/readiness` (nếu có; nếu camera dùng chung readiness thì chỉ một chỗ) trả thêm `default_cell_grid: str`, `cell_grid_range: [1, 32]`, `cell_grid_supported: bool` (`detection_engine == "h20saver"`).

- [ ] **Step 1: Viết test lỗi** trong `tests/test_cell_gate_settings.py`, dùng khuôn `tests/test_box_review.py` (`ConfidenceAndBoxReviewApiTests`, `CameraConfidenceTests`, helper `request`, `call`, `png`, `video_job`). Tên test và khẳng định:
  - `test_format_and_resolve`: `format_cell_grid((6, 4)) == "6x4"`, `format_cell_grid(None) == "off"`; `resolve_cell_grid(Settings.from_env({}), None) == (6, 4)`; `resolve_cell_grid(..., "off") is None`; `resolve_cell_grid(..., " 8X6 ") == (8, 6)`; `"0x4"`, `"33x4"`, `"abc"`, `"6x"` đều `ValueError`.
  - `test_video_job_stores_normalised_grid_and_rejects_a_changed_one`: POST `/api/v1/test/video` với `cell_grid=" 8X6 "` -> 200, `result["cell_grid"] == "8x6"`, `/api/v1/jobs` có `cell_grid == "8x6"`; cùng `Idempotency-Key` với `cell_grid=off` -> 409 và thông báo chứa `cell_grid`.
  - `test_invalid_grid_creates_no_job`: `cell_grid=0x4` -> 409; bảng `Job` (loại `inference`) rỗng.
  - `test_image_ignores_cell_grid`: POST `/api/v1/test/image?cell_grid=off` -> 200 và payload không có khoá `cell_grid` (đọc qua `/api/v1/jobs`: giá trị `None`).
  - `test_camera_start_stores_grid_and_rejects_invalid`: `camera/start` với `cell_grid="5x3"` -> 200 và payload job có `"cell_grid": "5x3"`; `cell_grid="9x"` -> 422, không tạo job.
  - `test_readiness_reports_grid_defaults`: `default_cell_grid == "6x4"`, `cell_grid_range == [1, 32]`, `cell_grid_supported` đúng theo `DETECTION_ENGINE` (mặc định không phải h20saver -> `False`; với `Settings` h20saver -> `True`).
  - `test_requested_grid_reaches_h20saver_loader`: dùng `unittest.mock.patch` cho `drowning.api.inference._h20saver_model` (trả `(identity, weights_path)` giả) và `drowning.inference.h20saver.load_h20saver_pipeline` (trả `(pipeline, identity)` giả); gọi `load_approved_pipeline(session, settings, cell_grid=None)` và khẳng định mock nhận `cell_grid=None`; gọi lại không truyền `cell_grid` và khẳng định mock nhận `(6, 4)`.

- [ ] **Step 2: Chạy để thấy lỗi:** `.\.venv\Scripts\python -m pytest tests\test_cell_gate_settings.py -v` — Expected: FAIL (thiếu `format_cell_grid`, `resolve_cell_grid`).

- [ ] **Step 3: Cài đặt** theo các chữ ký ở khối Interfaces. `submit` và `CameraService.start` gọi `resolve_cell_grid` trong cùng `try/except ValueError` với `resolve_confidence` (video -> `MediaRequestConflict`, camera -> `CameraSourceError`), lưu chuỗi đã `format_cell_grid`, so sánh với `existing.payload.get("cell_grid")` cho cùng `Idempotency-Key` theo mẫu `stored_threshold`. `app.py`: thêm `cell_grid: str | None = Query(None, max_length=16)` cho `/api/v1/test/video` và chuyển qua `execute_media`; thêm `cell_grid: str | None = Field(default=None, max_length=16)` vào `CameraStartRequest`; `/test/image` không nhận tham số này. Truyền kết quả `resolve_cell_grid` vào `load_approved_pipeline(cell_grid=...)` ở cả hai nơi gọi; chỉ truyền khi người dùng đã gửi (nếu không, để sentinel).

- [ ] **Step 4: Chạy test mới và test liên quan:** `.\.venv\Scripts\python -m pytest tests\test_cell_gate_settings.py tests\test_box_review.py tests\test_camera.py tests\test_media_api.py tests\test_cell_gate.py -v` — Expected: tất cả PASS.

---

### Task 2: fps trong tiến trình và tóm tắt video

**Files:**
- Modify: `src/drowning/inference/pipeline.py:289-321,368-373` (`infer_video`: hai chỗ gọi `progress`, và dict trả về)
- Test: `tests/test_inference_pipeline.py` (thêm vào cạnh `test_video_stride_processes_only_selected_frames_and_reports_progress`)

**Interfaces:**
- Produces: mỗi dict `progress` có thêm `"fps": float | None` = `processed / elapsed_s` làm tròn 1 chữ số, `None` khi `elapsed_s == 0` hoặc `processed == 0`. Dict kết quả của `infer_video` có thêm `"elapsed_s": float`, `"processing_fps": float | None` và `"fps_metadata": float` (fps gốc của video, đã có biến `fps` trong hàm). Các trường này tự đi vào `result_summary` của job vì `run_video_test` trải nguyên dict kết quả (`{**result, ...}`).

- [ ] **Step 1: Viết test lỗi** `test_video_progress_and_result_report_processing_fps` (skip khi thiếu cv2/torch như các test cạnh bên), dùng cùng fixture 10 khung/stride 3 như test láng giềng. Khẳng định: `reports[-1]["fps"]` là số dương và bằng `round(reports[-1]["processed"] / reports[-1]["elapsed_s"], 1)` (cho phép sai số làm tròn 0.1; nếu `elapsed_s` làm tròn bằng 0 thì khẳng định `fps is None`); `result["processing_fps"] > 0`; `result["fps_metadata"] == 10`; `"elapsed_s" in result`. Thêm `test_fps_is_none_without_elapsed_time` cho hàm trợ giúp riêng (xem Step 3) với `(processed=0, elapsed=0)` và `(processed=5, elapsed=0)` -> `None`, `(processed=20, elapsed=4.0)` -> `5.0`.

- [ ] **Step 2: Chạy để thấy lỗi:** `.\.venv\Scripts\python -m pytest tests\test_inference_pipeline.py -k fps -v` — Expected: FAIL.

- [ ] **Step 3: Cài đặt** hàm module-level `processing_fps(processed: int, elapsed_s: float) -> float | None` trong `pipeline.py` (làm tròn 1 chữ số, `None` nếu không chia được), dùng nó cho cả hai lời gọi `progress(...)` (khoá `"fps"`) và cho `"processing_fps"` trong dict trả về. Không đổi thứ tự hay tần suất gọi `progress`.

- [ ] **Step 4: Chạy:** `.\.venv\Scripts\python -m pytest tests\test_inference_pipeline.py tests\test_h20saver.py tests\test_media_api.py -v` — Expected: PASS.

---

### Task 3: Web — điều khiển lưới và hiển thị fps

**Files:**
- Modify: `web/src/api.ts` (hook `useCellGrid`, `cellGridParam`, hằng `CELL_GRID_MAX`)
- Modify: `web/src/ui.tsx` (component `CellGridControl`)
- Modify: `web/src/pages.css` (kiểu cho `.cell-grid-control`, theo `.confidence-control`)
- Modify: `web/src/App.tsx` (khối `media-test`: gửi `cell_grid` khi `mediaKind === 'video'`, hiện fps trong `result-info`; kiểu `TestReadiness` thêm `default_cell_grid`, `cell_grid_supported`)
- Modify: `web/src/pages/EvaluationPage.tsx:344` (URL test video + `CellGridControl` cạnh `ConfidenceControl` ở dòng ~423; hiện fps trong phần tóm tắt video)
- Modify: `web/src/pages/CameraPage.tsx:55` (body `camera/start` + `CellGridControl` cạnh `ConfidenceControl`; điều khiển bị mờ khi camera đang chạy)
- Modify: `web/src/LiveVideoPreview.tsx` (kiểu `Job.progress` thêm `fps`; thêm "Tốc độ model: X khung/giây")
- Modify: `web/src/overlayData.ts` (`VideoSummary` thêm `processing_fps?: number | null`, `elapsed_s?: number`)
- Modify: `IMPLEMENTATION_STATUS.md`

**Interfaces:**
- Consumes: Task 1 (`cell_grid` query/body, readiness `default_cell_grid`, `cell_grid_range`, `cell_grid_supported`); Task 2 (`progress.fps`, `summary.processing_fps`).
- Produces:
  - `useCellGrid(): readonly [CellGridChoice | null, (next: CellGridChoice) => void]` với `type CellGridChoice = { enabled: boolean; cols: number; rows: number }`; lưu `localStorage` khoá `cell-grid-choice` trong `try/catch`, ép `cols`/`rows` về số nguyên 1–32; `null` = dùng mặc định của server.
  - `cellGridParam(choice: CellGridChoice | null, fallback: string | undefined): string | undefined` -> `"off"`, `"<cols>x<rows>"`, hoặc `fallback` khi `choice` là `null`.
  - `CellGridControl({ value, onChange, disabled, hint }: { value: CellGridChoice; onChange(next: CellGridChoice): void; disabled?: boolean; hint?: ReactNode })`: công tắc "Tự chia vùng" (`input[type=checkbox]` dùng class `switch` có sẵn như trong `videoOverlay.tsx`) + hai `input[type=number]` "Cột" và "Hàng" (min 1, max 32, mờ khi tắt công tắc hoặc `disabled`). Mỗi ô có `aria-label`.

- [ ] **Step 1: Cài đặt hook và component** theo Interfaces. Giá trị hiển thị ban đầu = `choice ?? parse(readiness.default_cell_grid)`; nếu readiness chưa có thì 6×4 bật. `cell_grid_supported === false` -> trang không vẽ `CellGridControl` và không gửi tham số.

- [ ] **Step 2: Nối vào ba trang.** Mọi thay đổi lưới ở App/Evaluation đặt `setMediaKey(null)`/`setKey(null)` như `ConfidenceControl` (đổi lưới = request khác). Chỉ thêm `&cell_grid=...` vào URL video, không thêm vào URL ảnh. Camera: gửi `cell_grid` trong body `camera/start`; sau khi đã chạy, điều khiển `disabled` kèm gợi ý "Đổi lưới cần dừng rồi bật lại camera".

- [ ] **Step 3: Hiển thị fps.** `LiveVideoPreview`: cạnh "Khung a/b" hiện `Tốc độ model: {fps} khung/giây` khi `progress.fps != null`, ngược lại "—". Kết quả (App.tsx `result-info` và tóm tắt video ở `EvaluationPage`): khi `summary.processing_fps != null` hiện `tốc độ model {x} khung/giây (video gốc {fps_metadata} khung/giây, bước nhảy {frame_stride})`. Thêm lời chú nhỏ: "Đo trên máy chạy server, không phải chất lượng model."

- [ ] **Step 4: Kiểm tra build:** trong `web/`: `npm run build` rồi `npm run lint` — Expected: không lỗi TypeScript, không lỗi oxlint mới.

- [ ] **Step 5: Chạy cả suite Python:** `.\.venv\Scripts\python -m unittest discover -s tests` — Expected: tất cả đạt, số test >= 417 + các test mới.

- [ ] **Step 6: Cập nhật `IMPLEMENTATION_STATUS.md`:** ghi "code đã viết" (công tắc/số ô lưới, fps), "test đã chạy" (con số thực từ Step 5, `npm run build`, `npm run lint`), "kiểm tra dữ liệu thật": chưa (chưa thử trên video/webcam thật), "người dùng nghiệm thu": chưa.

---

## Self-review

- **Spec coverage:** công tắc + số ô (Task 1 backend, Task 3 UI); áp dụng test video + camera (Task 1, 3); lưu cùng kết quả + từ chối `Idempotency-Key` khác lưới (Task 1); lưới vẽ theo lượt chạy (đã có sẵn qua `cell_gate` trong detections, tắt lưới -> không có `cell_gate` -> UI không vẽ); fps trực tiếp (Task 2, 3); fps tổng kết kèm fps gốc và stride (Task 2, 3).
- **Nhất quán tên:** `cell_grid` (param/payload), `format_cell_grid`, `resolve_cell_grid`, `USE_SETTINGS`, `processing_fps`, `fps_metadata`, `progress.fps`, `useCellGrid`, `CellGridControl`, `cellGridParam` dùng nhất quán giữa các task.
- **Review Focus:** mỗi dòng đều có test: lưới sai/chuẩn hoá/ảnh bỏ qua/engine không hỗ trợ (Task 1), `elapsed==0` (Task 2); video bị dừng giữ `progress.fps` (hành vi tự nhiên của `progress`, không cần thêm mã).
