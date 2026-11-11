# Quét tab trình duyệt như camera trực tiếp — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trang Camera có nguồn "Quét tab/cửa sổ": trình duyệt chụp một tab (ví dụ YouTube), đẩy khung JPEG lên server, server xử lý như một camera live với đầy đủ khung nhận diện, cảnh báo và âm báo.

**Architecture:** Server thêm một nguồn `browser-tab` do lớp `PushCapture` (giao diện giống `cv2.VideoCapture`) cấp khung cho `CameraSession` hiện có, nên tracker/H20Saver/lọc ô/MJPEG/job/slot dùng lại nguyên. Endpoint `POST /api/v1/camera/frame` nhận JPEG và trả luôn trạng thái cảnh báo. Phía web, module singleton `screenShare.ts` đọc khung bằng `MediaStreamTrackProcessor` (không dùng timer), nén JPEG và gửi lên.

**Tech Stack:** Python 3.11, FastAPI, OpenCV, Pillow, SQLAlchemy; React 19 + Vite + TypeScript (Chrome/Edge: `getDisplayMedia`, `MediaStreamTrackProcessor`, `OffscreenCanvas`).

**Spec:** `docs/superpowers/specs/2026-11-11-browser-tab-capture-design.md`

## Global Constraints

- Chỉ model đã duyệt; một compute job toàn hệ thống; giữ owner/token fencing; không log/lưu secret hay khung hình.
- Không thêm đường train cục bộ. Test dùng pipeline/giả lập; không mô tả như kết quả chất lượng model.
- Khung hình chỉ nằm trong RAM (khung mới nhất), không ghi đĩa/DB/Drive.
- Nguồn đặc biệt có đúng chuỗi `browser-tab`; webcam `0–15` và URL `http/https/rtsp/rtmp` giữ nguyên hành vi; đường dẫn file vẫn bị từ chối.
- Mặc định gửi 10 khung/giây (`fps` 1–30), khung tối đa 1280 px cạnh dài từ trình duyệt, server chấp nhận tối đa 1920 px cạnh dài và 2 MB (`2_000_000` byte) mỗi khung.
- `PushCapture.read()`: chờ khung mới tối đa 1,0 s; lần đọc đầu tiên tối đa 20 s. `CameraSession` hiện tại coi 30 lần đọc hỏng liên tiếp là phiên `failed` (giữ nguyên).
- Chỉ hỗ trợ Chrome/Edge; trình duyệt khác hiện thông báo tiếng Việt, không gọi server.
- Chuỗi giao diện bằng tiếng Việt. Working tree đang có nhiều thay đổi chưa commit không liên quan: khi commit chỉ `git add` đúng các file của task, và chỉ commit khi người dùng đã yêu cầu commit.
- Cập nhật `IMPLEMENTATION_STATUS.md` ở cuối, tách rõ "code đã viết / test đã chạy / kiểm tra thật / người dùng chấp nhận".

## Review Focus

Các tình huống spec ngầm đòi nhưng dễ bị bỏ sót (mỗi dòng có test ở task nêu trong ngoặc):
1. JPEG khai kích thước khổng lồ (bomb giải nén) → 422, không giải mã, session vẫn sống (Task 2, 3).
2. Gửi khung khi không có phiên `browser-tab` (không có phiên, hoặc đang chạy webcam) → 409, không lưu gì (Task 3, 4).
3. Cửa sổ nguồn đổi kích thước giữa chừng → server từ chối khung sai kích thước (422) mà không giết phiên; client giữ canvas cố định (Task 3, 5).
4. Người dùng đóng tab app / bấm "Dừng chia sẻ" / ngừng gửi → phiên `failed`, slot được giải phóng, `status` không kẹt ở "active" (Task 3; Task 5 cho "Dừng chia sẻ").
5. Trình duyệt thiếu `MediaStreamTrackProcessor` hoặc người dùng hủy hộp thoại chọn tab → thông báo rõ, không tạo job (Task 5, 6).
6. Khung đến trong lúc model còn đang tải (trước khi `start` xong) → 409 tạm thời, không làm hỏng `start` (Task 3, 5).

---

### Task 1: Spike — tab app bị ẩn có làm hỏng việc gửi khung/âm báo không?

Bỏ đi sau khi xong; kết quả quyết định phần còn lại của Task 5–6. Task 2–4 (server) độc lập, có thể làm song song.

**Files:**
- Create (throwaway, ngoài repo): `C:\Users\ADMIN\AppData\Local\Temp\claude\3eb6e23e7cc347a3\d2b7daeb-ee8b-49c7-841a-e348c67fe633\scratchpad\spike-share.html`

**Interfaces:** Không có. Chỉ sinh ra số liệu.

- [ ] **Step 1: Viết trang spike** gồm nút "Chọn tab", gọi `getDisplayMedia({video:{frameRate:15}, audio:false, selfBrowserSurface:'exclude'})`, vòng đọc `MediaStreamTrackProcessor` đếm số `VideoFrame` mỗi giây (ghi bảng trên trang + `document.title`), đồng thời một `setInterval(1000)` ghi độ lệch thời gian thực của chính nó, và một nút phát beep lặp 1 s bằng `setInterval`.
- [ ] **Step 2: Chạy** `py -3.11 -m http.server 8765 --bind 127.0.0.1` trong thư mục scratchpad, mở `http://127.0.0.1:8765/spike-share.html` bằng Chrome.
- [ ] **Step 3: Người dùng đo** (Claude không bấm được hộp thoại chọn tab): chia sẻ một tab YouTube đang phát; (a) để tab spike ở nền sau tab YouTube 6 phút; (b) để tab spike ở cửa sổ riêng bị che hoàn toàn 6 phút; (c) trong lúc đó bật beep. Ghi: khung/giây trung bình của vòng `MediaStreamTrackProcessor`, độ lệch `setInterval`, beep có còn đều không.
- [ ] **Step 4: Quyết định và ghi lại** trong `IMPLEMENTATION_STATUS.md` (mục kiểm tra thật):
  - Nếu vòng đọc giữ ≥ 8 khung/giây khi tab ẩn → làm Task 5–6 đúng thiết kế.
  - Nếu bị giảm tốc → thêm vào Task 5 phương án chuyển vòng gửi vào Web Worker, và Task 6 hiện khuyến nghị "mở app thành cửa sổ riêng cạnh video" ở dạng cảnh báo nổi bật.
  - Nếu beep `setInterval` bị lệch → Task 6 dùng lịch beep dựng sẵn bằng đồng hồ `AudioContext` thay vì `setInterval`.

---

### Task 2: `browser-tab` source và `PushCapture`

**Files:**
- Modify: `src/drowning/inference/camera.py` (thêm hằng số, `parse_camera_source`, `PushCapture`, `decode_frame`)
- Test: `tests/test_camera.py` (thêm `PushCaptureTests`, mở rộng `SourceParsingTests`)

**Interfaces:**
- Produces:
  - `BROWSER_TAB_SOURCE = "browser-tab"`, `MAX_FRAME_BYTES = 2_000_000`, `MAX_FRAME_SIDE = 1920`
  - `parse_camera_source("browser-tab") -> ("browser-tab", "browser-tab")`
  - `decode_frame(jpeg: bytes, *, max_side: int = MAX_FRAME_SIDE) -> numpy.ndarray` (BGR uint8). Ném `CameraSourceError` nếu rỗng, quá `MAX_FRAME_BYTES`, không phải ảnh, hoặc cạnh dài > `max_side`. Đọc kích thước từ header bằng Pillow **trước** khi giải mã.
  - `class PushCapture(fps: float = 10.0, *, read_timeout: float = 1.0, first_frame_timeout: float = 20.0)` với `isOpened() -> bool`, `push(frame: ndarray) -> None`, `read() -> tuple[bool, ndarray | None]`, `release() -> None`, `get(prop_id: int) -> float` (trả `fps` khi `prop_id == 5`, ngược lại `0.0`), thuộc tính `shape: tuple[int, int] | None` (cao, rộng của khung đầu tiên đã push), `released: bool`.

- [ ] **Step 1: Viết test lỗi** trong `tests/test_camera.py`:
  - `SourceParsingTests.test_browser_tab_is_a_named_source`: `parse_camera_source("browser-tab") == ("browser-tab", "browser-tab")`; `"browser-tab "` (có khoảng trắng) cũng hợp lệ; `"browser-tabs"` và `"Browser-Tab"` vẫn bị `CameraSourceError`.
  - `PushCaptureTests.test_read_returns_only_new_frames_and_latest_wins`: push A, push B → `read()` trả B; `read()` kế tiếp (timeout nhỏ 0.05) trả `(False, None)`.
  - `test_first_read_waits_for_first_frame`: thread push sau 0,1 s, `PushCapture(first_frame_timeout=2)` → `read()` đầu trả khung; không có khung nào → `(False, None)` sau `first_frame_timeout` (đặt 0,1).
  - `test_release_unblocks_reader_and_marks_closed`: `release()` từ thread khác khi `read()` đang chờ → trả `(False, None)`; `isOpened()` là `False` sau đó.
  - `test_get_reports_nominal_fps`: `PushCapture(fps=12).get(5) == 12.0`; `get(3) == 0.0`.
  - `DecodeFrameTests`: JPEG 64×48 hợp lệ giải mã ra `shape == (48, 64, 3)`; bytes rác, `b""`, và payload `MAX_FRAME_BYTES + 1` → `CameraSourceError`; một JPEG hợp lệ có header 4000×4000 (tạo bằng Pillow, nén màu đơn) → `CameraSourceError` và `cv2.imdecode` **không** được gọi (patch `cv2.imdecode` bằng mock, assert chưa gọi).
- [ ] **Step 2: Chạy để thấy lỗi:** `.\.venv\Scripts\python -m pytest tests\test_camera.py -k "browser_tab or PushCapture or DecodeFrame" -v` — Expected: FAIL (ImportError / chưa định nghĩa).
- [ ] **Step 3: Cài đặt** trong `src/drowning/inference/camera.py`: `PushCapture` dùng `threading.Condition`, một `_seq` đếm khung, `read()` chờ `_seq` tăng so với lần đọc trước (lần đầu dùng `first_frame_timeout`, các lần sau `read_timeout`), trả bản sao khung để thread khác không sửa. `parse_camera_source` kiểm tra `value == BROWSER_TAB_SOURCE` sau `strip()` trước nhánh chỉ số webcam. `decode_frame` dùng `PIL.Image.open(BytesIO(jpeg)).size` để kiểm tra cạnh dài, rồi `cv2.imdecode(np.frombuffer(...), cv2.IMREAD_COLOR)`; `None` → `CameraSourceError`.
- [ ] **Step 4: Chạy lại:** cùng lệnh Step 2 — Expected: PASS; sau đó `.\.venv\Scripts\python -m pytest tests\test_camera.py -v` — Expected: toàn bộ PASS (các test webcam cũ không đổi).
- [ ] **Step 5: Commit** (nếu người dùng đã yêu cầu commit): `git add src/drowning/inference/camera.py tests/test_camera.py` — message `feat: add browser-tab camera source and PushCapture`.

---

### Task 3: `CameraService` hỗ trợ phiên `browser-tab`

**Files:**
- Modify: `src/drowning/api/camera.py` (`CameraService.__init__`, `start`, `probe`, `_forget`, thêm `push_frame`)
- Test: `tests/test_camera.py` (thêm vào `CameraApiTests`)

**Interfaces:**
- Consumes (Task 2): `BROWSER_TAB_SOURCE`, `PushCapture`, `decode_frame`, `CameraSourceError`.
- Produces:
  - `CameraService.__init__(..., push_read_timeout: float = 1.0, push_first_frame_timeout: float = 20.0)` (lưu thành thuộc tính cùng tên để test chỉnh được).
  - `CameraService.start(raw_source, idempotency_key, *, confidence=None, cell_grid=None, fps: int | None = None)`. Với `browser-tab`: tạo `PushCapture(fps or 10, read_timeout=..., first_frame_timeout=...)`, gán `self._push` **trước khi tải model** và **không cần `self._lock` để nhận khung**; `CameraSession(..., capture_factory=lambda _s: push)`; payload job thêm `"capture": "browser_tab"`. `fps` bị bỏ qua với nguồn khác.
  - `CameraService.push_frame(jpeg: bytes) -> dict[str, Any]` trả `{"alert_track_ids": list[str], "frames_processed": int}`. Không lấy `self._lock`. Ném `CameraNotRunning` nếu không có `self._push` hoặc nó đã `released`; `CameraSourceError` nếu `decode_frame` lỗi hoặc `frame.shape[:2] != push.shape` (khi `push.shape` đã có).
  - `probe("browser-tab")` ném `CameraSourceError("nguồn browser-tab không cần kiểm tra kết nối")`.
  - `_forget()` cũng đặt `self._push = None`.

- [ ] **Step 1: Viết test lỗi** trong `CameraApiTests` (dùng helper `feed()` mới: thread lặp `service.push_frame(jpeg_bytes)` mỗi 0,02 s, bỏ qua mọi exception, dừng bằng `Event` — mô phỏng client nên cũng chứng minh khung đến sớm bị bỏ qua an toàn; JPEG tạo bằng `cv2.imencode` từ mảng 48×64×3). Trong `setUp` đặt `self.app.state.camera.push_read_timeout = 0.05`:
  - `test_browser_tab_start_status_stop_cycle`: bật `feed`, `POST /camera/start` `{"source": "browser-tab", "idempotency_key": "camera-tab-1"}` → 200, `active`, job `running`, payload `capture == "browser_tab"` và `source == "browser-tab"`, slot bị giữ; `frames_processed > 0` sau 0,3 s; `stop` → `completed`, slot trống.
  - `test_frames_without_a_browser_tab_session_are_rejected`: không có phiên → `service.push_frame(jpeg)` ném `CameraNotRunning`; khởi động phiên webcam `"0"` (ScriptedCapture) rồi `push_frame` → vẫn `CameraNotRunning`.
  - `test_wrong_size_frame_is_rejected_but_session_survives`: sau khi chạy, `push_frame` ảnh 32×24 → `CameraSourceError`; `status.active` vẫn `True` và `frames_processed` tiếp tục tăng.
  - `test_huge_declared_size_frame_is_rejected`: JPEG header 4000×4000 → `CameraSourceError`; phiên vẫn chạy.
  - `test_stopping_the_feed_fails_the_session_and_frees_the_slot`: chạy `feed`, rồi dừng `feed`; `wait_for(lambda: not status["active"] , 8)` qua `GET /camera/status`; job `failed`, slot trống.
  - `test_start_with_no_frames_fails_and_frees_the_slot`: đặt `push_first_frame_timeout = 0.2`, không `feed` → `POST /camera/start` trả 502 (`CameraConnectError`), job `failed`, slot trống.
  - `test_probe_rejects_browser_tab`: `POST /camera/probe` `{"source": "browser-tab"}` → 422.
  - `test_fps_is_passed_to_the_pipeline_clock`: `start` với `fps=15` → `pipeline.configure_live_stream` (thêm vào `FakeStreamPipeline` ghi lại tham số) nhận `15.0`.
- [ ] **Step 2: Chạy để thấy lỗi:** `.\.venv\Scripts\python -m pytest tests\test_camera.py -k "browser_tab or wrong_size or huge_declared or stopping_the_feed or no_frames or frames_without or fps_is_passed" -v` — Expected: FAIL.
- [ ] **Step 3: Cài đặt** theo Interfaces. Với `start`: nhánh `browser-tab` bỏ qua `capture_factory` ngoài và thay bằng factory trả `PushCapture`; giữ nguyên toàn bộ luồng create_job/claim_slot/transition/lỗi (đã có test). `push_frame` đọc `push = self._push` một lần rồi dùng biến cục bộ; lấy `alert_track_ids` và `frames_processed` từ `self._session.summary()` nếu `self._session` đã có, ngược lại `[]` và `0`.
- [ ] **Step 4: Chạy lại:** cùng lệnh — Expected: PASS; rồi `.\.venv\Scripts\python -m pytest tests\test_camera.py tests\test_cell_gate_settings.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** (nếu được yêu cầu): `git add src/drowning/api/camera.py tests/test_camera.py` — `feat: run browser-tab sessions through CameraService`.

---

### Task 4: Route `POST /api/v1/camera/frame` và `fps` khi bắt đầu

**Files:**
- Modify: `src/drowning/api/app.py` (`CameraStartRequest`, route mới cạnh các route camera tại ~dòng 498–521)
- Test: `tests/test_camera.py` (thêm vào `CameraApiTests`)

**Interfaces:**
- Consumes (Task 3): `CameraService.push_frame`, `CameraService.start(..., fps=...)`; `MAX_FRAME_BYTES` từ Task 2; `camera_call` đã có (ánh xạ `CameraSourceError`→422, `CameraNotRunning`→409).
- Produces: `CameraStartRequest.fps: int | None = Field(default=None, ge=1, le=30)`; route `POST /api/v1/camera/frame` (body thô `image/jpeg`) → `200 {"alert_track_ids": [...], "frames_processed": n}`, `409` khi không có phiên browser-tab, `413` khi body > `MAX_FRAME_BYTES` (kiểm `Content-Length` rồi kiểm lại độ dài body thực), `422` khi ảnh sai. Route là `async def`, đọc `await request.body()` rồi gọi `push_frame` qua `run_in_threadpool`.

- [ ] **Step 1: Viết test lỗi** (`CameraApiTests`, dùng `call(self.app, "POST", "/api/v1/camera/frame", content=jpeg_bytes, headers={"Content-Type": "image/jpeg"})`):
  - `test_frame_route_returns_alert_state_and_counts`: có phiên chạy → 200, JSON có `alert_track_ids` (list) và `frames_processed` (int ≥ 0).
  - `test_frame_route_conflict_without_session`: 409 và JSON `detail` là chuỗi.
  - `test_frame_route_rejects_oversized_body`: body `b"x" * (MAX_FRAME_BYTES + 1)` → 413 (cả khi phiên đang chạy).
  - `test_frame_route_rejects_garbage_image`: 422.
  - `test_start_accepts_fps_and_validates_range`: `fps=0` và `fps=31` → 422; `fps=15` → 200.
- [ ] **Step 2: Chạy để thấy lỗi:** `.\.venv\Scripts\python -m pytest tests\test_camera.py -k "frame_route or accepts_fps" -v` — Expected: FAIL (404/422 sai).
- [ ] **Step 3: Cài đặt** route và trường `fps`; truyền `fps=body.fps` vào `service.start`. Cập nhật mô tả `source` của `CameraStartRequest` thêm `or 'browser-tab'`.
- [ ] **Step 4: Chạy lại** cùng lệnh — Expected: PASS; rồi `.\.venv\Scripts\python -m unittest discover -s tests` — Expected: toàn bộ PASS (nếu có test fail sẵn không liên quan, ghi lại đúng tên, không sửa hộ).
- [ ] **Step 5: Commit** (nếu được yêu cầu): `git add src/drowning/api/app.py tests/test_camera.py` — `feat: add camera frame upload endpoint`.

---

### Task 5: Bộ gửi khung phía trình duyệt `screenShare.ts`

Phụ thuộc kết quả Task 1.

**Files:**
- Create: `web/src/screenShare.ts`
- Modify: `web/src/api.ts` (chỉ nếu cần export thêm helper gửi nhị phân; ưu tiên dùng `fetch(api(...))` trực tiếp trong module mới)

**Interfaces:**
- Consumes: `api(path)` và `post(path, body?)` từ `web/src/api.ts`; endpoint Task 4.
- Produces (tất cả export từ `web/src/screenShare.ts`):
  - `type SharePhase = 'idle' | 'ready' | 'starting' | 'sending' | 'ended' | 'error'`
  - `type ShareState = { phase: SharePhase; alertTracks: string[]; framesSent: number; sendFps: number | null; surfaceLabel: string | null; error: string | null }`
  - `screenShareSupported(): boolean` — `true` khi có `navigator.mediaDevices?.getDisplayMedia`, `MediaStreamTrackProcessor`, `OffscreenCanvas`.
  - `pickSurface(): Promise<void>` — gọi `getDisplayMedia({ video: { frameRate: 15 }, audio: false, selfBrowserSurface: 'exclude' })`; thành công → `phase: 'ready'` và `surfaceLabel` = nhãn track; người dùng hủy (`NotAllowedError`) → `phase: 'idle'`, `error` = "Bạn chưa chọn tab nào để quét."; thiếu hỗ trợ → `phase: 'error'`, `error` = "Trình duyệt này chưa hỗ trợ quét tab. Hãy dùng Chrome hoặc Edge."
  - `beginSending(fps: number): void` — `phase: 'starting'`, bắt đầu vòng gửi.
  - `stopShare(): void` — dừng track, hủy vòng gửi, `phase: 'idle'` (không gọi API).
  - `getShareState(): ShareState`, `subscribeShare(listener: () => void): () => void`, `useScreenShare(): ShareState` (dùng `useSyncExternalStore`).
- Hành vi vòng gửi (đã chốt trong spec): đọc `VideoFrame` qua `MediaStreamTrackProcessor`; bỏ khung để giữ ≈ `fps` theo `VideoFrame.timestamp` (µs); canvas `OffscreenCanvas` cố định theo khung đầu tiên (co về cạnh dài ≤ 1280, giữ tỉ lệ), khung sau vẽ kiểu "contain" có viền đen; `convertToBlob({ type: 'image/jpeg', quality: 0.8 })`; gửi tuần tự (chỉ gửi khung tiếp theo sau khi khung trước có phản hồi); luôn `frame.close()`. Ở `phase 'starting'` phản hồi 409 → chờ 200 ms rồi thử lại; phản hồi 200 đầu tiên → `phase 'sending'`; từ đó mọi 409/422-không-phải-kích-thước/lỗi mạng liên tiếp 5 lần → `phase 'error'` + `stopShare()`. Phản hồi 200 cập nhật `alertTracks`, `framesSent`, `sendFps` (trung bình 2 s gần nhất). Khi track `ended` (người dùng bấm "Dừng chia sẻ"): `post('/api/v1/camera/stop')` (bỏ qua lỗi), `phase: 'ended'`, `error` = "Đã dừng chia sẻ tab; phiên camera đã được dừng.".
- Nếu Task 1 kết luận timer/rAF bị giảm tốc: vòng đọc đã không dựa vào timer nên giữ nguyên; chỉ chuyển nó vào Web Worker khi Task 1 cho thấy vòng promise cũng bị giảm tốc.

- [ ] **Step 1: Khai báo kiểu** cho `MediaStreamTrackProcessor` và tùy chọn `selfBrowserSurface` (không có trong `lib.dom`) ngay đầu `screenShare.ts` bằng `declare`/ép kiểu cục bộ, không sửa `tsconfig`.
- [ ] **Step 2: Cài đặt** các export trên theo Interfaces.
- [ ] **Step 3: Kiểm tra tĩnh:** `cd web; npm run build` — Expected: `tsc -b && vite build` thành công; `npm run lint` — Expected: không lỗi mới.
- [ ] **Step 4: Kiểm tra tay bằng bản chạy thật** (không có test JS runner): chạy API (`drownctl serve`) với model/fake như hướng dẫn trong `docs/SETUP.md`, mở Chrome, trong DevTools console gọi tạm module qua `import('/src/screenShare.ts')` ở chế độ `npm run dev` để thử `pickSurface()` → `beginSending(10)`; xác nhận request `POST /api/v1/camera/frame` ra khoảng 10/giây và trả 409 trước khi `start`. Ghi kết quả (không báo "đạt" nếu chưa chạy được).
- [ ] **Step 5: Commit** (nếu được yêu cầu): `git add web/src/screenShare.ts` — `feat: add browser tab frame sender`.

---

### Task 6: Giao diện trang Camera, âm báo và tiêu đề tab

**Files:**
- Modify: `web/src/pages/CameraPage.tsx`, `web/src/overlayData.ts` (thêm `useTitleAlert`; sửa `useAlarmSound` chỉ khi Task 1 yêu cầu), `web/src/pages.css`

**Interfaces:**
- Consumes (Task 5): `useScreenShare`, `screenShareSupported`, `pickSurface`, `beginSending`, `stopShare`; API `POST /api/v1/camera/start` nhận `source: 'browser-tab'` và `fps` (Task 4).
- Produces: `useTitleAlert(active: boolean, text = '⚠ ĐUỐI NƯỚC'): void` trong `web/src/overlayData.ts` — đổi `document.title` luân phiên với tiêu đề gốc mỗi giây khi `active`, khôi phục khi tắt/unmount.

- [ ] **Step 1: Bộ chọn nguồn** trong `CameraPage`: state `mode: 'camera' | 'tab'`, hai nút/segmented "Webcam / URL" và "Quét tab/cửa sổ" (khóa khi `active`). Chế độ `tab`: ẩn ô nhập nguồn và nút "Kiểm tra kết nối"; hiện nút "Chọn tab để quét…" (gọi `pickSurface`), nhãn tab đã chọn (`surfaceLabel`), và dòng hướng dẫn: "Mở video ở một tab khác rồi chọn tab đó. App tự loại trừ chính tab này nên không bị lặp hình." Nếu `!screenShareSupported()` hiện `callout warn` "Trình duyệt này chưa hỗ trợ quét tab. Hãy dùng Chrome hoặc Edge." và khóa nút.
- [ ] **Step 2: Bắt đầu/dừng** ở chế độ `tab`: nút "Bắt đầu" bật khi `share.phase === 'ready'` và `readiness?.ready_for_request`. Handler: gọi `beginSending(10)` **trước**, rồi `await post('/api/v1/camera/start', { source: 'browser-tab', fps: 10, idempotency_key: newKey('camera'), confidence, cell_grid: gridParam })`; nếu `start` lỗi → `stopShare()` rồi để `action` hiện lỗi. Nút "Dừng": `post('/api/v1/camera/stop')` rồi `stopShare()`. Khi `status.active` chuyển sang `false` trong lúc `share.phase` là `sending` (phiên chết phía server) → `stopShare()`.
- [ ] **Step 3: Cảnh báo:** `alerted` hợp nhất `status.alert_track_ids` với `share.alertTracks` (tránh trễ do polling 1 s); `useAlarmSound(sound && alerted.length > 0)` dùng giá trị hợp nhất; gọi `useTitleAlert(alerted.length > 0)`. Hiển thị `share.error` qua `Banner`/`callout` khi có.
- [ ] **Step 4: Nội dung hiển thị khi quét tab:** `monitor-source` hiện "Tab trình duyệt" thay vì `status.source`; thêm dưới thanh trạng thái một dòng `muted`: "Đã gửi {framesSent} khung · ~{sendFps} khung/giây tới model" khi `phase === 'sending'`; thêm `fine-print` cạnh khối khuyến nghị: "Để cảnh báo chắc chắn đến kịp, mở app thành cửa sổ riêng cạnh cửa sổ video (cửa sổ bị che hoàn toàn có thể bị trình duyệt giảm tốc)." Giữ phong cách CSS hiện có (`source-bar`, `callout`, `segmented` nếu đã có; nếu chưa, thêm class nhỏ vào `pages.css`).
- [ ] **Step 5: Kiểm tra tĩnh:** `cd web; npm run build; npm run lint` — Expected: sạch.
- [ ] **Step 6: Kiểm tra tay:** `drownctl serve`, mở trang Camera trong Chrome, chọn tab YouTube, Bắt đầu: thấy luồng MJPEG đã vẽ nhãn, bấm "Dừng chia sẻ" của trình duyệt → phiên dừng và thông báo hiện; tab app ở nền → âm báo/tiêu đề vẫn nháy khi có cảnh báo (dùng pipeline giả hoặc model thật nếu có; ghi rõ cái nào). Không có model/detector thật thì ghi "chưa kiểm tra với model thật".
- [ ] **Step 7: Commit** (nếu được yêu cầu): `git add web/src/pages/CameraPage.tsx web/src/overlayData.ts web/src/pages.css` — `feat: add browser tab source to camera page`. Lưu ý các file này đã có thay đổi chưa commit từ trước: xem `git diff` và hỏi người dùng trước khi gộp.

---

### Task 7: Tài liệu, trạng thái và kiểm tra tổng

**Files:**
- Modify: `IMPLEMENTATION_STATUS.md`, `docs/SETUP.md` (một đoạn ngắn)

- [ ] **Step 1: `docs/SETUP.md`** thêm mục "Quét tab trình duyệt" nêu: dùng Chrome/Edge, chạy trên `localhost`, nguồn `browser-tab`, khung không được lưu, mở app ở cửa sổ riêng khi cần.
- [ ] **Step 2: `IMPLEMENTATION_STATUS.md`** thêm bốn mục tách biệt: code đã viết; test đã chạy (liệt kê lệnh và kết quả thật); kiểm tra thật (kết quả spike Task 1, đo độ trễ chụp → model → hiển thị, "Dừng chia sẻ", tab ẩn — chỉ ghi điều đã đo); người dùng chấp nhận (chưa).
- [ ] **Step 3: Hồi quy đầy đủ:** `.\.venv\Scripts\python -m unittest discover -s tests` và `cd web; npm run build; npm run lint` — Expected: PASS / sạch; báo đúng nếu có test fail.
- [ ] **Step 4: Đo thật** (khi có model đã duyệt + detector): chạy một video YouTube qua tab, ghi số `latency_ms_p50/p95`, `processed_fps` từ trang Camera vào `IMPLEMENTATION_STATUS.md`, kèm ghi chú "đo đường ống, không phải chất lượng model".
- [ ] **Step 5: Commit** (nếu được yêu cầu): `git add IMPLEMENTATION_STATUS.md docs/SETUP.md` — `docs: record browser tab capture status`.
