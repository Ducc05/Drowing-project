# Kế hoạch: chạy test video nhanh hơn (bỏ khung + vẽ khung trên trình duyệt), có tiến độ và nút dừng

Trạng thái: **kế hoạch, chưa triển khai**. Ngày lập: 2026-10-26.

## Context

Test video chạy ở máy bạn bằng CPU rất chậm: video 93 giây (2.795 khung, 720p) chạy 20-40 phút và giao diện không hiện tiến độ. Số đo thật trên 40 khung (đo khi máy đang bận với một job khác, nên số tuyệt đối bị chậm đi, tỉ lệ vẫn đáng tin): **ghi video WebM (VP8) chiếm 78%** thời gian mỗi khung (281 ms), detector YOLO 18% (64 ms), phân loại 2% (7 ms), tracking chỉ 0,1% (0,3 ms).

Hai hướng đã chọn:
- **Cách 2:** bỏ khung, chỉ chạy detector và model mỗi N khung.
- **Cách 3:** không mã hóa lại video; lưu file nhãn JSON và vẽ khung/nhãn lên video gốc ngay trong trình duyệt.

Không xây lại hệ thống; chỉ chỉnh đường chạy video hiện có. Các việc khác (nhãn ổn định theo từng người, hiện phần trăm trên video đã xử lý ở đường camera, augmentation khi train) là kế hoạch riêng, chưa làm ở đây.

## Hiện trạng (đã đọc code)

- `POST /api/v1/test/video` → `LocalMediaTestJobs.submit` (`src/drowning/api/test_jobs.py`) chạy **đồng bộ** trong request: claim ô compute → nạp model → `run_video_test` (`src/drowning/api/inference.py`) → `PersonStateInference.infer_video` (`src/drowning/inference/pipeline.py`) đọc từng khung, `process_frame`, `draw_annotations`, `VideoWriter` VP8, ghi `annotated.webm` + `.detections.json`.
- Kết quả phục vụ qua `GET /api/v1/jobs/{id}/results/{role}` (`app.py:495`, `result_file` chỉ cho file trong thư mục kết quả). Trang Đánh giá (`web/src/pages/EvaluationPage.tsx`) phát `<video src=…/results/annotated>` và đọc `…/results/detections` để dựng đường thời gian.
- Không có tiến độ: chỉ 3 sự kiện (queued, claimed, started). Danh sách job (`app.py` ~dòng 625-640) không trả tiến độ. Hủy chỉ có cho job `queued`.
- Máy trạng thái job (`src/drowning/domain/jobs.py`) đã có `RUNNING → CANCELLATION_REQUESTED → CANCELLED`, dùng lại được, không thêm trạng thái.

## Thiết kế

### 1. Bỏ khung (`frame_stride`) trong `infer_video`

- Thêm tham số `frame_stride` (1-10, mặc định **3**: 30 fps xử lý 10 khung/giây). Vòng đọc: khung không phải bội của stride chỉ gọi `capture.grab()` (giải mã, không đổi màu, không detector, không model); khung bội của stride mới `retrieve()` rồi `process_frame`.
- Chỉ khung đã xử lý có bản ghi; mọi bản ghi giữ `frame_index` và `timestamp_ms` từ container như hiện nay. `ByteTracker` và `TrackAggregator` đã dùng `timestamp_ms` nên chạy được ở tốc độ thấp hơn (EMA theo `dt`, `max_gap_ms` 1.500 ms vẫn bao được khoảng cách 100 ms). `min_frames` đếm khung đã xử lý (5 khung ≈ 0,5 s ở stride 3).
- JSON kết quả thêm `frame_stride`, `processed_frames`, `width`, `height`, `fps_metadata`, `render`. Tham số stride đi qua `run_video_test` và query `frame_stride` của `/api/v1/test/video`; giá trị lưu vào payload job.
- Trang Đánh giá không đổi: `buildSegments` đã lấy `step` là trung vị khoảng cách giữa các bản ghi nên tự co theo stride.

### 2. Không mã hóa lại video (`render="overlay"`)

- Chế độ `overlay` **không tạo `VideoWriter` và không gọi `draw_annotations`**. Kết quả chỉ gồm `detections.json` và bản sao video gốc `source.<đuôi>` ghi vào thư mục kết quả (copy file, không mã hóa). `result_files` gồm `{"source": "source.mp4", "detections": "…json"}`; `result_file(job, "source")` phục vụ bản sao nên đi qua kiểm tra đường dẫn sẵn có, và `_sync_files` tự đẩy cả hai lên Drive như hiện nay.
- **Chỉ dùng overlay khi trình duyệt phát được video gốc:** container `.mp4`/`.webm` và codec đọc được bằng `cv2.CAP_PROP_FOURCC` thuộc `avc1/h264/vp80/vp90/av01`. Khác (mkv, avi, mov, HEVC) thì **tự rơi về chế độ `video` hiện tại** (mã hóa WebM, vẫn dùng stride) và báo `render` thật trong kết quả. Thêm hàm `browser_playable(path)` trong `src/drowning/api/media.py` (đã có kiểm tra giải mã video khi tải lên).
- Giữ nguyên đường `render="video"` cũ (test `test_video_pipeline_writes_annotated_video_and_marks_warmup` vẫn đạt).

### 3. Vẽ khung và nhãn trên trình duyệt (trang Đánh giá)

- Trong `EvaluationPage.tsx`, đặt `<canvas>` phủ lên `<video>` (cùng kích thước hiển thị). Mỗi khung (dùng `requestVideoFrameCallback`, nếu thiếu thì `requestAnimationFrame`) tra bản ghi theo `video.currentTime`: với mỗi `track_id`, lấy hai bản ghi đã xử lý kẹp quanh thời điểm, **nội suy tuyến tính hộp** giữa chúng nếu cách nhau ≤ `max_gap_ms`, ngoài ra giữ hộp cũ tối đa 1 bước stride rồi ẩn. Tọa độ nhân theo `videoWidth/videoHeight` → kích thước canvas.
- Chú thích mỗi người: `ID  nhãn  87%` (phần trăm = điểm của nhãn đang hiển thị, từ `class_scores`; ghi chú "điểm số mô hình, không phải xác suất đã hiệu chuẩn"). Màu theo nhãn dùng bảng màu sẵn có của trang (`labelColor`).
- Video phát từ `…/results/source` khi `render="overlay"`, từ `…/results/annotated` khi `render="video"` (đọc `summary.render`/`files`).
- Hệ quả tốt: không cần mã hóa lại ở bất kỳ stride nào; đổi cách vẽ (kích thước chữ, ẩn/hiện ID) không cần chạy lại model.
- Đường camera trực tiếp (`CameraPage`, MJPEG, `draw_annotations`) không đổi; chỉ thêm phần trăm vào caption `draw_annotations` cho thống nhất.

### 4. Tiến độ và nút dừng

- `infer_video(..., progress=None, should_stop=None)`. Cứ ~20 khung đã xử lý hoặc ~2 giây, gọi `progress({"frame": i, "total": n, "processed": k, "elapsed_s": …, "eta_s": …})`. `submit` truyền một hàm ghi `self._payload(job.id, progress=…)` (ghi DB thưa, không mỗi khung).
- Danh sách job (`app.py` serializer ~dòng 625-640) trả thêm `progress` từ payload. Trang Đánh giá, trong lúc POST đang chờ, **poll `/api/v1/jobs` mỗi 2 giây**, tìm job theo `idempotency_key` mà trang đã tự sinh (`newKey('eval-video')`), hiện thanh tiến độ "đã xử lý x/y khung, còn khoảng z phút" và nút **Dừng**. Không đổi `submit` thành bất đồng bộ.
- Dừng: `POST /api/v1/test/jobs/{id}/stop` đặt cờ trong bộ nhớ của `LocalMediaTestJobs` (`self._stop: set[UUID]`); `infer_video` kiểm tra cờ mỗi khung, raise `InferenceCancelled`; `submit` chuyển `RUNNING → CANCELLATION_REQUESTED → CANCELLED` và nhả ô compute (`_release`). Xóa file `.pending`. Request POST trả 409 `cancelled`. Job đã `completed` thì stop trả "đã xong".
- Nếu backend bị khởi động lại giữa chừng, job `running` mồ côi vẫn do cơ chế phục hồi sẵn có (supervisor, phần phục hồi job local) xử lý; không thay đổi.

## File sẽ sửa

`src/drowning/inference/pipeline.py` (stride, progress, stop, overlay, caption phần trăm), `src/drowning/api/inference.py` (`run_video_test`: chọn `render`, copy `source`), `src/drowning/api/media.py` (`browser_playable`), `src/drowning/api/test_jobs.py` (truyền tham số, progress, cờ dừng, hủy, result role `source`), `src/drowning/api/app.py` (query `frame_stride`, route stop, `progress` trong job view), `web/src/pages/EvaluationPage.tsx` (canvas overlay, tiến độ, nút Dừng, chọn nguồn video), `web/src/api.ts` nếu cần kiểu mới; tests: `tests/test_inference_pipeline.py`, `tests/test_media_api.py`, `tests/test_evaluation.py`.

## Kiểm chứng

1. **Test đơn vị:** stride 3 chỉ xử lý đúng các khung bội 3 và `frame_index/timestamp_ms` đúng; `render="overlay"` không tạo `.webm` và có `source.*`; video không phát được (mkv/avi) rơi về `render="video"`; `progress` được gọi, tăng dần, `eta_s` hợp lý; cờ dừng ngắt giữa chừng, trạng thái `cancelled`, ô compute được nhả, không còn `.pending`; job view có `progress`. Hồi quy đầy đủ `python -m unittest discover -s tests` (đặt `DATABASE_URL=` và `DATABASE_URL_UNPOOLED=` rỗng) vẫn đạt.
2. **Web:** `npm run build` và `npm run lint` sạch lỗi (cảnh báo cũ giữ nguyên).
3. **Đo tốc độ thật** (sau khi job test hiện tại xong và khởi động lại backend): chạy lại `ZoneA_Abovewater.mp4` (237 khung, trước đây 111 giây) và video 93 giây; ghi thời gian, so với bản cũ. Kỳ vọng giảm mạnh nhưng **chưa đo**; báo số thật.
4. **Xem bằng mắt ở trang Đánh giá:** khung/nhãn/phần trăm bám theo người, không giật rõ rệt ở stride 3; thanh tiến độ chạy; bấm Dừng thì ô compute được nhả (kiểm tra tác vụ camera/test kế tiếp chạy được).
5. **Lưu ý vận hành:** sửa code ngay được, nhưng **chỉ khởi động lại backend khi không còn job đang chạy** (kiểm tra ô compute rảnh), vì khởi động lại sẽ cắt job đang chạy.
