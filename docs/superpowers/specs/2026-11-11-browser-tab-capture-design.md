# Quét tab/cửa sổ trình duyệt như camera trực tiếp

## Ý định
Người dùng muốn mở video đuối nước (ví dụ YouTube) trên chính laptop đang chạy app, rồi để app xử lý nó như một luồng live: khung nhận diện, nhãn, cảnh báo đuối nước có âm báo, giống khi upload video và giống trang Camera. Dùng để demo/thử model với video có sẵn mà không cần camera thật.

## Mục tiêu
- Trang Camera có thêm nguồn "Quét tab/cửa sổ trình duyệt". Người dùng chọn đúng tab (hoặc cửa sổ) cần quét trong hộp thoại của trình duyệt.
- Khung hình của tab được gửi lên server, đi qua đúng `CameraSession` hiện có: detector, tracker, H20Saver, lọc ô lưới, ngưỡng confidence, luồng MJPEG đã vẽ nhãn, banner và âm báo.
- Không tự quét chính trang app (không gương lặp): `selfBrowserSurface: 'exclude'`.
- Vẫn một phiên camera mỗi lần, vẫn giữ ô compute chung, vẫn lưu job `inference` (modality camera) và release slot khi dừng hoặc khi nguồn bị đứt.
- Cảnh báo vẫn đến được khi tab app không nằm phía trước (xem "Rủi ro: tab app bị ẩn").

## Không làm (non-goal)
- Không chạy model trong trình duyệt; không thêm WebRTC.
- Không quét cả màn hình bằng server (cách 2 đã bị loại: không chọn riêng được tab, bị gương lặp).
- Không hỗ trợ Firefox/Safari (thiếu `MediaStreamTrackProcessor`); hiện thông báo rõ ràng. Chỉ Chrome/Edge.
- Không lưu khung hình lên đĩa, DB hay Drive. Chỉ giữ khung mới nhất trong RAM.
- Không quét âm thanh. Không thay đổi cách model/alert quyết định đuối nước.
- Không báo kết quả này là chất lượng model: video nén lại qua trình duyệt + JPEG, chỉ là kiểm tra đường ống.

## Anti-goal
- Tính năng "chạy được" nhưng cảnh báo trễ hàng chục giây hoặc im lặng khi người dùng đang xem tab YouTube.
- Cho phép ai đó đẩy ảnh vào server khi không có phiên browser-tab đang chạy.

## Ràng buộc từ dự án (CLAUDE.md / AGENTS.md)
- Chỉ model đã duyệt; một compute job toàn hệ thống; owner/token fencing; không log/lưu secret.
- Không thêm đường train cục bộ. Test dùng fixture/giả lập, không mô tả như kết quả model.
- Cập nhật `IMPLEMENTATION_STATUS.md`, tách rõ "code đã viết / test đã chạy / kiểm tra thật / người dùng chấp nhận".
- Đường dẫn Unicode-safe (không liên quan trực tiếp, nhưng đừng đưa path vào payload).

## Cách làm (đã chọn)
Hai cách được xem xét và bị loại: server tự chụp màn hình (không chọn tab, gương lặp); chạy model trong trình duyệt (model là Python/PyTorch). Chọn: **trình duyệt chụp tab, đẩy JPEG lên server, server coi đó là một "camera"**.

### Server
1. `inference/camera.py`
   - Nguồn đặc biệt `browser-tab`: `parse_camera_source("browser-tab")` trả `("browser-tab", "browser-tab")`. Webcam/URL giữ nguyên.
   - Lớp `PushCapture` có đúng giao diện `cv2.VideoCapture` mà `CameraSession` dùng: `isOpened()`, `read()`, `release()`, `get(CAP_PROP_FPS)`.
     - `push(frame_bgr)` giữ khung mới nhất (khung cũ bị thay, không xếp hàng) và đánh thức `read()`.
     - `read()` chờ khung *mới* tối đa 1 giây; hết giờ trả `(False, None)`. Lần đọc đầu tiên (lúc `start`) chờ tối đa 20 giây vì model đang tải. `CameraSession` đã có sẵn cơ chế: 30 lần đọc hỏng liên tiếp thì phiên `failed` và slot được giải phóng. Đó chính là cách xử lý khi trình duyệt đóng/ngừng gửi.
     - `get(CAP_PROP_FPS)` trả fps danh định mà trình duyệt gửi (mặc định 10) để `configure_live_stream` đặt đúng nhịp phiếu bầu.
     - `release()` đánh dấu đóng; `read()` sau đó trả hỏng.
2. `api/camera.py` (`CameraService`)
   - `start("browser-tab", ...)`: tạo `PushCapture` **trước khi tải model** và lưu vào `self._push` (không cần giữ `self._lock`); truyền `capture_factory=lambda _: push` cho `CameraSession`. Payload job thêm `"capture": "browser_tab"`.
   - `push_frame(jpeg_bytes)`: giải mã bằng `cv2.imdecode`; từ chối nếu không có phiên browser-tab đang chạy/đang khởi động, ảnh hỏng, kích thước khung khác khung đầu tiên, hoặc cạnh dài > 1920. Trả `{"alert_track_ids": [...], "frames_processed": n}` lấy từ `CameraSession.summary()`.
   - `probe("browser-tab")` trả 422 ("nguồn này không cần kiểm tra kết nối").
   - `stop()`/finalize: dùng nguyên đường hiện có; `_push` được xóa trong `_forget()`.
3. `api/app.py`
   - `POST /api/v1/camera/frame`: body thô `image/jpeg`, tối đa 2 MB (413 nếu vượt), 409 nếu chưa có phiên browser-tab, 422 nếu ảnh sai, 200 JSON như trên. Không cần `Idempotency-Key`.
   - `CameraStartRequest.source` nhận `"browser-tab"`; thêm `fps` tùy chọn (1–30, mặc định 10) chỉ dùng cho nguồn này.

### Web
4. `web/src/screenShare.ts` (module singleton, nằm ngoài component để đổi trang không làm dừng gửi)
   - `startScreenShare()` gọi `navigator.mediaDevices.getDisplayMedia({video: {frameRate: 15}, audio: false, selfBrowserSurface: 'exclude'})`.
   - Vòng lặp async đọc `VideoFrame` từ `MediaStreamTrackProcessor` (không dùng `setInterval`/`requestAnimationFrame`, nên không bị Chrome giảm tốc khi tab app ẩn). Bỏ khung theo timestamp để giữ ~10 khung/s; vẽ lên `OffscreenCanvas` có kích thước cố định theo khung đầu tiên (co về tối đa 1280 px cạnh dài, giữ tỉ lệ, letterbox nếu cửa sổ nguồn đổi kích thước) → `convertToBlob('image/jpeg', 0.8)` → `fetch POST /camera/frame`. Chỉ gửi khung kế tiếp sau khi khung trước đã được trả lời (tự bỏ khung nếu model chậm, tránh tích trễ).
   - Khi `track` kết thúc (người dùng bấm "Dừng chia sẻ" của trình duyệt) → gọi `POST /camera/stop` và báo trên giao diện.
   - Trong lúc `start` còn đang tải model, server trả 409 cho khung; bộ gửi thử lại mỗi 200 ms cho đến khi `start` xong hoặc lỗi.
   - Phát sự kiện/callback cho trang Camera: `alert_track_ids` từ phản hồi `POST /camera/frame`, lỗi, trạng thái.
5. `web/src/pages/CameraPage.tsx`
   - Bộ chọn nguồn: "Webcam / URL" | "Quét tab/cửa sổ". Chế độ sau ẩn ô nhập nguồn, có nút "Chọn tab để quét…" rồi "Bắt đầu". Dùng lại nguyên ngưỡng confidence và lưới ô.
   - Âm báo lấy trạng thái cảnh báo từ phản hồi `POST /camera/frame` (không phụ thuộc polling `setInterval` 1 s vốn bị giảm tốc khi tab ẩn); polling 1 s vẫn giữ để cập nhật số liệu khi tab hiện.
   - Hiện `document.title` nhấp nháy "⚠ ĐUỐI NƯỚC" khi có cảnh báo để người dùng thấy trên thanh tab.
   - Chuỗi tiếng Việt, đúng phong cách các trang khác.

### Rủi ro: tab app bị ẩn
Người dùng xem YouTube ở tab khác, nên tab app chạy nền. Biện pháp: vòng gửi không dựa vào timer; cảnh báo đến theo phản hồi của chính request gửi khung; âm báo dùng `AudioBufferSourceNode` lặp thay vì `setInterval` (hoặc giữ cách cũ nếu chạy tốt). Khuyến nghị trên giao diện: mở app thành cửa sổ riêng cạnh cửa sổ video (cửa sổ bị che hoàn toàn cũng bị Chrome giảm tốc). Task đầu tiên của kế hoạch là một spike đo thực tế điều này trên Chrome; kết quả quyết định phần còn lại.

## Để người xây quyết
Tên hàm/biến, kích thước giới hạn chính xác, định dạng thông báo lỗi, vị trí đặt hàm trong file, cách dựng giả lập `MediaStreamTrackProcessor` khi thử (không có test JS runner).

## Cách kiểm
- Unit: `PushCapture` (khung mới nhất thắng, `read` chờ/hết giờ, release), `parse_camera_source`, `push_frame` (từ chối đúng các trường hợp), `CameraService.start` với nguồn browser-tab dùng pipeline giả, phiên chết khi ngừng gửi khung và slot được giải phóng, route `/camera/frame` (200/409/413/422).
- Web: `npm run build` và `npm run lint` (không có test JS).
- Kiểm tra thật (tách riêng, ghi vào `IMPLEMENTATION_STATUS.md`): trên Chrome với YouTube thật, đo độ trễ đường chụp → model → hiển thị, kiểm tra cảnh báo khi tab app ẩn, kiểm tra "Dừng chia sẻ". Chỉ khi chạy được mới báo là đạt; chất lượng model không được suy ra từ đây.
