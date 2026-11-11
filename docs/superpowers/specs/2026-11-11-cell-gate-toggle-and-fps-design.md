# Công tắc "Tự chia vùng", chỉnh số ô và hiện fps video

## Ý định
Lớp lọc theo ô lưới (`inference/cells.py`, commit 0ea5c31) hiện chỉ đổi được qua `H20SAVER_CELL_GRID` trong `.env`. Người dùng cần bật/tắt và chỉnh số cột x hàng ngay trên web theo từng lượt chạy, và xem tốc độ model gán nhãn video (fps).

## Mục tiêu
- Công tắc "Tự chia vùng" + hai ô số cột/hàng (1-32) trên trang Đánh giá; áp dụng cho test video và lúc bắt đầu phiên camera.
- Tham số `cell_grid=6x4|off` đi qua API, dùng lại `parse_cell_grid`; không gửi thì giữ cấu hình hiện tại.
- Giá trị lưới lưu cùng kết quả; lưới vẽ trên video theo đúng lượt chạy; `Idempotency-Key` trùng mà khác lưới bị từ chối (như confidence).
- Tiến trình video có thêm `fps` = processed / elapsed_s; trình xem trực tiếp hiện "Tốc độ model: X khung/giây"; tóm tắt kết quả giữ fps trung bình, fps gốc và frame_stride.

## Không làm
- Không đổi lớp lọc chuyển động, không đổi cách model chạy, không hiện fps cho camera.
- Không ghi fps như kết quả đánh giá chất lượng model.

## Cách làm (đã chọn)
Đi theo đúng đường của `confidence`: query param -> `resolve_*` -> lưu trong payload job -> truyền vào `load_approved_pipeline(cell_grid=...)`. fps lấy từ `progress` đã có trong `inference/pipeline.py`.

## Để người xây quyết
Tên trường, vị trí hiển thị cụ thể, kiểm tra đầu vào ở UI.
