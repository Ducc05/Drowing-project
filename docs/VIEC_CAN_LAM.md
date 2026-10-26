# Việc cần làm để hoàn thiện hệ thống

Cập nhật: 2026-10-26. Kiểm tra bất cứ lúc nào (không in giá trị bí mật):

```powershell
cd D:\AI\DrowningProject
.\.venv\Scripts\python -m drowning.cli doctor --online
```

Chi tiết cấu hình từng dịch vụ: `docs/SETUP.md`. Thêm dataset: `docs/DATASET_ONBOARDING.md`.
Không dán token, mật khẩu hay `DATABASE_URL` vào chat; chỉ điền vào `.env` hoặc đặt file đúng chỗ.

## Đã xong

- [x] Kaggle: CLI, token (`~/.kaggle/access_token`), `KAGGLE_KERNEL_REF=vuduc792005/drowning-temporal`, `KAGGLE_DATASET_SOURCES=vuduc792005/alanoud-pool` (truy cập được)
- [x] PostgreSQL trên máy này đã kết nối và tạo bảng
- [x] Dữ liệu Alanoud (`ALANOUD_V4_DATA_ROOT`), PyTorch, OpenCV, ultralytics
- [x] `.env` và `.env.example` đã xếp theo nhóm (`.env.bak` là bản cũ, chứa mật khẩu: xóa khi thấy `.env` ổn)

## Việc 1. Google Drive (ĐÃ XONG, `doctor --online` báo truy cập được thư mục)

- [x] 1.1 `console.cloud.google.com` → New project `drowning` → APIs & Services → Library → bật **Google Drive API**
- [x] 1.2 OAuth consent screen: **External**, thêm Gmail của bạn (và người dùng chung) vào **Test users**
- [x] 1.3 Credentials → Create credentials → OAuth client ID → **Desktop app** → Download JSON → đổi tên `google-oauth-client.json` → đặt vào `D:\AI\DrowningProject\secrets\`
- [x] 1.4 `drive.google.com` → tạo thư mục `DrowningProject` → lấy ID sau `folders/` trong URL → điền `.env`: `GOOGLE_DRIVE_FOLDER_ID=<ID>`; chia sẻ thư mục cho người dùng chung (quyền Editor)
- [x] 1.5 `.\.venv\Scripts\python -m drowning.cli auth drive` (trình duyệt mở, chọn đúng Gmail; nếu có cảnh báo "unverified" chọn Advanced → Go to ... (unsafe))
- [x] 1.6 `doctor --online`: dòng "Truy cập được thư mục Drive" phải `[ OK ]`

## Việc 2. PostgreSQL dùng chung (ĐÃ XONG trên Neon: project frosty-flower-01574171, nhánh production)

Còn lại: gửi `DATABASE_URL_UNPOOLED` (hoặc `DATABASE_URL`) cho người dùng chung qua kênh riêng; họ chỉ chạy `db check`. Ứng dụng ưu tiên `DATABASE_URL_UNPOOLED` nếu có.

- [x] 2.1 `neon.tech` → Create project → Connect → copy connection string
- [x] 2.2 Đổi đầu chuỗi `postgresql://` thành `postgresql+psycopg://`, giữ `?sslmode=require`
- [x] 2.3 Điền vào `.env` dòng `DATABASE_URL=` (giữ `TEST_DATABASE_URL` trỏ về PostgreSQL trên máy bạn)
- [x] 2.4 `.\.venv\Scripts\python -m drowning.cli db upgrade` rồi `db check` (một máy chạy `upgrade`, máy khác chỉ `check`)
- [ ] 2.5 Gửi `DATABASE_URL` cho người kia qua kênh riêng; họ điền `.env`, chạy `db check` và tự chạy `auth drive` bằng Gmail của họ

## Việc 3. Detector người (ĐÃ XONG: `drownctl detector fetch` đã tải yolo11n.pt và điền `.env`)

- [x] 3.1 Tải `yolo11n.pt` từ `github.com/ultralytics/assets/releases` (mục Assets)
- [x] 3.2 Chép vào `D:\AI\DrowningProject\local-cache\models\`
- [x] 3.3 Điền `.env`: `PERSON_DETECTOR_WEIGHTS=D:\AI\DrowningProject\local-cache\models\yolo11n.pt`
- [ ] 3.4 Kiểm tra chất lượng trên video của bạn ở trang Camera/Đánh giá (chưa được kiểm chứng)

## Việc 4. Duyệt dữ liệu Alanoud (chỉ bạn làm được)

- [ ] 4.1 `.\.venv\Scripts\python -m drowning.cli serve` → mở `http://127.0.0.1:8000` → trang **Dữ liệu**
- [ ] 4.2 Xem ảnh mẫu từng nhãn (Drowning, Swimming, out of water). Báo cáo đã cho thấy: `out of water` chỉ 3,3% số khung; 512 cụm ảnh gần giống nhau nằm xuyên các tập gốc; nhiều crop mờ hoặc có vẻ gán sai
- [ ] 4.3 Nếu nhãn đáng tin: điền tên → **Duyệt ánh xạ này**. Nếu không đáng tin thì đừng duyệt và nói cho mình biết
- [ ] 4.4 Xác nhận nhóm ảnh trùng (cần viết cơ sở ít nhất 20 ký tự). Hệ thống vẫn ghi "chưa xác minh tách theo video nguồn"

## Việc 5. Chạy thử các bước Kaggle (khi Việc 4 xong thì mở khóa tiếp)

- [ ] 5.1 (ĐANG CHẠY trên Kaggle: job 1c0be65f, bắt đầu 06:24 UTC; giữ backend bật đến khi "hoàn tất") Trang **Train** → bước 1 `prepare_only` (chạy được ngay, không cần Drive) → xem báo cáo
- [ ] 5.2 Bước 2 `remote_smoke` (xác nhận GPU) — cần Việc 4 xong
- [ ] 5.3 Lần chạy Kaggle thật đầu tiên có thể lộ lỗi chưa lường trước (định dạng trạng thái, chờ GPU): chụp trang Train hoặc dán thông báo lỗi cho mình

## Việc 6. Dữ liệu video có nhãn (bắt buộc trước khi train nhánh temporal)

- [ ] 6.1 Tìm dataset video có **ID người và nhãn theo thời gian** (CVAT video, hoặc xuất ra `reviewed_jsonl`)
- [ ] 6.2 Đặt vào thư mục riêng, điền `.env`: `POOL_VIDEO_DATA_ROOT=` hoặc `USER_VIDEO_DATA_ROOT=`
- [ ] 6.3 Viết `configs/datasets/<tên>.yaml` theo `docs/DATASET_ONBOARDING.md` (nhờ mình viết giúp khi có dataset)
- [ ] 6.4 Tải dataset lên Kaggle (Private), thêm vào `KAGGLE_DATASET_SOURCES` (cách nhau bằng dấu phẩy)
- [ ] 6.5 Duyệt ánh xạ và nhóm của dataset mới ở trang Dữ liệu, rồi chạy lại `prepare_only`

## Việc 7. Train và đưa vào dùng (sau cùng)

- [ ] 7.1 Trang Train → bước 3 `train_candidate` (tích "phiên bản mới", ghi lý do)
- [ ] 7.2 Trang **Đánh giá**: tải model ứng viên về máy → chạy thử video của bạn → đánh dấu đoạn đúng/sai
- [ ] 7.3 Duyệt hoặc từ chối model; model đã duyệt mới dùng được ở trang Camera

## Biết trước giới hạn

- Chưa có lần chạy thật nào với Drive, PostgreSQL ở dịch vụ ngoài, GPU Kaggle, camera vật lý.
- Chưa có model nào được train. `train_candidate` bị chặn có chủ ý cho đến khi có video có nhãn và dữ liệu đã duyệt.
- Nhiều người dùng chung: mọi người dùng cùng checkpoint trên Drive (kiểm tra SHA-256), yêu cầu train trùng hệt không chạy lại. Train lại trên GPU Kaggle chỉ nên kỳ vọng giống hệt khi cùng loại GPU.
