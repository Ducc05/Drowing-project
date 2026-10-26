# Hướng dẫn cài đặt và sử dụng (dành cho người dùng chung hệ thống)

Tài liệu này dành cho bạn của chủ dự án. Đọc hết một lượt rồi làm theo thứ tự. Mọi lệnh chạy trong **PowerShell** trên Windows.

## 0. Hệ thống này là gì, chạy ở đâu

Hệ thống phát hiện **đuối nước** từ ảnh/video/camera. Mỗi người trong khung hình được gán một trong ba nhãn:

| Nhãn | Nghĩa |
|---|---|
| `drowning` | người có dấu hiệu đuối nước |
| `swimming` | người hoạt động bình thường trong nước |
| `out_of_water` | người ở ngoài nước |

Phân việc giữa các nơi:

| Việc | Chạy ở đâu |
|---|---|
| **Train** model (GPU) | Kaggle, tài khoản Kaggle **của chính bạn** |
| Lưu danh sách model, job, ô compute chung | PostgreSQL chung (Neon), mọi người cùng thấy |
| Lưu file model (checkpoint), báo cáo | Google Drive chung (một thư mục) |
| Chạy thử model trên ảnh/video/camera, xem giao diện | **máy của bạn** |

Quy tắc quan trọng:
- **Không có train trên máy bạn.** Train thật chỉ chạy trên Kaggle. Máy bạn chỉ chạy thử model đã train.
- **Chỉ một tác vụ compute (train hoặc chạy thử) cùng lúc trên toàn hệ thống.** Nếu người khác đang chạy, bạn sẽ thấy "bận" và phải đợi, hệ thống không chạy song song.
- Model mới train xong chỉ là **ứng viên (candidate)**. Nó chỉ thành model được duyệt khi có người chạy thử video thật và bấm duyệt.
- Con số độ chính xác chỉ có ý nghĩa khi đo trên video thật của bạn (trang Đánh giá). Hệ thống không hứa trước độ chính xác.

## 1. Cài phần mềm (một lần)

1. **Python 3.11** (tải từ python.org, tick *Add to PATH*). Kiểm tra: `py -3.11 --version`.
2. **Node.js 20 trở lên** (nodejs.org). Kiểm tra: `node --version`.
3. **Git** (git-scm.com). Kiểm tra: `git --version`.

## 2. Lấy mã nguồn

Chủ dự án mời bạn vào repo GitHub **private**. Chấp nhận lời mời trong email GitHub, rồi:

```powershell
cd D:\
git clone https://github.com/Ducc05/Drowing-project.git DrowningProject
cd D:\DrowningProject
```
(Bạn có thể clone vào thư mục khác; thư mục clone chính là thư mục gốc của dự án. Mọi lệnh bên dưới chạy trong thư mục đó.)

## 3. Cài thư viện và dựng giao diện

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[api,drive,vision,ml,detector,kaggle,dev]"
cd web
npm ci
npm run build
cd ..
```
Bước này tải khá nhiều (PyTorch vài trăm MB), mất vài phút. Nếu `pip` báo lỗi mạng, chạy lại đúng lệnh đó.

## 4. Tạo file `.env` và điền

`.env` là file cấu hình **riêng của bạn**, nằm ngoài git, **không bao giờ gửi cho ai và không commit**.

```powershell
copy .env.example .env
notepad .env
```

Chỉ cần điền các dòng dưới đây, các dòng khác để nguyên:

| Biến | Điền gì | Lấy ở đâu |
|---|---|---|
| `DATABASE_URL_UNPOOLED` | chuỗi kết nối PostgreSQL chung | chủ dự án gửi qua **tin nhắn riêng**. Dạng `postgresql://...neon.tech/...?sslmode=require` (hệ thống tự đổi sang driver đúng) |
| `GOOGLE_DRIVE_FOLDER_ID` | ID thư mục Drive chung | chủ dự án gửi. Là đoạn cuối của URL `drive.google.com/drive/folders/<ID>` |
| `KAGGLE_KERNEL_REF` | `<tên-Kaggle-của-BẠN>/drowning-temporal` | tên người dùng Kaggle của bạn (xem ở trang hồ sơ Kaggle). Notebook sẽ được tạo ở tài khoản của bạn |
| `KAGGLE_DATASET_SOURCES` | `vuduc792005/alanoud-pool` | dataset private của chủ dự án; chủ phải đã chia sẻ cho bạn (mục 6) |
| `PERSON_DETECTOR_WEIGHTS` | `./local-cache/models/yolo11n.pt` | tự tải ở mục 7 (lệnh tự điền dòng này nếu còn trống) |
| `ALANOUD_V4_DATA_ROOT` | thư mục chứa dataset Alanoud trên máy bạn | **tùy chọn**: chỉ cần nếu bạn muốn xem ảnh mẫu ở trang Dữ liệu. Train và chạy thử không cần |

Để **trống**: `DATABASE_URL`, `TEST_DATABASE_URL`, `WORKER_ID` (hệ thống tự đặt tên máy bạn).
Lưu ý: nếu chủ dự án gửi cả `DATABASE_URL` thì điền luôn, không sao; nhưng chỉ `DATABASE_URL_UNPOOLED` là đủ.

Không sửa: `KAGGLE_NOTEBOOK_PRIVATE=true`, `ALLOW_LOCAL_TRAIN=false`, `TRAIN_DEFAULT_EXECUTOR=kaggle`, `MAX_ACTIVE_COMPUTE_JOBS=1`.

## 5. Google Drive (đăng nhập một lần)

Cùng một ứng dụng Google và cùng một thư mục Drive với chủ dự án; bạn đăng nhập bằng **Gmail của bạn**.

Điều kiện (chủ dự án làm trước): Gmail của bạn đã nằm trong *Test users* của ứng dụng, và thư mục Drive đã chia sẻ cho Gmail đó với quyền **Editor**.

1. Chủ dự án gửi riêng cho bạn file `google-oauth-client.json`. Lưu nó vào thư mục `secrets\` (tạo thư mục nếu chưa có): `D:\AI\DrowningProject\secrets\google-oauth-client.json`.
2. Chạy:
   ```powershell
   .\.venv\Scripts\python -m drowning.cli auth drive
   ```
3. Trình duyệt mở ra. Chọn Gmail của bạn. Google có thể hiện *"Google hasn't verified this app"*: bấm **Advanced** → **Go to ... (unsafe)** → **Continue/Allow**. Đây là app thử nghiệm của chủ dự án nên màn hình này bình thường.
4. Xong sẽ có file `secrets\google-token.json` (token của riêng bạn, không gửi cho ai).

Nếu gặp `403 access_denied`: Gmail của bạn chưa nằm trong Test users. Nhờ chủ dự án thêm rồi thử lại.

## 6. Kaggle (tài khoản của bạn)

1. Đăng ký/đăng nhập kaggle.com. **Xác minh số điện thoại** (Settings → Phone verification). Không xác minh thì không bật được GPU và internet cho notebook.
2. Tạo token: kaggle.com → ảnh đại diện → **Settings** → mục **API** → tạo token mới. Lưu nội dung token vào file:
   `C:\Users\<tên-bạn>\.kaggle\access_token` (tạo thư mục `.kaggle` nếu chưa có; file không có đuôi). Nếu Kaggle cho tải file `kaggle.json` thay vì token chuỗi, đặt file đó vào cùng thư mục `.kaggle`.
3. Chủ dự án chia sẻ dataset `alanoud-pool` cho tài khoản Kaggle của bạn (Viewer). Bạn kiểm tra bằng cách mở `https://www.kaggle.com/datasets/vuduc792005/alanoud-pool` khi đã đăng nhập: phải thấy được.
4. Kaggle tính hạn mức GPU **theo từng tài khoản** (khoảng 30 giờ mỗi tuần). Job của bạn dùng hạn mức của bạn. Notebook `<tên-bạn>/drowning-temporal` được hệ thống tự tạo ở lần chạy đầu và luôn **private**.

## 7. Tải detector người (một lần)

Detector tìm người trong ảnh trước khi model phân loại. Hệ thống không tự tải ngầm; chạy lệnh rõ ràng này:

```powershell
.\.venv\Scripts\python -m drowning.cli detector fetch
```
Lệnh tải `yolo11n.pt` (khoảng 5,6 MB), kiểm tra SHA-256 cố định, lưu vào `local-cache\models\` và điền `PERSON_DETECTOR_WEIGHTS` nếu còn trống.

## 8. Kiểm tra mọi thứ đã sẵn sàng

```powershell
.\.venv\Scripts\python -m drowning.cli db check
.\.venv\Scripts\python -m drowning.cli doctor --online
```
- `db check` phải báo kết nối được và schema đúng phiên bản. **Không chạy `db upgrade`**: việc đó chỉ chủ dự án làm khi đổi cấu trúc.
- `doctor --online` in từng mục `[ OK ]` hoặc `[THIẾU]` kèm cách sửa. Kết thúc phải là `Thiếu bắt buộc: không`. Lệnh không in giá trị bí mật.

Nếu có `[THIẾU]`, làm theo gợi ý ngay dưới dòng đó rồi chạy lại.

## 9. Chạy hệ thống

```powershell
.\.venv\Scripts\python -m drowning.cli serve
```
Mở `http://127.0.0.1:8000`. Giữ cửa sổ PowerShell này mở khi dùng. Bên trong nó có **tiến trình nền** tự gửi job lên Kaggle, theo dõi và thu kết quả. **Job bạn tạo chỉ được theo dõi khi chương trình này đang chạy trên máy bạn.** Tắt máy giữa chừng thì job vẫn chạy trên Kaggle, và khi bạn mở lại, hệ thống nối lại theo dõi.

Khi chỉnh sửa giao diện (hiếm): `cd web; npm run dev` rồi mở địa chỉ Vite in ra.

## 10. Cách dùng từng trang

### Tổng quan
Cho biết cấu hình đã đủ chưa, job gần đây và model đang có. Không bấm gì ở đây.

### Dữ liệu
Danh sách nguồn dữ liệu (hiện có Alanoud: ảnh có sẵn nhãn). **Bạn không phải gán nhãn.** Bước duyệt ở đây chỉ để *xác nhận*, và **hoàn toàn tùy chọn**: bỏ qua thì vẫn train được, chỉ là nguồn được ghi là "chưa xác nhận" trong báo cáo và hồ sơ model.
- Nếu muốn xem: bấm chuẩn bị tài liệu duyệt (cần `ALANOUD_V4_DATA_ROOT`), xem ảnh mẫu từng nhãn, quy tắc tự động và nhóm ảnh trùng.
- Có thể nhờ một AI duyệt theo quy tắc: xem `docs/LUONG_DUYET.md`.

### Train (3 bước, chạy trên Kaggle)
Làm lần lượt từ trên xuống. Mỗi bước là một nút **Xếp vào hàng đợi**; tiến trình nền tự gửi lên Kaggle khi ô compute rảnh. Trang tự cập nhật trạng thái, không cần tải lại.

1. **prepare_only**: Kaggle đọc dataset, chuyển sang định dạng chuẩn, kiểm tra lỗi, đếm lớp. Xem *báo cáo lần gần nhất* (số ảnh, số khung, lớp nào ít). Nếu một bước đã đang chạy, nút không tạo thêm job mới.
2. **remote_smoke**: kiểm tra GPU bằng dữ liệu giả. Phải thấy "GPU xác nhận: có". Bước này **không** tạo model.
3. **train_candidate**: train thật.
   - Chế độ duy nhất hiện dùng là **chế độ ảnh**: nếu có nguồn video có nhãn thì video được cắt thành khung ảnh và train trước, rồi mới tới ảnh rời. Hiện chỉ có dữ liệu ảnh nên chỉ có giai đoạn ảnh.
   - Tích **"Tôi yêu cầu train một phiên bản mới"** và nhập lý do (ít nhất 10 ký tự).
   - Tùy chọn: số epoch khung video (train trước), số epoch ảnh (train sau), và khởi tạo encoder bằng trọng số ImageNet (cần internet trên Kaggle, được ghi vào hồ sơ model).
   - Nếu **ai đó đã train đúng cùng dữ liệu, nhãn, cấu hình, mã nguồn, tùy chọn và seed**, hệ thống trả về model đó thay vì train lại: "same_training_exists". Đây là chủ ý, để mọi người dùng cùng một model.
   - Train mất từ vài chục phút đến vài giờ. Khi xong, file model được đẩy lên Drive (có kiểm tra SHA-256) và ghi vào PostgreSQL.

Hủy job: chỉ hủy được khi job **chưa** lên Kaggle. Job đã lên Kaggle phải dừng bằng web Kaggle (Kaggle CLI không có lệnh hủy); hệ thống giữ ô compute cho đến khi thấy nó dừng.

### Đánh giá
Nơi quyết định model có dùng được không.
1. Chọn model ứng viên → **tải về máy** (kiểm tra SHA-256).
2. Tải video của bạn lên và chạy. Kết quả là video đã vẽ khung và nhãn cho từng người (kèm số ID do tracker gán), cùng đường thời gian nhãn theo từng người.
3. **Đánh dấu đúng/sai** từng đoạn. Các dấu này chính là bằng chứng đo độ chính xác thật.
4. Duyệt hoặc từ chối. Chỉ model được duyệt mới thành model dùng chung.

Cách model dùng video ở chế độ ảnh: mỗi khung hình, model nhìn từng người riêng lẻ; tracker gán một ID cho mỗi người xuyên suốt các khung; điểm số được làm mượt theo ID đó, và chỉ báo động khi điểm "đuối nước" giữ cao đủ lâu. Hạn chế cần biết: model **không hiểu chuyển động**; tracker đổi ID thì lịch sử làm mượt của người đó bắt đầu lại; các tham số làm mượt là giá trị khởi điểm chưa được chỉnh trên video có nhãn.

### Camera
Bấm Start/Stop để chạy model đã duyệt trên webcam hoặc luồng `http/https/rtsp/rtmp`. Chỉ dùng được khi đã có model được duyệt và đã tải về máy. Thông tin đăng nhập trong URL luồng bị che khi hiển thị.

## 11. Cập nhật khi chủ dự án đổi mã nguồn

```powershell
git pull
.\.venv\Scripts\python -m pip install -e ".[api,drive,vision,ml,detector,kaggle,dev]"
cd web; npm ci; npm run build; cd ..
.\.venv\Scripts\python -m drowning.cli db check
```
Nếu `db check` báo schema cũ: nhờ chủ dự án chạy `db upgrade` (chỉ một người làm việc này), rồi kiểm tra lại. Khởi động lại `serve`.

Cập nhật mã nguồn làm "mã băm code" đổi, nên yêu cầu train sau đó có thể không còn trùng với lần train trước. Đó là bình thường.

## 12. Những điều không được làm

- Không gửi, đăng hay commit: `.env`, `secrets\*`, `google-token.json`, `~\.kaggle\access_token`, chuỗi `DATABASE_URL`.
- Không sửa, di chuyển hay xóa dữ liệu gốc (`sources/` và thư mục dataset). Hệ thống chỉ đọc chúng.
- Không bật train trên máy bạn, không đổi notebook Kaggle sang public.
- Không chạy `db upgrade`.
- Bản duyệt nhãn nằm trong thư mục `reviews\` của git, không nằm trong database. Nếu bạn duyệt thêm, commit và đẩy thư mục đó để người khác thấy.

## 13. Báo cáo chi tiết và dữ liệu cục bộ

Báo cáo chi tiết của một lần chạy được lưu ở máy **đã chạy lệnh** (`local-cache\results`) và bản gói được đẩy lên Drive. Trên máy khác, trang Train hiển thị phần tóm tắt trong database; muốn xem đầy đủ hãy mở file trên Drive hoặc nhờ người chạy gửi.

## 14. Lỗi thường gặp

| Hiện tượng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `403 access_denied` khi `auth drive` | Gmail chưa có trong Test users | nhờ chủ dự án thêm |
| Drive báo không có quyền ghi | thư mục chưa chia sẻ Editor | nhờ chủ dự án chia sẻ |
| `db check` không kết nối | sai chuỗi, thiếu `?sslmode=require`, mạng chặn cổng 5432 | kiểm tra lại chuỗi; thử mạng khác |
| `schema chưa ở bản mới nhất` | chủ dự án đã đổi cấu trúc | `git pull`, nhờ chủ chạy `db upgrade` |
| doctor: dataset Kaggle không truy cập được | dataset chưa chia sẻ cho tài khoản Kaggle của bạn | nhờ chủ dự án chia sẻ |
| Kaggle 403 khi đẩy notebook | `KAGGLE_KERNEL_REF` không phải tên Kaggle của bạn | sửa thành `<tên-bạn>/drowning-temporal` |
| Notebook không bật được GPU/internet | chưa xác minh số điện thoại | xác minh trong Settings Kaggle |
| Nút Train bị khóa | có mục chặn hiển thị ngay trên nút | đọc dòng màu cam, làm theo |
| Trang báo "bận" | người khác đang train/chạy thử | đợi họ xong; hệ thống không chạy song song |
| `serve` báo cổng 8000 đang dùng | đã có một `serve` khác đang chạy | đóng cái cũ, hoặc đặt `API_PORT` khác trong `.env` |
| Lệnh in lỗi font tiếng Việt | console Windows cũ | bản mới đã tự xử lý; nếu vẫn lỗi, chạy `chcp 65001` trước |
| Kaggle báo trạng thái lạ, job ở `remote_unconfirmed` | dạng trạng thái chưa biết | **không** bấm chạy lại; chụp màn hình trang Train gửi chủ dự án |

## 15. Thuật ngữ

- **candidate (ứng viên)**: model vừa train, chưa ai duyệt.
- **approved (đã duyệt)**: model được người chạy thử và bấm duyệt; là model dùng chung.
- **checkpoint**: file trọng số model, lưu trên Drive, bản ghi trỏ tới nó nằm trong PostgreSQL kèm mã băm để kiểm chứng.
- **ô compute**: khóa toàn hệ thống đảm bảo một tác vụ nặng chạy một lúc.
- **advisory (tư vấn)**: chế độ duyệt dữ liệu hiện tại; duyệt là tùy chọn, không chặn train.
