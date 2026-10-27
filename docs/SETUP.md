# Cấu hình PostgreSQL chung, Google Drive và Kaggle

Mọi giá trị bí mật chỉ nằm trong file `.env` (đã bị bỏ qua khỏi git) hoặc file đăng nhập riêng của từng dịch vụ.
Hệ thống không in giá trị bí mật ra màn hình, log hay báo cáo. Kiểm tra bất cứ lúc nào:

```powershell
.\.venv\Scripts\python -m drowning.cli doctor            # thiếu gì, cách sửa (không in giá trị)
.\.venv\Scripts\python -m drowning.cli doctor --online   # thử kết nối thật tới PostgreSQL, Drive, dataset Kaggle
```

## Bảng biến `.env` bạn cần điền

| Biến | Bắt buộc cho | Bạn điền gì |
|---|---|---|
| `DATABASE_URL` | mọi thứ dùng job/registry | URL PostgreSQL **dùng chung** (mục 1) |
| `TEST_DATABASE_URL` | chỉ khi chạy test PostgreSQL thật | URL database **riêng** để test, không bao giờ trùng database dùng chung |
| `WORKER_ID` | không | Để trống: hệ thống tự sinh `host-<mã máy>`. Nếu đặt, mỗi máy một tên khác nhau |
| `GOOGLE_DRIVE_FOLDER_ID` | xuất bản model, lưu báo cáo/kết quả | ID thư mục Drive chung (mục 2) |
| `GOOGLE_OAUTH_CLIENT_SECRET_FILE` | chỉ lúc đăng nhập Drive lần đầu | đường dẫn file OAuth client JSON |
| `GOOGLE_OAUTH_TOKEN_FILE` | dùng Drive | nơi lưu token (mặc định `./secrets/google-token.json`) |
| `KAGGLE_KERNEL_REF` | chạy Kaggle | `tên-kaggle/tên-notebook`, ví dụ `dung123/drowning-temporal` (mục 3) |
| `KAGGLE_DATASET_SOURCES` | chạy Kaggle | các dataset **private** gắn vào notebook, `owner/slug`, cách nhau bằng dấu phẩy |
| `KAGGLE_KERNEL_DIR` | không | thư mục chứa kernel tạo cho từng job (mặc định `./notebooks/kernels`) |
| `KAGGLE_NOTEBOOK_PRIVATE` | không | để `true`; hệ thống từ chối push notebook công khai |
| `SUPERVISOR_ENABLED`, `SUPERVISOR_INTERVAL_SECONDS` | không | tiến trình nền theo dõi job; mặc định bật, 30 giây |
| `PERSON_DETECTOR_WEIGHTS` | chạy thử ảnh/video/camera ở máy | file trọng số detector người (ví dụ YOLO `.pt`) |
| `APPROVED_MODEL_BUNDLE_DIR` | chạy thử | nơi lưu model đã tải về (mặc định `./local-cache/models/approved`) |
| `ALANOUD_V4_DATA_ROOT` (và các `*_DATA_ROOT` khác) | duyệt/convert dữ liệu ở máy này | thư mục dataset gốc, chỉ đọc |

Không cần điền: `TRAIN_DEFAULT_EXECUTOR` (luôn `kaggle`), `ALLOW_LOCAL_TRAIN` (luôn `false`; train thật chỉ chạy trên Kaggle).

## 1. PostgreSQL dùng chung

Mọi máy dùng **một** PostgreSQL để cùng thấy model, job và ô compute chung (chỉ một job Train/Test chạy toàn hệ thống).

1. Chọn nơi chạy PostgreSQL (≥ 14) mà mọi máy truy cập được: dịch vụ có sẵn (Neon, Supabase, Render, RDS…), một VPS tự cài, hoặc máy trong cùng mạng LAN.
2. Tạo database `drowning` và một người dùng riêng, ví dụ `drowning_app`, chỉ có quyền trên database đó. Tạo thêm database `drowning_test` **ở nơi khác hoặc trên máy bạn** cho test.
3. Điền `.env`:
   ```
   DATABASE_URL=postgresql+psycopg://drowning_app:MẬT_KHẨU@HOST:5432/drowning?sslmode=require
   ```
   - Mật khẩu có ký tự đặc biệt (`@ : / ? #`) phải mã hóa URL (ví dụ `@` → `%40`).
   - Database ở máy khác thì giữ `sslmode=require` (`doctor` sẽ cảnh báo nếu thiếu).
4. Máy đầu tiên chạy migration một lần: `drownctl db upgrade`. Các máy sau chỉ cần `drownctl db check`.
5. Mỗi máy: `pip install -e ".[api]"`, rồi `drownctl serve`. Mỗi máy chạy backend và web riêng, cùng dùng database này.

Giới hạn đã biết: chưa kiểm tra trên PostgreSQL ở dịch vụ ngoài hay nhiều máy thật. Hai tiến trình trên một máy với PostgreSQL local đã được test.

## 2. Google Drive (lưu model và kết quả)

1. Google Cloud Console → tạo project → bật **Google Drive API**.
2. OAuth consent screen: loại External, thêm email của bạn (và của người dùng chung) vào Test users.
3. Credentials → Create credentials → **OAuth client ID** → loại **Desktop app** → tải JSON, lưu thành `secrets/google-oauth-client.json`.
4. Trên Drive tạo thư mục (ví dụ `DrowningProject`), chia sẻ cho mọi tài khoản dùng chung với quyền **Editor**. ID là phần cuối URL `drive.google.com/drive/folders/<ID>` → `GOOGLE_DRIVE_FOLDER_ID`.
5. `pip install -e ".[drive]"` rồi `drownctl auth drive` (mở trình duyệt một lần, ghi token vào `GOOGLE_OAUTH_TOKEN_FILE`). Mỗi máy/người làm một lần.
6. `drownctl doctor --online` kiểm tra tài khoản có quyền thêm file vào thư mục.

Lưu ý: hệ thống xin quyền `drive` đầy đủ vì cần ghi vào thư mục có sẵn của bạn. Token là bí mật như mật khẩu; không chia sẻ, không đưa vào repo.

## 3. Kaggle

1. Tài khoản Kaggle đã xác minh số điện thoại (cần để bật GPU và internet trong notebook).
2. `pip install kaggle`. Kaggle → Settings → API → tạo token, lưu `kaggle.json` vào `%USERPROFILE%\.kaggle\` (hoặc đặt `KAGGLE_USERNAME`/`KAGGLE_KEY`).
3. Tải dữ liệu lên Kaggle dạng **dataset private** (Create → New Dataset → upload thư mục dataset gốc, giữ nguyên cấu trúc; không sửa dữ liệu trên máy bạn). Điền `KAGGLE_DATASET_SOURCES=ten-kaggle/ten-dataset` (nhiều dataset: cách nhau dấu phẩy).
4. Chọn tên notebook: `KAGGLE_KERNEL_REF=ten-kaggle/drowning-temporal`. Lần đẩy đầu tiên Kaggle tạo notebook **private** với GPU và internet bật (internet cần để cài thư viện nhỏ và tải trọng số ImageNet nếu bạn chọn).
5. Notebook tự tìm dataset đã gắn trong `/kaggle/input` theo `data.yaml`/annotation của từng nguồn; nếu thấy nhiều thư mục khớp, nó **dừng và báo** thay vì chọn đại.
6. `drownctl doctor --online` kiểm tra CLI, đăng nhập và quyền truy cập từng dataset.

Đã kiểm tra với Kaggle CLI 2.2.5: `kernels push -t` giới hạn thời gian chạy, `kernels status` và `kernels output` nhận `owner/kernel/version`, `datasets status` có sẵn.
Chưa kiểm chứng với tài khoản thật (cần theo dõi lần chạy đầu tiên):
- Kaggle CLI không có lệnh hủy kernel. Job đã tới Kaggle chỉ hủy được trên web Kaggle; hệ thống giữ ô compute cho đến khi thấy trạng thái đã dừng.
- Dạng dòng trạng thái do CLI in ra: hệ thống nhận vài dạng đã biết, dạng lạ thì giữ job ở `remote_unconfirmed` thay vì đoán.
- Token kiểu mới: dán vào file `%USERPROFILE%\.kaggleccess_token` (hoặc biến `KAGGLE_API_TOKEN`).

## Thứ tự chạy lần đầu

1. `drownctl doctor` → điền hết mục bắt buộc → `drownctl db upgrade` → `drownctl serve` → mở `http://127.0.0.1:8000`.
2. Trang **Dữ liệu**: bấm *Chuẩn bị tài liệu duyệt* → xem ảnh mẫu, duyệt ánh xạ nhãn → xem báo cáo chất lượng → xác nhận nhóm ảnh trùng.
3. Trang **Train**: bước 1 *prepare_only* (chạy trên Kaggle, xem báo cáo) → bước 2 *remote_smoke* (xác nhận GPU) → bước 3 *train_candidate* (chỉ khi có video có nhãn theo người và thời gian).
4. Trang **Đánh giá**: tải model ứng viên về máy, chạy thử video của bạn, đánh dấu đoạn đúng/sai, rồi mới duyệt hoặc từ chối.
5. Trang **Camera**: Start/Stop với model đã duyệt.

## Chuẩn bị tăng dần theo từng nguồn

Mỗi nguồn prepare thành công được lưu kèm checksum dữ liệu, cấu hình nguồn (bao gồm ánh xạ nhãn và nhóm), recipe, code xử lý và phiên bản thư viện. Lượt prepare sau đưa các snapshot báo cáo đã kiểm chứng từ output cũ vào notebook, rồi đối chiếu checksum của dữ liệu hiện tại trên Kaggle.

- Nguồn mới hoặc có thay đổi: chạy chuyển đổi, audit, phân tích chất lượng và tạo ảnh mẫu.
- Nguồn không đổi: dùng lại báo cáo và ảnh mẫu, giữ nguyên thông tin đã/chưa duyệt; không chạy lại các bước xử lý trên.
- Snapshot lỗi, thiếu file, checksum sai hoặc không còn archive đã kiểm chứng trên máy: chuẩn bị lại nguồn đó.
- Báo cáo cũ chưa có `prepare-cache.json` phải chuẩn bị lại một lần để tạo mốc đối chiếu. Không tự xác nhận dữ liệu cũ chỉ vì tên hoặc số phiên bản trong YAML không đổi.

Web đánh dấu mỗi nguồn là **chuẩn bị mới** hoặc **dùng lại kết quả đã kiểm chứng**. Việc đọc file để đối chiếu checksum vẫn diễn ra; cache này tái sử dụng báo cáo prepare, không kế thừa trọng số model và không thay thế các bước tạo input/split/crop trong lượt train.

Cache được khôi phục từ archive prepare đã tải và kiểm chứng trên máy đang gửi job. Máy chưa có archive sẽ chuẩn bị lại; chưa có cơ chế tự tải cache từ Drive ở bước gửi prepare.

## Checkpoint dùng chung và kết quả giống hệt nhau giữa nhiều người

PostgreSQL **không chứa file checkpoint** (file `.pt` nặng và bất biến); nó chứa bản ghi trỏ tới file và các mã băm để kiểm chứng. File nằm trên Google Drive.

Mỗi phiên bản model trong bảng `model_versions` kèm `artifacts` có: ID và URL file trên Drive, SHA-256 của gói, SHA-256 của trọng số, kích thước, mã băm dữ liệu/recipe/code đã dùng, tùy chọn train, seed, phiên bản thư viện, kernel Kaggle và số phiên bản. Xem ở trang Đánh giá hoặc `GET /api/v1/models/{id}/checkpoint`.

Ba cơ chế để mọi người ra cùng kết quả:
1. **Cùng một checkpoint:** máy nào cũng tải đúng file ghi trong bản ghi và chỉ dùng khi SHA-256 và danh tính model khớp. Đây là cách đảm bảo chắc chắn.
2. **Không train lại thứ đã train:** mỗi yêu cầu `train_candidate` có một mã `training_identity` (băm của dữ liệu, ánh xạ nhãn, recipe, code, tùy chọn, seed). Yêu cầu trùng hệt trả về job/model đã có thay vì train lại. Mã băm code không phụ thuộc máy hay giờ sửa file.
3. **Train có tính tất định:** seed cố định, thuật toán tất định, xáo dữ liệu có seed. Test cục bộ xác nhận hai lần train cùng đầu vào cho trọng số giống hệt từng bit, đổi seed thì khác. Trên GPU Kaggle chỉ nên kỳ vọng giống hệt khi cùng loại GPU và cùng phiên bản thư viện; Kaggle không cho chọn GPU, nên nếu cần kết quả chắc chắn hãy dùng cơ chế 1 và 2.

## Train tiếp từ model chính

1. Đăng ký dataset mới trong `configs/datasets/`, khai báo ánh xạ nhãn và gắn dataset private vào `KAGGLE_DATASET_SOURCES`. Chạy lại bước chuẩn bị dữ liệu và kiểm tra GPU khi dữ liệu thay đổi.
2. Ở Tổng quan, đồng bộ model đã duyệt về máy. Chỉ model chính hiện tại được dùng làm gốc; model phải cùng loại và cùng hợp đồng tiền xử lý/kiến trúc với chế độ train.
3. Ở Train, yêu cầu phiên bản mới, nhập lý do, chọn **Train tiếp model chính**, rồi đánh dấu các dataset dùng trong lượt này. Các dataset không chọn sẽ không được đưa vào training inputs. Chưa có form thêm nguồn dataset trực tiếp trên web.
4. Job ghim ID và checksum của model gốc, lưu bản sao manifest/trọng số riêng. Khi dispatch, hệ thống tạo một dataset checkpoint **private** trên Kaggle dành riêng cho job, chờ dataset sẵn sàng rồi mới gửi notebook. Notebook chỉ chứa ID/checksum, đọc trọng số từ dataset gắn vào và kiểm tra lại trước khi tạo optimizer mới. Không đưa token Drive vào notebook. Việc duyệt model khác sau khi tạo job không đổi checkpoint của job đã xếp hàng. Nếu upload checkpoint thất bại hoặc chưa sẵn sàng trong thời gian chờ, job thất bại và nhả ô compute; kernel chưa được gửi. Dataset của một upload bị ngắt có thể còn trên tài khoản Kaggle để người dùng dọn sau.
5. Kết quả là candidate mới, có `parent_model_version`, `parent_weights_sha256` và `initialization=approved_checkpoint` trong hồ sơ. Model chính đang dùng chỉ đổi khi người dùng duyệt candidate mới.

Đây là train tiếp từ trọng số đã học: epoch, optimizer và trạng thái ngẫu nhiên bắt đầu một lượt mới. Gói output hiện tại chưa lưu trạng thái đầy đủ để resume đúng epoch bị ngắt.

API `POST /api/v1/train/submit` nhận thêm `continue_from_approved` (mặc định false) và `source_ids` (null: tất cả nguồn đã đăng ký; danh sách không rỗng: chỉ các nguồn được chọn). Không dùng `pretrained=true` cùng với `continue_from_approved=true`. Preflight trả `capabilities` để giao diện chặn thao tác nếu backend còn chạy phiên bản cũ chưa hỗ trợ các tùy chọn này. Sau khi cập nhật backend, chỉ khởi động lại lúc job hiện tại đã hoàn tất.

## Dùng chung PostgreSQL cho nhiều người
1. Tạo PostgreSQL ở nơi cả hai người truy cập được. Cách nhanh: dịch vụ miễn phí như Neon (neon.tech): tạo project, lấy *connection string*.
2. Đổi đầu chuỗi thành `postgresql+psycopg://` và giữ `?sslmode=require`. Điền cùng một `DATABASE_URL` vào `.env` của mỗi máy. `TEST_DATABASE_URL` giữ trỏ tới database cục bộ riêng.
3. Một máy chạy `drownctl db upgrade`; các máy khác chạy `drownctl db check`.
4. Mỗi người tự cấp Drive cho mình (`drownctl auth drive`) và được chia sẻ cùng thư mục Drive với quyền Editor.
