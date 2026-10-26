# Việc của riêng bạn trước khi gửi cho bạn của bạn

Tài liệu cho **bạn (chủ dự án)**. Bạn của bạn đọc `docs/HUONG_DAN_BAN.md` (cài đặt, điền `.env`, cách dùng từng trang). File này liệt kê bạn phải làm và gửi gì.

## Tóm tắt: dùng chung gì, riêng gì

Theo kế hoạch của bạn, bạn của bạn dùng **giống bạn** mọi thứ, trừ tài khoản Kaggle:

| Thứ | Dùng chung hay riêng | Ghi chú |
|---|---|---|
| Mã nguồn, cấu hình, bản duyệt `reviews\` | chung (qua git) | |
| PostgreSQL Neon | **chung** | họ cần chuỗi kết nối |
| Thư mục Google Drive | **chung** | chia sẻ thư mục cho Gmail của họ |
| Ứng dụng Google (OAuth client) | **chung** | gửi file `google-oauth-client.json` |
| Đăng nhập Google Drive (token) | **riêng**, mỗi người một token | xem lý do bên dưới |
| Dataset Kaggle `vuduc792005/alanoud-pool` | **chung** | chia sẻ dataset cho tài khoản Kaggle của họ |
| Tài khoản Kaggle, token Kaggle, notebook, hạn mức GPU | **riêng** | họ đổi `KAGGLE_KERNEL_REF` thành tên của họ |
| Detector `yolo11n.pt` | tự tải | `drownctl detector fetch` |

### Vì sao không dùng chung "nick" Google Drive (không gửi `google-token.json`)

Token Drive của hệ thống có quyền `drive` đầy đủ, nghĩa là cầm token là đọc/sửa/xóa được **toàn bộ Drive của bạn**, không chỉ thư mục dự án. Cách đúng là chia sẻ **thư mục** cho Gmail của họ (Editor) và để họ tự đăng nhập bằng Gmail của họ. Họ vẫn thấy cùng checkpoint và báo cáo. Nếu bạn vẫn muốn gửi token, hãy coi như đưa mật khẩu Drive của bạn.

## Bước 0. Mã nguồn trên git (đã làm cho bạn)

Cấu trúc git đã được sửa: **thư mục gốc của repo là `D:\AI\DrowningProject`** (không còn là `D:\AI`). Remote: `https://github.com/Ducc05/Drowing-project.git`, nhánh `master`. `.gitignore` nằm ở gốc mới và đã gộp từ hai file cũ.

- Repo cũ ở `D:\AI` được sao lưu nguyên vẹn trong `D:\AI\_git-backup-old-root\` (thư mục `dot-git` và `root.gitignore`). Xóa được khi bạn chắc chắn không cần.
- `codex-cli-handoff` (tài liệu đặc tả) **không còn nằm trong repo**, vì nó ở ngoài thư mục dự án. Nó vẫn nằm trên máy bạn tại `D:\AI\codex-cli-handoff`. Bạn của bạn không cần nó để chạy hệ thống.
- Không file bí mật nào được theo dõi: `.env`, `.env.*` (trừ `.env.example`), `secrets\`, `.neon`, `*.pt`, `*.mp4`, `*.zip`, `local-cache\`, thư mục job Kaggle đều bị `.gitignore` chặn.

Mỗi lần sau muốn đẩy thêm:
```powershell
cd D:\AI\DrowningProject
git status                  # đọc kỹ cả "Untracked" lẫn "Changed"
git add -A
git commit -m "mô tả thay đổi"
git push
```
**Không được thấy** trong danh sách: `.env`, `.env.bak`, `.env.pre-neon.bak`, `secrets/`, `local-cache/`, `.venv/`, `node_modules/`, file `.pt`. Thấy thì dừng, không commit. Nếu lỡ commit một bí mật: xóa file đi **không** xóa nó khỏi lịch sử git; phải đổi bí mật đó (đổi mật khẩu Neon, tạo lại token).

Repo phải là **Private**: GitHub → repo → Settings → General → Danger Zone → "Change visibility" đang ở Private. Mời bạn: Settings → Collaborators → Add people.

## Bước 1. Chuỗi kết nối Neon (gửi kênh riêng)

Gửi giá trị `DATABASE_URL_UNPOOLED` trong `.env` của bạn qua **tin nhắn riêng** (không đăng nhóm, không đưa lên git). Họ dán vào `.env` của họ.

Khuyến nghị an toàn hơn (tùy chọn): trong Neon console → project → **Roles** → tạo một role riêng cho bạn của bạn, cấp quyền trên database dự án, và đưa chuỗi của role đó. Khi cần thu hồi quyền chỉ phải xóa role đó, không phải đổi mật khẩu của bạn. Lúc đó chủ dự án vẫn giữ chuỗi gốc.

Thu hồi sau này: Neon console → Roles → Reset password.

Họ **không** chạy `db upgrade`. Chỉ bạn chạy khi đổi cấu trúc database.

## Bước 2. Google Drive

1. Mở Drive → chuột phải thư mục `DrowningProject` (thư mục có ID bạn điền ở `GOOGLE_DRIVE_FOLDER_ID`) → **Share** → thêm Gmail của họ → quyền **Editor**.
2. Google Cloud Console → **APIs & Services** → **OAuth consent screen** (hoặc *Audience*) → **Test users** → **Add users** → thêm đúng Gmail đó. Nếu không, họ gặp `403 access_denied` như bạn đã gặp.
3. Gửi riêng file `secrets\google-oauth-client.json` (định danh ứng dụng, không phải mật khẩu Google của bạn, nhưng đừng đăng công khai).
4. Gửi họ `GOOGLE_DRIVE_FOLDER_ID` (đoạn cuối URL thư mục).

Họ tự chạy `auth drive` và đăng nhập bằng Gmail của họ.

## Bước 3. Kaggle

1. kaggle.com → Datasets → `alanoud-pool` → **Settings/Share** → thêm tên người dùng Kaggle của họ với quyền xem (Viewer). Kiểm tra: họ mở được `https://www.kaggle.com/datasets/vuduc792005/alanoud-pool`.
2. Họ phải xác minh số điện thoại trên Kaggle và tự tạo token API.
3. Họ đặt `KAGGLE_KERNEL_REF=<tên-Kaggle-của-họ>/drowning-temporal` và `KAGGLE_DATASET_SOURCES=vuduc792005/alanoud-pool`.
4. Hạn mức GPU Kaggle tính theo từng tài khoản; job của họ dùng hạn mức của họ.

Điều này thay đổi cách cũ ("chỉ bạn chạy Kaggle"): giờ **mỗi người chạy Kaggle bằng tài khoản của mình**, vẫn chỉ có **một job compute trên toàn hệ thống** nhờ khóa trong PostgreSQL chung. Hai người không chạy song song.

## Bước 4. Những thứ gửi qua kênh riêng (tổng kết)

**Gói có sẵn:** `D:\AI\gui-cho-ban.zip` gồm các thứ riêng tư bên dưới đã điền sẵn (chuỗi database, ID thư mục Drive, tên dataset, file OAuth), hướng dẫn `DOC_TRUOC_TIEN.md` (đặt file ở đâu, test thế nào) và script kiểm tra. Gửi file zip này qua kênh riêng, dặn họ xóa sau khi cài. Bạn vẫn phải làm các việc trên web ở Bước 0 đến 3 (mời GitHub, chia sẻ Drive, Test users, chia sẻ dataset Kaggle). Nếu bạn đổi chuỗi database hoặc ID Drive sau này, nhờ mình đóng gói lại.

| Gửi | Cách gửi |
|---|---|
| `DATABASE_URL_UNPOOLED` | tin nhắn riêng |
| `google-oauth-client.json` | tin nhắn riêng / file riêng |
| `GOOGLE_DRIVE_FOLDER_ID` | tin nhắn thường được |
| Lời mời GitHub, chia sẻ Drive, chia sẻ dataset Kaggle, Test users | làm trên web, không phải gửi file |

**Tuyệt đối không gửi:** `.env`, `.env.bak`, `.env.pre-neon.bak`, `secrets\google-token.json`, `~\.kaggle\access_token`.

## Bước 5. Cùng kiểm tra sau khi họ cài xong

Cách nhanh: nhờ họ chạy `.\.venv\Scripts\python scripts\kiem_tra_cai_dat.py` và gửi ảnh chụp kết quả (script không in bí mật; dòng cuối phải là `0 loi`).

Hoặc từng lệnh riêng, gửi **ảnh chụp** (không có bí mật trong đầu ra):
```powershell
.\.venv\Scripts\python -m drowning.cli doctor --online
```
Phải thấy `Thiếu bắt buộc: không`. Sau đó cả hai cùng mở trang Train: phải thấy **cùng danh sách job** (vì dùng chung PostgreSQL). Bạn có thể xem bên đó kết quả `prepare_only` bạn đã chạy.

## Bước 6. Khi bạn thay đổi hệ thống sau này

- Đổi mã nguồn: commit và push. Họ `git pull`, cài lại, `npm run build`, chạy `db check`.
- Đổi cấu trúc database: bạn chạy `db upgrade` **một lần**, rồi báo họ `git pull`.
- Duyệt thêm nhãn/nhóm: commit thư mục `reviews\` để họ có cùng trạng thái duyệt (bản duyệt không nằm trong database).
- Duyệt model ứng viên (trang Đánh giá) là quyết định chung: thống nhất ai duyệt.

## Điều bạn cần biết về giới hạn khi dùng chung

- Tiến trình nền (gửi, theo dõi, thu kết quả job Kaggle) chạy trong `serve` **của máy đã tạo job**. Ai tạo job thì máy đó phải mở `serve` để theo dõi; job vẫn chạy trên Kaggle nếu tắt máy và được nối lại khi mở lại.
- Báo cáo chi tiết của một lần chạy nằm ở máy đã thu kết quả (`local-cache\results`) và gói trên Drive. Máy kia thấy bản tóm tắt trong database.
- Ảnh mẫu ở trang Dữ liệu cần dataset gốc trên máy đó (`ALANOUD_V4_DATA_ROOT`). Train và chạy thử không cần.
- Đã kiểm tra thật: PostgreSQL Neon, Drive, Kaggle trên **tài khoản của bạn**. Chưa kiểm tra thật với tài khoản Kaggle/Gmail thứ hai; lần đầu bạn của bạn cài có thể lộ lỗi nhỏ, hãy nhờ họ chụp màn hình lỗi.

## Nếu họ gặp lỗi

Xem bảng cuối `docs/HUONG_DAN_BAN.md` (mục 14).
