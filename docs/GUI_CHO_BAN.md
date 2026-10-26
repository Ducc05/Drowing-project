# Việc cần gửi cho bạn của bạn (làm SAU khi hệ thống của bạn chạy ổn)

Chưa cần làm gì trong file này cho đến khi bạn tự thấy hệ thống chạy ổn.

## Khi nào nên gửi (điều kiện)

Gửi khi **cả bốn** mục sau đúng:
- [ ] `drownctl doctor --online` không còn dòng `[THIẾU]`
- [ ] Bạn đã chạy `prepare_only` ít nhất một lần trên Kaggle và xem báo cáo
- [ ] Bạn đã duyệt ánh xạ nhãn của Alanoud (hoặc quyết định chưa dùng nó)
- [ ] Bạn thấy trang Dữ liệu, Train, Đánh giá, Camera mở được và không lỗi

## Phần 1. Thứ bạn gửi cho bạn

| Gửi gì | Cách gửi | Ghi chú |
|---|---|---|
| Mã nguồn dự án | **Git repo PRIVATE** (mục "Đưa lên git" bên dưới), hoặc zip | Nếu dùng zip, tạo bằng lệnh bên dưới. **Không** kèm `.env`, `secrets`, `.venv`, `node_modules`, `local-cache` |
| Thư mục `reviews\` | nằm trong zip | Bản duyệt nhãn/nhóm của bạn, để bạn của bạn thấy cùng trạng thái duyệt |
| `DATABASE_URL_UNPOOLED` của Neon | **kênh riêng** (tin nhắn riêng, không đăng nhóm) | Chứa mật khẩu. Họ điền vào `.env` của họ |
| Quyền Editor vào thư mục Drive `DrowningProject` | Drive → Share | Họ dùng chung checkpoint và báo cáo |
| Email Gmail của họ thêm vào Test users | Google Cloud Console của bạn | Nếu không, họ gặp lỗi 403 `access_denied` như bạn đã gặp |
| File detector `yolo11n.pt` (nếu họ chạy thử video/camera) | cùng cách với zip | Hoặc họ tự tải |

**Tuyệt đối không gửi:** file `.env`, `secrets\google-token.json`, `secrets\google-oauth-client.json`, `~\.kaggle\access_token`, `.env.bak`, `.env.pre-neon.bak`. Đó là thông tin đăng nhập của riêng bạn.

## Đưa lên git (khuyến nghị: repo PRIVATE)

Trong repo có tên tài khoản Kaggle, ID project Neon và đường dẫn máy bạn. Đó không phải mật khẩu nhưng bạn nên để repo **private** và chỉ mời đúng bạn của bạn.

```powershell
cd D:\AI\DrowningProject
git init
git add .
git status          # đọc kỹ danh sách file TRƯỚC khi commit
```
Danh sách phải **không** có: `.env`, `.env.bak`, `.env.pre-neon.bak`, `secrets/`, `local-cache/`, `.venv/`, `node_modules/`, file `.pt`. (`.env.example` thì được.) Nếu thấy một file bí mật trong danh sách thì dừng lại và báo mình, đừng commit.
```powershell
git commit -m "Hệ thống phát hiện đuối nước"
```
Trên github.com tạo repo mới, chọn **Private**, không tick tạo README. Rồi chạy các lệnh GitHub hiện ra (`git remote add origin ...`, `git push -u origin main`). Mời bạn của bạn ở **Settings → Collaborators**. Họ chỉ cần `git clone`.

Lưu ý: nếu lỡ commit một bí mật thì xóa file đi sau đó **không** xóa được nó khỏi lịch sử git; phải đổi bí mật đó (đổi mật khẩu Neon, tạo lại token). Vì vậy hãy kiểm tra `git status` trước khi commit đầu tiên.

Tạo zip mã nguồn (PowerShell), nếu không dùng git:
```powershell
cd D:\AI
$exclude = @('.venv','node_modules','local-cache','secrets','.neon','__pycache__','.env','.env.bak','.env.pre-neon.bak','dist')
$files = Get-ChildItem DrowningProject -Recurse -File -Force | Where-Object {
  $p = $_.FullName; -not ($exclude | Where-Object { $p -match [regex]::Escape("\$_\") -or $p.EndsWith("\$_") }) }
Compress-Archive -Path $files.FullName -DestinationPath D:\AI\drowning-share.zip -Force
```
Mở zip kiểm tra: không thấy `.env`, `secrets`, `.venv` trước khi gửi.

## Phần 2. Bạn của bạn cần làm trên máy họ

1. Cài Python 3.11 và Node.js. Giải nén zip, mở PowerShell trong thư mục dự án.
2. Tạo môi trường và cài thư viện:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python -m pip install -e ".[api,drive,vision,ml]"
   cd web; npm ci; npm run build; cd ..
   ```
3. Tạo `.env` từ mẫu: `copy .env.example .env`, rồi điền **chỉ** các dòng sau:
   ```
   DATABASE_URL_UNPOOLED=<chuỗi bạn gửi qua kênh riêng>
   GOOGLE_DRIVE_FOLDER_ID=<cùng ID thư mục Drive của bạn>
   PERSON_DETECTOR_WEIGHTS=<đường dẫn file .pt trên máy họ>
   ```
   Để trống `WORKER_ID` (tự sinh theo máy). `TEST_DATABASE_URL` để trống, họ không cần chạy test PostgreSQL thật.
4. Cấp Drive bằng tài khoản Gmail của họ: đặt `google-oauth-client.json` (cùng file bạn tải) vào `secrets\`. Bạn có thể gửi riêng file này cho họ, vì đó là định danh ứng dụng, không phải mật khẩu của bạn. Rồi chạy:
   ```powershell
   .\.venv\Scripts\python -m drowning.cli auth drive
   ```
5. Kiểm tra: `.\.venv\Scripts\python -m drowning.cli db check` (**không** chạy `db upgrade`) và `.\.venv\Scripts\python -m drowning.cli doctor --online`.
6. Chạy: `.\.venv\Scripts\python -m drowning.cli serve` rồi mở `http://127.0.0.1:8000`.

## Phần 3. Họ làm được gì và không làm được gì

- **Làm được:** xem model và job chung, tải model đã duyệt về máy (kiểm tra SHA-256), chạy thử ảnh/video/camera, đánh dấu đoạn đúng/sai, duyệt dữ liệu.
- **Không nên làm:** chạy bước Kaggle (prepare/smoke/train) từ máy họ. Notebook `vuduc792005/drowning-temporal` thuộc tài khoản Kaggle của bạn, nên họ cần token Kaggle của bạn mới đẩy được. **Chỉ bạn chạy Kaggle**, để một nơi giữ ô compute.
- Chỉ một tác vụ compute chạy trên toàn hệ thống: nếu bạn đang train thì máy họ không chạy thử model được cho đến khi xong (hệ thống báo bận, không tự chạy song song).
- Bản duyệt nhãn nằm trong thư mục `reviews\`, **không** nằm trong database. Nếu bạn duyệt thêm sau này, gửi lại thư mục `reviews\` cho họ.

## Nếu họ gặp lỗi

| Lỗi | Nguyên nhân thường gặp |
|---|---|
| `403 access_denied` khi `auth drive` | Gmail của họ chưa nằm trong Test users |
| `db check` báo không kết nối | Sai chuỗi, thiếu `?sslmode=require`, hoặc mạng chặn |
| `schema chưa ở bản mới nhất` | Bạn đã cập nhật mã nguồn và đổi schema; bạn chạy `db upgrade`, họ cập nhật mã nguồn |
| Drive: không có quyền ghi | Bạn chưa chia sẻ thư mục Drive cho họ quyền Editor |
