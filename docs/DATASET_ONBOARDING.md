# Thêm dataset đã có nhãn

Bạn không phải tự gán nhãn. Bạn tìm dataset đã gán nhãn; hệ thống đọc, kiểm tra và báo cáo, còn bạn chỉ xem ảnh mẫu thật
rồi xác nhận nghĩa của nhãn và cách nhóm dữ liệu. Dữ liệu gốc luôn chỉ đọc.

## Hai loại dữ liệu và việc chúng dùng được

| Loại dataset | Dùng để | Không dùng để |
|---|---|---|
| Ảnh rời có khung và nhãn (YOLO, COCO, Pascal VOC) | train **nhánh ảnh** | train nhánh temporal (không có ID người theo thời gian) |
| Video có khung, **ID người** và nhãn theo thời gian (CVAT video, `reviewed_jsonl`) | train **nhánh temporal** | |
| Video chưa có nhãn / ảnh không có khung | thử model, không train | train có giám sát |

Train ứng viên cần cả hai: ít nhất một nguồn ảnh và một nguồn video có nhãn theo người và thời gian, đủ ba lớp
(`drowning`, `swimming`, `out_of_water`) ở cả tập train và validation sau khi gộp các nguồn đã duyệt.

## Cách thêm một nguồn

1. Đặt dữ liệu ở một thư mục riêng và điền biến `*_DATA_ROOT` tương ứng trong `.env` (chỉ để duyệt ở máy này).
2. Tạo file `configs/datasets/<ten>.yaml` (mẫu bên dưới). `mapping` liệt kê từng **tên nhãn gốc** → nhãn chuẩn; nhãn không rõ nghĩa
   để `action: review`, không đoán.
3. Trang **Dữ liệu** → *Chuẩn bị tài liệu duyệt*. Xem báo cáo chất lượng, ảnh mẫu từng nhãn, duyệt ánh xạ, xác nhận nhóm.
4. Tải dataset lên Kaggle, thêm vào `KAGGLE_DATASET_SOURCES`, rồi chạy các bước ở trang **Train**.

Tên nhãn dễ hiểu nhầm (`normal`, `person`, `not drowning`, `standing`, `floating`, `diving`…) chỉ được ánh xạ khi bạn viết định nghĩa
bằng lời cho nhãn đó. Đổi ánh xạ thì bản duyệt cũ tự mất hiệu lực.

## Mẫu descriptor

Ảnh YOLO:
```yaml
schema_version: 1
source_id: ten_nguon
source_version: "1"
format: yolo_detection
modality: image
paths: {root_env: TEN_NGUON_DATA_ROOT, class_list: "data.yaml:names"}
source_labels: {expected_names: [Drowning, Swimming, out of water]}   # giúp notebook Kaggle nhận đúng thư mục
mapping:
  Drowning: {action: map, target: drowning}
  Swimming: {action: map, target: swimming}
  out of water: {action: map, target: out_of_water}
mapping_review: {status: pending_sample_review, sample_review_required: true}
```

Nhiều video CVAT (mỗi `tasks[]` một video):
```yaml
format: cvat_video
modality: video
paths: {root_env: TEN_VIDEO_DATA_ROOT}
tasks:
  - {annotations: "a/annotations.xml", video: "a/video.mp4", timestamps: "a/pts.json", global_group_id: "buoi-1-camera-A"}
  - {annotations: "b/annotations.xml", video: "b/video.mp4", timestamps: "b/pts.json"}   # bỏ global_group_id nếu chưa biết
timing: {mode: explicit_timestamps}          # hoặc {mode: constant_frame_rate, cfr_verified: true}
identity_review: {status: verified, sample_review_required: false}   # chỉ khi bạn đã kiểm tra ID người là đúng
mapping: {...}
```
Trạng thái của từng người đọc từ thuộc tính `state` (hoặc `water_state`, `class`) của từng box, nếu không có thì dùng nhãn của track.
Hộp `outside="1"` ngắt chuỗi của người đó. Hộp nội suy bị bỏ qua trừ khi bạn khai báo `interpolation: {policy: reviewed_interpolation, reviewed: true}`.

`reviewed_jsonl` (dành cho dataset có định dạng riêng; viết một script nhỏ để xuất ra hai file này):
```yaml
format: reviewed_jsonl
modality: mixed        # image | video | mixed
paths: {root_env: TEN_DATA_ROOT, media: media.jsonl, annotations: annotations.jsonl}
identity_review: {status: verified, sample_review_required: false}
mapping: {...}
```
`media.jsonl`, mỗi dòng: `{"path": "videos/a.mp4", "modality": "video", "group_id": "buoi-1"}` (`group_id` tùy chọn).
`annotations.jsonl`, mỗi dòng:
- ảnh: `{"path": "img/1.jpg", "label": "Swimming", "bbox": [x1, y1, x2, y2]}`
- video: `{"path": "videos/a.mp4", "frame": 12, "timestamp_ms": 480, "person_id": "p3", "label": "Swimming", "bbox": [x1, y1, x2, y2], "occluded": false}`
- người ra khỏi khung: `{"path": "videos/a.mp4", "frame": 40, "timestamp_ms": 1600, "person_id": "p3", "label": "Swimming", "outside": true}`

Hệ thống **không đoán**: thiếu `timestamp_ms`, `person_id`, hộp ngoài khung hay nhãn rỗng đều làm việc chuyển đổi dừng lại và báo dòng lỗi.

## Dataset YOLO không có `data.yaml` hoặc có tên file chứa video gốc

Ví dụ `configs/datasets/dataset_test_1.yaml` (dataset Roboflow tải lên Kaggle chỉ có `train/images` và `train/labels`):

```yaml
paths:
  root_env: DATASET_TEST_1_DATA_ROOT
  marker: train/labels               # thư mục (hoặc file) phải có ở gốc dataset để notebook Kaggle tìm đúng nguồn
  input_dir_contains: dataset-test-1 # tên dataset gắn vào notebook; bắt buộc khi hai dataset có cấu trúc giống nhau
source_labels:
  class_names: {0: Drowning, 1: Person out of water, 2: Swimming}   # chỉ số lớp → tên gốc, thay cho data.yaml
mapping:
  Drowning: {action: map, target: drowning}
  Person out of water: {action: map, target: out_of_water}
  Swimming: {action: map, target: swimming}
grouping:
  path_regex: '^train/images/(.+?)-\d+_jpg\.rf\.[0-9a-f]+\.[A-Za-z0-9]+$'   # nhóm 1 = tên video gốc
```

- `class_names` do **bạn xác nhận**: hệ thống không đoán chỉ số lớp. Chỉ số sai thì nhãn sai toàn bộ nguồn.
- `path_regex` gom mọi bản tăng cường và mọi khung của cùng một video vào cùng một nhóm để chia tập không rò rỉ. Nhóm lấy từ tên file, không ai xác nhận, nên báo cáo ghi `filename_pattern_unconfirmed`. File không khớp biểu thức để nhóm "chưa biết" (chế độ tư vấn sẽ tự gom bằng ảnh gần giống).
- Không có `marker` hoặc `input_dir_contains` thì notebook báo "ambiguous" và dừng khi có hai dataset trùng cấu trúc, không chọn bừa.

## Các cổng kiểm tra trước khi train

| Cổng | Ý nghĩa |
|---|---|
| Ánh xạ đã duyệt | một người có tên đã xem ≥ 10 ảnh mẫu mỗi nhãn và xác nhận; gắn với đúng bản ánh xạ hiện tại |
| Nhóm nguồn đã xác nhận | ảnh/video cùng nguồn không rơi vào nhiều tập. Với ảnh rời, hệ thống **đề xuất** nhóm bằng so khớp ảnh nhưng chỉ ghi nhận khi bạn xác nhận. Đề xuất này không chứng minh video nguồn khác nhau, nên báo cáo luôn ghi "chưa xác minh tách theo video nguồn" |
| Kiểm tra cấu trúc (audit) | khung hợp lệ, tham chiếu đúng, không trùng ID |
| Đủ ba lớp | ở tập train và validation, riêng cho nhánh ảnh và nhánh temporal, sau khi gộp các nguồn đã duyệt |
| Ảnh trùng nhãn khác nhau | bị loại khỏi mọi tập và báo cáo, không giữ theo thứ tự tải |

Nguồn chưa qua cổng nào bị **loại khỏi train kèm lý do** trong báo cáo, không bị dùng ngầm.
