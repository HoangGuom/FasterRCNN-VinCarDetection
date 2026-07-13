# VF-Class VinFast Vehicle Detection

Dự án nhận diện xe VinFast bằng Faster R-CNN. Mô hình chỉ có một lớp đối tượng `vin_car`; kết quả gồm bounding box và confidence score.

Repository đã được rút gọn về duy nhất phiên bản tốt nhất sau ba lần huấn luyện 50 epoch:

- Backbone: ConvNeXt-Tiny.
- Neck: Feature Pyramid Network (FPN).
- Detector: Faster R-CNN.
- Kích thước train: 1024 px, tối đa 1280 px.
- Best validation loss đã ghi nhận: `0.018583477`.
- Checkpoint sử dụng: `weights/best.pt`.

Checkpoint không được đưa lên GitHub vì có dung lượng lớn. File checkpoint tốt nhất hiện vẫn được giữ ở máy trong `weights/best.pt`.

Phần dataset công khai lấy từ repository ban đầu đã được đưa trực tiếp vào dự án, gồm:

- 671 ảnh test trong `public/test/images`.
- 2.683 file nhãn YOLO trong `public/train/labels`.
- Annotation CSV trong `public/train_labels.csv`.
- File submission mẫu và mô tả dataset.

Nguồn ban đầu không chứa 2.683 ảnh train, vì vậy cần bổ sung ảnh train vào `public/train/images` nếu muốn huấn luyện lại.

## Cấu trúc dự án

```text
FasterRCNN-PyTorch/
├── config/
│   └── final.yaml              # Cấu hình cuối cùng
├── model/
│   ├── __init__.py
│   └── faster_rcnn.py          # ConvNeXt-T + FPN + Faster R-CNN
├── tools/
│   ├── __init__.py
│   ├── train_final.py          # Huấn luyện bản cuối
│   ├── infer.py                # Detect nhiều ảnh, xuất CSV
│   ├── infer_single.py         # Detect một ảnh, vẽ bounding box
│   └── soft_nms.py             # Gộp bounding box từ các lát ảnh
├── public/
│   ├── train/
│   │   └── labels/             # 2.683 nhãn YOLO
│   ├── test/
│   │   └── images/             # 671 ảnh test
│   ├── train_labels.csv
│   └── sample_submission.csv
├── requirements.txt
├── LICENSE
└── README.md
```

Các thư mục cục bộ được tạo khi chạy nhưng không commit lên GitHub:

```text
weights/                # best.pt và output train
single_infer_results/   # Ảnh đã vẽ bounding box
```

## Cài đặt

```powershell
pip install -r requirements.txt
```

Đặt checkpoint đã train tại:

```text
weights/best.pt
```

## Detect một ảnh

Chạy lệnh:

```powershell
python tools\infer_single.py
```

Terminal sẽ yêu cầu nhập hoặc kéo-thả đường dẫn ảnh. Chỉ nhập đường dẫn, ví dụ:

```text
D:\images\vinfast.jpg
```

Hoặc truyền ảnh trực tiếp trong lệnh:

```powershell
python tools\infer_single.py --image "D:\images\vinfast.jpg"
```

Ảnh có bounding box được lưu trong `single_infer_results/` và tự mở bằng trình xem ảnh mặc định. Nếu không có box, có thể giảm ngưỡng:

```powershell
python tools\infer_single.py --image "D:\images\vinfast.jpg" --conf-thresh 0.1
```

## Detect cả thư mục

```powershell
python tools\infer.py --input-dir "D:\images\test"
```

Kết quả mặc định được lưu tại `weights/submission.csv`.

## Huấn luyện lại bản cuối

Chuẩn bị dữ liệu theo cấu trúc:

```text
public/
├── train/
│   ├── images/          # Cần bổ sung ảnh train
│   └── labels/          # Đã có 2.683 file nhãn
├── test/
│   └── images/          # Đã có 671 ảnh test
└── train_labels.csv
```

File `train_labels.csv` cần các cột:

```csv
image_id,class_id,x_min,y_min,x_max,y_max
```

Sau đó chạy:

```powershell
python tools\train_final.py --config config\final.yaml
```

Script train 50 epoch, dùng augmentation trực tiếp khi đọc ảnh và lưu model tốt nhất vào `weights/best.pt`.

## Luồng xử lý ảnh

1. `tools/infer_single.py` nhận đường dẫn ảnh từ terminal.
2. `tools/infer.py` chia ảnh thành các lát 512 x 512 có overlap.
3. `model/faster_rcnn.py` chạy Faster R-CNN trên từng lát ảnh.
4. Soft-NMS gộp các dự đoán bị chồng lấn.
5. Bounding box được vẽ và lưu vào `single_infer_results/`.

## Bảo mật dữ liệu

`.gitignore` loại trừ checkpoint, output, cache Python, dữ liệu cục bộ trong `data/` và cấu hình editor. Không đưa ảnh riêng tư, annotation nội bộ, token hoặc credential lên repository public.

## License

Dự án sử dụng giấy phép MIT. Xem [LICENSE](LICENSE).
