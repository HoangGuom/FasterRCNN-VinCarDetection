# VinFast Car Detection Challenge

## Dữ liệu hiện có trong repository

- `test/images`: 671 ảnh test.
- `train/labels`: 2.683 file nhãn YOLO.
- `train_labels.csv`: annotation bounding box dạng CSV.
- `train/images`: chưa có trong nguồn repository ban đầu và cần được bổ sung để train.

## Dataset Structure
- `train/images`: Training images (cần bổ sung)
- `train/labels`: YOLO format labels (class 0: vin_car)
- `test/images`: Test images (labels held out)
- `train_labels.csv`: Bounding box annotations in CSV format
- `vinfast_1class.yaml`: Configuration for YOLO training

## Classes
- 0: vin_car

## Submission Format
CSV file with columns: `image_id`, `PredictionString`
`PredictionString` format: `class_id score x_min y_min x_max y_max ...`
