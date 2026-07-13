# VinFast Car Detection Dataset

> **Lưu ý về dữ liệu trong repository:** nguồn repository ban đầu chỉ chứa 671 ảnh test, 2.683 file nhãn train và annotation CSV. Ảnh train không có trong Git nên phải được bổ sung riêng vào `public/train/images` trước khi huấn luyện.

## Overview
This dataset is designed for the **VinFast Car Detection Challenge**. The goal is to detect VinFast vehicles in various real-world scenarios. The dataset has been curated to focus effectively on a single class object detection task.

## Dataset Statistics
- **Total Images**: 3,354
- **Classes**: 1 (`vin_car`)
- **Split Ratio**: 80% Train / 20% Test
- **Total Objects**: Includes instances of VinFast cars with bounding box annotations.

## Data Split
The dataset is split into Training and Testing sets using a **stratified split** method to ensure a balanced distribution of images containing the target class versus background images.

- **Training Set (`train`)**: 2,683 images. Contains images and associated YOLO-format labels.
- **Test Set (`test`)**: 671 images. Images only; ground truth labels are withheld for scoring.

## Directory Structure
```
dataset/
├── train/
│   ├── images/          # Training images (.jpg/.png)
│   └── labels/          # YOLO annotations (.txt)
├── test/
│   └── images/          # Test images (.jpg/.png)
├── train_labels.csv     # Training annotations in CSV format
├── sample_submission.csv # Example submission format
├── vinfast_1class.yaml  # Configuration for YOLO training
└── README.md            # Basic instructions
```

## Annotation Format

### 1. YOLO Format (`train/labels/*.txt`)
Each text file corresponds to an image and contains one row per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
- `class_id`: Always `0` (representing `vin_car`).
- Coordinates are normalized (0-1) relative to image dimensions.

### 2. CSV Format (`train_labels.csv`)
For convenience and visualization, training labels are also provided in CSV format with pixel coordinates:
```csv
image_id,class_id,x_min,y_min,x_max,y_max
IMG_123.jpg,0,100,50,200,150
...
```

## Submission Format
Competitors must predict bounding boxes for images in the `test` folder. The submission file should be a CSV with the following header:
```csv
image_id,PredictionString
```
- `image_id`: Filename of the test image (e.g., `IMG_999.jpg`).
- `PredictionString`: A space-separated string of bounding box predictions in the format: `0 score x_min y_min x_max y_max`.
    - `0`: Class ID
    - `score`: Confidence score (0.0 - 1.0)
    - `x_min, y_min, x_max, y_max`: Pixel coordinates of the bounding box.

Example:
```csv
IMG_999.jpg,0 0.95 10 10 50 50 0 0.88 100 100 200 200
```
(This string predicts two cars in the image).

## Challenge Goals
- Build a robust object detection model to identify VinFast cars.
- Handle varying lighting conditions, angles, and occlusions.
- Optimize for Mean Average Precision (mAP) on the test set.
