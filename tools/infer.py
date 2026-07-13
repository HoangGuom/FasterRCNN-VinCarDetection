import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.faster_rcnn import build_model
from tools.soft_nms import soft_nms_torch


def load_trained_model(config_path, checkpoint_path, device):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {checkpoint_path}. "
            "Hãy đặt file best.pt vào thư mục weights."
        )

    model = build_model(config, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, config


@torch.inference_mode()
def sliced_prediction(
    model,
    image_array,
    device,
    slice_size=512,
    overlap=100,
    conf_thresh=0.25,
):
    height, width = image_array.shape[:2]
    stride = slice_size - overlap
    predictions = []

    for top in range(0, height, stride):
        for left in range(0, width, stride):
            right = min(width, left + slice_size)
            bottom = min(height, top + slice_size)
            image_slice = image_array[top:bottom, left:right].copy()

            if image_slice.shape[:2] != (slice_size, slice_size):
                padded = np.zeros((slice_size, slice_size, 3), dtype=image_slice.dtype)
                padded[: image_slice.shape[0], : image_slice.shape[1]] = image_slice
                image_slice = padded

            tensor = (
                torch.from_numpy(image_slice)
                .permute(2, 0, 1)
                .float()
                .to(device)
                / 255.0
            )
            output = model([tensor])[0]
            boxes = output["boxes"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if int(label) != 1 or float(score) < conf_thresh:
                    continue
                x1, y1, x2, y2 = box
                predictions.append(
                    [
                        float(max(0, min(width, x1 + left))),
                        float(max(0, min(height, y1 + top))),
                        float(max(0, min(width, x2 + left))),
                        float(max(0, min(height, y2 + top))),
                        float(score),
                    ]
                )
    return predictions


def apply_soft_nms(predictions, iou_threshold=0.4, sigma=0.5, keep_top_k=100):
    if not predictions:
        return []

    values = np.asarray(predictions, dtype=np.float32)
    boxes = torch.tensor(values[:, :4], dtype=torch.float32)
    scores = torch.tensor(values[:, 4], dtype=torch.float32)
    keep_indices, updated_scores = soft_nms_torch(
        boxes,
        scores,
        iou_threshold=iou_threshold,
        sigma=sigma,
        keep_top_k=keep_top_k,
    )
    return [
        [*map(float, boxes[index].tolist()), float(updated_scores[index])]
        for index in keep_indices
    ]


def predict_image(model, image, device, conf_thresh, slice_size, overlap):
    predictions = sliced_prediction(
        model,
        np.asarray(image),
        device,
        slice_size=slice_size,
        overlap=overlap,
        conf_thresh=conf_thresh,
    )
    return apply_soft_nms(predictions)


def prediction_string(predictions):
    return " ".join(
        f"0 {score:.6f} {round(x1)} {round(y1)} {round(x2)} {round(y2)}"
        for x1, y1, x2, y2, score in predictions
    )


def main():
    parser = argparse.ArgumentParser(description="Detect toàn bộ ảnh trong một thư mục.")
    parser.add_argument("--config", default="config/final.yaml")
    parser.add_argument("--checkpoint", default="weights/best.pt")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output", default="weights/submission.csv")
    parser.add_argument("--conf-thresh", type=float, default=0.25)
    parser.add_argument("--slice-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_trained_model(args.config, args.checkpoint, device)
    input_dir = Path(args.input_dir or config["dataset_params"]["im_test_path"])
    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong: {input_dir}")

    rows = []
    for image_path in tqdm(image_paths, desc="Detect"):
        image = Image.open(image_path).convert("RGB")
        predictions = predict_image(
            model,
            image,
            device,
            args.conf_thresh,
            args.slice_size,
            args.overlap,
        )
        rows.append(
            {"image_id": image_path.name, "PredictionString": prediction_string(predictions)}
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Đã lưu kết quả: {output_path.resolve()}")


if __name__ == "__main__":
    main()
