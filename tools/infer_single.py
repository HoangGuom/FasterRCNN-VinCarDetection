import argparse
import os
import subprocess
import sys

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.infer import load_trained_model, predict_image


def draw_predictions(image, predictions, output_path):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for x1, y1, x2, y2, score in predictions:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"vin_car {score:.2f}"

        draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_y = max(0, y1 - text_h - 6)

        draw.rectangle((x1, text_y, x1 + text_w + 8, text_y + text_h + 6), fill="lime")
        draw.text((x1 + 4, text_y + 3), label, fill="black", font=font)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    image.save(output_path)


def open_output_image(output_path):
    abs_output_path = os.path.abspath(output_path)
    if sys.platform.startswith("win"):
        os.startfile(abs_output_path)
    elif sys.platform == "darwin":
        subprocess.run(["open", abs_output_path], check=False)
    else:
        subprocess.run(["xdg-open", abs_output_path], check=False)


def normalize_terminal_path(path):
    path = path.strip()
    if path.startswith("&"):
        path = path[1:].strip()
    if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
        path = path[1:-1]
    return path.strip()


def main():
    parser = argparse.ArgumentParser(description="Chạy Faster R-CNN trên 1 ảnh và lưu ảnh đã vẽ bounding box.")
    parser.add_argument("--image", default=None, help="Đường dẫn ảnh cần kiểm tra.")
    parser.add_argument("--config", default="config/final.yaml", help="Cấu hình model cuối.")
    parser.add_argument("--output", default=None, help="Đường dẫn ảnh output đã vẽ bounding box.")
    parser.add_argument(
        "--checkpoint",
        default="weights/best.pt",
        help="Checkpoint bản cuối.",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.25,
        help="Ngưỡng confidence ban đầu (khuyên dùng 0.1-0.3 để không bỏ sót xe nhỏ).",
    )
    parser.add_argument("--slice-size", type=int, default=512, help="Kích thước mỗi slice khi inference.")
    parser.add_argument("--overlap", type=int, default=100, help="Số pixel overlap giữa các slice.")
    parser.add_argument("--no-show", action="store_true", help="Chỉ lưu ảnh kết quả, không tự mở ảnh.")
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        image_path = input("Nhập hoặc kéo-thả đường dẫn ảnh vào đây rồi nhấn Enter: ")
    image_path = normalize_terminal_path(image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")
    model, _ = load_trained_model(args.config, args.checkpoint, device)
    print(f"Đã load checkpoint: {args.checkpoint}")

    image = Image.open(image_path).convert("RGB")
    predictions = predict_image(
        model, image, device, args.conf_thresh, args.slice_size, args.overlap
    )
    used_conf_thresh = args.conf_thresh

    # Tự động nới ngưỡng nếu không có box để tránh output trắng khi xe nhỏ/xa.
    if len(predictions) == 0 and args.conf_thresh > 0.05:
        fallback_conf = max(0.05, min(0.2, args.conf_thresh * 0.5))
        print(
            f"Không có box với conf={args.conf_thresh:.2f}. "
            f"Thử lại với conf={fallback_conf:.2f}..."
        )
        predictions = predict_image(
            model, image, device, fallback_conf, args.slice_size, args.overlap
        )
        used_conf_thresh = fallback_conf

    if len(predictions) == 0:
        print(
            "Không có bounding box nào sau Soft-NMS. "
            "Bạn có thể thử --conf-thresh 0.1 hoặc kiểm tra đúng checkpoint."
        )

    output_path = args.output
    if output_path is None:
        image_name = os.path.basename(image_path)
        output_path = os.path.join("single_infer_results", image_name)

    draw_predictions(image, predictions, output_path)

    print(f"Tìm thấy {len(predictions)} bounding box (conf đã dùng: {used_conf_thresh:.2f}).")
    for idx, (x1, y1, x2, y2, score) in enumerate(predictions, start=1):
        print(f"{idx}: score={score:.4f}, box=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")
    print(f"Đã lưu ảnh kết quả tại: {output_path}")

    if not args.no_show:
        open_output_image(output_path)
        print("Đã mở ảnh kết quả bằng trình xem ảnh mặc định.")


if __name__ == "__main__":
    main()
