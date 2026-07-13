from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.faster_rcnn import build_model


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return tuple(zip(*batch))


class VinCarDataset(Dataset):
    def __init__(self, image_dir: Path, annotation_csv: Path | None, augment: bool):
        self.image_paths = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        self.augment = augment
        self.boxes_by_name: dict[str, list[list[float]]] = defaultdict(list)

        if annotation_csv is not None:
            frame = pd.read_csv(annotation_csv)
            for row in frame.itertuples(index=False):
                self.boxes_by_name[str(row.image_id)].append(
                    [float(row.x_min), float(row.y_min), float(row.x_max), float(row.y_max)]
                )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        boxes = torch.tensor(self.boxes_by_name.get(path.name, []), dtype=torch.float32)
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = boxes.reshape(-1, 4)

        if self.augment and random.random() < 0.5:
            width = image.width
            image = TF.hflip(image)
            if len(boxes):
                old_x1 = boxes[:, 0].clone()
                old_x2 = boxes[:, 2].clone()
                boxes[:, 0] = width - old_x2
                boxes[:, 2] = width - old_x1

        if self.augment:
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            image = TF.adjust_brightness(image, brightness)
            image = TF.adjust_contrast(image, contrast)

        image_tensor = TF.to_tensor(image)
        target = {
            "boxes": boxes,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return image_tensor, target, path.name


def split_indices(length: int, seed: int, val_ratio: float) -> tuple[list[int], list[int]]:
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    val_count = max(1, round(length * val_ratio))
    return indices[val_count:], indices[:val_count]


def make_optimizer(model: FasterRCNN, config: dict):
    train_cfg = config["train_params"]
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if train_cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
    return torch.optim.SGD(
        params,
        lr=float(train_cfg["lr"]),
        momentum=float(train_cfg.get("momentum", 0.9)),
        weight_decay=float(train_cfg["weight_decay"]),
    )


def make_scheduler(optimizer, config: dict):
    train_cfg = config["train_params"]
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(train_cfg["num_epochs"]),
        eta_min=float(train_cfg["lr"]) * 0.01,
    )


def move_targets(targets, device):
    return [
        {key: value.to(device) for key, value in target.items()}
        for target in targets
    ]


def run_loss_epoch(
    model,
    loader,
    device,
    scaler,
    optimizer=None,
    progress_label: str = "",
    progress_path: Path | None = None,
    use_amp: bool = True,
    max_grad_norm: float | None = None,
) -> float:
    training = optimizer is not None
    model.train()
    total_loss = 0.0
    batches = 0
    total_batches = len(loader)
    started = time.time()

    for batch_index, (images, targets, _) in enumerate(loader, start=1):
        images = [image.to(device, non_blocking=True) for image in images]
        targets = move_targets(targets, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with autocast(enabled=device.type == "cuda" and use_amp):
                losses = model(images, targets)
                loss = sum(losses.values())

        if training:
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss in {progress_label} at batch "
                    f"{batch_index}/{total_batches}: "
                    + ", ".join(
                        f"{name}={float(value.detach().cpu())}"
                        for name, value in losses.items()
                    )
                )
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.detach().cpu())
        batches += 1

        if batch_index == 1 or batch_index % 50 == 0 or batch_index == total_batches:
            elapsed = time.time() - started
            rate = batch_index / max(elapsed, 1e-6)
            remaining = (total_batches - batch_index) / max(rate, 1e-6)
            message = (
                f"{progress_label}: batch {batch_index}/{total_batches} "
                f"({batch_index / total_batches:.1%}), "
                f"loss={total_loss / batches:.4f}, "
                f"ETA={remaining / 60:.1f} min"
            )
            print(message, flush=True)
            if progress_path is not None:
                progress_path.write_text(message + "\n", encoding="utf-8")

    return total_loss / max(1, batches)


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, best_val_loss, config):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )


def write_metrics(path: Path, epoch: int, best_val_loss: float):
    path.write_text(
        json.dumps(
            {
                "version": "final",
                "status": "trained",
                "epochs": epoch,
                "checkpoint": "weights/best.pt",
                "best_val_loss": best_val_loss,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def prediction_string(output: dict, threshold: float) -> str:
    parts: list[str] = []
    boxes = output["boxes"].detach().cpu()
    scores = output["scores"].detach().cpu()
    labels = output["labels"].detach().cpu()
    for box, score, label in zip(boxes, scores, labels):
        if int(label) != 1 or float(score) < threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        parts.extend(
            [
                "0",
                f"{float(score):.6f}",
                str(max(0, round(x1))),
                str(max(0, round(y1))),
                str(max(0, round(x2))),
                str(max(0, round(y2))),
            ]
        )
    return " ".join(parts)


@torch.inference_mode()
def create_submission(model, config: dict, output_path: Path, device) -> None:
    test_dir = ROOT / config["dataset_params"]["im_test_path"]
    dataset = VinCarDataset(test_dir, None, augment=False)
    model.eval()
    rows = []
    threshold = float(config["model_params"]["thresholds"]["eval_score"])
    for index in range(len(dataset)):
        image, _, name = dataset[index]
        output = model([image.to(device)])[0]
        rows.append(
            {"image_id": name, "PredictionString": prediction_string(output, threshold)}
        )
        if (index + 1) % 100 == 0 or index + 1 == len(dataset):
            print(f"Predicted {index + 1}/{len(dataset)}")
    pd.DataFrame(rows).to_csv(output_path, index=False)


def train_model(config_path: Path, device) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_cfg = config["train_params"]
    seed = int(train_cfg["seed"])
    total_epochs = int(train_cfg["num_epochs"])
    output_dir = ROOT / train_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    image_dir = ROOT / config["dataset_params"]["im_train_path"]
    annotation_csv = ROOT / config["dataset_params"]["ann_train_path"]
    if not image_dir.is_dir() or not annotation_csv.is_file():
        raise FileNotFoundError(
            "Không tìm thấy dữ liệu train. Kiểm tra im_train_path và ann_train_path "
            f"trong {config_path}."
        )
    full_train = VinCarDataset(image_dir, annotation_csv, augment=True)
    full_val = VinCarDataset(image_dir, annotation_csv, augment=False)
    train_indices, val_indices = split_indices(
        len(full_train), seed, float(train_cfg["val_ratio"])
    )

    batch_size = int(train_cfg["batch_size"])
    train_loader = DataLoader(
        Subset(full_train, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    print(
        f"\nFinal model: {len(train_indices)} train, {len(val_indices)} val, "
        f"{total_epochs} epochs"
    )
    model = build_model(config, pretrained=True).to(device)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    scaler = GradScaler(enabled=False)
    start_epoch = 1
    best_val_loss = math.inf
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    log_path = output_dir / "train_log.csv"

    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        print(f"Tiếp tục train từ epoch {start_epoch}")

    if start_epoch == 1:
        with log_path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                ["epoch", "train_loss", "val_loss", "lr", "seconds"]
            )

    for epoch in range(start_epoch, total_epochs + 1):
        started = time.time()
        progress_path = output_dir / "progress.txt"
        train_loss = run_loss_epoch(
            model,
            train_loader,
            device,
            scaler,
            optimizer,
            progress_label=f"final epoch {epoch}/{total_epochs} train",
            progress_path=progress_path,
            use_amp=False,
            max_grad_norm=5.0,
        )
        # Giới hạn tập validation mỗi epoch để thời gian train vẫn thực tế.
        val_subset_count = min(100, len(val_indices))
        val_loss = run_loss_epoch(
            model,
            DataLoader(
                Subset(full_val, val_indices[:val_subset_count]),
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            ),
            device,
            scaler,
            progress_label=f"final epoch {epoch}/{total_epochs} val",
            progress_path=progress_path,
            use_amp=False,
        )
        scheduler.step()
        elapsed = time.time() - started
        lr = optimizer.param_groups[0]["lr"]
        with log_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{lr:.10g}", f"{elapsed:.2f}"]
            )

        save_checkpoint(
            last_path, model, optimizer, scheduler, epoch, best_val_loss, config
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                best_path, model, optimizer, scheduler, epoch, best_val_loss, config
            )
        write_metrics(output_dir / "metrics.json", epoch, best_val_loss)
        print(
            f"final epoch {epoch}/{total_epochs}: train={train_loss:.4f}, "
            f"val={val_loss:.4f}, {elapsed / 60:.1f} min"
        )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dir = ROOT / config["dataset_params"]["im_test_path"]
    if test_dir.is_dir():
        create_submission(model, config, output_dir / "submission.csv", device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN bản cuối.")
    parser.add_argument("--config", default="config/final.yaml")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    train_model(ROOT / args.config, torch.device("cuda"))


if __name__ == "__main__":
    main()
