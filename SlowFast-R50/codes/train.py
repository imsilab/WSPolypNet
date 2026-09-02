from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cam import evaluate_corloc
from dataset import BinaryVideoDataset, LocalizationVideoDataset
from model import SlowFastR50Binary, count_trainable_parameters
from utils import save_checkpoint, seed_everything, seed_worker, update_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "datasets"
DEFAULT_NEGATIVE_ROOT = DATA_ROOT / "TrainVaild(video)_without_polyp"
DEFAULT_POSITIVE_ROOT = DATA_ROOT / "TrainValid(video)_with_polyp"
DEFAULT_VALIDATION_ROOT = DATA_ROOT / "ValidationData"
DEFAULT_ROI_MANIFEST = DATA_ROOT / "roi_manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_optimizer(
    model: SlowFastR50Binary,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    # Register every parameter once. Frozen backbone parameters receive no
    # gradients during epochs 1-10 and begin updating immediately when unfrozen.
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def train_one_epoch(
    model: SlowFastR50Binary,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    backbone_trainable: bool,
    progress_interval: int,
) -> tuple[float, float]:
    model.set_training_mode(backbone_trainable)
    loss_sum = 0.0
    correct = 0
    samples = 0
    for step, (video, label, _) in enumerate(loader, start=1):
        # batch_size=1 permits the original variable T for every whole video.
        video = video.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        binary_target = label.float().view(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(video)
            assert isinstance(logits, torch.Tensor)
            loss = criterion(logits, binary_target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = label.numel()
        loss_sum += float(loss.detach()) * batch_size
        prediction = (logits.view(-1) >= 0).long()
        correct += int((prediction == label).sum())
        samples += batch_size
        if progress_interval and step % progress_interval == 0:
            print(
                f"  train: {step}/{len(loader)} videos, "
                f"loss={loss_sum / samples:.6f}, accuracy={correct / samples:.6f}",
                flush=True,
            )
    return loss_sum / samples, correct / samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one-logit Kinetics-pretrained SlowFast R50 and evaluate CAM CorLoc."
    )
    parser.add_argument("--negative-root", type=Path, default=DEFAULT_NEGATIVE_ROOT)
    parser.add_argument("--positive-root", type=Path, default=DEFAULT_POSITIVE_ROOT)
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--roi-manifest", type=Path, default=DEFAULT_ROI_MANIFEST)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "Checkpoints")
    parser.add_argument("--log-csv", type=Path, default=PROJECT_ROOT / "logs" / "train.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=10,
        help="Freeze the backbone for this many initial epochs (default: 10).",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--cam-threshold", type=float, default=0.5)
    parser.add_argument("--spatial-size", type=int, default=224)
    parser.add_argument("--canvas-scale-min", type=float, default=0.90)
    parser.add_argument("--canvas-scale-max", type=float, default=1.00)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.freeze_epochs < 0:
        raise ValueError("epochs must be >=1 and freeze-epochs must be >=0")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("lr must be positive and weight-decay must be non-negative")
    if not 0.0 <= args.cam_threshold <= 1.0:
        raise ValueError("cam-threshold must be between 0 and 1")
    if not 0 < args.canvas_scale_min <= args.canvas_scale_max <= 1:
        raise ValueError("canvas scales must satisfy 0 < min <= max <= 1")
    seed_everything(args.seed)
    roi_manifest_sha256 = file_sha256(args.roi_manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    training_dataset = BinaryVideoDataset(
        args.negative_root,
        args.positive_root,
        spatial_size=args.spatial_size,
        canvas_scale_range=(args.canvas_scale_min, args.canvas_scale_max),
        roi_manifest_path=args.roi_manifest,
    )
    validation_dataset = LocalizationVideoDataset(
        args.validation_root / "video",
        args.validation_root / "annotation",
        spatial_size=args.spatial_size,
        roi_manifest_path=args.roi_manifest,
    )
    generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    print(
        f"Training videos: {len(training_dataset)} "
        f"(negative={training_dataset.class_counts[0]}, "
        f"positive={training_dataset.class_counts[1]}); no split"
    )
    print(f"Validation videos: {len(validation_dataset)}")
    print(f"ROI manifest: {args.roi_manifest}")
    print(f"ROI manifest SHA-256: {roi_manifest_sha256}")

    model = SlowFastR50Binary(pretrained=args.resume is None).to(device)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 1
    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        checkpoint_roi_hash = resume_checkpoint.get("roi_manifest_sha256")
        if checkpoint_roi_hash != roi_manifest_sha256:
            raise RuntimeError(
                "Cannot resume with a different or unrecorded ROI manifest: "
                f"checkpoint={checkpoint_roi_hash}, current={roi_manifest_sha256}"
            )
        checkpoint_args = resume_checkpoint.get("args", {})
        if int(checkpoint_args.get("spatial_size", -1)) != args.spatial_size:
            raise RuntimeError(
                "Cannot resume with a different spatial size: "
                f"checkpoint={checkpoint_args.get('spatial_size')}, "
                f"current={args.spatial_size}"
            )
        for name in ("canvas_scale_min", "canvas_scale_max"):
            if float(checkpoint_args.get(name, -1)) != float(getattr(args, name)):
                raise RuntimeError(
                    f"Cannot resume with a different {name.replace('_', '-')}: "
                    f"checkpoint={checkpoint_args.get(name)}, current={getattr(args, name)}"
                )
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        if "scaler_state_dict" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        print(f"Resuming after epoch {start_epoch - 1}: {args.resume}")

    model.set_backbone_trainable(start_epoch > args.freeze_epochs)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)
    if resume_checkpoint is not None:
        if "scheduler_state_dict" not in resume_checkpoint:
            raise RuntimeError(
                "Checkpoint predates the cosine schedule and cannot be resumed safely"
            )
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        print("Restored AdamW optimizer and cosine scheduler state")

    active_stage: str | None = None
    for epoch in range(start_epoch, args.epochs + 1):
        full_finetuning = epoch > args.freeze_epochs
        stage = "full" if full_finetuning else "head"
        if stage != active_stage:
            model.set_backbone_trainable(full_finetuning)
            active_stage = stage
            trainable, total = count_trainable_parameters(model)
            optimizer_ids = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            backbone_is_in_optimizer = any(
                id(parameter) in optimizer_ids
                for name, parameter in model.named_parameters()
                if "backbone.blocks.6.proj" not in name
            )
            print(
                f"Stage -> {stage}: trainable={trainable}/{total}, "
                f"backbone_in_optimizer={backbone_is_in_optimizer}, "
                f"lr={optimizer.param_groups[0]['lr']:.8g}"
            )
            if full_finetuning and not backbone_is_in_optimizer:
                raise RuntimeError("Backbone was not added to optimizer at unfreeze")

        started = time.time()
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        print(
            f"Epoch {epoch}/{args.epochs} [{stage}] lr={epoch_learning_rate:.8g}",
            flush=True,
        )
        train_loss, train_accuracy = train_one_epoch(
            model,
            training_loader,
            optimizer,
            criterion,
            scaler,
            device,
            full_finetuning,
            args.progress_interval,
        )
        corloc = evaluate_corloc(
            model,
            validation_dataset,
            device,
            args.cam_threshold,
        )
        metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "corloc_0.3": corloc.corloc_03,
            "corloc_0.5": corloc.corloc_05,
            "corloc_0.7": corloc.corloc_07,
        }
        scheduler.step()
        checkpoint_path = args.checkpoint_dir / f"epoch_{epoch:03d}.pth"
        save_checkpoint(
            checkpoint_path,
            {
                **metrics,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "optimizer_stage": stage,
                "output_logits": 1,
                "learning_rate": epoch_learning_rate,
                "scaler_state_dict": scaler.state_dict(),
                "args": vars(args),
                "roi_manifest_sha256": roi_manifest_sha256,
                "evaluated_frames": corloc.evaluated_frames,
                "mean_iou": corloc.mean_iou,
            },
        )
        update_csv(args.log_csv, metrics)
        print(
            f"Epoch {epoch} complete in {(time.time() - started) / 60:.2f} min: "
            f"loss={train_loss:.6f}, accuracy={train_accuracy:.6f}, "
            f"CorLoc@0.3={corloc.corloc_03:.6f}, "
            f"@0.5={corloc.corloc_05:.6f}, @0.7={corloc.corloc_07:.6f}; "
            f"evaluated_frames={corloc.evaluated_frames}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
