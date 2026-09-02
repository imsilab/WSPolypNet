#!/usr/bin/env python3
"""Evaluate positive-video recall with exact model ROI preprocessing."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_CLASSES = {
    "Slow-R50": "SlowR50Binary",
    "SlowFast-R50": "SlowFastR50Binary",
    "R3D-18": "R3D18Binary",
    "R(2+1)D-18": "R2Plus1D18Binary",
    "X3D": "X3DMBinary",
}
MODEL_SIZE = 224
CROP_SIZE = 144
CORNERS = ((0, 0), (80, 0), (0, 80), (80, 80))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(MODEL_CLASSES))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--view-mode", choices=("single", "multi"), default="single")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "ValidationData",
    )
    parser.add_argument(
        "--roi-manifest",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "roi_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def view_probabilities(
    model: torch.nn.Module,
    video: torch.Tensor,
    include_corners: bool,
) -> list[float]:
    views = [video]
    if include_corners:
        for left, top in CORNERS:
            crop = video[:, :, top : top + CROP_SIZE, left : left + CROP_SIZE]
            zoomed = F.interpolate(
                crop.permute(1, 0, 2, 3),
                size=(MODEL_SIZE, MODEL_SIZE),
                mode="bilinear",
                align_corners=False,
            ).permute(1, 0, 2, 3)
            views.append(zoomed)
    probabilities = []
    for view in views:
        logit = model(view.unsqueeze(0))
        probabilities.append(float(torch.sigmoid(logit.reshape(-1)[0]).item()))
    return probabilities


def main() -> int:
    args = parse_args()
    codes = PROJECT_ROOT / args.model / "codes"
    sys.path.insert(0, str(codes))
    dataset_module = importlib.import_module("dataset")
    model_module = importlib.import_module("model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = getattr(model_module, MODEL_CLASSES[args.model])
    model = model_class(pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    roi_hash = file_sha256(args.roi_manifest)
    if checkpoint.get("roi_manifest_sha256") != roi_hash:
        raise RuntimeError("Checkpoint and current ROI manifest do not match")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    dataset = dataset_module.LocalizationVideoDataset(
        args.validation_root / "video",
        args.validation_root / "annotation",
        spatial_size=MODEL_SIZE,
        roi_manifest_path=args.roi_manifest,
    )
    if len(dataset) != 182:
        raise RuntimeError(f"Expected the pruned 182-video evaluation set, found {len(dataset)}")

    rows: list[dict[str, str | int | float]] = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            video, _, _, _, path = dataset[index]
            video = video.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                probabilities = view_probabilities(
                    model,
                    video,
                    include_corners=args.view_mode == "multi",
                )
            probability = max(probabilities)
            rows.append(
                {
                    "video_index": index + 1,
                    "video_path": path,
                    "ground_truth_label": 1,
                    "probability": probability,
                    "predicted_label": int(probability >= args.threshold),
                    "full_view_probability": probabilities[0],
                    "corner_tl_probability": probabilities[1] if args.view_mode == "multi" else "",
                    "corner_tr_probability": probabilities[2] if args.view_mode == "multi" else "",
                    "corner_bl_probability": probabilities[3] if args.view_mode == "multi" else "",
                    "corner_br_probability": probabilities[4] if args.view_mode == "multi" else "",
                }
            )
            if (index + 1) % 50 == 0 or index + 1 == len(dataset):
                recall = sum(int(row["predicted_label"]) for row in rows) / len(rows)
                print(
                    f"[{index + 1}/{len(dataset)}] positive_recall={recall:.6f}",
                    flush=True,
                )

    probabilities = [float(row["probability"]) for row in rows]
    true_positives = sum(int(row["predicted_label"]) for row in rows)
    summary = {
        "model": args.model,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "view_mode": args.view_mode,
        "multi_view_aggregation": "max_probability" if args.view_mode == "multi" else None,
        "classification_threshold": args.threshold,
        "population": "positive-only pruned validation videos",
        "videos": len(rows),
        "positive_videos": len(rows),
        "negative_videos": 0,
        "true_positives": true_positives,
        "false_negatives": len(rows) - true_positives,
        "positive_recall": true_positives / len(rows),
        "mean_positive_probability": sum(probabilities) / len(probabilities),
        "minimum_probability": min(probabilities),
        "maximum_probability": max(probabilities),
        "accuracy_available": False,
        "roi_manifest_sha256": roi_hash,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_video.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
