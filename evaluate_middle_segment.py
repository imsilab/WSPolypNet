#!/usr/bin/env python3
"""Single-view CAM CorLoc evaluation for validation videos 183 through 364."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VALIDATION_ROOT = PROJECT_ROOT / "datasets" / "ValidationData"
DEFAULT_ROI_MANIFEST = PROJECT_ROOT / "datasets" / "roi_manifest.json"
MODEL_CLASSES = {
    "Slow-R50": "SlowR50Binary",
    "SlowFast-R50": "SlowFastR50Binary",
    "R3D-18": "R3D18Binary",
    "R(2+1)D-18": "R2Plus1D18Binary",
    "X3D": "X3DMBinary",
}
START_INDEX = 182  # zero-based; validation video 183
STOP_INDEX = 364   # exclusive; validation video 364


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one model on the middle 182 validation videos with single-view CAM."
    )
    parser.add_argument("model", choices=tuple(MODEL_CLASSES))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--selection-corloc-0.5",
        dest="selection_corloc_05",
        type=float,
        required=True,
    )
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--roi-manifest", type=Path, default=DEFAULT_ROI_MANIFEST)
    parser.add_argument("--cam-threshold", type=float, default=0.5)
    parser.add_argument("--spatial-size", type=int, default=224)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_root = PROJECT_ROOT / args.model
    codes = model_root / "codes"
    sys.path.insert(0, str(codes))

    cam_module = importlib.import_module("cam")
    dataset_module = importlib.import_module("dataset")
    model_module = importlib.import_module("model")
    model_class = getattr(model_module, MODEL_CLASSES[args.model])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    current_roi_hash = file_sha256(args.roi_manifest)
    checkpoint_roi_hash = checkpoint.get("roi_manifest_sha256")
    if checkpoint_roi_hash != current_roi_hash:
        raise RuntimeError(
            "Checkpoint and evaluation ROI manifests do not match: "
            f"checkpoint={checkpoint_roi_hash}, current={current_roi_hash}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    full_dataset = dataset_module.LocalizationVideoDataset(
        args.validation_root / "video",
        args.validation_root / "annotation",
        spatial_size=args.spatial_size,
        roi_manifest_path=args.roi_manifest,
    )
    if len(full_dataset) != 546:
        raise RuntimeError(f"Expected 546 validation videos, found {len(full_dataset)}")
    middle_dataset = Subset(full_dataset, range(START_INDEX, STOP_INDEX))
    if len(middle_dataset) != 182:
        raise RuntimeError(f"Expected 182 middle videos, found {len(middle_dataset)}")

    result = cam_module.evaluate_corloc(
        model,
        middle_dataset,
        device,
        args.cam_threshold,
        progress_interval=50,
    )
    payload = {
        "model": args.model,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_selection_metric": "full_validation_corloc_0.5",
        "checkpoint_selection_value": args.selection_corloc_05,
        "pipeline": "single_view_cam",
        "medsam2_used": False,
        "five_view_used": False,
        "cam_threshold": args.cam_threshold,
        "validation_video_start_1_based": START_INDEX + 1,
        "validation_video_end_1_based": STOP_INDEX,
        "validation_video_count": len(middle_dataset),
        "evaluated_frames": result.evaluated_frames,
        "mean_iou": result.mean_iou,
        "corloc_0.3": result.corloc_03,
        "corloc_0.5": result.corloc_05,
        "corloc_0.7": result.corloc_07,
        "roi_manifest_sha256": current_roi_hash,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
