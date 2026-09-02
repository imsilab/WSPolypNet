from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from cam import evaluate_corloc
from dataset import LocalizationVideoDataset
from model import SlowFastR50Binary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROI_MANIFEST = PROJECT_ROOT.parent / "datasets" / "roi_manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with CAM CorLoc.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--validation-root", type=Path, default=PROJECT_ROOT.parent / "datasets" / "ValidationData"
    )
    parser.add_argument("--cam-threshold", type=float, default=0.5)
    parser.add_argument("--spatial-size", type=int, default=224)
    parser.add_argument("--roi-manifest", type=Path, default=DEFAULT_ROI_MANIFEST)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SlowFastR50Binary(pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    current_roi_hash = file_sha256(args.roi_manifest)
    checkpoint_roi_hash = checkpoint.get("roi_manifest_sha256")
    if checkpoint_roi_hash != current_roi_hash:
        raise RuntimeError(
            "Checkpoint and evaluation ROI manifests do not match: "
            f"checkpoint={checkpoint_roi_hash}, current={current_roi_hash}"
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = LocalizationVideoDataset(
        args.validation_root / "video",
        args.validation_root / "annotation",
        spatial_size=args.spatial_size,
        roi_manifest_path=args.roi_manifest,
    )
    result = evaluate_corloc(model, dataset, device, args.cam_threshold)
    print(f"Evaluated annotated frames: {result.evaluated_frames}")
    print(f"CorLoc@0.3: {result.corloc_03:.6f}")
    print(f"CorLoc@0.5: {result.corloc_05:.6f}")
    print(f"CorLoc@0.7: {result.corloc_07:.6f}")
    print(f"Mean IoU: {result.mean_iou:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
