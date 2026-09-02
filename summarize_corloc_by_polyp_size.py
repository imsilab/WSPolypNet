#!/usr/bin/env python3
"""Split per-frame CorLoc metrics by the pipeline's 5% GT-area criterion."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
X3D_CODES = PROJECT_ROOT / "X3D" / "codes"
sys.path.insert(0, str(X3D_CODES))

from dataset import LocalizationVideoDataset  # noqa: E402
from roi import make_canvas_transform  # noqa: E402


MODEL_SIZE = 224
SMALL_BOX_AREA_FRACTION = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--current-validation-only",
        action="store_true",
        help="Skip result rows whose videos are no longer in the current validation set.",
    )
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
    return parser.parse_args()


def summarize(rows: list[dict[str, str]]) -> dict[str, float | int]:
    count = len(rows)
    if count == 0:
        return {
            "frames": 0,
            "mean_iou": 0.0,
            "corloc_0.3_count": 0,
            "corloc_0.3": 0.0,
            "corloc_0.5_count": 0,
            "corloc_0.5": 0.0,
            "corloc_0.7_count": 0,
            "corloc_0.7": 0.0,
        }
    result: dict[str, float | int] = {
        "frames": count,
        "mean_iou": sum(float(row["max_iou"]) for row in rows) / count,
    }
    for threshold in ("0.3", "0.5", "0.7"):
        correct = sum(int(row[f"correct_{threshold}"]) for row in rows)
        result[f"corloc_{threshold}_count"] = correct
        result[f"corloc_{threshold}"] = correct / count
    return result


def main() -> int:
    args = parse_args()
    dataset = LocalizationVideoDataset(
        args.validation_root / "video",
        args.validation_root / "annotation",
        spatial_size=MODEL_SIZE,
        roi_manifest_path=args.roi_manifest,
    )
    transforms = {
        str(sample.video_path.resolve()): make_canvas_transform(
            sample.roi,
            MODEL_SIZE,
            scale=1.0,
            position_x=0.5,
            position_y=0.5,
            horizontal_flip=False,
        )
        for sample in dataset.samples
    }

    with args.result_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No result rows: {args.result_csv}")

    enriched: list[dict[str, str]] = []
    skipped_rows = 0
    for row in rows:
        video_path = str(Path(row["video_path"]).resolve())
        transform = transforms.get(video_path)
        if transform is None:
            if args.current_validation_only:
                skipped_rows += 1
                continue
            raise RuntimeError(f"Result video is not in current validation set: {video_path}")
        boxes = json.loads(row["ground_truth_boxes_xyxy"])
        if not boxes:
            raise RuntimeError("Per-frame CorLoc row has no ground-truth box")
        mapped = [transform.original_box_to_canvas(tuple(box)) for box in boxes]
        largest_area_fraction = max(
            (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
            for x1, y1, x2, y2 in mapped
        ) / float(MODEL_SIZE**2)
        size_class = (
            "small" if largest_area_fraction <= SMALL_BOX_AREA_FRACTION else "large"
        )
        enriched.append(
            {
                **row,
                "largest_gt_area_fraction": f"{largest_area_fraction:.10f}",
                "polyp_size": size_class,
            }
        )

    grouped = {
        size: [row for row in enriched if row["polyp_size"] == size]
        for size in ("small", "large")
    }
    summary = {
        "source_result_csv": str(args.result_csv.resolve()),
        "criterion_source": str(
            (PROJECT_ROOT / "standalone_localization_eval.py").resolve()
        ),
        "criterion": (
            "largest GT box area after the evaluation ROI/canvas transform, "
            "divided by 224^2; small <= 0.05, large > 0.05"
        ),
        "multiple_gt_rule": "use the largest transformed GT box in the frame",
        "source_rows": len(rows),
        "skipped_rows_outside_current_validation": skipped_rows,
        "all": summarize(enriched),
        "small": summarize(grouped["small"]),
        "large": summarize(grouped["large"]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "per_frame_corloc_by_size.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(enriched[0]))
        writer.writeheader()
        writer.writerows(enriched)
    summary_path = args.output_dir / "size_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
