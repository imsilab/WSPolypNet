from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import read_boxes
from roi import load_roi_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "datasets"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit how much validation GT geometry is retained by folder ROIs."
    )
    parser.add_argument(
        "--annotation-root", type=Path, default=DATA_ROOT / "ValidationData/annotation"
    )
    parser.add_argument("--roi-manifest", type=Path, default=DATA_ROOT / "roi_manifest.json")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "logs/roi_validation_coverage.json"
    )
    args = parser.parse_args()

    rois = load_roi_manifest(args.roi_manifest)
    total_boxes = 0
    center_inside_count = 0
    fully_inside_count = 0
    retention_sum = 0.0
    minimum_retention = 1.0
    below_90_count = 0
    below_50_count = 0
    outside_center_examples: list[dict[str, object]] = []
    margin_fractions = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25)
    sensitivity = {
        fraction: {
            "center_inside_count": 0,
            "fully_inside_count": 0,
            "retention_sum": 0.0,
            "minimum_retention": 1.0,
            "below_90_count": 0,
            "below_50_count": 0,
        }
        for fraction in margin_fractions
    }

    folders = sorted(
        [path for path in args.annotation_root.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )
    for index, folder in enumerate(folders, start=1):
        roi = rois[f"validation/{folder.name}"]
        fov_mask = np.zeros(
            (roi.source_height, roi.source_width), dtype=np.uint8
        )
        cv2.fillPoly(
            fov_mask,
            [np.asarray(roi.fov_polygon_xy, dtype=np.int32)],
            1,
        )
        for annotation_path in sorted(folder.glob("*.txt")):
            for box in read_boxes(annotation_path):
                x1, y1, x2, y2 = box
                pixel_x1 = max(0, int(np.floor(x1)))
                pixel_y1 = max(0, int(np.floor(y1)))
                pixel_x2 = min(roi.source_width, int(np.ceil(x2)))
                pixel_y2 = min(roi.source_height, int(np.ceil(y2)))
                area = max(0, pixel_x2 - pixel_x1) * max(
                    0, pixel_y2 - pixel_y1
                )
                retained_pixels = int(
                    fov_mask[pixel_y1:pixel_y2, pixel_x1:pixel_x2].sum()
                )
                retention = retained_pixels / area if area > 0 else 0.0
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                center_pixel_x = min(
                    roi.source_width - 1, max(0, int(center_x))
                )
                center_pixel_y = min(
                    roi.source_height - 1, max(0, int(center_y))
                )
                center_inside = bool(fov_mask[center_pixel_y, center_pixel_x])
                fully_inside = retention >= 1.0
                total_boxes += 1
                center_inside_count += int(center_inside)
                fully_inside_count += int(fully_inside)
                retention_sum += retention
                minimum_retention = min(minimum_retention, retention)
                below_90_count += int(retention < 0.90)
                below_50_count += int(retention < 0.50)
                if not center_inside and len(outside_center_examples) < 50:
                    outside_center_examples.append(
                        {
                            "folder": folder.name,
                            "annotation": annotation_path.name,
                            "gt_xyxy": list(box),
                            "roi_xyxy": list(roi.xyxy),
                            "roi_policy": "FOV hull polygon",
                            "area_retention": retention,
                        }
                    )
                for fraction, metrics in sensitivity.items():
                    margin_x = round(roi.width * fraction)
                    margin_y = round(roi.height * fraction)
                    expanded_x1 = max(0, roi.x1 - margin_x)
                    expanded_y1 = max(0, roi.y1 - margin_y)
                    expanded_x2 = min(roi.source_width, roi.x2 + margin_x)
                    expanded_y2 = min(roi.source_height, roi.y2 + margin_y)
                    expanded_intersection = max(
                        0.0, min(x2, expanded_x2) - max(x1, expanded_x1)
                    ) * max(0.0, min(y2, expanded_y2) - max(y1, expanded_y1))
                    expanded_retention = expanded_intersection / area if area > 0 else 0.0
                    expanded_center = (
                        expanded_x1 <= center_x < expanded_x2
                        and expanded_y1 <= center_y < expanded_y2
                    )
                    expanded_full = (
                        x1 >= expanded_x1
                        and y1 >= expanded_y1
                        and x2 <= expanded_x2
                        and y2 <= expanded_y2
                    )
                    metrics["center_inside_count"] += int(expanded_center)
                    metrics["fully_inside_count"] += int(expanded_full)
                    metrics["retention_sum"] += expanded_retention
                    metrics["minimum_retention"] = min(
                        metrics["minimum_retention"], expanded_retention
                    )
                    metrics["below_90_count"] += int(expanded_retention < 0.90)
                    metrics["below_50_count"] += int(expanded_retention < 0.50)
        print(f"[{index}/{len(folders)}] validation/{folder.name}", flush=True)

    if total_boxes == 0:
        raise RuntimeError("No GT boxes were found")
    result = {
        "total_gt_boxes": total_boxes,
        "center_inside_count": center_inside_count,
        "center_inside_rate": center_inside_count / total_boxes,
        "fully_inside_count": fully_inside_count,
        "fully_inside_rate": fully_inside_count / total_boxes,
        "mean_area_retention": retention_sum / total_boxes,
        "minimum_area_retention": minimum_retention,
        "boxes_below_90_percent_retention": below_90_count,
        "boxes_below_50_percent_retention": below_50_count,
        "outside_center_examples": outside_center_examples,
        "outward_margin_sensitivity": {
            f"{fraction:.2f}": {
                "center_inside_rate": metrics["center_inside_count"] / total_boxes,
                "fully_inside_rate": metrics["fully_inside_count"] / total_boxes,
                "mean_area_retention": metrics["retention_sum"] / total_boxes,
                "minimum_area_retention": metrics["minimum_retention"],
                "boxes_below_90_percent_retention": metrics["below_90_count"],
                "boxes_below_50_percent_retention": metrics["below_50_count"],
            }
            for fraction, metrics in sensitivity.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Coverage report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

