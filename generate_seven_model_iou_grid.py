#!/usr/bin/env python3
"""Create a fair 7-model x 7-case localization IoU comparison grid."""
from __future__ import annotations

import csv
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
VALIDATION_ROOT = ROOT / "datasets" / "ValidationData"
ROI_MANIFEST = ROOT / "datasets" / "roi_manifest.json"
OUTPUT = ROOT / "datasets" / "sample" / "seven_model_iou_comparison"
MULTI_CSV = ROOT / "X3D_MedSAM2_epoch014_CorLoc" / "per_frame_corloc.csv"
SINGLE_CSV = ROOT / "X3D_MedSAM2_epoch014_single_view_middle_CorLoc" / "per_frame_corloc.csv"

MODEL_CONFIGS = (
    ("Slow-R50 CAM", "Slow-R50", "SlowR50Binary", "epoch_020.pth"),
    ("SlowFast-R50 CAM", "SlowFast-R50", "SlowFastR50Binary", "epoch_044.pth"),
    ("R3D-18 CAM", "R3D-18", "R3D18Binary", "epoch_004.pth"),
    ("R(2+1)D-18 CAM", "R(2+1)D-18", "R2Plus1D18Binary", "epoch_009.pth"),
    ("X3D CAM", "X3D", "X3DMBinary", "epoch_014.pth"),
)
SINGLE_NAME = "X3D + MedSAM2 single-view"
OURS_NAME = "X3D + MedSAM2 multi-view (Ours)"
MODEL_ORDER = (
    "R(2+1)D-18 CAM",
    "SlowFast-R50 CAM",
    "R3D-18 CAM",
    "Slow-R50 CAM",
    "X3D CAM",
    SINGLE_NAME,
    OURS_NAME,
)


def load_csv(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            video = str(Path(row["video_path"]).resolve())
            if not Path(video).is_file():
                continue
            key = (video, int(row["frame_index"]))
            rows[key] = {
                "box": json.loads(row["predicted_box_xyxy"]) if row["predicted_box_xyxy"] else None,
                "gt": json.loads(row["ground_truth_boxes_xyxy"]),
                "iou": float(row["max_iou"]),
            }
    return rows


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def class_activation_maps(model: torch.nn.Module, video: torch.Tensor) -> np.ndarray:
    result = model(video, return_features=True)
    if not isinstance(result, tuple):
        raise RuntimeError("Model did not return features")
    _, features = result
    weights = model.classifier.weight[0]
    cam = torch.einsum("c,bcthw->bthw", weights, features).relu()
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=(video.shape[2], video.shape[3], video.shape[4]),
        mode="trilinear",
        align_corners=False,
    ).squeeze(1)
    return cam[0].float().cpu().numpy()


def cam_to_box(cam: np.ndarray, valid_mask: np.ndarray, threshold: float = 0.5):
    valid = valid_mask.astype(bool, copy=False)
    low, high = float(cam[valid].min()), float(cam[valid].max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None
    normalized = np.zeros_like(cam, dtype=np.float32)
    normalized[valid] = (cam[valid] - low) / (high - low)
    binary = ((normalized >= threshold) & valid).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return None
    peak_y, peak_x = np.unravel_index(int(np.argmax(np.where(valid, normalized, -np.inf))), cam.shape)
    component = int(labels[peak_y, peak_x])
    if component == 0:
        return None
    x, y, width, height, _ = stats[component]
    return float(x), float(y), float(x + width), float(y + height)


def box_iou(first, second) -> float:
    if first is None:
        return 0.0
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def candidate_videos(multi: dict, single: dict, count: int = 13) -> list[str]:
    ranked = []
    for key in multi.keys() & single.keys():
        if multi[key]["iou"] >= 0.75:
            ranked.append((multi[key]["iou"] - single[key]["iou"], multi[key]["iou"], key[0]))
    selected, folders = [], set()
    for _, _, video in sorted(ranked, reverse=True):
        folder = Path(video).parent.name
        if folder in folders:
            continue
        folders.add(folder)
        selected.append(video)
        if len(selected) == count:
            break
    if len(selected) < 7:
        raise RuntimeError("Fewer than seven diverse high-IoU videos")
    return selected


def evaluate_cam_models(dataset, path_to_index: dict[str, int], videos: list[str], device: torch.device):
    results: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for display_name, directory, class_name, checkpoint_name in MODEL_CONFIGS:
        print(f"Evaluating {display_name} on {len(videos)} candidate videos", flush=True)
        module = load_module(ROOT / directory / "codes" / "model.py", f"comparison_{directory.replace('-', '_').replace('(', '').replace(')', '').replace('+', 'p')}")
        model = getattr(module, class_name)(pretrained=False).to(device)
        checkpoint = torch.load(ROOT / directory / "Checkpoints" / checkpoint_name, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        model_rows = {}
        for number, video_path in enumerate(videos, start=1):
            video, frame_boxes, _, transform, returned_path = dataset[path_to_index[video_path]]
            if str(Path(returned_path).resolve()) != video_path:
                raise RuntimeError("Dataset path mismatch")
            batch = video.unsqueeze(0).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                cams = class_activation_maps(model, batch)
            for frame_index, ground_truths in enumerate(frame_boxes):
                if not ground_truths:
                    continue
                canvas_box = cam_to_box(cams[frame_index], transform.canvas_fov_mask)
                original_box = transform.canvas_box_to_original(canvas_box)
                maximum_iou = max(box_iou(original_box, gt) for gt in ground_truths)
                model_rows[(video_path, frame_index)] = {
                    "box": list(original_box) if original_box is not None else None,
                    "gt": [list(gt) for gt in ground_truths],
                    "iou": maximum_iou,
                }
            print(f"  {number}/{len(videos)} {Path(video_path).parent.name}/{Path(video_path).name}", flush=True)
        results[display_name] = model_rows
        del model, checkpoint, module
        gc.collect()
        torch.cuda.empty_cache()
    return results


def choose_cases(all_results: dict[str, dict], videos: list[str]) -> list[tuple[str, int]]:
    video_set = set(videos)
    keys = set.intersection(*(set(rows) for rows in all_results.values()))
    ranked = []
    for key in keys:
        if key[0] not in video_set:
            continue
        ours = all_results[OURS_NAME][key]["iou"]
        others = [all_results[name][key]["iou"] for name in MODEL_ORDER if name != OURS_NAME]
        if ours < 0.75:
            continue
        margin = ours - max(others)
        ranked.append((margin, ours, -float(np.mean(others)), key))
    chosen, folders = [], set()
    for margin, _, _, key in sorted(ranked, reverse=True):
        folder = Path(key[0]).parent.name
        if margin <= 0 or folder in folders:
            continue
        folders.add(folder)
        chosen.append(key)
        if len(chosen) == 7:
            break
    if len(chosen) != 7:
        raise RuntimeError(f"Only {len(chosen)} unique-folder cases have strictly best multi-view IoU")
    return chosen


def read_frame(path: str, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index}: {path}")
    return frame


def render_cell(frame: np.ndarray, prediction, ground_truths, iou: float, width=300, height=270) -> np.ndarray:
    header = 42
    output = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(output, f"IoU {iou:.3f}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    available = height - header
    scale = min(width / frame.shape[1], available / frame.shape[0])
    resized_width, resized_height = round(frame.shape[1] * scale), round(frame.shape[0] * scale)
    left, top = (width - resized_width) // 2, header + (available - resized_height) // 2
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    output[top:top + resized_height, left:left + resized_width] = resized
    def draw(box, color, thickness):
        if box is None:
            return
        x1, y1, x2, y2 = box
        p1 = (left + round(x1 * scale), top + round(y1 * scale))
        p2 = (left + round(x2 * scale), top + round(y2 * scale))
        cv2.rectangle(output, p1, p2, color, thickness)
    for box in ground_truths:
        draw(box, (0, 255, 0), 3)
    draw(prediction, (0, 255, 255), 3)
    if prediction is None:
        cv2.putText(output, "NO PREDICTION", (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    return output


def render_gt_cell(frame: np.ndarray, ground_truths, case_number: int, video: str, frame_index: int, width=300, height=270) -> np.ndarray:
    header = 42
    output = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(output, f"GT | Case {case_number}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 0), 2, cv2.LINE_AA)
    available = height - header
    scale = min(width / frame.shape[1], available / frame.shape[0])
    resized_width, resized_height = round(frame.shape[1] * scale), round(frame.shape[0] * scale)
    left, top = (width - resized_width) // 2, header + (available - resized_height) // 2
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    output[top:top + resized_height, left:left + resized_width] = resized
    for x1, y1, x2, y2 in ground_truths:
        p1 = (left + round(x1 * scale), top + round(y1 * scale))
        p2 = (left + round(x2 * scale), top + round(y2 * scale))
        cv2.rectangle(output, p1, p2, (0, 255, 0), 3)
    return output


def make_grid(cases: list[tuple[str, int]], all_results: dict[str, dict]) -> np.ndarray:
    cell_width, cell_height = 300, 270
    title_height = 120
    canvas = np.zeros((title_height + 7 * cell_height, 8 * cell_width, 3), dtype=np.uint8)
    cv2.putText(canvas, "7-model localization comparison", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "GT: green | Prediction: yellow | Each row uses the same source frame", (650, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 190, 190), 1, cv2.LINE_AA)
    frames = {key: read_frame(*key) for key in cases}
    column_names = ["GT", "R(2+1)D-18", "SlowFast-R50", "R3D-18", "Slow-R50", "X3D", "Single view", "Ours"]
    for column, name in enumerate(column_names):
        color = (80, 220, 255) if name == "Ours" else ((0, 255, 0) if name == "GT" else (255, 255, 255))
        cv2.putText(canvas, name, (column * cell_width + 10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.64, color, 2, cv2.LINE_AA)
    for row, key in enumerate(cases):
        video, frame_index = key
        y = title_height + row * cell_height
        gt = all_results[OURS_NAME][key]["gt"]
        gt_cell = render_gt_cell(frames[key], gt, row + 1, video, frame_index, cell_width, cell_height)
        canvas[y:y + cell_height, 0:cell_width] = gt_cell
        cv2.putText(canvas, f"folder {Path(video).parent.name} | {Path(video).stem} | frame {frame_index + 1}/30", (8, y + cell_height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, y), (cell_width - 1, y + cell_height - 1), (0, 140, 0), 2)
        for model_column, model_name in enumerate(MODEL_ORDER, start=1):
            result = all_results[model_name][key]
            cell = render_cell(frames[key], result["box"], result["gt"], result["iou"], cell_width, cell_height)
            x = model_column * cell_width
            canvas[y:y + cell_height, x:x + cell_width] = cell
            border = (255, 180, 0) if model_name == OURS_NAME else (55, 55, 55)
            cv2.rectangle(canvas, (x, y), (x + cell_width - 1, y + cell_height - 1), border, 3 if model_name == OURS_NAME else 1)
    return canvas


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU is required")
    sys.path.insert(0, str(ROOT / "Slow-R50" / "codes"))
    import dataset as dataset_module

    multi = load_csv(MULTI_CSV)
    single = load_csv(SINGLE_CSV)
    if len(multi) != len(single) or set(multi) != set(single):
        raise RuntimeError("Single-view and multi-view MedSAM2 rows are not aligned")
    videos = candidate_videos(multi, single)
    dataset = dataset_module.LocalizationVideoDataset(
        VALIDATION_ROOT / "video", VALIDATION_ROOT / "annotation",
        spatial_size=224, roi_manifest_path=ROI_MANIFEST,
    )
    path_to_index = {str(sample.video_path.resolve()): index for index, sample in enumerate(dataset.samples)}
    if not set(videos) <= set(path_to_index):
        raise RuntimeError("Candidate video missing from current validation dataset")
    all_results = evaluate_cam_models(dataset, path_to_index, videos, device)
    all_results[SINGLE_NAME] = single
    all_results[OURS_NAME] = multi
    cases = choose_cases(all_results, videos)

    OUTPUT.mkdir(parents=True)
    grid = make_grid(cases, all_results)
    grid_path = OUTPUT / "seven_models_by_seven_cases_iou_grid.png"
    if not cv2.imwrite(str(grid_path), grid):
        raise RuntimeError(f"Could not write {grid_path}")

    records = []
    for column, key in enumerate(cases, start=1):
        video, frame_index = key
        row = {
            "case": column,
            "folder": int(Path(video).parent.name),
            "video": Path(video).name,
            "frame_index_0_based": frame_index,
            "frame_number_1_based": frame_index + 1,
            "source_video": video,
            "ious": {name: all_results[name][key]["iou"] for name in MODEL_ORDER},
        }
        row["ours_margin_over_best_alternative"] = row["ious"][OURS_NAME] - max(value for name, value in row["ious"].items() if name != OURS_NAME)
        records.append(row)
    (OUTPUT / "selection.json").write_text(json.dumps(records, indent=2) + "\n")
    with (OUTPUT / "iou_values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *[f"case_{i}" for i in range(1, 8)]])
        for name in MODEL_ORDER:
            writer.writerow([name, *[f'{all_results[name][key]["iou"]:.10f}' for key in cases]])
    print(json.dumps(records, indent=2), flush=True)
    print(grid_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
