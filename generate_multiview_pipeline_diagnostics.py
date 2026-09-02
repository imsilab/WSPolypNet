#!/usr/bin/env python3
"""Export 14-stage X3D + MedSAM2 five-view diagnostic images per video."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import overall_best_pipeline as pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "X3D" / "Checkpoints" / "epoch_014.pth"
DEFAULT_RESULT_CSV = PROJECT_ROOT / "X3D_MedSAM2_epoch014_CorLoc" / "per_frame_corloc.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "sample" / "x3d_medsam2_multiview_diagnostics"
CORNERS = (
    ("top_left", 0, 0),
    ("top_right", 80, 0),
    ("bottom_left", 0, 80),
    ("bottom_right", 80, 80),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--result-csv", type=Path, default=DEFAULT_RESULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--validation-root", type=Path, default=pipeline.DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--roi-manifest", type=Path, default=pipeline.DEFAULT_ROI_MANIFEST)
    parser.add_argument("--medsam-repository", type=Path, default=pipeline.DEFAULT_MEDSAM_REPOSITORY)
    parser.add_argument("--medsam-checkpoint", type=Path, default=pipeline.DEFAULT_MEDSAM_CHECKPOINT)
    return parser.parse_args()


def label(image: np.ndarray, text: str, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 27), (0, 0, 0), thickness=-1)
    cv2.putText(output, text, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    return output


def make_grid(
    images: list[np.ndarray],
    title: str,
    columns: int = 5,
    tile_size: int = 224,
) -> np.ndarray:
    if not images:
        raise ValueError("Cannot make an empty grid")
    rows = math.ceil(len(images) / columns)
    title_height = 52
    grid = np.zeros((title_height + rows * tile_size, columns * tile_size, 3), dtype=np.uint8)
    cv2.putText(grid, title, (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    for index, image in enumerate(images):
        tile = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        tile = label(tile, f"frame {index + 1:02d}/{len(images):02d}")
        top = title_height + (index // columns) * tile_size
        left = (index % columns) * tile_size
        grid[top : top + tile_size, left : left + tile_size] = tile
    return grid


def heat_overlay(frame: np.ndarray, raw_map: np.ndarray) -> np.ndarray:
    heat = pipeline.smooth_heat(raw_map, sigma=4.0)
    colored = cv2.applyColorMap(np.uint8(np.clip(heat, 0.0, 1.0) * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.55, colored, 0.45, 0.0)


def mask_overlay(frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    output = frame.copy()
    if mask is None or not np.asarray(mask).any():
        return label(output, "empty mask", (0, 0, 255))
    binary = np.asarray(mask, dtype=bool)
    box = pipeline.mask_box(binary)
    if box is not None:
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return output


def write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


def select_candidates(
    dataset: pipeline.LocalizationVideoDataset,
    result_csv: Path,
    count: int,
) -> list[dict[str, Any]]:
    current = {str(sample.video_path.resolve()): sample for sample in dataset.samples}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with result_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            path = str(Path(row["video_path"]).resolve())
            if path in current:
                grouped[path].append(row)
    candidates = []
    for order, sample in enumerate(dataset.samples, start=1):
        path = str(sample.video_path.resolve())
        rows = grouped.get(path, [])
        if len(sample.annotation_paths) != 30 or not rows:
            continue
        candidates.append(
            {
                "dataset_order": order,
                "sample": sample,
                "path": path,
                "annotated_frames": len(rows),
                "mean_iou": sum(float(row["max_iou"]) for row in rows) / len(rows),
                "corloc_0.5": sum(int(row["correct_0.5"]) for row in rows) / len(rows),
                "corloc_0.7": sum(int(row["correct_0.7"]) for row in rows) / len(rows),
            }
        )
    candidates.sort(
        key=lambda item: (item["mean_iou"], item["corloc_0.7"], item["corloc_0.5"]),
        reverse=True,
    )
    selected = []
    used_folders: set[int] = set()
    for item in candidates:
        folder = int(item["sample"].video_path.parent.name)
        if folder in used_folders:
            continue
        selected.append(item)
        used_folders.add(folder)
        if len(selected) == count:
            break
    if len(selected) < count:
        selected_paths = {item["path"] for item in selected}
        selected.extend(item for item in candidates if item["path"] not in selected_paths)
        selected = selected[:count]
    if len(selected) < count:
        raise RuntimeError(f"Only {len(selected)} eligible 30-frame videos were found")
    return selected


@torch.inference_mode()
def intermediate_views(
    model: pipeline.X3DMScoreModel,
    video_cpu: torch.Tensor,
    frames_bgr: list[np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    video = video_cpu.to(device)
    frame_count = video.shape[1]
    full_map = pipeline._positive_evidence(model, video, frame_count, pipeline.MODEL_SIZE)
    fused = full_map.clone()
    crop_frames: dict[str, list[np.ndarray]] = {}
    crop_maps: dict[str, np.ndarray] = {}
    for name, left, top in CORNERS:
        crop = video[:, :, top : top + 144, left : left + 144]
        zoomed = F.interpolate(
            crop.permute(1, 0, 2, 3),
            size=(pipeline.MODEL_SIZE, pipeline.MODEL_SIZE),
            mode="bilinear",
            align_corners=False,
        ).permute(1, 0, 2, 3)
        local_map = pipeline._positive_evidence(model, zoomed, frame_count, 144)
        projected = torch.zeros_like(full_map)
        projected[:, top : top + 144, left : left + 144] = local_map
        fused = torch.maximum(fused, projected)
        display_maps = F.interpolate(
            local_map.unsqueeze(1),
            size=(pipeline.MODEL_SIZE, pipeline.MODEL_SIZE),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        crop_maps[name] = display_maps.float().cpu().numpy()
        crop_frames[name] = [
            cv2.resize(
                frame[top : top + 144, left : left + 144],
                (pipeline.MODEL_SIZE, pipeline.MODEL_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
            for frame in frames_bgr
        ]
    return {
        "full_map": full_map.float().cpu().numpy(),
        "crop_frames": crop_frames,
        "crop_maps": crop_maps,
        "fused_map": fused.float().cpu().numpy(),
    }


def single_frame_canvas(image: np.ndarray, title: str, details: list[str]) -> np.ndarray:
    size = 720
    header = 115
    canvas = np.zeros((header + size, size, 3), dtype=np.uint8)
    cv2.putText(canvas, title, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    for line_index, detail in enumerate(details):
        cv2.putText(canvas, detail, (16, 65 + line_index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
    canvas[header:, :] = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return canvas


def export_video(
    item: dict[str, Any],
    dataset: pipeline.LocalizationVideoDataset,
    dataset_index: int,
    model: pipeline.X3DMScoreModel,
    video_predictor: Any,
    image_predictor: Any,
    device: torch.device,
    output_root: Path,
    rank: int,
) -> dict[str, Any]:
    video, frame_boxes, _, transform, path_string = dataset[dataset_index]
    frames_bgr = pipeline.decode_canvas_frames(Path(path_string), transform)
    if len(frames_bgr) != 30 or video.shape[1] != 30:
        raise RuntimeError(f"Expected exactly 30 frames: {path_string}")

    result = pipeline.localize_best_pipeline(
        frames_bgr,
        video,
        model,
        video_predictor,
        image_predictor,
        device,
        use_five_view=True,
    )
    intermediates = intermediate_views(model, video, frames_bgr, device)
    if not np.allclose(result["fused_score_maps"], intermediates["fused_map"], rtol=1e-5, atol=1e-6):
        raise RuntimeError("Diagnostic fused map does not match the pipeline fused map")

    folder = Path(path_string).parent.name
    stem = Path(path_string).stem
    output = output_root / f"rank_{rank:02d}__folder_{folder}__{stem}"
    output.mkdir(parents=True, exist_ok=False)
    prefix = f"folder {folder} | {stem} | eval order {item['dataset_order']} | prior mean IoU {item['mean_iou']:.4f}"

    write_image(output / "01_full_view_roi_input_grid.png", make_grid(frames_bgr, f"01 Full ROI/model input | {prefix}"))
    for file_index, (name, _, _) in enumerate(CORNERS, start=2):
        write_image(
            output / f"{file_index:02d}_{name}_crop_input_grid.png",
            make_grid(intermediates["crop_frames"][name], f"{file_index:02d} {name} crop 144->224 | {prefix}"),
        )

    full_cam_frames = [
        heat_overlay(frame, intermediates["full_map"][index])
        for index, frame in enumerate(frames_bgr)
    ]
    write_image(output / "06_full_view_cam_grid.png", make_grid(full_cam_frames, f"06 Full-view CAM | {prefix}"))
    for file_index, (name, _, _) in enumerate(CORNERS, start=7):
        cam_frames = [
            heat_overlay(frame, intermediates["crop_maps"][name][index])
            for index, frame in enumerate(intermediates["crop_frames"][name])
        ]
        write_image(
            output / f"{file_index:02d}_{name}_crop_cam_grid.png",
            make_grid(cam_frames, f"{file_index:02d} {name} crop CAM | {prefix}"),
        )

    fused_frames = [
        heat_overlay(frame, intermediates["fused_map"][index])
        for index, frame in enumerate(frames_bgr)
    ]
    write_image(output / "11_five_view_fused_cam_grid.png", make_grid(fused_frames, f"11 Five-view max-fused CAM | {prefix}"))

    seed_index = int(result["seed_frame_index"])
    selected_frame = fused_frames[seed_index].copy()
    cv2.rectangle(selected_frame, (2, 2), (221, 221), (0, 255, 255), 4)
    write_image(
        output / "12_selected_seed_frame.png",
        single_frame_canvas(
            selected_frame,
            "12 Selected seed frame CAM",
            [f"frame index: {seed_index + 1}/30", prefix],
        ),
    )

    point_frame = fused_frames[seed_index].copy()
    heat = pipeline.smooth_heat(intermediates["fused_map"][seed_index], sigma=4.0)
    candidates = pipeline.nms_peaks(heat, count=5, minimum_distance=16)
    height, width = frames_bgr[seed_index].shape[:2]
    chosen = int(result["selected_candidate_index"])
    for candidate_index, (x, y, _) in enumerate(candidates):
        px, py = pipeline.model_to_frame((x, y), width, height)
        center = (int(round(px)), int(round(py)))
        selected = candidate_index == chosen
        cv2.circle(point_frame, center, 10 if selected else 6, (0, 0, 255) if selected else (0, 255, 255), 3)
        cv2.putText(
            point_frame,
            f"P{candidate_index + 1}",
            (center[0] + 7, center[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255) if selected else (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    selected_point = result["selected_seed_point_xy"]
    write_image(
        output / "13_selected_point_on_seed_frame.png",
        single_frame_canvas(
            point_frame,
            "13 Candidate points and selected point",
            [
                f"selected: P{chosen + 1}, xy=({selected_point[0]:.1f}, {selected_point[1]:.1f})",
                "red=selected, yellow=other NMS candidates",
            ],
        ),
    )

    final_mask_frames = [
        mask_overlay(frame, result["masks"].get(index))
        for index, frame in enumerate(frames_bgr)
    ]
    write_image(output / "14_final_medsam2_mask_grid.png", make_grid(final_mask_frames, f"14 Final MedSAM2 bounding boxes | {prefix}"))

    manifest = {
        "rank_by_prior_five_view_mean_iou": rank,
        "source_video": path_string,
        "validation_dataset_order_after_pruning": item["dataset_order"],
        "folder": int(folder),
        "video_name": Path(path_string).name,
        "frame_count": len(frames_bgr),
        "prior_evaluation": {
            "annotated_frames": item["annotated_frames"],
            "mean_iou": item["mean_iou"],
            "corloc_0.5": item["corloc_0.5"],
            "corloc_0.7": item["corloc_0.7"],
        },
        "pipeline": "X3D-M epoch 14 + five-view max fusion + MedSAM2",
        "five_views": ["full", "top_left", "top_right", "bottom_left", "bottom_right"],
        "seed_frame_index_0_based": seed_index,
        "seed_frame_number_1_based": seed_index + 1,
        "selected_candidate_index_0_based": chosen,
        "selected_candidate_number_1_based": chosen + 1,
        "selected_seed_point_xy": list(selected_point),
        "candidate_track_scores": result["candidate_track_scores"],
        "files": [path.name for path in sorted(output.glob("*.png"))],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU is required")

    dataset = pipeline.LocalizationVideoDataset(
        args.validation_root / "video",
        args.validation_root / "annotation",
        spatial_size=pipeline.MODEL_SIZE,
        roi_manifest_path=args.roi_manifest,
    )
    selected = select_candidates(dataset, args.result_csv, args.count)
    args.output_dir.mkdir(parents=True)
    selection = [
        {
            "rank": rank,
            "dataset_order": item["dataset_order"],
            "folder": int(item["sample"].video_path.parent.name),
            "video_name": item["sample"].video_path.name,
            "path": item["path"],
            "frame_count": len(item["sample"].annotation_paths),
            "annotated_frames": item["annotated_frames"],
            "mean_iou": item["mean_iou"],
            "corloc_0.5": item["corloc_0.5"],
            "corloc_0.7": item["corloc_0.7"],
        }
        for rank, item in enumerate(selected, start=1)
    ]
    (args.output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2), flush=True)

    model = pipeline.X3DMScoreModel()
    model.load_checkpoint(args.checkpoint, args.roi_manifest)
    model.to(device).eval()
    video_predictor, image_predictor = pipeline.load_medsam2(
        args.medsam_repository,
        args.medsam_checkpoint,
        device,
    )
    path_to_index = {
        str(sample.video_path.resolve()): index
        for index, sample in enumerate(dataset.samples)
    }
    manifests = []
    for rank, item in enumerate(selected, start=1):
        print(f"[{rank}/{len(selected)}] exporting {item['path']}", flush=True)
        manifests.append(
            export_video(
                item,
                dataset,
                path_to_index[item["path"]],
                model,
                video_predictor,
                image_predictor,
                device,
                args.output_dir,
                rank,
            )
        )
    (args.output_dir / "index.json").write_text(json.dumps(manifests, indent=2) + "\n")
    print(f"Exported {len(manifests)} videos to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
