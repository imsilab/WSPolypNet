from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from cam import box_iou, cam_to_box, class_activation_maps
from dataset import KINETICS_MEAN, KINETICS_STD, LocalizationVideoDataset
from model import R3D18Binary
from roi import Box, FOVCanvasTransform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "datasets"
PANEL_SIZE = 224
HEADER_HEIGHT = 24


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def original_box_to_input(
    box: Box, transform: FOVCanvasTransform
) -> tuple[int, int, int, int]:
    mapped = transform.original_box_to_canvas(box)
    return tuple(int(round(np.clip(value, 0, PANEL_SIZE - 1))) for value in mapped)


def tensor_frames_to_bgr(video: torch.Tensor) -> list[np.ndarray]:
    array = video.permute(1, 2, 3, 0).cpu().numpy()
    mean = np.asarray(KINETICS_MEAN, dtype=np.float32)
    std = np.asarray(KINETICS_STD, dtype=np.float32)
    rgb = np.clip((array * std + mean) * 255.0, 0, 255).astype(np.uint8)
    return [np.ascontiguousarray(frame[:, :, ::-1]) for frame in rgb]


def normalized_cam(cam: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if valid_mask.shape != cam.shape or not valid_mask.any():
        raise ValueError(f"Invalid CAM mask shape: cam={cam.shape}, mask={valid_mask.shape}")
    minimum = float(cam[valid_mask].min())
    maximum = float(cam[valid_mask].max())
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        return np.zeros_like(cam, dtype=np.float32)
    normalized = np.zeros_like(cam, dtype=np.float32)
    normalized[valid_mask] = (cam[valid_mask] - minimum) / (maximum - minimum)
    return normalized


def choose_annotated_frames(frame_boxes: list[list[Box]], count: int) -> list[int]:
    annotated = [index for index, boxes in enumerate(frame_boxes) if boxes]
    if not annotated:
        raise RuntimeError("Selected validation video has no annotated frames")
    if len(annotated) <= count:
        return annotated
    positions = np.linspace(0, len(annotated) - 1, count).round().astype(int)
    return [annotated[index] for index in positions]


def label_bar(text: str, color: tuple[int, int, int]) -> np.ndarray:
    bar = np.full((HEADER_HEIGHT, PANEL_SIZE, 3), 18, dtype=np.uint8)
    cv2.putText(
        bar,
        text,
        (5, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        color,
        1,
        cv2.LINE_AA,
    )
    return bar


@torch.inference_mode()
def make_grid(
    model: R3D18Binary,
    dataset: LocalizationVideoDataset,
    video_index: int,
    device: torch.device,
    threshold: float,
    frames_per_grid: int,
) -> tuple[np.ndarray, str]:
    video, frame_boxes, original_size, transform, path = dataset[video_index]
    batch = video.unsqueeze(0).to(device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits, cams = class_activation_maps(model, batch, target_class=1)
    probability = float(torch.sigmoid(logits.float())[0, 0].cpu())
    prediction = "polyp" if probability >= 0.5 else "normal"
    cams_np = cams[0].float().cpu().numpy()
    frames = tensor_frames_to_bgr(video)
    selected = choose_annotated_frames(frame_boxes, frames_per_grid)
    valid_mask = transform.canvas_fov_mask
    rows: list[np.ndarray] = []

    for frame_index in selected:
        frame = frames[frame_index]
        ground_truths = frame_boxes[frame_index]
        cam_norm = normalized_cam(cams_np[frame_index], valid_mask)
        heatmap = cv2.applyColorMap(
            np.round(cam_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        activation = cv2.addWeighted(frame, 0.52, heatmap, 0.48, 0.0)

        gt_panel = frame.copy()
        for box in ground_truths:
            gx1, gy1, gx2, gy2 = original_box_to_input(box, transform)
            cv2.rectangle(gt_panel, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)

        predicted_input = cam_to_box(
            cams_np[frame_index], threshold, valid_mask=valid_mask
        )
        predicted_original = transform.canvas_box_to_original(predicted_input)
        maximum_iou = max(box_iou(predicted_original, box) for box in ground_truths)
        prediction_panel = frame.copy()
        if predicted_input is not None:
            px1, py1, px2, py2 = (int(round(value)) for value in predicted_input)
            cv2.rectangle(
                prediction_panel,
                (max(0, px1), max(0, py1)),
                (min(PANEL_SIZE - 1, px2), min(PANEL_SIZE - 1, py2)),
                (0, 255, 0),
                2,
            )

        left = np.vstack(
            [label_bar(f"GT f={frame_index + 1:04d} boxes={len(ground_truths)}", (80, 150, 255)), gt_panel]
        )
        middle = np.vstack(
            [label_bar(f"POLYP CAM threshold={threshold:.2f}", (0, 210, 255)), activation]
        )
        right = np.vstack(
            [label_bar(f"PRED {prediction} p={probability:.3f} IoU={maximum_iou:.3f}", (80, 255, 80)), prediction_panel]
        )
        rows.append(np.hstack([left, middle, right]))

    return np.vstack(rows), path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create GT/CAM/prediction grids for validation videos.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "logs/grid/epoch1")
    parser.add_argument("--video-count", type=int, default=10)
    parser.add_argument("--frames-per-grid", type=int, default=10)
    parser.add_argument("--cam-threshold", type=float, default=0.5)
    parser.add_argument("--spatial-size", type=int, default=PANEL_SIZE)
    parser.add_argument("--roi-manifest", type=Path, default=DATA_ROOT / "roi_manifest.json")
    parser.add_argument("--validation-root", type=Path, default=DATA_ROOT / "ValidationData")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.video_count < 1 or args.frames_per_grid < 1:
        raise ValueError("video-count and frames-per-grid must be positive")
    if args.spatial_size != PANEL_SIZE:
        raise ValueError(f"This grid layout requires spatial-size={PANEL_SIZE}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LocalizationVideoDataset(
        args.validation_root / "video",
        args.validation_root / "annotation",
        spatial_size=args.spatial_size,
        roi_manifest_path=args.roi_manifest,
    )
    if args.video_count > len(dataset):
        raise ValueError(f"Requested {args.video_count} videos from a dataset of {len(dataset)}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    current_roi_hash = file_sha256(args.roi_manifest)
    checkpoint_roi_hash = checkpoint.get("roi_manifest_sha256")
    if checkpoint_roi_hash != current_roi_hash:
        raise RuntimeError(
            "Checkpoint and grid ROI manifests do not match: "
            f"checkpoint={checkpoint_roi_hash}, current={current_roi_hash}"
        )
    model = R3D18Binary(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    indices = np.linspace(0, len(dataset) - 1, args.video_count).round().astype(int)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for old in args.output_dir.glob("video_*_grid.jpg"):
        old.unlink()
    for output_index, dataset_index in enumerate(indices, start=1):
        grid, source = make_grid(
            model,
            dataset,
            int(dataset_index),
            device,
            args.cam_threshold,
            args.frames_per_grid,
        )
        output = args.output_dir / f"video_{output_index:02d}_grid.jpg"
        if not cv2.imwrite(str(output), grid, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Failed to save grid: {output}")
        print(f"[{output_index}/{args.video_count}] dataset_index={dataset_index}, source={source} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

