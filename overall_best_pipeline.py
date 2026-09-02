#!/usr/bin/env python3
"""X3D-M five-view + MedSAM2 bounding-box CorLoc evaluation pipeline.

Pipeline (no annotation, training, metric, or ROI-preprocessing code):
  1. a frozen ROI-trained X3D-M checkpoint produces a score map for the full view;
  2. score maps from four 144x144 corner views are projected back and max-fused;
  3. use the frame with the greatest fused evidence, then select five NMS peaks;
  4. prompt MedSAM2 with each peak, propagate each mask through time, and score
     each track with frozen X3D-M evidence;
  5. keep X3D-M top-1 unless another track exceeds it by 20%; for low-confidence
     temporal masks (<0.755), replace only that frame with framewise MedSAM2.

The CLI uses the exact Experiment2 validation videos, shared ROI manifest,
aspect-preserving centered 224x224 FOV canvas, annotations, and Kinetics
normalization used to train X3D-M.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent
X3D_CODES = PROJECT_ROOT / "X3D" / "codes"
sys.path.insert(0, str(X3D_CODES))

from cam import box_iou  # noqa: E402
from dataset import (  # noqa: E402
    KINETICS_MEAN,
    KINETICS_STD,
    LocalizationVideoDataset,
)
from model import X3DMBinary  # noqa: E402
from roi import FOVCanvasTransform  # noqa: E402

MODEL_SIZE = 224
DEFAULT_X3D_CHECKPOINT = PROJECT_ROOT / "X3D" / "Checkpoints" / "epoch_014.pth"
DEFAULT_MEDSAM_REPOSITORY = PROJECT_ROOT / "external" / "MedSAM2"
DEFAULT_MEDSAM_CHECKPOINT = (
    DEFAULT_MEDSAM_REPOSITORY / "checkpoints" / "MedSAM2_latest.pt"
)
DEFAULT_VALIDATION_ROOT = PROJECT_ROOT / "datasets" / "ValidationData"
DEFAULT_ROI_MANIFEST = PROJECT_ROOT / "datasets" / "roi_manifest.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "X3D_MedSAM2_epoch014_CorLoc"


class X3DMScoreModel(nn.Module):
    """Frozen X3D-M classifier exposed as a dense one-logit score map."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier_model = X3DMBinary(pretrained=False)
        self.checkpoint_epoch: int | None = None

    def score_map(self, video: torch.Tensor) -> torch.Tensor:
        _, features = self.classifier_model(video, return_features=True)
        weights = self.classifier_model.classifier.weight[0]
        bias = self.classifier_model.classifier.bias[0]
        return torch.einsum("c,bcthw->bthw", weights, features) + bias

    def load_checkpoint(
        self, path: str | Path, roi_manifest: str | Path
    ) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        epoch = int(checkpoint.get("epoch", -1))
        if epoch < 1:
            raise RuntimeError(f"Invalid X3D checkpoint epoch: {epoch}")
        digest = hashlib.sha256(Path(roi_manifest).read_bytes()).hexdigest()
        checkpoint_digest = checkpoint.get("roi_manifest_sha256")
        if checkpoint_digest != digest:
            raise RuntimeError(
                "X3D checkpoint and ROI manifest do not match: "
                f"checkpoint={checkpoint_digest}, current={digest}"
            )
        self.classifier_model.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
        self.checkpoint_epoch = epoch


def preprocess(frames_bgr: list[np.ndarray]) -> torch.Tensor:
    """Apply the exact X3D-M training transform to ROI-cropped BGR frames."""
    tensors: list[torch.Tensor] = []
    mean = torch.tensor(KINETICS_MEAN).view(3, 1, 1)
    std = torch.tensor(KINETICS_STD).view(3, 1, 1)
    for frame in frames_bgr:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR
        )
        image = torch.from_numpy(resized.copy())
        image = image.permute(2, 0, 1).float().div_(255.0)
        tensors.append((image - mean) / std)
    if not tensors:
        raise ValueError("zero input frames")
    return torch.stack(tensors, dim=1)  # (C, T, 224, 224)


def _spatial_attention(logits: torch.Tensor, temperature: float | None) -> tuple[torch.Tensor, torch.Tensor]:
    flat = logits.flatten(2)
    if temperature is None:
        return flat.mean(dim=2), torch.full_like(flat, 1.0 / flat.shape[2]).view_as(logits)
    weights = torch.softmax(flat / temperature, dim=2)
    return (weights * flat).sum(dim=2), weights.view_as(logits)


def _positive_evidence(model: X3DMScoreModel, video: torch.Tensor, frames: int, size: int) -> torch.Tensor:
    logits = model.score_map(video.unsqueeze(0))
    _, attention = _spatial_attention(logits, temperature=0.5)
    evidence = attention * torch.sigmoid(logits)
    return F.interpolate(evidence.unsqueeze(1), size=(frames, size, size), mode="trilinear", align_corners=False)[0, 0]


@torch.inference_mode()
def fused_five_view_maps(model: X3DMScoreModel, video_cpu: torch.Tensor, device: torch.device) -> np.ndarray:
    """Full view + four overlapping 144px corners, fused by a pixelwise maximum."""
    video = video_cpu.to(device)
    frames = video.shape[1]
    full = _positive_evidence(model, video, frames, MODEL_SIZE)
    fused = full.clone()
    for left, top in ((0, 0), (80, 0), (0, 80), (80, 80)):
        crop = video[:, :, top:top + 144, left:left + 144]
        zoomed = F.interpolate(crop.permute(1, 0, 2, 3), size=(MODEL_SIZE, MODEL_SIZE), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
        crop_map = _positive_evidence(model, zoomed, frames, 144)
        projected = torch.zeros_like(full)
        projected[:, top:top + 144, left:left + 144] = crop_map
        fused = torch.maximum(fused, projected)
    return fused.cpu().numpy()


@torch.inference_mode()
def single_view_maps(
    model: X3DMScoreModel,
    video_cpu: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Use only the full 224x224 ROI canvas without corner crops or fusion."""
    video = video_cpu.to(device)
    frames = video.shape[1]
    return _positive_evidence(model, video, frames, MODEL_SIZE).cpu().numpy()


def smooth_heat(raw: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    raw = raw.astype(np.float32)
    low, high = float(raw.min()), float(raw.max())
    heat = (raw - low) / (high - low + 1e-8)
    return cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)


def nms_peaks(heat: np.ndarray, count: int = 5, minimum_distance: int = 16) -> list[tuple[int, int, float]]:
    work = heat.copy()
    peaks: list[tuple[int, int, float]] = []
    for _ in range(count):
        y, x = np.unravel_index(int(np.argmax(work)), work.shape)
        score = float(work[y, x])
        if not np.isfinite(score):
            break
        peaks.append((int(x), int(y), score))
        cv2.circle(work, (int(x), int(y)), minimum_distance, float("-inf"), thickness=-1)
    return peaks


def model_to_frame(point: tuple[int, int], width: int, height: int) -> tuple[float, float]:
    x = point[0] * width / MODEL_SIZE
    y = point[1] * height / MODEL_SIZE
    return min(max(x, 0.0), width - 1.0), min(max(y, 0.0), height - 1.0)


def frame_to_model_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, x2 = round(box[0] * MODEL_SIZE / width), round(box[2] * MODEL_SIZE / width)
    y1, y2 = round(box[1] * MODEL_SIZE / height), round(box[3] * MODEL_SIZE / height)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(MODEL_SIZE - 1, x2), min(MODEL_SIZE - 1, y2)
    return (x1, y1, x2, y2) if x2 >= x1 and y2 >= y1 else None


def mask_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask)
    return None if len(xs) == 0 else (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def load_medsam2(repository: str | Path, checkpoint: str | Path, device: torch.device) -> tuple[Any, Any]:
    repository = Path(repository).resolve()
    checkpoint = Path(checkpoint).resolve()
    if not (repository / "sam2" / "build_sam.py").is_file():
        raise FileNotFoundError(f"Invalid MedSAM2 repository: {repository}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"MedSAM2 checkpoint not found: {checkpoint}")
    sys.path.insert(0, str(repository))
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    video_predictor = build_sam2_video_predictor(
        config_file="configs/sam2.1_hiera_t512.yaml",
        ckpt_path=str(checkpoint),
        device=str(device),
        apply_postprocessing=False,
    )
    return video_predictor, SAM2ImagePredictor(video_predictor)


def point_prompt(image_predictor: Any, frame_bgr: np.ndarray, point: tuple[float, float]) -> tuple[np.ndarray, float]:
    image_predictor.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    masks, scores, _ = image_predictor.predict(
        point_coords=np.asarray([point], dtype=np.float32), point_labels=np.asarray([1], dtype=np.int64), multimask_output=False,
    )
    return np.asarray(masks[0], dtype=bool), float(scores[0])


def propagate(video_predictor: Any, frames_bgr: list[np.ndarray], seed_index: int, seed_mask: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Bidirectionally propagate a MedSAM2 seed mask through one sequence."""
    masks: dict[int, np.ndarray] = {}
    confidences: dict[int, float] = {}
    with tempfile.TemporaryDirectory(prefix="medsam2_sequence_") as directory:
        for index, frame in enumerate(frames_bgr):
            if not cv2.imwrite(str(Path(directory) / f"{index:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                raise RuntimeError("could not stage frame for MedSAM2")
        state = video_predictor.init_state(directory, offload_video_to_cpu=True, offload_state_to_cpu=False, async_loading_frames=False)
        video_predictor.add_new_mask(state, seed_index, obj_id=1, mask=seed_mask)
        for reverse in (False, True):
            for index, _, logits in video_predictor.propagate_in_video(state, start_frame_idx=seed_index, reverse=reverse):
                logits = logits[0].squeeze()
                mask = (logits > 0).detach().cpu().numpy()
                masks[int(index)] = mask
                confidences[int(index)] = float(torch.sigmoid(logits[logits > 0]).mean().item()) if mask.any() else 0.0
        video_predictor.reset_state(state)
    return masks, confidences


def track_score(track: dict[int, np.ndarray], fused_maps: np.ndarray, width: int, height: int) -> float:
    values: list[float] = []
    for index, fmap in enumerate(fused_maps):
        mask = track.get(index)
        box = None if mask is None else mask_box(mask)
        model_box = None if box is None else frame_to_model_box(box, width, height)
        if model_box is None:
            values.append(0.0)
            continue
        x1, y1, x2, y2 = model_box
        values.append(float(fmap[y1:y2 + 1, x1:x2 + 1].mean()))
    return float(np.mean(values))


def localize_best_pipeline(
    frames_bgr: list[np.ndarray],
    video_cpu: torch.Tensor,
    model: X3DMScoreModel,
    video_predictor: Any,
    image_predictor: Any,
    device: torch.device,
    use_five_view: bool = True,
) -> dict[str, Any]:
    """Return the adopted final mask/bbox sequence for one already-ROI-cropped video."""
    if video_cpu.shape != (3, len(frames_bgr), MODEL_SIZE, MODEL_SIZE):
        raise RuntimeError(
            f"Frame/model input mismatch: video={tuple(video_cpu.shape)}, "
            f"frames={len(frames_bgr)}"
        )
    fused_maps = (
        fused_five_view_maps(model, video_cpu, device)
        if use_five_view
        else single_view_maps(model, video_cpu, device)
    )
    seed_index = int(np.argmax([fmap.max() for fmap in fused_maps]))
    heat = smooth_heat(fused_maps[seed_index], sigma=4.0)
    height, width = frames_bgr[0].shape[:2]
    candidates = nms_peaks(heat, count=5, minimum_distance=16)
    tracks, confidences, points = [], [], []
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for x, y, _ in candidates:
            point = model_to_frame((x, y), width, height)
            seed_mask, _ = point_prompt(image_predictor, frames_bgr[seed_index], point)
            track, confidence = propagate(video_predictor, frames_bgr, seed_index, seed_mask)
            tracks.append(track); confidences.append(confidence); points.append(point)
    scores = [track_score(track, fused_maps, width, height) for track in tracks]
    chosen = 0
    if len(scores) > 1:
        alternative = 1 + int(np.argmax(scores[1:]))
        if scores[alternative] > scores[0] * 1.2:  # adopted top-1 margin
            chosen = alternative
    masks, confidence = tracks[chosen], confidences[chosen]
    # Empty/low-confidence temporal masks only: use independent point-prompted MedSAM2 for that frame.
    for index, frame in enumerate(frames_bgr):
        mask = masks.get(index)
        if mask is None or not mask.any() or confidence.get(index, 0.0) < 0.755:
            x, y = np.unravel_index(int(np.argmax(smooth_heat(fused_maps[index], sigma=4.0))), (MODEL_SIZE, MODEL_SIZE))
            fallback_point = model_to_frame((int(y), int(x)), width, height)
            masks[index], confidence[index] = point_prompt(image_predictor, frame, fallback_point)
    return {
        "masks": masks,
        "boxes_xyxy": {index: mask_box(mask) for index, mask in masks.items()},
        "seed_frame_index": seed_index,
        "selected_candidate_index": chosen,
        "selected_seed_point_xy": points[chosen],
        "candidate_track_scores": scores,
        "fused_score_maps": fused_maps,
    }


def decode_canvas_frames(
    path: Path, transform: FOVCanvasTransform
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(transform.apply(frame, path))
    capture.release()
    if not frames:
        raise RuntimeError(f"No decodable frames: {path}")
    return frames


def canvas_box_to_original(
    box: tuple[float, float, float, float] | None,
    transform: FOVCanvasTransform,
) -> tuple[float, float, float, float] | None:
    return transform.canvas_box_to_original(box)


def format_box(box: tuple[float, float, float, float] | None) -> str:
    return "" if box is None else json.dumps([round(value, 4) for value in box])


def write_rows_atomic(path: Path, rows: dict[tuple[str, int], dict[str, str]]) -> None:
    columns = (
        "video_path",
        "frame_index",
        "predicted_box_xyxy",
        "ground_truth_boxes_xyxy",
        "max_iou",
        "correct_0.3",
        "correct_0.5",
        "correct_0.7",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])
    os.replace(temporary, path)


def load_existing_rows(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        loaded = list(csv.DictReader(handle))
    return {
        (row["video_path"], int(row["frame_index"])): row
        for row in loaded
    }


def metrics_from_rows(rows: dict[tuple[str, int], dict[str, str]]) -> dict[str, float | int]:
    values = list(rows.values())
    if not values:
        return {
            "evaluated_frames": 0,
            "mean_iou": 0.0,
            "corloc_0.3": 0.0,
            "corloc_0.5": 0.0,
            "corloc_0.7": 0.0,
        }
    count = len(values)
    return {
        "evaluated_frames": count,
        "mean_iou": sum(float(row["max_iou"]) for row in values) / count,
        "corloc_0.3": sum(int(row["correct_0.3"]) for row in values) / count,
        "corloc_0.5": sum(int(row["correct_0.5"]) for row in values) / count,
        "corloc_0.7": sum(int(row["correct_0.7"]) for row in values) / count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate X3D-M + MedSAM2 boxes with single-view or five-view CAM evidence."
    )
    parser.add_argument("--x3d-checkpoint", type=Path, default=DEFAULT_X3D_CHECKPOINT)
    parser.add_argument("--medsam-repository", type=Path, default=DEFAULT_MEDSAM_REPOSITORY)
    parser.add_argument("--medsam-checkpoint", type=Path, default=DEFAULT_MEDSAM_CHECKPOINT)
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--roi-manifest", type=Path, default=DEFAULT_ROI_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument(
        "--single-view",
        action="store_true",
        help="Use only the full 224x224 view; disable four corner crops and max fusion.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_videos is not None and args.max_videos < 1:
        raise ValueError("max-videos must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU is required for MedSAM2 evaluation")

    dataset = LocalizationVideoDataset(
        args.validation_root / "video",
        args.validation_root / "annotation",
        spatial_size=MODEL_SIZE,
        roi_manifest_path=args.roi_manifest,
    )
    model = X3DMScoreModel()
    model.load_checkpoint(args.x3d_checkpoint, args.roi_manifest)
    model.to(device).eval()
    video_predictor, image_predictor = load_medsam2(
        args.medsam_repository, args.medsam_checkpoint, device
    )

    output_csv = args.output_dir / "per_frame_corloc.csv"
    summary_path = args.output_dir / "summary.json"
    if output_csv.exists() and not args.resume:
        raise FileExistsError(
            f"Output exists; pass --resume or choose a new output-dir: {output_csv}"
        )
    rows = load_existing_rows(output_csv) if args.resume else {}
    completed_videos = {key[0] for key in rows}
    indices = list(range(len(dataset)))
    if args.max_videos is not None:
        indices = indices[: args.max_videos]

    started = time.time()
    for position, dataset_index in enumerate(indices, start=1):
        video, frame_boxes, original_size, transform, path_string = dataset[dataset_index]
        roi = transform.roi
        if path_string in completed_videos:
            print(f"[{position}/{len(indices)}] resume skip: {path_string}", flush=True)
            continue
        path = Path(path_string)
        frames_bgr = decode_canvas_frames(path, transform)
        if len(frames_bgr) != video.shape[1] or len(frame_boxes) != video.shape[1]:
            raise RuntimeError(
                f"Frame mismatch for {path}: bgr={len(frames_bgr)}, "
                f"model={video.shape[1]}, annotations={len(frame_boxes)}"
            )
        if original_size != (roi.source_height, roi.source_width):
            raise RuntimeError(f"ROI/source mismatch for {path}")

        result = localize_best_pipeline(
            frames_bgr,
            video,
            model,
            video_predictor,
            image_predictor,
            device,
            use_five_view=not args.single_view,
        )
        predicted_canvas_boxes = result["boxes_xyxy"]
        for frame_index, ground_truths in enumerate(frame_boxes):
            if not ground_truths:
                continue
            predicted = canvas_box_to_original(
                predicted_canvas_boxes.get(frame_index), transform
            )
            maximum_iou = max(box_iou(predicted, box) for box in ground_truths)
            rows[(path_string, frame_index)] = {
                "video_path": path_string,
                "frame_index": str(frame_index),
                "predicted_box_xyxy": format_box(predicted),
                "ground_truth_boxes_xyxy": json.dumps(ground_truths),
                "max_iou": f"{maximum_iou:.10f}",
                "correct_0.3": str(int(maximum_iou >= 0.3)),
                "correct_0.5": str(int(maximum_iou >= 0.5)),
                "correct_0.7": str(int(maximum_iou >= 0.7)),
            }
        completed_videos.add(path_string)
        write_rows_atomic(output_csv, rows)
        metrics = metrics_from_rows(rows)
        summary = {
            "model": "X3D-M",
            "x3d_checkpoint": str(args.x3d_checkpoint),
            "x3d_epoch": model.checkpoint_epoch,
            "medsam_checkpoint": str(args.medsam_checkpoint),
            "view_mode": "single_view" if args.single_view else "five_view",
            "five_view_used": not args.single_view,
            "processed_videos": len(completed_videos),
            **metrics,
        }
        temporary_summary = summary_path.with_suffix(".tmp.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_summary.write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(temporary_summary, summary_path)
        print(
            f"[{position}/{len(indices)}] {path.name}: "
            f"frames={metrics['evaluated_frames']}, "
            f"CorLoc@0.3={metrics['corloc_0.3']:.6f}, "
            f"@0.5={metrics['corloc_0.5']:.6f}, "
            f"@0.7={metrics['corloc_0.7']:.6f}, "
            f"elapsed={(time.time() - started) / 60:.2f} min",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
