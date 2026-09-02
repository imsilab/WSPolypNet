from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from roi import (
    FOVCanvasTransform,
    FolderROI,
    load_roi_manifest,
    make_canvas_transform,
    roi_for_video,
)


# PyTorchVideo Slow R50 Kinetics-400 normalization values.
KINETICS_MEAN = (0.45, 0.45, 0.45)
KINETICS_STD = (0.225, 0.225, 0.225)


def natural_key(value: str | Path) -> tuple[tuple[int, object], ...]:
    text = value.name if isinstance(value, Path) else str(value)
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", text)
        if part
    )


def natural_paths(paths: Sequence[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: tuple(natural_key(x) for x in p.parts))


def discover_mp4s(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Video root does not exist: {root}")
    return natural_paths(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4"]
    )


def video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    if count <= 0:
        raise RuntimeError(f"Invalid frame count ({count}) in {path}")
    return count


def decode_entire_video(
    path: Path,
    spatial_size: int,
    canvas_transform: FOVCanvasTransform,
) -> tuple[torch.Tensor, tuple[int, int], FOVCanvasTransform]:
    """Decode every frame and apply one fixed FOV/canvas transform to the video."""
    if canvas_transform.canvas_size != spatial_size:
        raise ValueError(
            f"Canvas/spatial-size mismatch: {canvas_transform.canvas_size} vs {spatial_size}"
        )
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frames: list[np.ndarray] = []
    original_size: tuple[int, int] | None = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        if original_size is None:
            original_size = (height, width)
        elif original_size != (height, width):
            capture.release()
            raise RuntimeError(
                f"Resolution changes within {path}: {original_size} -> {(height, width)}"
            )
        frame = canvas_transform.apply(frame, path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    capture.release()

    if not frames or original_size is None:
        raise RuntimeError(f"No decodable frames in {path}")

    # [T,H,W,C] -> [C,T,H,W]
    array = np.stack(frames).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()
    mean = torch.tensor(KINETICS_MEAN, dtype=tensor.dtype).view(3, 1, 1, 1)
    std = torch.tensor(KINETICS_STD, dtype=tensor.dtype).view(3, 1, 1, 1)
    tensor = (tensor - mean) / std
    return tensor, original_size, canvas_transform


@dataclass(frozen=True)
class TrainingSample:
    video_path: Path
    label: int
    roi: FolderROI


class BinaryVideoDataset(Dataset):
    """All positive and negative MP4s; one entire video is one sample."""

    def __init__(
        self,
        negative_root: Path,
        positive_root: Path,
        spatial_size: int = 224,
        horizontal_flip_probability: float = 0.5,
        canvas_scale_range: tuple[float, float] = (0.90, 1.00),
        roi_manifest_path: Path | None = None,
    ) -> None:
        negatives = discover_mp4s(negative_root)
        positives = discover_mp4s(positive_root)
        if not negatives or not positives:
            raise RuntimeError(
                f"Both classes must be non-empty: negative={len(negatives)}, "
                f"positive={len(positives)}"
            )
        if roi_manifest_path is None:
            raise ValueError("roi_manifest_path is required for ROI-based training")
        roi_entries = load_roi_manifest(roi_manifest_path)
        self.samples = [
            TrainingSample(path, 0, roi_for_video(roi_entries, "negative", path))
            for path in negatives
        ] + [
            TrainingSample(path, 1, roi_for_video(roi_entries, "positive", path))
            for path in positives
        ]
        self.spatial_size = spatial_size
        self.horizontal_flip_probability = horizontal_flip_probability
        self.canvas_scale_range = canvas_scale_range
        minimum_scale, maximum_scale = canvas_scale_range
        if not 0 < minimum_scale <= maximum_scale <= 1:
            raise ValueError(
                f"canvas_scale_range must satisfy 0 < min <= max <= 1, got {canvas_scale_range}"
            )
        self.class_counts = {0: len(negatives), 1: len(positives)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        flip = bool(
            self.horizontal_flip_probability > 0
            and torch.rand(()) < self.horizontal_flip_probability
        )
        minimum_scale, maximum_scale = self.canvas_scale_range
        scale = minimum_scale + (maximum_scale - minimum_scale) * float(torch.rand(()))
        canvas_transform = make_canvas_transform(
            sample.roi,
            self.spatial_size,
            scale=scale,
            position_x=float(torch.rand(())),
            position_y=float(torch.rand(())),
            horizontal_flip=flip,
        )
        video, _, _ = decode_entire_video(
            sample.video_path,
            self.spatial_size,
            canvas_transform,
        )
        return video, sample.label, str(sample.video_path)


Box = tuple[float, float, float, float]


def read_boxes(path: Path) -> list[Box]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Empty annotation: {path}")
    try:
        declared = int(lines[0])
        boxes = [tuple(float(value) for value in line.split()) for line in lines[1:]]
    except ValueError as exc:
        raise RuntimeError(f"Malformed annotation {path}: {exc}") from exc
    if declared != len(boxes) or any(len(box) != 4 for box in boxes):
        raise RuntimeError(
            f"Annotation count mismatch in {path}: declared={declared}, parsed={len(boxes)}"
        )
    return [box for box in boxes]  # type: ignore[list-item]


@dataclass(frozen=True)
class ValidationSample:
    video_path: Path
    annotation_paths: tuple[Path, ...]
    folder_number: int
    roi: FolderROI


class LocalizationVideoDataset(Dataset):
    """Validation videos mapped sequentially to their frame-level bbox text files."""

    def __init__(
        self,
        video_root: Path,
        annotation_root: Path,
        spatial_size: int = 224,
        roi_manifest_path: Path | None = None,
    ) -> None:
        self.spatial_size = spatial_size
        self.samples: list[ValidationSample] = []
        if roi_manifest_path is None:
            raise ValueError("roi_manifest_path is required for ROI-based validation")
        roi_entries = load_roi_manifest(roi_manifest_path)

        video_folders = sorted(
            [path for path in video_root.iterdir() if path.is_dir() and path.name.isdigit()],
            key=lambda p: int(p.name),
        )
        for video_folder in video_folders:
            folder_number = int(video_folder.name)
            annotation_folder = annotation_root / video_folder.name
            if not annotation_folder.is_dir():
                raise RuntimeError(f"Missing annotation folder: {annotation_folder}")
            videos = sorted(video_folder.glob("*.mp4"), key=natural_key)
            annotations = sorted(annotation_folder.glob("*.txt"), key=natural_key)
            offset = 0
            for video_path in videos:
                frame_count = video_frame_count(video_path)
                chunk = annotations[offset : offset + frame_count]
                if len(chunk) != frame_count:
                    raise RuntimeError(
                        f"Not enough annotations for {video_path}: "
                        f"frames={frame_count}, annotations={len(chunk)}"
                    )
                self.samples.append(
                    ValidationSample(
                        video_path,
                        tuple(chunk),
                        folder_number,
                        roi_for_video(roi_entries, "validation", video_path),
                    )
                )
                offset += frame_count
            if offset != len(annotations):
                raise RuntimeError(
                    f"Unused annotations in folder {folder_number}: "
                    f"used={offset}, available={len(annotations)}"
                )
        if not self.samples:
            raise RuntimeError(f"No validation videos found under {video_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        list[list[Box]],
        tuple[int, int],
        FOVCanvasTransform,
        str,
    ]:
        sample = self.samples[index]
        canvas_transform = make_canvas_transform(
            sample.roi,
            self.spatial_size,
            scale=1.0,
            position_x=0.5,
            position_y=0.5,
            horizontal_flip=False,
        )
        video, original_size, canvas_transform = decode_entire_video(
            sample.video_path,
            self.spatial_size,
            canvas_transform,
        )
        if video.shape[1] != len(sample.annotation_paths):
            raise RuntimeError(
                f"Decoded frame/annotation mismatch for {sample.video_path}: "
                f"{video.shape[1]} vs {len(sample.annotation_paths)}"
            )
        boxes = [read_boxes(path) for path in sample.annotation_paths]
        return video, boxes, original_size, canvas_transform, str(sample.video_path)
