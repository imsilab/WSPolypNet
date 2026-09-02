from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from dataset import natural_key


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "datasets"
DEFAULT_DATASETS = {
    "negative": DATA_ROOT / "TrainVaild(video)_without_polyp",
    "positive": DATA_ROOT / "TrainValid(video)_with_polyp",
    "validation": DATA_ROOT / "ValidationData" / "video",
}


def largest_inner_rectangle(binary: np.ndarray) -> tuple[int, int, int, int]:
    """Largest axis-aligned rectangle entirely inside a binary FOV mask."""
    height, width = binary.shape
    histogram = np.zeros(width, dtype=np.int32)
    best_area = 0
    best = (0, 0, width, height)
    for row in range(height):
        histogram = np.where(binary[row] > 0, histogram + 1, 0)
        stack: list[int] = []
        for column in range(width + 1):
            current = int(histogram[column]) if column < width else 0
            while stack and int(histogram[stack[-1]]) > current:
                index = stack.pop()
                rectangle_height = int(histogram[index])
                left = stack[-1] + 1 if stack else 0
                area = rectangle_height * (column - left)
                if area > best_area:
                    best_area = area
                    best = (left, row - rectangle_height + 1, column, row + 1)
            stack.append(column)
    return best


def detect_fov(
    frames: list[np.ndarray],
    brightness_threshold: int,
    minimum_bright_fraction: float,
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """Detect the stable endoscopic FOV using temporal brightness voting."""
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])
    bright_fraction = np.mean(gray > brightness_threshold, axis=0)
    mask = (bright_fraction >= minimum_bright_fraction).astype(np.uint8) * 255
    scale = max(mask.shape) / 768.0
    close_size = max(5, int(round(21 * scale)) | 1)
    open_size = max(3, int(round(5 * scale)) | 1)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise RuntimeError("No foreground FOV component detected")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component_mask = (labels == component).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    hull_mask = np.zeros_like(component_mask)
    cv2.fillPoly(hull_mask, [hull], 255)
    inner = largest_inner_rectangle(hull_mask)
    if (inner[2] - inner[0]) * (inner[3] - inner[1]) < 0.2 * hull_mask.size:
        raise RuntimeError(f"Detected FOV is unexpectedly small: {inner}")
    return inner, hull_mask


def roi_from_fov(
    hull_mask: np.ndarray,
) -> tuple[tuple[int, int, int, int], list[list[int]] | None, str]:
    """Use the same complete-hull crop and zero background for every namespace."""
    x, y, width, height = cv2.boundingRect(hull_mask)
    contours, _ = cv2.findContours(
        hull_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygon = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(int).tolist()
    return (
        (x, y, x + width, y + height),
        polygon,
        "complete FOV hull bbox crop with exact-zero outside-hull masking",
    )


def video_metadata(path: Path) -> tuple[int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid video metadata: {path} "
            f"(frames={frame_count}, size={width}x{height})"
        )
    return frame_count, width, height


def sample_folder_frames(
    folder: Path,
    maximum_frames: int,
) -> tuple[list[np.ndarray], list[Path], int, tuple[int, int]]:
    videos = sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"],
        key=natural_key,
    )
    if not videos:
        raise RuntimeError(f"No MP4 videos in folder: {folder}")

    metadata = [video_metadata(path) for path in videos]
    sizes = {(width, height) for _, width, height in metadata}
    if len(sizes) != 1:
        details = ", ".join(
            f"{path.name}={width}x{height}"
            for path, (_, width, height) in zip(videos, metadata)
        )
        raise RuntimeError(f"Resolution mismatch in {folder}: {details}")
    width, height = next(iter(sizes))
    total_frames = sum(frame_count for frame_count, _, _ in metadata)
    sample_count = min(maximum_frames, total_frames)
    selected = set(
        np.linspace(0, total_frames - 1, num=sample_count)
        .round()
        .astype(np.int64)
        .tolist()
    )

    sampled: list[np.ndarray] = []
    global_offset = 0
    for path, (declared_count, _, _) in zip(videos, metadata):
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        local_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                capture.release()
                raise RuntimeError(
                    f"Resolution changes while decoding {path}: "
                    f"expected={width}x{height}, got={frame.shape[1]}x{frame.shape[0]}"
                )
            if global_offset + local_index in selected:
                sampled.append(frame)
            local_index += 1
        capture.release()
        if local_index != declared_count:
            raise RuntimeError(
                f"Frame-count metadata mismatch in {path}: "
                f"declared={declared_count}, decoded={local_index}"
            )
        global_offset += local_index

    if len(sampled) != len(selected):
        raise RuntimeError(
            f"Sampling mismatch in {folder}: requested={len(selected)}, decoded={len(sampled)}"
        )
    return sampled, videos, total_frames, (width, height)


def rectangle_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def analyze_folder(
    namespace: str,
    folder: Path,
    maximum_frames: int,
    brightness_threshold: int,
    minimum_bright_fraction: float,
) -> dict[str, object]:
    frames, videos, total_frames, (width, height) = sample_folder_frames(
        folder, maximum_frames
    )
    base_roi, hull_mask = detect_fov(
        frames,
        brightness_threshold,
        minimum_bright_fraction,
    )
    roi, polygon, roi_policy = roi_from_fov(hull_mask)

    warnings: list[str] = []
    temporal_half_iou: float | None = None
    if len(frames) >= 8:
        midpoint = len(frames) // 2
        try:
            _, first_mask = detect_fov(
                frames[:midpoint], brightness_threshold, minimum_bright_fraction
            )
            _, second_mask = detect_fov(
                frames[midpoint:], brightness_threshold, minimum_bright_fraction
            )
            first_roi, _, _ = roi_from_fov(first_mask)
            second_roi, _, _ = roi_from_fov(second_mask)
            temporal_half_iou = rectangle_iou(first_roi, second_roi)
            if temporal_half_iou < 0.80:
                combined = cv2.bitwise_or(first_mask, second_mask)
                contours, _ = cv2.findContours(
                    combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                combined_hull = cv2.convexHull(
                    np.concatenate(contours, axis=0)
                )
                hull_mask = np.zeros_like(combined)
                cv2.fillPoly(hull_mask, [combined_hull], 255)
                base_roi = largest_inner_rectangle(hull_mask)
                roi, polygon, _ = roi_from_fov(hull_mask)
                roi_policy = (
                    "complete temporal-union FOV hull bbox crop with exact-zero "
                    "outside-hull masking"
                )
                warnings.append(
                    f"Temporal-half ROI IoU is low ({temporal_half_iou:.4f} < 0.80); "
                    "using the convex-hull union to preserve the complete moving FOV"
                )
        except RuntimeError as exc:
            warnings.append(f"Temporal stability check failed: {exc}")
    else:
        warnings.append("Fewer than 8 sampled frames; temporal stability not evaluated")

    x1, y1, x2, y2 = roi
    roi_width = x2 - x1
    roi_height = y2 - y1
    area_fraction = (roi_width * roi_height) / (width * height)
    center_is_inside = x1 <= width / 2 < x2 and y1 <= height / 2 < y2
    if not center_is_inside:
        warnings.append("ROI does not contain the original frame center")
    if area_fraction < 0.20:
        raise RuntimeError(
            f"ROI area is too small in {folder}: {area_fraction:.4f} < 0.20"
        )

    return {
        "key": f"{namespace}/{folder.name}",
        "namespace": namespace,
        "folder": folder.name,
        "source_size": [width, height],
        "video_count": len(videos),
        "total_frames": total_frames,
        "sampled_frames": len(frames),
        "base_roi_xyxy": list(base_roi),
        "fov_hull_polygon_xy": polygon,
        "roi_policy": roi_policy,
        "roi_xyxy": [x1, y1, x2, y2],
        "roi_normalized_xyxy": [
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height,
        ],
        "roi_size": [roi_width, roi_height],
        "roi_area_fraction": area_fraction,
        "center_is_inside": center_is_inside,
        "temporal_half_iou": temporal_half_iou,
        "qa_status": "warning" if warnings else "pass",
        "warnings": warnings,
        "source_videos": [path.name for path in videos],
    }


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute one content-derived FOV ROI per numeric source-video folder."
    )
    parser.add_argument("--negative-root", type=Path, default=DEFAULT_DATASETS["negative"])
    parser.add_argument("--positive-root", type=Path, default=DEFAULT_DATASETS["positive"])
    parser.add_argument(
        "--validation-root", type=Path, default=DEFAULT_DATASETS["validation"]
    )
    parser.add_argument("--manifest", type=Path, default=DATA_ROOT / "roi_manifest.json")
    parser.add_argument("--max-sampled-frames", type=int, default=120)
    parser.add_argument("--brightness-threshold", type=int, default=40)
    parser.add_argument("--minimum-bright-fraction", type=float, default=0.50)
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional keys such as negative/1 positive/50 validation/101.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed entries in an existing manifest and process only missing/error entries.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_sampled_frames < 1:
        raise ValueError("max-sampled-frames must be >= 1")
    if not 0 <= args.brightness_threshold <= 255:
        raise ValueError("brightness-threshold must be in [0,255]")
    if not 0 < args.minimum_bright_fraction <= 1:
        raise ValueError("minimum-bright-fraction must be in (0,1]")
    roots = {
        "negative": args.negative_root,
        "positive": args.positive_root,
        "validation": args.validation_root,
    }
    only = set(args.only or [])
    jobs: list[tuple[str, Path]] = []
    for namespace, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        folders = sorted(
            [path for path in root.iterdir() if path.is_dir() and path.name.isdigit()],
            key=lambda path: int(path.name),
        )
        jobs.extend(
            (namespace, folder)
            for folder in folders
            if not only or f"{namespace}/{folder.name}" in only
        )
    if only:
        found = {f"{namespace}/{folder.name}" for namespace, folder in jobs}
        missing = sorted(only - found)
        if missing:
            raise RuntimeError(f"Requested folder keys were not found: {missing}")

    configuration = {
        "brightness_threshold": args.brightness_threshold,
        "minimum_bright_fraction": args.minimum_bright_fraction,
        "max_sampled_frames_per_folder": args.max_sampled_frames,
        "roi_coordinate_convention": "xyxy, zero-based, x2/y2 exclusive",
        "all_namespaces_roi_policy": (
            "complete FOV hull bbox crop plus exact-zero outside-hull masking"
        ),
        "unstable_fov_policy": (
            "when temporal-half ROI IoU is below 0.80, use their convex-hull union"
        ),
        "training_canvas_policy": (
            "preserve aspect ratio; random scale 0.90-1.00 and random valid position"
        ),
        "evaluation_canvas_policy": (
            "preserve aspect ratio; scale 1.00 and deterministic centered position"
        ),
    }
    if args.resume and args.manifest.is_file():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest.get("configuration") != configuration:
            raise RuntimeError(
                "Existing manifest configuration differs from the requested configuration"
            )
        print(
            f"Resuming manifest with {len(manifest.get('entries', {}))} completed entries: "
            f"{args.manifest}",
            flush=True,
        )
    else:
        manifest = {
            "version": 2,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "method": (
                "temporal brightness voting -> morphology -> largest component -> "
                "convex hull -> common crop/mask policy for every namespace"
            ),
            "configuration": configuration,
            "dataset_roots": {name: str(path) for name, path in roots.items()},
            "expected_entries": len(jobs),
            "entries": {},
            "errors": {},
        }

    entries = manifest["entries"]
    errors = manifest["errors"]
    assert isinstance(entries, dict) and isinstance(errors, dict)
    for index, (namespace, folder) in enumerate(jobs, start=1):
        key = f"{namespace}/{folder.name}"
        if key in entries:
            print(f"[{index}/{len(jobs)}] {key}: SKIP (already complete)", flush=True)
            continue
        errors.pop(key, None)
        try:
            result = analyze_folder(
                namespace,
                folder,
                args.max_sampled_frames,
                args.brightness_threshold,
                args.minimum_bright_fraction,
            )
            entries[key] = result
            print(
                f"[{index}/{len(jobs)}] {key}: "
                f"roi={tuple(result['roi_xyxy'])}, "
                f"area={float(result['roi_area_fraction']):.4f}, "
                f"qa={result['qa_status']}",
                flush=True,
            )
        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"
            print(f"[{index}/{len(jobs)}] {key}: ERROR: {exc}", flush=True)
        manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(args.manifest, manifest)

    warning_count = sum(
        entry.get("qa_status") == "warning" for entry in entries.values()
    )
    manifest["summary"] = {
        "entries_created": len(entries),
        "qa_pass": len(entries) - warning_count,
        "qa_warning": warning_count,
        "errors": len(errors),
    }
    write_manifest(args.manifest, manifest)
    print(f"Manifest: {args.manifest}")
    print(
        f"Summary: expected={len(jobs)}, created={len(entries)}, "
        f"pass={len(entries) - warning_count}, warnings={warning_count}, "
        f"errors={len(errors)}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
