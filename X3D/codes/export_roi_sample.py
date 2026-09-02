from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from roi import load_roi_manifest, make_canvas_transform, roi_for_video


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "datasets"


def decode_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames: list[np.ndarray] = []
    expected_size: tuple[int, int] | None = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        size = (frame.shape[1], frame.shape[0])
        if expected_size is None:
            expected_size = size
        elif size != expected_size:
            capture.release()
            raise RuntimeError(
                f"Resolution changes within {path}: {expected_size} -> {size}"
            )
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames


def encode_mp4(frames: list[np.ndarray], output: Path, fps: float) -> None:
    if not frames:
        raise ValueError("Cannot encode an empty frame list")
    height, width = frames[0].shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.mp4")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
        "-bf", "0", "-crf", "18", "-pix_fmt", "yuv420p", str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None and process.stderr is not None
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace")
        return_code = process.wait()
    except Exception:
        process.kill()
        temporary.unlink(missing_ok=True)
        raise
    if return_code != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {output}: {stderr.strip()}")
    os.replace(temporary, output)


def export_roi_sample(
    source: Path,
    namespace: str,
    manifest_path: Path,
    output: Path,
    spatial_size: int = 224,
    fps: float = 1.0,
    scale: float = 1.0,
    position_x: float = 0.5,
    position_y: float = 0.5,
) -> None:
    entries = load_roi_manifest(manifest_path)
    roi = roi_for_video(entries, namespace, source)
    transform = make_canvas_transform(
        roi,
        spatial_size,
        scale=scale,
        position_x=position_x,
        position_y=position_y,
    )
    processed = [
        transform.apply(frame, source)
        for frame in decode_video(source)
    ]
    encode_mp4(processed, output, fps)
    print(
        f"{source} -> {output}; frames={len(processed)}, fps={fps:g}, "
        f"roi={roi.xyxy}, fov={transform.resized_width}x{transform.resized_height}, "
        f"offset=({transform.offset_x},{transform.offset_y}), "
        f"output={spatial_size}x{spatial_size}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export one manifest-ROI sample video.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--namespace", choices=("negative", "positive", "validation"), required=True)
    parser.add_argument("--manifest", type=Path, default=DATA_ROOT / "roi_manifest.json")
    parser.add_argument("--spatial-size", type=int, default=224)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--position-x", type=float, default=0.5)
    parser.add_argument("--position-y", type=float, default=0.5)
    args = parser.parse_args()
    if args.spatial_size < 1 or args.fps <= 0:
        raise ValueError("spatial-size and fps must be positive")
    export_roi_sample(
        args.source,
        args.namespace,
        args.manifest,
        args.output,
        args.spatial_size,
        args.fps,
        args.scale,
        args.position_x,
        args.position_y,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

