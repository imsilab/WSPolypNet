#!/usr/bin/env python3
"""Create original-image five-view grids matching the R3D-18 CAM views."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "R3D-18" / "codes"))

from roi import load_roi_manifest, make_canvas_transform, roi_for_video  # noqa: E402


MODEL_SIZE = 224
CROP_SIZE = 144
VIEW_SPECS = (
    ("full", None),
    ("top_left", (0, 0)),
    ("top_right", (80, 0)),
    ("bottom_left", (0, 80)),
    ("bottom_right", (80, 80)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--roi-manifest",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "roi_manifest.json",
    )
    parser.add_argument("--roi-namespace", default="positive")
    parser.add_argument("--frames", type=int, default=5)
    return parser.parse_args()


def make_views(frame, transform, video_path: Path):
    full = transform.apply(frame, video_path)
    views = [("full", full)]
    for name, position in VIEW_SPECS[1:]:
        left, top = position
        crop = full[top : top + CROP_SIZE, left : left + CROP_SIZE]
        enlarged = cv2.resize(
            crop, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR
        )
        views.append((name, enlarged))
    return views


def make_grid(views, frame_number: int):
    title_height = 38
    panels = []
    for name, image in views:
        panel = cv2.copyMakeBorder(
            image,
            title_height,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(25, 25, 25),
        )
        label = f"Frame {frame_number} - {name}"
        cv2.putText(
            panel,
            label,
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    return cv2.hconcat(panels)


def main() -> int:
    args = parse_args()
    if args.frames < 1:
        raise ValueError("--frames must be positive")
    entries = load_roi_manifest(args.roi_manifest)
    roi = roi_for_video(entries, args.roi_namespace, args.video)
    transform = make_canvas_transform(
        roi, MODEL_SIZE, scale=1.0, position_x=0.5, position_y=0.5
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    written_grids = 0
    written_views = 0
    for frame_number in range(1, args.frames + 1):
        ok, frame = capture.read()
        if not ok:
            break
        views = make_views(frame, transform, args.video)
        frame_dir = args.output_dir / f"frame_{frame_number:02d}_views"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for view_number, (name, image) in enumerate(views, start=1):
            output = frame_dir / f"view_{view_number}_{name}.png"
            if not cv2.imwrite(str(output), image):
                raise RuntimeError(f"Could not write: {output}")
            written_views += 1
        grid_path = args.output_dir / f"frame_{frame_number:02d}_five_view_grid.png"
        if not cv2.imwrite(str(grid_path), make_grid(views, frame_number)):
            raise RuntimeError(f"Could not write: {grid_path}")
        written_grids += 1
    capture.release()

    if written_grids != args.frames or written_views != args.frames * 5:
        raise RuntimeError(
            f"Incomplete output: grids={written_grids}, views={written_views}"
        )
    print(f"grids={written_grids} individual_views={written_views}")
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
