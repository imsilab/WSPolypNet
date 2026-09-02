from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class FolderROI:
    key: str
    source_width: int
    source_height: int
    x1: int
    y1: int
    x2: int
    y2: int
    fov_polygon_xy: tuple[tuple[int, int], ...] | None = None

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def validate_frame(self, frame: np.ndarray, path: Path) -> None:
        height, width = frame.shape[:2]
        if (width, height) != (self.source_width, self.source_height):
            raise RuntimeError(
                f"ROI source-size mismatch for {path}: manifest="
                f"{self.source_width}x{self.source_height}, decoded={width}x{height}"
            )
        if not (
            0 <= self.x1 < self.x2 <= width
            and 0 <= self.y1 < self.y2 <= height
        ):
            raise RuntimeError(
                f"ROI is outside the decoded frame for {path}: {self.xyxy}, "
                f"frame={width}x{height}"
            )

    @cached_property
    def crop_mask(self) -> np.ndarray | None:
        """Return the optional FOV mask in cropped-ROI coordinates."""
        if self.fov_polygon_xy is None:
            return None
        polygon = np.asarray(self.fov_polygon_xy, dtype=np.int32).copy()
        polygon[:, 0] -= self.x1
        polygon[:, 1] -= self.y1
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        return mask

    def apply(self, frame: np.ndarray, path: Path) -> np.ndarray:
        """Crop the full FOV and set every pixel outside its hull to exact zero."""
        self.validate_frame(frame, path)
        cropped = frame[self.y1 : self.y2, self.x1 : self.x2].copy()
        if cropped.size == 0:
            raise RuntimeError(f"ROI produced an empty frame for {path}: {self.xyxy}")
        if self.crop_mask is not None:
            cropped[self.crop_mask == 0] = 0
        return cropped


@dataclass(frozen=True)
class FOVCanvasTransform:
    """Place one complete folder FOV on a square zero-valued model canvas."""

    roi: FolderROI
    canvas_size: int
    resized_width: int
    resized_height: int
    offset_x: int
    offset_y: int
    horizontal_flip: bool = False

    def __post_init__(self) -> None:
        if self.canvas_size < 1:
            raise ValueError("canvas_size must be positive")
        if not (
            1 <= self.resized_width <= self.canvas_size
            and 1 <= self.resized_height <= self.canvas_size
        ):
            raise ValueError(
                f"Invalid resized FOV: {self.resized_width}x{self.resized_height} "
                f"for canvas {self.canvas_size}"
            )
        if not (
            0 <= self.offset_x <= self.canvas_size - self.resized_width
            and 0 <= self.offset_y <= self.canvas_size - self.resized_height
        ):
            raise ValueError(
                f"Invalid canvas offset ({self.offset_x}, {self.offset_y}) for "
                f"FOV {self.resized_width}x{self.resized_height} on {self.canvas_size}"
            )

    @property
    def content_xyxy(self) -> tuple[int, int, int, int]:
        return (
            self.offset_x,
            self.offset_y,
            self.offset_x + self.resized_width,
            self.offset_y + self.resized_height,
        )

    @cached_property
    def canvas_fov_mask(self) -> np.ndarray:
        """Boolean mask of valid endoscopic pixels on the model-input canvas."""
        crop_mask = self.roi.crop_mask
        if crop_mask is None:
            crop_mask = np.full(
                (self.roi.height, self.roi.width), 255, dtype=np.uint8
            )
        resized = cv2.resize(
            crop_mask,
            (self.resized_width, self.resized_height),
            interpolation=cv2.INTER_NEAREST,
        )
        canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        x1, y1, x2, y2 = self.content_xyxy
        canvas[y1:y2, x1:x2] = resized
        if self.horizontal_flip:
            canvas = np.ascontiguousarray(canvas[:, ::-1])
        return canvas > 0

    def apply(self, frame: np.ndarray, path: Path) -> np.ndarray:
        """Apply the FOV mask and place the uncropped content on a zero canvas."""
        fov = self.roi.apply(frame, path)
        resized = cv2.resize(
            fov,
            (self.resized_width, self.resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        canvas = np.zeros(
            (self.canvas_size, self.canvas_size, frame.shape[2]), dtype=frame.dtype
        )
        x1, y1, x2, y2 = self.content_xyxy
        canvas[y1:y2, x1:x2] = resized
        if self.horizontal_flip:
            canvas = np.ascontiguousarray(canvas[:, ::-1])
        # Linear resize can blend bright boundary pixels into a nominally black
        # neighbour. Reapply the resized hull so background is exactly zero.
        canvas[~self.canvas_fov_mask] = 0
        return canvas

    def original_box_to_canvas(self, box: Box) -> Box:
        """Map an original-frame box into model-input canvas coordinates."""
        x1, y1, x2, y2 = box
        mapped_x1 = self.offset_x + (x1 - self.roi.x1) * self.resized_width / self.roi.width
        mapped_x2 = self.offset_x + (x2 - self.roi.x1) * self.resized_width / self.roi.width
        mapped_y1 = self.offset_y + (y1 - self.roi.y1) * self.resized_height / self.roi.height
        mapped_y2 = self.offset_y + (y2 - self.roi.y1) * self.resized_height / self.roi.height
        if self.horizontal_flip:
            mapped_x1, mapped_x2 = (
                self.canvas_size - mapped_x2,
                self.canvas_size - mapped_x1,
            )
        return mapped_x1, mapped_y1, mapped_x2, mapped_y2

    def canvas_box_to_original(self, box: Box | None) -> Box | None:
        """Undo canvas placement, resize, and FOV crop for a predicted box."""
        if box is None:
            return None
        x1, y1, x2, y2 = box
        if self.horizontal_flip:
            x1, x2 = self.canvas_size - x2, self.canvas_size - x1
        content_x1, content_y1, content_x2, content_y2 = self.content_xyxy
        x1 = min(max(x1, content_x1), content_x2)
        x2 = min(max(x2, content_x1), content_x2)
        y1 = min(max(y1, content_y1), content_y2)
        y2 = min(max(y2, content_y1), content_y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return (
            self.roi.x1 + (x1 - self.offset_x) * self.roi.width / self.resized_width,
            self.roi.y1 + (y1 - self.offset_y) * self.roi.height / self.resized_height,
            self.roi.x1 + (x2 - self.offset_x) * self.roi.width / self.resized_width,
            self.roi.y1 + (y2 - self.offset_y) * self.roi.height / self.resized_height,
        )


def make_canvas_transform(
    roi: FolderROI,
    canvas_size: int,
    scale: float = 1.0,
    position_x: float = 0.5,
    position_y: float = 0.5,
    horizontal_flip: bool = False,
) -> FOVCanvasTransform:
    """Fit the complete FOV on a canvas without cropping or aspect distortion."""
    if not 0 < scale <= 1:
        raise ValueError(f"scale must be in (0,1], got {scale}")
    if not 0 <= position_x <= 1 or not 0 <= position_y <= 1:
        raise ValueError("position_x and position_y must be in [0,1]")
    target_max = max(1, min(canvas_size, round(canvas_size * scale)))
    resize_factor = target_max / max(roi.width, roi.height)
    resized_width = max(1, min(canvas_size, round(roi.width * resize_factor)))
    resized_height = max(1, min(canvas_size, round(roi.height * resize_factor)))
    offset_x = round((canvas_size - resized_width) * position_x)
    offset_y = round((canvas_size - resized_height) * position_y)
    return FOVCanvasTransform(
        roi=roi,
        canvas_size=canvas_size,
        resized_width=resized_width,
        resized_height=resized_height,
        offset_x=offset_x,
        offset_y=offset_y,
        horizontal_flip=horizontal_flip,
    )


def load_roi_manifest(path: Path) -> dict[str, FolderROI]:
    if not path.is_file():
        raise FileNotFoundError(f"ROI manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError(f"ROI manifest has no entries: {path}")
    errors = payload.get("errors", {})
    if errors:
        raise RuntimeError(f"ROI manifest contains unresolved errors: {errors}")

    result: dict[str, FolderROI] = {}
    for key, entry in entries.items():
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"Malformed ROI entry {key}: expected an object")
        try:
            source_width, source_height = (int(value) for value in entry["source_size"])
            x1, y1, x2, y2 = (int(value) for value in entry["roi_xyxy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Malformed ROI coordinates for {key}: {entry}") from exc
        raw_polygon = entry.get("fov_hull_polygon_xy")
        if raw_polygon is None:
            raise RuntimeError(
                f"ROI entry lacks the required complete-FOV hull polygon for {key}. "
                "Rebuild the manifest with codes/build_roi_manifest.py."
            )
        try:
            polygon = tuple((int(point[0]), int(point[1])) for point in raw_polygon)
        except (TypeError, ValueError, IndexError) as exc:
            raise RuntimeError(f"Malformed FOV polygon for {key}") from exc
        if len(polygon) < 3:
            raise RuntimeError(f"FOV polygon has fewer than 3 points for {key}")
        if any(
            not (0 <= x < source_width and 0 <= y < source_height)
            for x, y in polygon
        ):
            raise RuntimeError(f"FOV polygon is outside source bounds for {key}")
        roi = FolderROI(
            key=str(key),
            source_width=source_width,
            source_height=source_height,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            fov_polygon_xy=polygon,
        )
        if not (
            0 <= roi.x1 < roi.x2 <= roi.source_width
            and 0 <= roi.y1 < roi.y2 <= roi.source_height
        ):
            raise RuntimeError(
                f"Invalid ROI bounds for {key}: roi={roi.xyxy}, "
                f"source={roi.source_width}x{roi.source_height}"
            )
        if entry.get("qa_status") not in {"pass", "warning"}:
            raise RuntimeError(f"ROI entry did not pass QA for {key}: {entry.get('qa_status')}")
        result[str(key)] = roi
    return result


def roi_for_video(
    entries: Mapping[str, FolderROI], namespace: str, video_path: Path
) -> FolderROI:
    folder = video_path.parent.name
    if not folder.isdigit():
        raise RuntimeError(f"Video is not directly inside a numeric folder: {video_path}")
    key = f"{namespace}/{folder}"
    try:
        return entries[key]
    except KeyError as exc:
        raise RuntimeError(f"Missing ROI manifest entry for {key}: {video_path}") from exc


