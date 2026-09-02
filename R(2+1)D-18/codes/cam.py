from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset import Box, LocalizationVideoDataset
from model import R2Plus1D18Binary


def class_activation_maps(
    model: R2Plus1D18Binary,
    video: torch.Tensor,
    target_class: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute positive-class CAM from the single binary-logit weight."""
    if target_class != 1:
        raise ValueError("The one-logit model exposes only positive-class CAM")
    result = model(video, return_features=True)
    assert isinstance(result, tuple)
    logits, features = result
    class_weights = model.classifier.weight[0]
    cam = torch.einsum("c,bcthw->bthw", class_weights, features)
    cam = F.relu(cam)
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=(video.shape[2], video.shape[3], video.shape[4]),
        mode="trilinear",
        align_corners=False,
    ).squeeze(1)
    return logits, cam


def cam_to_box(
    cam: np.ndarray,
    threshold: float,
    valid_mask: np.ndarray | None = None,
) -> Box | None:
    """Select the thresholded component containing the valid-FOV CAM maximum."""
    if cam.ndim != 2:
        raise ValueError(f"CAM must be 2D, got {cam.shape}")
    if valid_mask is None:
        valid = np.ones(cam.shape, dtype=bool)
    else:
        if valid_mask.shape != cam.shape:
            raise ValueError(
                f"CAM/mask shape mismatch: {cam.shape} vs {valid_mask.shape}"
            )
        valid = valid_mask.astype(bool, copy=False)
    if not valid.any():
        return None
    minimum = float(cam[valid].min())
    maximum = float(cam[valid].max())
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        return None
    normalized = np.zeros_like(cam, dtype=np.float32)
    normalized[valid] = (cam[valid] - minimum) / (maximum - minimum)
    binary = ((normalized >= threshold) & valid).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if component_count <= 1:
        return None
    maximum_search = np.where(valid, normalized, -np.inf)
    max_y, max_x = np.unravel_index(
        int(np.argmax(maximum_search)), maximum_search.shape
    )
    component = int(labels[max_y, max_x])
    if component == 0:
        return None
    x, y, width, height, _ = stats[component]
    return float(x), float(y), float(x + width), float(y + height)


def box_iou(box_a: Box | None, box_b: Box) -> float:
    if box_a is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class CorLocResult:
    corloc_03: float
    corloc_05: float
    corloc_07: float
    evaluated_frames: int
    mean_iou: float


@torch.inference_mode()
def evaluate_corloc(
    model: R2Plus1D18Binary,
    dataset: LocalizationVideoDataset,
    device: torch.device,
    cam_threshold: float,
    progress_interval: int = 50,
) -> CorLocResult:
    """Frame-level CorLoc over frames having >=1 GT box; use maximum GT IoU."""
    model.eval()
    correct = {0.3: 0, 0.5: 0, 0.7: 0}
    evaluated = 0
    iou_sum = 0.0

    for video_index in range(len(dataset)):
        video, frame_boxes, original_size, transform, path = dataset[video_index]
        batch = video.unsqueeze(0).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            _, cams = class_activation_maps(model, batch, target_class=1)
        cams_np = cams[0].float().cpu().numpy()
        roi = transform.roi
        valid_mask = transform.canvas_fov_mask
        original_height, original_width = original_size
        if (original_width, original_height) != (
            roi.source_width,
            roi.source_height,
        ):
            raise RuntimeError(
                f"Validation ROI/source mismatch for {path}: "
                f"roi={roi.source_width}x{roi.source_height}, "
                f"video={original_width}x{original_height}"
            )

        for frame_index, ground_truths in enumerate(frame_boxes):
            if not ground_truths:
                continue
            input_height, input_width = cams_np[frame_index].shape
            if valid_mask.shape != (input_height, input_width):
                raise RuntimeError(
                    f"CAM/transform shape mismatch for {path}: "
                    f"cam={input_width}x{input_height}, mask={valid_mask.shape[::-1]}"
                )
            predicted_in_input = cam_to_box(
                cams_np[frame_index], cam_threshold, valid_mask=valid_mask
            )
            predicted = transform.canvas_box_to_original(predicted_in_input)
            maximum_iou = max(box_iou(predicted, box) for box in ground_truths)
            evaluated += 1
            iou_sum += maximum_iou
            for threshold in correct:
                if maximum_iou >= threshold:
                    correct[threshold] += 1

        if progress_interval and (video_index + 1) % progress_interval == 0:
            print(
                f"  CAM validation: {video_index + 1}/{len(dataset)} videos, "
                f"{evaluated} annotated frames",
                flush=True,
            )

    if evaluated == 0:
        raise RuntimeError("No validation frames with at least one GT box")
    return CorLocResult(
        corloc_03=correct[0.3] / evaluated,
        corloc_05=correct[0.5] / evaluated,
        corloc_07=correct[0.7] / evaluated,
        evaluated_frames=evaluated,
        mean_iou=iou_sum / evaluated,
    )

