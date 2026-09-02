#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""처음부터 끝까지 한 파일로 도는 Slow R50 + MedSAM2 국소화(corloc) 평가 스크립트.

이 파일 하나 + 가중치 2개 + LDPolypVideo TrainValid 데이터만 있으면,
프로젝트 본체(`slow_r50_fullvideo/src/*`)의 여러 파일에 흩어져 있던

    FOV(내시경 원형 시야) 자동 검출/크롭
      -> 25fps 이미지 시퀀스를 1fps 로 다운샘플
      -> ROI 로 크롭한 프레임을 Slow R50 에 넣어 5-view(전체+4코너 144px) score map 융합
      -> 증거가 가장 큰 프레임을 seed 로, NMS 로 상위 5개 peak 추출
      -> 각 peak 를 MedSAM2 point 프롬프트로 마스크화하고 양방향 트래킹
      -> Slow 증거로 트랙 점수를 매겨 top-1 유지(대안이 20% 이상 높을 때만 교체)
      -> 트랙이 비었거나 SAM2 신뢰도 < 0.755 인 프레임만 프레임별 MedSAM2 로 대체
      -> GT bbox 를 열어 IoU>=0.5 프레임 비율(corloc) 계산

까지를 **원본 코드와 수치적으로 동일하게** 재현한다.

원본 대응:
  slow_r50_fullvideo/src/medsam2_localization.py  (평가 루프)
  slow_r50_fullvideo/scripts/run_medsam2_localization.sh  (하이퍼파라미터 고정값)
  + dataset.py / multiscale_crop_localization.py / model.py / gradcam.py
    / sam2_temporal_localization.py / sam_pseudo_localization.py
    / topk_pointing_recall.py / track_quality_seed_screen.py
    / score_attention_grid.py / score_attention_postprocess_grid.py
    / (repo root) src/verify_generic_fov.py

주의 — "떡같은 값"이 안 나오는 흔한 이유:
  1. GPU 종류. 우리 채택 수치는 전부 NVIDIA A100 에서 나온 값이다. L40S / A6000 등
     다른 GPU 에서는 코드가 완전히 같아도 corloc 이 1~2%p 흔들린다(같은 GPU 안에서는
     비트 단위로 재현됨).
  2. MedSAM2 체크포인트/레포 커밋. 아래 고정값과 정확히 같아야 마스크가 같다.
  3. FOV 크롭과 1fps 다운샘플을 빼먹으면(원본 프레임을 그대로 넣으면) 전부 어긋난다.
     이 스크립트는 그 둘을 원본과 똑같이 안에서 처리한다 — 직접 하지 말 것.

세 가지 평가 모드:
  --mode sandl-chunk (기본)  LDPolypVideo TrainValid 케이스 1~100 을 30프레임 청크로
                             잘라서 평가. SANDL_Experiment2 실험이 이것. corloc ~= 0.423.
                             (build_sandl_chunk_frames.py + medsam2_localization.py
                              --annotated-sequence-source-fps 1.0 을 한 파일로 재현.
                              SANDL 원본 데이터 없이 LDPolypVideo 만으로 청크를 재구성한다.)
  --mode trainvalid          케이스 전체 시퀀스를 25fps->1fps 로 평가. corloc ~= 0.50.
  --mode validation-cam      Experiment2 ValidationData MP4를 Slow R50의 5-view
                             fused score map만으로 평가. MedSAM2는 로드하지 않음.
  --mode validation-medsam   같은 ValidationData에서 5-view Slow R50과 MedSAM2
                             point prompt/양방향 tracking을 함께 평가.

사용법:
  python standalone_localization_eval.py \
      --mode sandl-chunk \
      --slow-checkpoint    weights/slow_r50_gap_roicrop.pt \
      --medsam2-checkpoint weights/MedSAM2_latest.pt \
      --medsam2-repo       /path/to/MedSAM2 \
      --images-root        /path/to/LDPolypVideo/TrainValid/Images \
      --annotations-root   /path/to/LDPolypVideo/TrainValid/Annotations \
      --output-dir         out_sandl_chunk

  # --medsam2-repo 를 생략하면 bowang-lab/MedSAM2 를 고정 커밋으로 자동 clone 한다.
  # --medsam2-checkpoint 파일이 없으면 HuggingFace(wanglab/MedSAM2)에서 자동 다운로드한다.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn

# ----------------------------------------------------------------------------
# run_medsam2_localization.sh 의 고정 하이퍼파라미터 (건드리지 말 것)
# ----------------------------------------------------------------------------
PYTORCHVIDEO_HUB = "facebookresearch/pytorchvideo:f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0"
MEDSAM2_GIT_URL = "https://github.com/bowang-lab/MedSAM2"
MEDSAM2_GIT_COMMIT = "332f30d"  # "medsam2 with recist marker"
MEDSAM2_CKPT_URL = "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"
MEDSAM2_CONFIG = "configs/sam2.1_hiera_t512.yaml"

TARGET_FPS = 1.0
SOURCE_FPS = 25.0                 # TrainValid 이미지 시퀀스의 실제 촬영 fps 가정
RESIZE_SHORT_SIDE = 256
CROP_SIZE = 224
TEMPERATURE = 0.5
CROP_WINDOW_SIZE = 144
SMOOTHING_SIGMA = 4.0
TOP_K_CANDIDATES = 5
MIN_PEAK_DISTANCE = 16
TEMPORAL_SEEDS = 1
TOP1_MARGIN = 0.2
FALLBACK_CONFIDENCE_THRESHOLD = 0.755
IOU_THRESHOLD = 0.5
SMALL_BOX_AREA_FRACTION = 0.05

KINETICS_MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1)
KINETICS_STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# FOV 검출 상수 (dataset.py / verify_generic_fov.py)
FOV_AREA_FRACTION_MIN = 0.12
FOV_AREA_FRACTION_MAX = 0.85
FOV_DETECTION_FRAMES = 5
FOV_DARK_THRESHOLD = 30
FOV_OPEN_KERNEL = 15
FOV_PAD = 8


# ============================================================================
# 1. FOV(내시경 시야) 검출  —  src/verify_generic_fov.py + dataset.consistent_fov_bbox
# ============================================================================
def detect_fov_bbox(frame_bgr: np.ndarray) -> tuple[int, int, int, int, float]:
    """밝기 임계 + 모폴로지 opening 후 가장 큰 연결영역을 시야로 인정."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    denoised = cv2.medianBlur(gray, 5)
    mask = (denoised > FOV_DARK_THRESHOLD).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FOV_OPEN_KERNEL, FOV_OPEN_KERNEL))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if n_labels <= 1:
        return 0, 0, w - 1, h - 1, 1.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    x, y, bw, bh, area = stats[largest]
    x0, y0 = max(0, x - FOV_PAD), max(0, y - FOV_PAD)
    x1, y1 = min(w - 1, x + bw - 1 + FOV_PAD), min(h - 1, y + bh - 1 + FOV_PAD)
    return int(x0), int(y0), int(x1), int(y1), float(area) / (h * w)


def consistent_fov_bbox(frames_bgr: list[np.ndarray]) -> tuple[int, int, int, int]:
    """프레임 시퀀스에서 안정적인 하나의 FOV 크롭 상자를 검출."""
    if not frames_bgr:
        raise ValueError("cannot detect an FOV from zero frames")
    indices = np.linspace(0, len(frames_bgr) - 1, num=min(FOV_DETECTION_FRAMES, len(frames_bgr)), dtype=int)
    detections = [detect_fov_bbox(frames_bgr[int(i)]) for i in indices]
    coordinates = np.asarray([d[:4] for d in detections])
    x0, y0, x1, y1 = np.median(coordinates, axis=0).astype(int)
    median_area_fraction = float(np.median([d[4] for d in detections]))
    height, width = frames_bgr[0].shape[:2]
    if not FOV_AREA_FRACTION_MIN <= median_area_fraction <= FOV_AREA_FRACTION_MAX:
        return 0, 0, width - 1, height - 1
    return int(x0), int(y0), int(x1), int(y1)


# ============================================================================
# 2. 데이터 로딩  —  dataset._read_sequence + dataset._transform(crop_fov=True)
# ============================================================================
def _natural_key(path: Path) -> list[object]:
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", path.name)]


def _sampled_indices(count: int, step: float) -> list[int]:
    """dataset._read_sequence / gradcam.sequence_image_paths 와 동일한 인덱스 선택.
    step = source_fps / target_fps. trainvalid = 25.0, sandl-chunk = 1.0(전부)."""
    selected: list[int] = []
    next_index = 0.0
    while round(next_index) < count:
        index = round(next_index)
        if not selected or index != selected[-1]:
            selected.append(index)
        next_index += step
    return selected


def _case_image_paths(directory: Path) -> list[Path]:
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=_natural_key,
    )


def load_frames_rgb(paths: list[Path]) -> list[np.ndarray]:
    """PIL 로 로드한 RGB 프레임. dataset._read_sequence 와 동일한 로더."""
    return [np.asarray(Image.open(p).convert("RGB")) for p in paths]


def transform_to_model_video(frames_rgb: list[np.ndarray]) -> torch.Tensor:
    """dataset._transform(crop_fov=True, train=False) 재현: (C, T, 224, 224)."""
    if not frames_rgb:
        raise RuntimeError("decoder produced zero frames")
    frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames_rgb]
    x0, y0, x1, y1 = consistent_fov_bbox(frames_bgr)
    frames = [f[y0 : y1 + 1, x0 : x1 + 1] for f in frames_rgb]

    images = [Image.fromarray(f) for f in frames]
    width, height = images[0].size
    scale = RESIZE_SHORT_SIDE / min(width, height)
    new_size = (round(width * scale), round(height * scale))
    images = [im.resize(new_size, Image.Resampling.BILINEAR) for im in images]
    width, height = images[0].size
    if width < CROP_SIZE or height < CROP_SIZE:
        raise RuntimeError(f"resize too small for crop: {(width, height)}")
    top, left = (height - CROP_SIZE) // 2, (width - CROP_SIZE) // 2
    tensors = []
    for im in images:
        im = im.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
        t = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float().div_(255.0)
        tensors.append((t - KINETICS_MEAN) / KINETICS_STD)
    return torch.stack(tensors, dim=1)  # (C, T, H, W)


def read_roi_frames(image_paths: list[Path], roi: tuple[int, int, int, int]) -> list[np.ndarray]:
    """MedSAM2 에 넣을 프레임: cv2 로 로드한 BGR 을 ROI 로 크롭. sam2_temporal.read_roi_frames 와 동일."""
    x0, y0, x1, y1 = roi
    frames: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"cannot read sequence frame: {image_path}")
        cropped = image[y0 : y1 + 1, x0 : x1 + 1]
        if shape is None:
            shape = cropped.shape[:2]
        elif cropped.shape[:2] != shape:
            raise RuntimeError(f"inconsistent frame size in sequence: {image_path}")
        frames.append(cropped)
    return frames


# ============================================================================
# 3. Slow R50 score model  —  model.SlowR50TopKMILClassifier (global_avg 체크포인트)
# ============================================================================
class SlowR50ScoreModel(nn.Module):
    """model.SlowR50TopKMILClassifier 와 동일 구조: Kinetics 백본에서 마지막 head 블록만
    떼어내고 1x1x1 Conv3d score_head 를 붙인다. global_avg 체크포인트의 최종 선형층
    (.proj)을 score_head 로 변환해 로드한다."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.hub.load(
            PYTORCHVIDEO_HUB, "slow_r50",
            pretrained=False, trust_repo=True, skip_validation=True,
        )
        self.backbone.blocks = nn.ModuleList(list(self.backbone.blocks[:-1]))
        self.score_head = nn.Conv3d(2048, 1, kernel_size=1, bias=True)

    def feature_map(self, video: torch.Tensor) -> torch.Tensor:
        features = video
        for block in self.backbone.blocks:
            features = block(features)
        return features

    def score_map(self, video: torch.Tensor) -> torch.Tensor:
        return self.score_head(self.feature_map(video)).squeeze(1)  # (B, T', H', W')

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)["model"]
        if "score_head.weight" in state:  # 이미 이 클래스 형식으로 저장된 경우
            self.load_state_dict(state)
            return
        # model.initialize_from_global_average_checkpoint 재현
        self.load_state_dict(state, strict=False)
        weight = next(
            v for k, v in state.items()
            if k.endswith(".proj.weight") and v.numel() == self.score_head.weight.numel()
        )
        bias = next(
            v for k, v in state.items()
            if k.endswith(".proj.bias") and v.numel() == self.score_head.bias.numel()
        )
        with torch.no_grad():
            self.score_head.weight.copy_(weight.reshape_as(self.score_head.weight))
            self.score_head.bias.copy_(bias.reshape_as(self.score_head.bias))


# --- score_attention_grid.spatial_pool -------------------------------------
def spatial_pool(local_logits: torch.Tensor, temperature: float | None) -> tuple[torch.Tensor, torch.Tensor]:
    if local_logits.ndim != 4:
        raise ValueError(f"expected (B,T,H,W), got {tuple(local_logits.shape)}")
    if temperature is None:
        weights = torch.full_like(
            local_logits, 1.0 / (local_logits.shape[2] * local_logits.shape[3])
        )
    else:
        weights = torch.softmax(local_logits.flatten(2) / temperature, dim=2).reshape_as(local_logits)
    frame_logits = (weights * local_logits).sum(dim=(2, 3))
    return frame_logits.mean(dim=1), weights


# --- multiscale_crop_localization ----------------------------------------
def corner_windows(canvas_size: int, window_size: int) -> list[tuple[int, int, int, int]]:
    offset = canvas_size - window_size
    return [
        (0, 0, window_size, window_size),
        (offset, 0, canvas_size, window_size),
        (0, offset, window_size, canvas_size),
        (offset, offset, canvas_size, canvas_size),
    ]


def resize_video_spatial(video: torch.Tensor, output_size: int) -> torch.Tensor:
    frames = video.permute(1, 0, 2, 3)
    frames = F.interpolate(frames, size=(output_size, output_size), mode="bilinear", align_corners=False)
    return frames.permute(1, 0, 2, 3).contiguous()


def positive_evidence(
    model: SlowR50ScoreModel, video: torch.Tensor, temperature: float, output_frames: int, output_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    local_logits = model.score_map(video.unsqueeze(0))
    global_logit, _ = spatial_pool(local_logits, None)
    _, spatial_weights = spatial_pool(local_logits, temperature)
    evidence = spatial_weights * torch.sigmoid(local_logits)
    resized = F.interpolate(
        evidence.unsqueeze(1), size=(output_frames, output_size, output_size),
        mode="trilinear", align_corners=False,
    )[0, 0]
    return resized, global_logit


@torch.no_grad()
def localization_maps(
    model: SlowR50ScoreModel, video_cpu: torch.Tensor, device: torch.device,
    temperature: float, crop_window_size: int,
) -> tuple[np.ndarray, float]:
    """multiscale_crop_localization.localization_maps 의 'multiscale_crop_max' 만 반환."""
    video = video_cpu.to(device)
    frames = int(video.shape[1])
    canvas_size = int(video.shape[2])
    if video.shape[2] != video.shape[3]:
        raise ValueError(f"expected square model input, got {tuple(video.shape[2:])}")
    full_map, global_logit = positive_evidence(model, video, temperature, frames, canvas_size)
    fused = full_map.clone()
    for left, top, right, bottom in corner_windows(canvas_size, crop_window_size):
        crop = video[:, :, top:bottom, left:right]
        zoomed = resize_video_spatial(crop, canvas_size)
        crop_map, _ = positive_evidence(model, zoomed, temperature, frames, crop_window_size)
        canvas = torch.zeros_like(full_map)
        canvas[:, top:bottom, left:right] = crop_map
        fused = torch.maximum(fused, canvas)
    probability = float(torch.sigmoid(global_logit[0]).item())
    return fused.cpu().numpy(), probability


# ============================================================================
# 4. heatmap 후처리 / peak / 좌표변환  —  gradcam + sam_pseudo + topk_pointing
# ============================================================================
def normalized_cam(cam: np.ndarray) -> np.ndarray:
    low, high = float(cam.min()), float(cam.max())
    return (cam - low) / (high - low) if high > low else np.zeros_like(cam)


def smooth_heatmap(heat: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0:
        return heat
    smoothed = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return normalized_cam(smoothed)


def map_heat(raw_map: np.ndarray, smoothing_sigma: float) -> np.ndarray:
    return smooth_heatmap(normalized_cam(raw_map), smoothing_sigma)


def spatial_nms_peaks(heat: np.ndarray, max_peaks: int, min_distance: int) -> list[tuple[int, int, float]]:
    if heat.ndim != 2:
        raise ValueError(f"expected a 2-D heatmap, got {heat.shape}")
    scores = np.asarray(heat, dtype=np.float64).copy()
    peaks: list[tuple[int, int, float]] = []
    yy, xx = np.ogrid[: scores.shape[0], : scores.shape[1]]
    for _ in range(max_peaks):
        flat_index = int(np.argmax(scores))
        score = float(scores.flat[flat_index])
        if not np.isfinite(score):
            break
        y, x = np.unravel_index(flat_index, scores.shape)
        peaks.append((int(x), int(y), score))
        suppressed = (xx - x) ** 2 + (yy - y) ** 2 < min_distance**2
        scores[suppressed] = -np.inf
    return peaks


def model_point_to_original(
    x: int, y: int, original_width: int, original_height: int, resize_short_side: int, crop_size: int
) -> tuple[float, float]:
    scale = resize_short_side / min(original_width, original_height)
    resized_width = round(original_width * scale)
    resized_height = round(original_height * scale)
    crop_left = (resized_width - crop_size) // 2
    crop_top = (resized_height - crop_size) // 2
    original_x = (float(x) + crop_left) / scale
    original_y = (float(y) + crop_top) / scale
    return (
        min(max(original_x, 0.0), float(original_width - 1)),
        min(max(original_y, 0.0), float(original_height - 1)),
    )


def mask_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def crop_box(
    box: tuple[float, float, float, float], width: int, height: int, resize_short_side: int, crop_size: int
) -> tuple[int, int, int, int] | None:
    scale = resize_short_side / min(width, height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    left, top = (resized_width - crop_size) // 2, (resized_height - crop_size) // 2
    x1, y1, x2, y2 = box
    x1, x2 = x1 * scale - left, x2 * scale - left
    y1, y2 = y1 * scale - top, y2 * scale - top
    x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    x2, y2 = min(crop_size - 1, int(np.ceil(x2))), min(crop_size - 1, int(np.ceil(y2)))
    return (x1, y1, x2, y2) if x2 >= x1 and y2 >= y1 else None


def box_iou(first: tuple[float, float, float, float] | None, second: tuple[int, int, int, int]) -> float:
    if first is None:
        return 0.0
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_first = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_second = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_first + area_second - intersection
    return intersection / denom if denom > 0 else 0.0


def heatmap_component_box(
    heat: np.ndarray, threshold: float
) -> tuple[float, float, float, float] | None:
    """Return the thresholded component containing the global heatmap peak."""
    normalized = normalized_cam(heat)
    if not np.isfinite(normalized).all() or float(normalized.max()) <= 0.0:
        return None
    binary = (normalized >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return None
    peak_y, peak_x = np.unravel_index(int(np.argmax(normalized)), normalized.shape)
    component = int(labels[peak_y, peak_x])
    if component == 0:
        return None
    x, y, width, height, _ = stats[component]
    return float(x), float(y), float(x + width), float(y + height)


# --- GT 어노테이션 (LDPolypVideo, CRLF 파일) ------------------------------
def parse_boxes(path: Path) -> list[tuple[float, float, float, float]]:
    """`count` + `x1 y1 x2 y2` 형식. .splitlines()+.strip() 으로 CRLF(\\r\\n) 안전."""
    if not path.is_file():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []
    count = int(lines[0])
    boxes = []
    for line in lines[1 : count + 1]:
        values = [float(v) for v in line.split()]
        if len(values) != 4:
            raise ValueError(f"expected four bbox coordinates in {path}: {line}")
        x1, y1, x2, y2 = values
        boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return boxes


def boxes_in_fov_crop(
    boxes: list[tuple[float, float, float, float]], roi: tuple[int, int, int, int]
) -> list[tuple[float, float, float, float]]:
    roi_x0, roi_y0, roi_x1, roi_y1 = roi
    adjusted: list[tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in boxes:
        cx1, cy1 = max(x1, roi_x0), max(y1, roi_y0)
        cx2, cy2 = min(x2, roi_x1), min(y2, roi_y1)
        if cx2 < cx1 or cy2 < cy1:
            continue
        adjusted.append((cx1 - roi_x0, cy1 - roi_y0, cx2 - roi_x0, cy2 - roi_y0))
    return adjusted


# ============================================================================
# 5. 시간축 seed 선택  —  medsam2_localization.selected_temporal_seed_indices
# ============================================================================
def temporal_local_maxima(evidence: np.ndarray) -> list[int]:
    if evidence.ndim != 1 or not len(evidence):
        raise ValueError("evidence must be a non-empty 1-D array")
    maxima: list[int] = []
    start = 0
    while start < len(evidence):
        end = start
        while end + 1 < len(evidence) and evidence[end + 1] == evidence[start]:
            end += 1
        left = evidence[start - 1] if start else -np.inf
        right = evidence[end + 1] if end + 1 < len(evidence) else -np.inf
        if evidence[start] > left and evidence[end] > right:
            maxima.append((start + end) // 2)
        start = end + 1
    return maxima


def selected_temporal_seed_indices(evidence: np.ndarray, count: int) -> list[int]:
    global_index = int(np.argmax(evidence))
    ranked_local = sorted(temporal_local_maxima(evidence), key=lambda i: (-float(evidence[i]), i))
    return [global_index, *[i for i in ranked_local if i != global_index]][:count]


# ============================================================================
# 6. MedSAM2  —  sam2_temporal_localization / sam_pseudo_localization
# ============================================================================
def load_medsam2(repository: Path, checkpoint: Path, device: torch.device) -> tuple[Any, Any]:
    repository = repository.resolve()
    if not (repository / "sam2" / "__init__.py").is_file():
        raise FileNotFoundError(f"MedSAM2 repository not found (no sam2/ package): {repository}")
    config = repository / "sam2" / MEDSAM2_CONFIG
    if not config.is_file():
        raise FileNotFoundError(f"MedSAM2 config not found: {config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"MedSAM2 checkpoint not found: {checkpoint}")
    sys.path.insert(0, str(repository))
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    video_predictor = build_sam2_video_predictor(
        config_file=MEDSAM2_CONFIG, ckpt_path=str(checkpoint),
        device=str(device), apply_postprocessing=False,
    )
    return video_predictor, SAM2ImagePredictor(video_predictor)


def predict_point_mask(image_predictor: Any, image_bgr: np.ndarray, point: tuple[float, float]) -> tuple[np.ndarray, float]:
    """단일 foreground point, multimask_output=False. sam_pseudo.predict_point_mask 와 동일."""
    image_predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    masks, scores, _ = image_predictor.predict(
        point_coords=np.asarray([point], dtype=np.float32),
        point_labels=np.asarray([1], dtype=np.int64),
        box=None,
        multimask_output=False,
    )
    if len(masks) != 1 or len(scores) != 1:
        raise RuntimeError(f"unexpected MedSAM2 point output: masks={masks.shape}, scores={scores.shape}")
    return np.asarray(masks[0], dtype=bool), float(scores[0])


def track_from_seed(
    video_predictor: Any, frames_bgr: list[np.ndarray], seed_index: int, seed_mask: np.ndarray
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """seed 마스크를 양방향으로 전파. sam2_temporal_localization.track_from_seed 와 동일."""
    masks: dict[int, np.ndarray] = {}
    confidences: dict[int, float] = {}
    with tempfile.TemporaryDirectory(prefix="slowr50_sam2_") as temporary:
        frame_dir = Path(temporary)
        for frame_index, image_bgr in enumerate(frames_bgr):
            destination = frame_dir / f"{frame_index:05d}.jpg"
            if not cv2.imwrite(str(destination), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                raise RuntimeError(f"failed to stage SAM 2 frame: {destination}")
        state = video_predictor.init_state(
            str(frame_dir), offload_video_to_cpu=True, offload_state_to_cpu=False, async_loading_frames=False,
        )
        video_predictor.add_new_mask(state, seed_index, obj_id=1, mask=seed_mask)

        def retain(frame_index: int, mask_logits: torch.Tensor) -> None:
            logits = mask_logits[0]
            while logits.ndim > 2 and logits.shape[0] == 1:
                logits = logits[0]
            if logits.ndim != 2:
                raise RuntimeError(f"unexpected SAM 2 mask shape: {mask_logits.shape}")
            mask = (logits > 0.0).detach().cpu().numpy()
            masks[int(frame_index)] = mask
            if bool(mask.any()):
                confidences[int(frame_index)] = float(torch.sigmoid(logits[logits > 0.0]).mean().item())
            else:
                confidences[int(frame_index)] = 0.0

        for frame_index, _, mask_logits in video_predictor.propagate_in_video(
            state, start_frame_idx=seed_index, reverse=False
        ):
            retain(frame_index, mask_logits)
        for frame_index, _, mask_logits in video_predictor.propagate_in_video(
            state, start_frame_idx=seed_index, reverse=True
        ):
            retain(frame_index, mask_logits)
        video_predictor.reset_state(state)
    if len(masks) != len(frames_bgr):
        missing = sorted(set(range(len(frames_bgr))) - set(masks))
        raise RuntimeError(f"SAM 2 propagation missed frames: {missing}")
    return masks, confidences


# --- track_quality_seed_screen -------------------------------------------
def track_quality_score(
    track_masks: dict[int, np.ndarray], fused_maps: np.ndarray, width: int, height: int,
    resize_short_side: int, crop_size: int,
) -> float:
    frame_scores: list[float] = []
    for time_index, frame_map in enumerate(fused_maps):
        mask = track_masks[time_index]
        if not bool(np.asarray(mask).any()):
            frame_scores.append(0.0)
            continue
        original_box = mask_box(mask)
        model_box = None if original_box is None else crop_box(original_box, width, height, resize_short_side, crop_size)
        if model_box is None:
            frame_scores.append(0.0)
            continue
        x1, y1, x2, y2 = model_box
        region = frame_map[y1 : y2 + 1, x1 : x2 + 1]
        frame_scores.append(float(region.mean()) if region.size > 0 else 0.0)
    return float(np.mean(frame_scores)) if frame_scores else 0.0


def select_track_position(track_scores: list[float], margin: float, threshold: float | None) -> int:
    if len(track_scores) <= 1:
        return 0
    if threshold is not None:
        best = int(np.argmax(track_scores))
        return best if track_scores[best] >= threshold else 0
    alternative = 1 + int(np.argmax(track_scores[1:]))
    if track_scores[alternative] > track_scores[0] * (1.0 + margin):
        return alternative
    return 0


# --- gradcam.spatial_average_precision ----------------------------------
def spatial_average_precision(
    detections: list[dict[str, object]], gt_by_frame: dict[str, list[tuple[int, int, int, int]]], iou_threshold: float
) -> float:
    total_gt = sum(len(b) for b in gt_by_frame.values())
    if total_gt == 0:
        return math.nan
    matched: set[tuple[str, int]] = set()
    true_positive, false_positive = [], []
    for detection in sorted(detections, key=lambda item: float(item["score"]), reverse=True):
        frame_key, prediction = str(detection["frame_key"]), detection["box"]
        best_iou, best_index = 0.0, -1
        for index, ground_truth in enumerate(gt_by_frame.get(frame_key, [])):
            if (frame_key, index) not in matched:
                iou = box_iou(prediction, ground_truth)  # type: ignore[arg-type]
                if iou > best_iou:
                    best_iou, best_index = iou, index
        if best_iou >= iou_threshold:
            true_positive.append(1.0)
            false_positive.append(0.0)
            matched.add((frame_key, best_index))
        else:
            true_positive.append(0.0)
            false_positive.append(1.0)
    if not true_positive:
        return 0.0
    tp, fp = np.cumsum(true_positive), np.cumsum(false_positive)
    recall = np.concatenate(([0.0], tp / total_gt, [1.0]))
    precision = np.concatenate(([0.0], tp / np.maximum(tp + fp, 1e-12), [0.0]))
    for index in range(len(precision) - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


# ============================================================================
# 7. 한 시퀀스 평가  —  medsam2_localization.evaluate 의 루프 본문
# ============================================================================
METHOD_FRAMEWISE = "medsam2_box_framewise"
METHOD_TEMPORAL = "medsam2_box_temporal"
METHOD_FALLBACK = "medsam2_box_temporal_empty_framewise_fallback"
METHODS = (METHOD_FRAMEWISE, METHOD_TEMPORAL, METHOD_FALLBACK)


@torch.inference_mode()
def evaluate_sequence(
    sample_id: str, image_paths: list[Path], annotation_dir: Path,
    model: SlowR50ScoreModel, video_predictor: Any, image_predictor: Any, device: torch.device,
) -> list[dict[str, Any]]:
    frames_rgb = load_frames_rgb(image_paths)
    video = transform_to_model_video(frames_rgb)                 # (C, T, 224, 224)
    fused_maps, video_probability = localization_maps(model, video, device, TEMPERATURE, CROP_WINDOW_SIZE)

    if len(image_paths) != int(video.shape[1]):
        raise RuntimeError(f"temporal mismatch for {sample_id}: {len(image_paths)} paths vs {video.shape[1]} frames")

    # ROI 재검출: cv2 로 로드한 전체 프레임 5장 기준 (원본 evaluate 와 동일 경로)
    inspection_indices = np.linspace(0, len(image_paths) - 1, num=min(5, len(image_paths)), dtype=int)
    inspection_frames = [cv2.imread(str(image_paths[int(i)])) for i in inspection_indices]
    if any(f is None for f in inspection_frames):
        raise RuntimeError(f"cannot read FOV frames for {sample_id}")
    sequence_roi = consistent_fov_bbox(inspection_frames)  # type: ignore[arg-type]
    frames_bgr = read_roi_frames(image_paths, sequence_roi)
    height, width = frames_bgr[0].shape[:2]

    autocast_enabled = device.type == "cuda"

    # --- 시간축 seed + 공간 peak 후보 ---
    frame_evidence = np.asarray([float(fm.max()) for fm in fused_maps], dtype=np.float64)
    temporal_seed_indices = selected_temporal_seed_indices(frame_evidence, TEMPORAL_SEEDS)
    if not temporal_seed_indices:
        raise RuntimeError("no temporal local maximum was found")

    candidate_seed_indices: list[int] = []
    candidate_points: list[tuple[float, float]] = []
    for tsi in temporal_seed_indices:
        seed_heat = map_heat(fused_maps[tsi], SMOOTHING_SIGMA)
        for peak_x, peak_y, _ in spatial_nms_peaks(seed_heat, TOP_K_CANDIDATES, MIN_PEAK_DISTANCE):
            point = model_point_to_original(int(peak_x), int(peak_y), width, height, RESIZE_SHORT_SIDE, CROP_SIZE)
            candidate_seed_indices.append(tsi)
            candidate_points.append(point)

    # --- 각 후보: MedSAM2 초기 마스크 + 양방향 트랙 ---
    candidate_tracks: list[dict[int, np.ndarray]] = []
    candidate_confidences: list[dict[int, float]] = []
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
        for seed_index, point in zip(candidate_seed_indices, candidate_points):
            initial_mask, _ = predict_point_mask(image_predictor, frames_bgr[seed_index], point)
            track_masks, track_confidences = track_from_seed(video_predictor, frames_bgr, seed_index, initial_mask)
            candidate_tracks.append(track_masks)
            candidate_confidences.append(track_confidences)

    track_scores = [
        track_quality_score(t, fused_maps, width, height, RESIZE_SHORT_SIDE, CROP_SIZE)
        for t in candidate_tracks
    ]
    selected_position = select_track_position(track_scores, TOP1_MARGIN, threshold=None)
    selected_track = candidate_tracks[selected_position]
    selected_confidences = candidate_confidences[selected_position]

    # --- 프레임별 독립 예측 (GT 열기 전에 확정) ---
    framewise_masks: dict[int, np.ndarray] = {}
    framewise_scores: dict[int, float] = {}
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
        for time_index, frame_map in enumerate(fused_maps):
            heat = map_heat(frame_map, SMOOTHING_SIGMA)
            peak_y, peak_x = np.unravel_index(int(np.argmax(heat)), heat.shape)
            point = model_point_to_original(int(peak_x), int(peak_y), width, height, RESIZE_SHORT_SIDE, CROP_SIZE)
            mask, score = predict_point_mask(image_predictor, frames_bgr[time_index], point)
            framewise_masks[time_index] = mask
            framewise_scores[time_index] = score

    # --- GT 읽고 프레임별 채점 ---
    rows: list[dict[str, Any]] = []
    for time_index, image_path in enumerate(image_paths):
        gt_native = boxes_in_fov_crop(parse_boxes(annotation_dir / f"{image_path.stem}.txt"), sequence_roi)
        gt_boxes = [
            b for b in (
                crop_box(box, width, height, RESIZE_SHORT_SIDE, CROP_SIZE) for box in gt_native
            ) if b is not None
        ]
        if not gt_boxes:
            continue
        largest_area = max((x2 - x1 + 1) * (y2 - y1 + 1) for x1, y1, x2, y2 in gt_boxes) / float(CROP_SIZE**2)
        is_small = largest_area <= SMALL_BOX_AREA_FRACTION

        temporal_mask = selected_track[time_index]
        temporal_confidence = selected_confidences[time_index]
        temporal_empty = not bool(np.asarray(temporal_mask).any())
        use_fallback = temporal_empty or (temporal_confidence < FALLBACK_CONFIDENCE_THRESHOLD)

        method_predictions = {
            METHOD_FRAMEWISE: (framewise_masks[time_index], framewise_scores[time_index], False),
            METHOD_TEMPORAL: (temporal_mask, temporal_confidence, False),
            METHOD_FALLBACK: (
                framewise_masks[time_index] if use_fallback else temporal_mask,
                framewise_scores[time_index] if use_fallback else temporal_confidence,
                use_fallback,
            ),
        }
        for method, (mask, confidence, fallback_used) in method_predictions.items():
            original_box = mask_box(mask)
            predicted_box = None if original_box is None else crop_box(
                original_box, width, height, RESIZE_SHORT_SIDE, CROP_SIZE
            )
            iou = max((box_iou(predicted_box, gt) for gt in gt_boxes), default=0.0)
            rows.append({
                "method": method,
                "sample_id": sample_id,
                "frame": image_path.name,
                "small_polyp": int(is_small),
                "slow_video_probability": float(video_probability),
                "empty_temporal_fallback_used": int(fallback_used),
                "prediction_confidence": float(confidence),
                "iou": float(iou),
                "corloc": int(iou >= IOU_THRESHOLD),
                "frame_key": f"{sample_id}:{image_path.name}",
                "predicted_box": predicted_box,
                "gt_boxes": gt_boxes,
            })
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in METHODS:
        for scope, keep in (("all_frames", lambda r: True), ("small_polyps", lambda r: r["small_polyp"] == 1)):
            records = [r for r in rows if r["method"] == method and keep(r)]
            if not records:
                out[f"{method}::{scope}"] = {"frames": 0, "corloc": None, "iou": None}
                continue
            detections, gt_by_frame = [], {}
            for r in records:
                gt_by_frame[r["frame_key"]] = r["gt_boxes"]
                if r["predicted_box"] is not None:
                    detections.append({
                        "frame_key": r["frame_key"], "box": r["predicted_box"],
                        "score": r["slow_video_probability"] * r["prediction_confidence"],
                    })
            out[f"{method}::{scope}"] = {
                "frames": len(records),
                "corloc_count": sum(r["corloc"] for r in records),
                "corloc": float(np.mean([r["corloc"] for r in records])),
                "iou": float(np.mean([r["iou"] for r in records])),
                "spatial_ap": spatial_average_precision(detections, gt_by_frame, IOU_THRESHOLD),
                "empty_temporal_fallback_count": sum(r["empty_temporal_fallback_used"] for r in records),
            }
    return out


# ============================================================================
# 8. Experiment2 ValidationData CAM-only 평가 (MedSAM2 미사용)
# ============================================================================
@torch.inference_mode()
def evaluate_validation_cam(
    validation_root: Path,
    roi_manifest: Path,
    model: SlowR50ScoreModel,
    device: torch.device,
    output_dir: Path,
    cam_threshold: float,
    max_sequences: int | None,
) -> dict[str, Any]:
    codes = Path(__file__).resolve().parent / "R3D-18" / "codes"
    sys.path.insert(0, str(codes))
    from dataset import LocalizationVideoDataset

    dataset = LocalizationVideoDataset(
        validation_root / "video",
        validation_root / "annotation",
        spatial_size=CROP_SIZE,
        roi_manifest_path=roi_manifest,
    )
    samples = dataset.samples[:max_sequences] if max_sequences else dataset.samples
    rows: list[dict[str, Any]] = []
    correct = {0.3: 0, 0.5: 0, 0.7: 0}
    iou_sum = 0.0
    positive_videos = 0

    for sequence_index, sample in enumerate(samples, start=1):
        capture = cv2.VideoCapture(str(sample.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open validation video: {sample.video_path}")
        frames_bgr: list[np.ndarray] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames_bgr.append(frame)
        capture.release()
        if len(frames_bgr) != len(sample.annotation_paths):
            raise RuntimeError(
                f"frame/annotation mismatch for {sample.video_path}: "
                f"{len(frames_bgr)} vs {len(sample.annotation_paths)}"
            )

        frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
        video = transform_to_model_video(frames_rgb)
        fused_maps, video_probability = localization_maps(
            model, video, device, TEMPERATURE, CROP_WINDOW_SIZE
        )
        positive_videos += int(video_probability >= 0.5)

        sequence_roi = consistent_fov_bbox(frames_bgr)
        roi_x0, roi_y0, roi_x1, roi_y1 = sequence_roi
        roi_width = roi_x1 - roi_x0 + 1
        roi_height = roi_y1 - roi_y0 + 1
        for frame_index, (frame_map, annotation_path) in enumerate(
            zip(fused_maps, sample.annotation_paths)
        ):
            native_boxes = parse_boxes(annotation_path)
            if not native_boxes:
                continue
            visible_boxes = boxes_in_fov_crop(native_boxes, sequence_roi)
            gt_boxes = [
                converted
                for converted in (
                    crop_box(box, roi_width, roi_height, RESIZE_SHORT_SIDE, CROP_SIZE)
                    for box in visible_boxes
                )
                if converted is not None
            ]
            heat = map_heat(frame_map, SMOOTHING_SIGMA)
            predicted = heatmap_component_box(heat, cam_threshold)
            maximum_iou = max(
                (box_iou(predicted, ground_truth) for ground_truth in gt_boxes),
                default=0.0,
            )
            iou_sum += maximum_iou
            for threshold in correct:
                correct[threshold] += int(maximum_iou >= threshold)
            rows.append({
                "video_path": str(sample.video_path),
                "frame_index": frame_index,
                "video_probability": video_probability,
                "predicted_box_xyxy": "" if predicted is None else json.dumps(predicted),
                "ground_truth_boxes_xyxy": json.dumps(gt_boxes),
                "max_iou": maximum_iou,
                "correct_0.3": int(maximum_iou >= 0.3),
                "correct_0.5": int(maximum_iou >= 0.5),
                "correct_0.7": int(maximum_iou >= 0.7),
            })
        if sequence_index % 25 == 0 or sequence_index == len(samples):
            running = correct[0.5] / len(rows) if rows else 0.0
            print(
                f"[{sequence_index}/{len(samples)}] annotated_frames={len(rows)}, "
                f"CAM CorLoc@0.5={running:.6f}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("no annotated validation frames were evaluated")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_csv = output_dir / "cam_frame_metrics.csv"
    with frame_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    count = len(rows)
    summary = {
        "mode": "validation-cam",
        "medsam2_used": False,
        "processed_videos": len(samples),
        "evaluated_frames": count,
        "cam_threshold": cam_threshold,
        "mean_iou": iou_sum / count,
        "corloc_0.3": correct[0.3] / count,
        "corloc_0.5": correct[0.5] / count,
        "corloc_0.7": correct[0.7] / count,
        "positive_video_rate_at_0.5": positive_videos / len(samples),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def stage_validation_video(
    video_path: Path,
    annotation_paths: tuple[Path, ...],
    directory: Path,
) -> tuple[list[Path], Path]:
    """Stage one MP4 as lossless frames with matching annotation names."""
    frame_dir = directory / "frames"
    annotation_dir = directory / "annotations"
    frame_dir.mkdir()
    annotation_dir.mkdir()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open validation video: {video_path}")
    image_paths: list[Path] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index >= len(annotation_paths):
            raise RuntimeError(f"more video frames than annotations: {video_path}")
        stem = f"{frame_index:06d}"
        image_path = frame_dir / f"{stem}.png"
        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError(f"could not stage validation frame: {image_path}")
        (annotation_dir / f"{stem}.txt").symlink_to(
            annotation_paths[frame_index].resolve()
        )
        image_paths.append(image_path)
        frame_index += 1
    capture.release()
    if not image_paths or len(image_paths) != len(annotation_paths):
        raise RuntimeError(
            f"frame/annotation mismatch for {video_path}: "
            f"{len(image_paths)} vs {len(annotation_paths)}"
        )
    return image_paths, annotation_dir


def evaluate_validation_medsam(
    validation_root: Path,
    roi_manifest: Path,
    model: SlowR50ScoreModel,
    video_predictor: Any,
    image_predictor: Any,
    device: torch.device,
    output_dir: Path,
    slow_checkpoint: Path,
    medsam2_checkpoint: Path,
    max_sequences: int | None,
) -> dict[str, Any]:
    """Evaluate Slow R50 + MedSAM2 on Experiment2 ValidationData MP4s."""
    codes = Path(__file__).resolve().parent / "R3D-18" / "codes"
    sys.path.insert(0, str(codes))
    from dataset import LocalizationVideoDataset

    dataset = LocalizationVideoDataset(
        validation_root / "video",
        validation_root / "annotation",
        spatial_size=CROP_SIZE,
        roi_manifest_path=roi_manifest,
    )
    samples = dataset.samples[:max_sequences] if max_sequences else dataset.samples
    all_rows: list[dict[str, Any]] = []
    started = time.time()
    for sequence_index, sample in enumerate(samples, start=1):
        sample_id = str(sample.video_path)
        with tempfile.TemporaryDirectory(prefix="validation_medsam_") as temporary:
            image_paths, annotation_dir = stage_validation_video(
                sample.video_path, sample.annotation_paths, Path(temporary)
            )
            rows = evaluate_sequence(
                sample_id,
                image_paths,
                annotation_dir,
                model,
                video_predictor,
                image_predictor,
                device,
            )

        present = {
            int(Path(str(row["frame"])).stem)
            for row in rows
            if row["method"] == METHOD_FALLBACK
        }
        # Center cropping can remove a GT completely. Keep those annotated
        # frames in the same 12,933-frame denominator as the CAM-only metric.
        for frame_index, annotation_path in enumerate(sample.annotation_paths):
            native_boxes = parse_boxes(annotation_path)
            if not native_boxes or frame_index in present:
                continue
            for method in METHODS:
                rows.append({
                    "method": method,
                    "sample_id": sample_id,
                    "frame": f"{frame_index:06d}.png",
                    "small_polyp": 0,
                    "slow_video_probability": 0.0,
                    "empty_temporal_fallback_used": 0,
                    "prediction_confidence": 0.0,
                    "iou": 0.0,
                    "corloc": 0,
                    "frame_key": f"{sample_id}:{frame_index:06d}.png",
                    "predicted_box": None,
                    "gt_boxes": [],
                })
        all_rows.extend(rows)

        if sequence_index % 10 == 0 or sequence_index == len(samples):
            adopted = [row for row in all_rows if row["method"] == METHOD_FALLBACK]
            running = float(np.mean([row["corloc"] for row in adopted]))
            print(
                f"[{sequence_index}/{len(samples)}] annotated_frames={len(adopted)}, "
                f"MedSAM2 CorLoc@0.5={running:.6f}, "
                f"elapsed={(time.time() - started) / 60:.2f} min",
                flush=True,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_csv = output_dir / "localization_frame_metrics.csv"
    fields = (
        "method", "sample_id", "frame", "small_polyp",
        "slow_video_probability", "empty_temporal_fallback_used",
        "prediction_confidence", "iou", "corloc",
    )
    with frame_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row[key] for key in fields})
    payload = {
        "mode": "validation-medsam",
        "medsam2_used": True,
        "slow_checkpoint": str(slow_checkpoint),
        "medsam2_checkpoint": str(medsam2_checkpoint),
        "processed_videos": len(samples),
        "adopted_method": METHOD_FALLBACK,
        "localization": summarize(all_rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(payload["localization"], ensure_ascii=False, indent=2), flush=True)
    return payload


# ============================================================================
# 9. 자산 준비 (가중치/레포 자동 확보) + main
# ============================================================================
def ensure_medsam2_repo(path: Path | None, workdir: Path) -> Path:
    if path is not None:
        return path
    target = workdir / "MedSAM2"
    if not (target / "sam2" / "__init__.py").is_file():
        print(f"[assets] MedSAM2 레포를 clone 합니다 -> {target}", flush=True)
        subprocess.run(["git", "clone", MEDSAM2_GIT_URL, str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", MEDSAM2_GIT_COMMIT], check=True)
    return target


def ensure_medsam2_checkpoint(path: Path) -> Path:
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[assets] MedSAM2 체크포인트를 내려받습니다 -> {path}", flush=True)
    urllib.request.urlretrieve(MEDSAM2_CKPT_URL, path)
    if path.stat().st_size == 0:
        raise RuntimeError(f"downloaded checkpoint is empty: {path}")
    return path


def _annotation_first_line_is_zero(path: Path) -> bool:
    """build_sandl_chunk_frames.frames_have_box 와 동일: 첫 줄(박스 개수) 판정. CRLF 안전."""
    with open(path, encoding="utf-8") as handle:
        return handle.readline().strip() == "0"


def _last_annotation_index(annotation_dir: Path) -> int:
    return max((int(p.stem) for p in annotation_dir.glob("*.txt")), default=0)


def _sequences_from_chunk_manifest(
    manifest: Path, annotations_root: Path, limit: int | None,
) -> list[tuple[str, list[Path], Path]]:
    """이미 만들어 둔 sandl_chunks_manifest.csv 를 그대로 사용한다.

    medsam2_localization.py 가 소비하는 것과 완전히 같은 입력:
    frames_dir 행마다 source_path 폴더의 이미지를 전부(다운샘플 없음) 쓰고,
    GT 는 annotations_root / <폴더이름> 에서 읽는다. SANDL 원본/심볼릭 링크가
    있는 환경이면 이 경로가 가장 정확하다(청크 재구성 로직을 안 탄다)."""
    manifest_dir = manifest.resolve().parent
    out: list[tuple[str, list[Path], Path]] = []
    with open(manifest, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("input_type") != "frames_dir":
                continue
            src = Path(row["source_path"])
            if not src.is_absolute():
                # manifest 의 상대경로는 (레포 루트 기준) — manifest 위치에서 거슬러 올라가 해석
                for base in (Path.cwd(), manifest_dir, *manifest_dir.parents):
                    if (base / src).is_dir():
                        src = base / src
                        break
            frame_paths = _case_image_paths(src)
            if not frame_paths:
                raise FileNotFoundError(f"chunk 폴더에 이미지가 없습니다: {src}")
            out.append((row["sample_id"], frame_paths, annotations_root / src.name))
    if not out:
        raise ValueError(f"chunk manifest 에 frames_dir 행이 없습니다: {manifest}")
    print(f"[chunk-manifest] {len(out)}개 청크", flush=True)
    return out[:limit] if limit else out


def build_sequences(
    mode: str, images_root: Path, annotations_root: Path, limit: int | None,
    chunk_manifest: Path | None = None,
) -> list[tuple[str, list[Path], Path]]:
    """평가할 시퀀스 목록. (sample_id, 프레임경로들, GT어노테이션폴더)."""
    if chunk_manifest is not None:
        return _sequences_from_chunk_manifest(chunk_manifest, annotations_root, limit)

    case_dirs = sorted((d for d in images_root.iterdir() if d.is_dir()), key=_natural_key)

    if mode == "trainvalid":
        # 케이스 전체 시퀀스를 25fps -> 1fps 다운샘플. (원본 ~50% corloc)
        step = SOURCE_FPS / TARGET_FPS
        out: list[tuple[str, list[Path], Path]] = []
        for d in case_dirs:
            paths = _case_image_paths(d)
            selected = [paths[i] for i in _sampled_indices(len(paths), step)]
            out.append((f"TrainValid_{d.name}", selected, annotations_root / d.name))
        return out[:limit] if limit else out

    if mode == "sandl-chunk":
        # SANDL_Experiment2 with_polyp = LDPolypVideo TrainValid 케이스 1~100 을
        # 30프레임씩 자른 청크. 마지막 청크는 짧을 수 있음. 30프레임 전 구간 bbox 0개인
        # 청크는 제외(build_sandl_chunk_frames.py 규칙, 62개). 청크 안 프레임은 전부 사용.
        # (SANDL 30프레임 청크 corloc ~= 0.423)
        out = []
        excluded = 0
        for d in case_dirs:
            if not d.name.isdigit() or not (1 <= int(d.name) <= 100):
                continue
            case = d.name
            annotation_dir = annotations_root / case
            n_frames = _last_annotation_index(annotation_dir)
            if n_frames == 0:
                continue
            n_chunks = math.ceil(n_frames / 30)
            for chunk_index in range(1, n_chunks + 1):
                start = (chunk_index - 1) * 30 + 1
                end = min(start + 29, n_frames)
                window = list(range(start, end + 1))
                if all(_annotation_first_line_is_zero(annotation_dir / f"{i:04d}.txt") for i in window):
                    excluded += 1
                    continue
                frame_paths = [d / f"{i:04d}.jpg" for i in window]
                out.append((f"sandl_chunk__{case}__video_{chunk_index:04d}", frame_paths, annotation_dir))
        print(f"[sandl-chunk] {len(out)}개 청크 (bbox 0개로 제외: {excluded}개)", flush=True)
        return out[:limit] if limit else out

    raise ValueError(f"unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    parser.add_argument("--slow-checkpoint", type=Path, default=here / "weights" / "slow_r50_gap_roicrop.pt",
                        help="우리가 학습한 Slow R50(global_avg, ROI-crop) 체크포인트")
    parser.add_argument("--medsam2-checkpoint", type=Path, default=here / "weights" / "MedSAM2_latest.pt",
                        help="없으면 HuggingFace wanglab/MedSAM2 에서 자동 다운로드")
    parser.add_argument("--medsam2-repo", type=Path, default=None,
                        help=f"없으면 {MEDSAM2_GIT_URL} 를 {MEDSAM2_GIT_COMMIT} 커밋으로 자동 clone")
    parser.add_argument("--images-root", type=Path, default=None,
                        help="LDPolypVideo TrainValid/Images (--chunk-manifest 를 안 쓸 때 필요)")
    parser.add_argument("--annotations-root", type=Path, default=None,
                        help="LDPolypVideo TrainValid/Annotations")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("sandl-chunk", "trainvalid", "validation-cam", "validation-medsam"), default="sandl-chunk",
                        help="sandl-chunk: 30프레임 청크 평가 (SANDL 실험, corloc~0.42, 기본). "
                             "trainvalid: 케이스 전체 시퀀스 1fps 평가 (corloc~0.50). "
                             "validation-cam: ValidationData에서 Slow R50 CAM만 평가. "
                             "validation-medsam: ValidationData에서 Slow R50+MedSAM2 평가. "
                             "--chunk-manifest 를 주면 무시된다.")
    parser.add_argument("--chunk-manifest", type=Path, default=None,
                        help="이미 만들어 둔 sandl_chunks_manifest.csv 경로. 있으면 이걸 그대로 쓴다"
                             "(청크 재구성 로직을 안 타므로 SANDL 환경에서 가장 정확).")
    parser.add_argument("--max-sequences", type=int, default=None, help="빠른 확인용 (기본: 전체)")
    parser.add_argument("--validation-root", type=Path, default=here / "datasets" / "ValidationData")
    parser.add_argument("--roi-manifest", type=Path, default=here / "datasets" / "roi_manifest.json")
    parser.add_argument("--cam-threshold", type=float, default=0.5)
    args = parser.parse_args()
    validation_modes = {"validation-cam", "validation-medsam"}
    if args.mode not in validation_modes and args.annotations_root is None:
        parser.error("--annotations-root is required unless --mode validation-cam")
    if args.mode not in validation_modes and args.chunk_manifest is None and args.images_root is None:
        parser.error("--chunk-manifest 또는 --images-root 중 하나는 필요합니다")
    if not 0.0 <= args.cam_threshold <= 1.0:
        parser.error("--cam-threshold must be between 0 and 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("이 평가는 CUDA GPU 가 필요합니다. (우리 채택 수치는 A100 기준)")
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = "unknown"
    if "A100" not in gpu_name:
        print(f"[warn] 현재 GPU = {gpu_name}. 우리 공식 corloc 수치는 A100 기준입니다 — "
              f"다른 GPU 에서는 코드가 같아도 1~2%p 차이가 날 수 있습니다.", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.slow_checkpoint.is_file():
        raise FileNotFoundError(
            f"Slow 체크포인트가 없습니다: {args.slow_checkpoint}\n"
            "  이건 우리가 학습한 파일이라 자동 다운로드가 안 됩니다. "
            "share/weights/slow_r50_gap_roicrop.pt 를 받아서 넣어주세요."
        )
    print("[load] Slow R50 score model", flush=True)
    model = SlowR50ScoreModel()
    model.load_checkpoint(args.slow_checkpoint)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if args.mode == "validation-cam":
        evaluate_validation_cam(
            validation_root=args.validation_root,
            roi_manifest=args.roi_manifest,
            model=model,
            device=device,
            output_dir=args.output_dir,
            cam_threshold=args.cam_threshold,
            max_sequences=args.max_sequences,
        )
        return

    medsam2_repo = ensure_medsam2_repo(args.medsam2_repo, args.output_dir)
    medsam2_checkpoint = ensure_medsam2_checkpoint(args.medsam2_checkpoint)

    print("[load] MedSAM2", flush=True)
    video_predictor, image_predictor = load_medsam2(medsam2_repo, medsam2_checkpoint, device)

    if args.mode == "validation-medsam":
        evaluate_validation_medsam(
            validation_root=args.validation_root,
            roi_manifest=args.roi_manifest,
            model=model,
            video_predictor=video_predictor,
            image_predictor=image_predictor,
            device=device,
            output_dir=args.output_dir,
            slow_checkpoint=args.slow_checkpoint,
            medsam2_checkpoint=medsam2_checkpoint,
            max_sequences=args.max_sequences,
        )
        return

    sequences = build_sequences(
        args.mode, args.images_root, args.annotations_root, args.max_sequences,
        chunk_manifest=args.chunk_manifest,
    )
    run_mode = "chunk-manifest" if args.chunk_manifest else args.mode
    print(f"[run] mode={run_mode}, {len(sequences)} sequences", flush=True)

    all_rows: list[dict[str, Any]] = []
    frame_csv = args.output_dir / "localization_frame_metrics.csv"
    for i, (sample_id, seq_image_paths, annotation_dir) in enumerate(sequences, 1):
        rows = evaluate_sequence(sample_id, seq_image_paths, annotation_dir, model, video_predictor, image_predictor, device)
        all_rows.extend(rows)
        fb = [r for r in rows if r["method"] == METHOD_FALLBACK]
        running = np.mean([r["corloc"] for r in all_rows if r["method"] == METHOD_FALLBACK]) if all_rows else 0.0
        print(f"[{i}/{len(sequences)}] {sample_id}: frames={len(fb)} "
              f"seq_corloc={np.mean([r['corloc'] for r in fb]) if fb else float('nan'):.3f} "
              f"running_corloc={running:.4f}", flush=True)

    # CSV
    if all_rows:
        fields = ["method", "sample_id", "frame", "small_polyp", "slow_video_probability",
                  "empty_temporal_fallback_used", "prediction_confidence", "iou", "corloc"]
        with frame_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for r in all_rows:
                writer.writerow({k: r[k] for k in fields})

    summary = {
        "gpu": gpu_name,
        "mode": run_mode,
        "slow_checkpoint": str(args.slow_checkpoint),
        "medsam2_checkpoint": str(medsam2_checkpoint),
        "medsam2_repo_commit": MEDSAM2_GIT_COMMIT,
        "hyperparameters": {
            "target_fps": TARGET_FPS,
            "frame_sampling": "all frames (source_fps=1)" if args.mode == "sandl-chunk"
                              else f"source_fps={SOURCE_FPS} -> 1fps",
            "resize_short_side": RESIZE_SHORT_SIDE, "crop_size": CROP_SIZE,
            "temperature": TEMPERATURE, "crop_window_size": CROP_WINDOW_SIZE,
            "smoothing_sigma": SMOOTHING_SIGMA, "top_k_candidates": TOP_K_CANDIDATES,
            "min_peak_distance": MIN_PEAK_DISTANCE, "temporal_seeds": TEMPORAL_SEEDS,
            "top1_margin": TOP1_MARGIN, "fallback_confidence_threshold": FALLBACK_CONFIDENCE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD, "prompt_mode": "point",
        },
        "adopted_method": METHOD_FALLBACK,
        "sequences": len(sequences),
        "localization": summarize(all_rows),
        "reference_a100": {
            "note": "우리 A100 재현값. GPU 가 다르면 1~2%p 차이가 정상.",
            "sandl-chunk (809 chunks, ~20417 frames)": {
                METHOD_FALLBACK: "≈ 0.423", METHOD_TEMPORAL: "≈ 0.42", METHOD_FRAMEWISE: "≈ 0.37",
            },
            "trainvalid (val 100 cases, ~835 frames)": {
                METHOD_FALLBACK: "≈ 0.50", METHOD_TEMPORAL: "≈ 0.49", METHOD_FRAMEWISE: "≈ 0.30",
            },
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary["localization"], ensure_ascii=False, indent=2), flush=True)
    print(f"\n[done] {frame_csv}\n[done] {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
