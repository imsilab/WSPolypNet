from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cam import box_iou, cam_to_box, class_activation_maps
from dataset import BinaryVideoDataset, LocalizationVideoDataset, video_frame_count
from model import SlowFastR50Binary
from train import (
    DEFAULT_NEGATIVE_ROOT,
    DEFAULT_POSITIVE_ROOT,
    DEFAULT_ROI_MANIFEST,
    DEFAULT_VALIDATION_ROOT,
    build_optimizer,
)
from utils import CSV_COLUMNS, save_checkpoint, seed_everything, update_csv


def first_backbone_parameter(model: SlowFastR50Binary) -> torch.nn.Parameter:
    return next(
        parameter
        for name, parameter in model.named_parameters()
        if "backbone.blocks.6.proj" not in name
    )


def one_update(
    model: SlowFastR50Binary,
    video: torch.Tensor,
    label: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    backbone_trainable: bool,
) -> None:
    model.set_training_mode(backbone_trainable)
    optimizer.zero_grad(set_to_none=True)
    logits = model(video)
    assert isinstance(logits, torch.Tensor)
    nn.BCEWithLogitsLoss()(logits, label.float().view(-1, 1)).backward()
    optimizer.step()


def main() -> int:
    seed_everything(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Sanity check expects the configured GPU environment"

    training = BinaryVideoDataset(
        DEFAULT_NEGATIVE_ROOT,
        DEFAULT_POSITIVE_ROOT,
        spatial_size=224,
        horizontal_flip_probability=0.0,
        roi_manifest_path=DEFAULT_ROI_MANIFEST,
    )
    assert len(training) == 1486
    assert training.class_counts == {0: 615, 1: 871}
    negative_video, negative_label, _ = training[0]
    short_index = next(
        index
        for index, sample in enumerate(training.samples)
        if sample.label == 1 and video_frame_count(sample.video_path) < 30
    )
    short_video, short_label, _ = training[short_index]
    assert negative_video.shape == (3, 30, 224, 224)
    assert short_video.shape[0] == 3 and short_video.shape[1] < 30
    print(
        f"Loader shapes: negative={tuple(negative_video.unsqueeze(0).shape)}, "
        f"short_positive={tuple(short_video.unsqueeze(0).shape)}"
    )

    model = SlowFastR50Binary(pretrained=True).to(device)
    short_batch = short_video.unsqueeze(0).to(device)
    target = torch.tensor([short_label], device=device)
    with torch.inference_mode():
        output = model(negative_video.unsqueeze(0).to(device))
    assert isinstance(output, torch.Tensor) and output.shape == (1, 1)
    print("SlowFast R50 whole-video forward: OK")

    backbone_parameter = first_backbone_parameter(model)
    model.set_backbone_trainable(False)
    frozen_before = backbone_parameter.detach().clone()
    head_before = model.classifier.weight.detach().clone()
    head_optimizer = build_optimizer(model, 1e-4, 0.01)
    one_update(model, short_batch, target, head_optimizer, False)
    assert backbone_parameter.grad is None
    assert torch.equal(frozen_before, backbone_parameter.detach())
    assert not torch.equal(head_before, model.classifier.weight.detach())
    print("Frozen stage: backbone unchanged, head updated")

    model.set_backbone_trainable(True)
    full_optimizer = build_optimizer(model, 1e-4, 0.01)
    optimizer_ids = {
        id(parameter)
        for group in full_optimizer.param_groups
        for parameter in group["params"]
    }
    assert id(backbone_parameter) in optimizer_ids
    unfrozen_before = backbone_parameter.detach().clone()
    one_update(model, short_batch, target, full_optimizer, True)
    assert backbone_parameter.grad is not None
    assert not torch.equal(unfrozen_before, backbone_parameter.detach())
    print("Unfrozen stage: backbone is in optimizer and updated")

    validation = LocalizationVideoDataset(
        DEFAULT_VALIDATION_ROOT / "video",
        DEFAULT_VALIDATION_ROOT / "annotation",
        spatial_size=224,
        roi_manifest_path=DEFAULT_ROI_MANIFEST,
    )
    validation_video, boxes, original_size, transform, _ = validation[0]
    roi = transform.roi
    assert original_size == (roi.source_height, roi.source_width)
    with torch.inference_mode():
        _, cams = class_activation_maps(
            model.eval(), validation_video.unsqueeze(0).to(device), target_class=1
        )
    assert tuple(cams.shape) == (
        1,
        validation_video.shape[1],
        224,
        224,
    )
    assert len(boxes) == validation_video.shape[1]
    mapped_full_roi = transform.canvas_box_to_original(
        tuple(float(value) for value in transform.content_xyxy)
    )
    assert mapped_full_roi == tuple(float(value) for value in roi.xyxy)
    synthetic = np.zeros((20, 30), dtype=np.float32)
    synthetic[3:10, 5:15] = 1.0
    assert cam_to_box(synthetic, 0.5) == (5.0, 3.0, 15.0, 10.0)
    masked_cam = np.zeros((20, 30), dtype=np.float32)
    masked_cam[1:4, 1:4] = 10.0
    masked_cam[8:12, 10:16] = 1.0
    valid_mask = np.zeros_like(masked_cam, dtype=bool)
    valid_mask[5:18, 7:25] = True
    assert cam_to_box(masked_cam, 0.5, valid_mask) == (10.0, 8.0, 16.0, 12.0)
    expected_iou = 25.0 / 175.0
    assert abs(box_iou((0, 0, 10, 10), (5, 5, 15, 15)) - expected_iou) < 1e-9
    print(f"CAM shape={tuple(cams.shape)}, bbox and IoU: OK")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint = root / "epoch_001.pth"
        save_checkpoint(
            checkpoint,
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": full_optimizer.state_dict(),
            },
        )
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert loaded["epoch"] == 1
        metrics = {
            "epoch": 1,
            "train_loss": 0.5,
            "train_accuracy": 0.6,
            "corloc_0.3": 0.3,
            "corloc_0.5": 0.2,
            "corloc_0.7": 0.1,
        }
        csv_path = root / "train.csv"
        update_csv(csv_path, metrics)
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            assert tuple(reader.fieldnames or ()) == CSV_COLUMNS
            assert len(list(reader)) == 1
    print("Checkpoint and train.csv atomic write: OK")
    print("ALL SANITY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

