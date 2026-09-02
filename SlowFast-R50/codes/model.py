from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from pytorchvideo.models.hub import slowfast_r50
from torch import nn


class SlowFastR50Binary(nn.Module):
    """Kinetics-pretrained SlowFast R50 with one binary logit."""

    slowfast_alpha = 4

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = slowfast_r50(pretrained=pretrained, progress=True)
        in_features = self.backbone.blocks[-1].proj.in_features
        self.backbone.blocks[-1].proj = nn.Linear(in_features, 1)

    def forward(
        self, video: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        frame_count = video.shape[2]
        slow_frame_count = max(1, math.ceil(frame_count / self.slowfast_alpha))
        slow_indices = torch.linspace(
            0,
            frame_count - 1,
            steps=slow_frame_count,
            device=video.device,
        ).long()
        pathways: list[torch.Tensor] = [video.index_select(2, slow_indices), video]
        # Blocks 0-4 are the pretrained dual-pathway backbone and fusion stages.
        for block in self.backbone.blocks[:5]:
            pathways = block(pathways)
        slow_features, fast_features = pathways
        slow_aligned = F.interpolate(
            slow_features,
            size=fast_features.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        features = torch.cat((slow_aligned, fast_features), dim=1)

        # Skip the fixed 8/32x7x7 Kinetics pools. Fast retains every input frame;
        # adaptive pooling accepts the original variable video duration.
        head = self.backbone.blocks[-1]
        classifier_input = head.dropout(features) if head.dropout is not None else features
        pooled = F.adaptive_avg_pool3d(classifier_input, output_size=1).flatten(1)
        logits = self.classifier(pooled)
        if return_features:
            return logits, features
        return logits

    @property
    def classifier(self) -> nn.Linear:
        return self.backbone.blocks[-1].proj

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

    def set_training_mode(self, backbone_trainable: bool) -> None:
        self.train()
        if not backbone_trainable:
            # Keep frozen BatchNorm running statistics unchanged as well.
            for block in self.backbone.blocks[:-1]:
                block.eval()
            self.backbone.blocks[-1].eval()
            self.classifier.train()


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
