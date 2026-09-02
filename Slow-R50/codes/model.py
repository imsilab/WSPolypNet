from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorchvideo.models.hub import slow_r50
from torch import nn


class SlowR50Binary(nn.Module):
    """Kinetics-pretrained Slow R50 with one binary-classification logit."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = slow_r50(pretrained=pretrained, progress=True)
        in_features = self.backbone.blocks[-1].proj.in_features
        self.backbone.blocks[-1].proj = nn.Linear(in_features, 1)

    def forward(
        self, video: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = video
        # Stop before PyTorchVideo's fixed 8x7x7 Kinetics head. Adaptive global
        # pooling preserves the complete, variable-length temporal sequence.
        for block in self.backbone.blocks[:-1]:
            x = block(x)
        features = x
        pooled = F.adaptive_avg_pool3d(features, output_size=1).flatten(1)
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
