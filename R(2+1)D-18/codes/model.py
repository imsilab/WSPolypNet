from __future__ import annotations

import torch
from torch import nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18


class R2Plus1D18Binary(nn.Module):
    """Kinetics-pretrained R(2+1)D-18 with one binary-classification logit."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None
        self.backbone = r2plus1d_18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(
        self, video: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone.stem(video)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        features = self.backbone.layer4(x)
        pooled = self.backbone.avgpool(features).flatten(1)
        logits = self.backbone.fc(pooled)
        if return_features:
            return logits, features
        return logits

    @property
    def classifier(self) -> nn.Linear:
        return self.backbone.fc

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

    def set_training_mode(self, backbone_trainable: bool) -> None:
        self.train()
        if not backbone_trainable:
            # Keep frozen BatchNorm running statistics unchanged as well.
            self.backbone.stem.eval()
            self.backbone.layer1.eval()
            self.backbone.layer2.eval()
            self.backbone.layer3.eval()
            self.backbone.layer4.eval()
            self.backbone.avgpool.eval()
            self.classifier.train()


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
