"""perceptual loss - Multi-layer VGG19 feature matching with ImageNet normalization. Frozen VGG — no gradients through VGG parameters"""
from typing import List
import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers: List[int] = None):
        super().__init__()
        if feature_layers is None:
            feature_layers = [2, 7, 16, 25, 34]

        vgg_feats = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        for p in vgg_feats.parameters():
            p.requires_grad = False

        # build list of sub-networks up to each target layer
        self.slice_nets = nn.ModuleList()
        self.layer_indices = feature_layers

        start = 0
        for end in sorted(feature_layers):
            self.slice_nets.append(nn.Sequential(*list(vgg_feats.children())[start:end + 1]))
            start = end + 1

        # imageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

        self.criterion = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0.0
        x, y = pred, target
        for net in self.slice_nets:
            x = net(x)
            y = net(y)
            loss = loss + self.criterion(x, y.detach())

        return loss / len(self.slice_nets)
