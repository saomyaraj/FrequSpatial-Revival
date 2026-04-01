"""combined loss manager"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from .perceptual import PerceptualLoss
from .frequency import FrequencyLoss
from .adversarial import AdversarialLoss
from config import TrainConfig

class LossManager(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg

        self.l1   = nn.L1Loss()
        self.perc = PerceptualLoss(feature_layers=cfg.vgg_layers)
        self.freq = FrequencyLoss()
        self.adv  = AdversarialLoss()

    def generator_loss(self, pred: torch.Tensor, target: torch.Tensor, fake_global: Optional[torch.Tensor] = None, fake_local:  Optional[torch.Tensor] = None,) -> Dict[str, torch.Tensor]:
        """compute all generator losses. Returns dict of individual + total"""
        losses = {}

        losses["l1"]   = self.cfg.l1_weight        * self.l1(pred, target)
        losses["perc"] = self.cfg.perceptual_weight * self.perc(pred, target)
        losses["freq"] = self.cfg.freq_weight       * self.freq(pred, target)

        if fake_global is not None and fake_local is not None:
            losses["adv"] = self.cfg.adv_weight * self.adv.generator_loss(fake_global, fake_local)
        else:
            losses["adv"] = torch.tensor(0.0, device=pred.device)

        losses["total"] = sum(v for v in losses.values())
        return losses

    def discriminator_loss(self, real_global: torch.Tensor, real_local:  torch.Tensor, fake_global: torch.Tensor, fake_local:  torch.Tensor,) -> Dict[str, torch.Tensor]:
        loss = self.adv.discriminator_loss(real_global, real_local, fake_global, fake_local)
        return {"disc": loss, "total": loss}
