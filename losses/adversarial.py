"""adversarial loss (Hinge)"""

import torch
import torch.nn as nn

class AdversarialLoss(nn.Module):
    def __init__(self, local_weight: float = 1.0, global_weight: float = 1.0):
        super().__init__()
        self.local_weight  = local_weight
        self.global_weight = global_weight

    def _hinge_d(self, real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
        return (torch.relu(1.0 - real_scores).mean() + torch.relu(1.0 + fake_scores).mean())

    def _hinge_g(self, fake_scores: torch.Tensor) -> torch.Tensor:
        return -fake_scores.mean()

    def discriminator_loss(self,
        real_global: torch.Tensor,  # [B, 1]
        real_local:  torch.Tensor,  # [B, 1, H, W]
        fake_global: torch.Tensor, fake_local:  torch.Tensor,) -> torch.Tensor:
        loss_global = self._hinge_d(real_global, fake_global)
        loss_local  = self._hinge_d(real_local,  fake_local)
        return self.global_weight * loss_global + self.local_weight * loss_local

    def generator_loss(self, fake_global: torch.Tensor, fake_local:  torch.Tensor,) -> torch.Tensor:
        return (self.global_weight * self._hinge_g(fake_global) + self.local_weight  * self._hinge_g(fake_local))
