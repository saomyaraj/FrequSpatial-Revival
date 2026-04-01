"""frequency domain loss"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyLoss(nn.Module):
    """combined frequency domain loss:
      1. Log-magnitude L1 (handles dynamic range)
      2. Phase L1 (wrapped) (circular-aware, no ±π artifacts)
      3. multi-scale phase coherence (new — checks phase at 3 resolutions)"""

    def __init__(self, mag_weight: float = 1.0, phase_weight: float = 0.5, ms_phase_weight: float = 0.3, scales: int = 3,):
        super().__init__()
        self.mag_weight = mag_weight
        self.phase_weight = phase_weight
        self.ms_phase_weight = ms_phase_weight
        self.scales = scales

    @staticmethod
    def _wrapped_phase_loss(pred_phase: torch.Tensor, tgt_phase: torch.Tensor) -> torch.Tensor:
        """phase-aware L1: maps difference to [-π, π] before taking absolute value. without wrapping, a phase of +π vs -π would give error 2π, but they're actually the same angle"""
        diff = pred_phase - tgt_phase
        diff = torch.remainder(diff + torch.pi, 2 * torch.pi) - torch.pi
        return diff.abs().mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: [B, C, H, W] in [0, 1]"""
        pred   = pred.float()
        target = target.float()

        total_loss = torch.tensor(0.0, device=pred.device)

        # full-resolution frequency loss
        pred_ft = torch.fft.rfft2(pred, norm="ortho")
        target_ft = torch.fft.rfft2(target, norm="ortho")

        pred_mag = torch.abs(pred_ft)
        target_mag = torch.abs(target_ft)

        # log-magnitude loss
        mag_loss = F.l1_loss(torch.log(pred_mag + 1e-8), torch.log(target_mag + 1e-8),)
        total_loss = total_loss + self.mag_weight * mag_loss

        # wrapped phase loss
        pred_phase = torch.angle(pred_ft)
        target_phase = torch.angle(target_ft)
        phase_loss = self._wrapped_phase_loss(pred_phase, target_phase)
        total_loss = total_loss + self.phase_weight * phase_loss

        # multi-scale phase-coherence loss
        B, C, H, W = pred.shape
        ms_phase_loss = torch.tensor(0.0, device=pred.device)

        for s in range(1, self.scales + 1):
            scale_factor = 2 ** s
            if H // scale_factor < 8 or W // scale_factor < 8:
                break
            H_s, W_s = H // scale_factor, W // scale_factor

            pred_s = F.interpolate(pred, size=(H_s, W_s), mode="bilinear", align_corners=False)
            target_s = F.interpolate(target, size=(H_s, W_s), mode="bilinear", align_corners=False)

            ft_pred_s = torch.fft.rfft2(pred_s, norm="ortho")
            ft_target_s = torch.fft.rfft2(target_s, norm="ortho")

            ms_phase_loss = ms_phase_loss + self._wrapped_phase_loss(torch.angle(ft_pred_s), torch.angle(ft_target_s))

        ms_phase_loss = ms_phase_loss / self.scales
        total_loss = total_loss + self.ms_phase_weight * ms_phase_loss

        return total_loss
