"""checkpoint & EMA utils"""
import os
import copy
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

# EMA
class EMA:
    """maintains an exponential moving average of model parameters. shadow weights live on CPU to save GPU memory"""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}

        # initialize shadow with current model params(detached, on CPU)
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().cpu().float()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """call once per generator step"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (self.decay * self.shadow[name] + (1.0 - self.decay) * param.data.cpu().float())

    def apply_shadow(self, model: nn.Module):
        """swap model weights with EMA shadow weights. call before validation / inference"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.device).to(param.dtype))

    def restore(self, model: nn.Module):
        """restore original (non-EMA) weights. call after validation to resume training"""
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> Dict[str, Any]:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: Dict[str, Any]):
        self.shadow = state["shadow"]
        self.decay  = state.get("decay", self.decay)


# checkpoint save/load
def save_checkpoint(save_path: str, epoch: int, generator: nn.Module, discriminator: nn.Module, opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer, sched_g, sched_d, ema: EMA, best_psnr: float, train_losses: list, val_psnrs: list,):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"epoch": epoch, "generator": generator.state_dict(), "discriminator": discriminator.state_dict(), "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(), "sched_g": sched_g.state_dict(), "sched_d": sched_d.state_dict(), "ema": ema.state_dict(),
            "best_psnr": best_psnr, "train_losses": train_losses, "val_psnrs": val_psnrs,}, save_path,)
    print(f"[ckpt] saved -> {save_path}")


def load_checkpoint(load_path: str, generator: nn.Module, discriminator: nn.Module, opt_g: torch.optim.Optimizer, opt_d: torch.optim.Optimizer,
    sched_g, sched_d, ema: EMA, device: torch.device,):
    """loads checkpoint. returns (start_epoch, best_psnr, train_losses, val_psnrs). returns (0, 0.0, [], []) if path doesn't exist"""
    if not os.path.isfile(load_path):
        print(f"  [ckpt] No checkpoint at {load_path}, starting fresh.")
        return 0, 0.0, [], []

    print(f"[ckpt] loading from {load_path}")
    ckpt = torch.load(load_path, map_location=device)

    # handle DataParallel prefix
    def strip_ddp(sd):
        return {k.replace("module.", ""): v for k, v in sd.items()}

    generator.load_state_dict(strip_ddp(ckpt["generator"]))
    discriminator.load_state_dict(strip_ddp(ckpt["discriminator"]))
    opt_g.load_state_dict(ckpt["opt_g"])
    opt_d.load_state_dict(ckpt["opt_d"])
    sched_g.load_state_dict(ckpt["sched_g"])
    sched_d.load_state_dict(ckpt["sched_d"])
    ema.load_state_dict(ckpt["ema"])

    start_epoch = ckpt.get("epoch", -1) + 1
    best_psnr = ckpt.get("best_psnr", 0.0)
    train_losses = ckpt.get("train_losses", [])
    val_psnrs = ckpt.get("val_psnrs", [])

    print(f"[ckpt] resumed from epoch {start_epoch}, best PSNR = {best_psnr:.4f} dB")
    return start_epoch, best_psnr, train_losses, val_psnrs
