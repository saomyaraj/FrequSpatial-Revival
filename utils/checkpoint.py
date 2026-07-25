"""checkpoint & EMA utils"""
import os
import copy
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

# EMA
class EMA:
    """exponential moving average of model parameters.

    Shadow weights live on the model's device by default and are updated in-place with
    torch.lerp — no per-step GPU->CPU sync (the old version copied every param to CPU each
    step, stalling training). Set cpu_offload=True only for low-VRAM runs.
    `update_every` amortizes the update over N steps for further speedup."""
    def __init__(self, model: nn.Module, decay: float = 0.999,
                 update_every: int = 1, cpu_offload: bool = False, start_step_reset: bool = True):
        self.decay = decay
        self.update_every = max(1, int(update_every))
        self.cpu_offload = cpu_offload
        self._step = 0
        # The shadow is seeded here from the *randomly initialized* model, but updates only begin at
        # `ema_start_epoch`. Without a reset, that random init survives with weight decay^n for
        # thousands of steps and the EMA-based validation (and best.pth selection) is contaminated.
        # start_step_reset=True re-seeds the shadow from the live weights on the first update.
        self._needs_reset = bool(start_step_reset)
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}

        # initialize shadow with current params (detached, fp32 master copy)
        for name, param in model.named_parameters():
            if param.requires_grad:
                t = param.detach().float().clone()
                self.shadow[name] = t.cpu() if cpu_offload else t

    @torch.no_grad()
    def update(self, model: nn.Module):
        """call once per generator step (actual update fires every `update_every` steps).
        Effective decay is rescaled so the time-constant is independent of update_every."""
        self._step += 1
        if self._step % self.update_every != 0:
            return
        # first real update: re-seed from the live weights so the random init doesn't linger
        if self._needs_reset:
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    src = param.detach().float()
                    self.shadow[name].copy_(src.cpu() if self.cpu_offload else src)
            self._needs_reset = False
            return
        # rescale decay: applying once per k steps with d^k matches per-step decay d
        d = self.decay ** self.update_every
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                src = param.detach().float()
                if self.cpu_offload:
                    src = src.cpu()
                # shadow = d*shadow + (1-d)*src  == lerp(shadow, src, 1-d)
                self.shadow[name].lerp_(src, 1.0 - d)

    def apply_shadow(self, model: nn.Module):
        """swap model weights with EMA shadow weights. call before validation / inference"""
        assert not self._backup, "apply_shadow() called twice without restore() — would lose raw weights"
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
        # store on CPU for portable checkpoints
        return {"shadow": {k: v.cpu() for k, v in self.shadow.items()},
                "decay": self.decay, "update_every": self.update_every, "step": self._step,
                "needs_reset": self._needs_reset}

    def load_state_dict(self, state: Dict[str, Any]):
        # restore onto whatever device the existing shadow tensors live on
        for k, v in state["shadow"].items():
            if k in self.shadow:
                self.shadow[k] = v.to(self.shadow[k].device)
            else:
                self.shadow[k] = v
        self.decay = state.get("decay", self.decay)
        self.update_every = state.get("update_every", self.update_every)
        self._step = state.get("step", 0)
        self._needs_reset = state.get("needs_reset", False)


# checkpoint save/load
def save_checkpoint(save_path: str, epoch: int, generator: nn.Module, opt_g: torch.optim.Optimizer,
    sched_g, ema: EMA, best_psnr: float, train_losses: list, val_psnrs: list,):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"epoch": epoch, "generator": generator.state_dict(), "opt_g": opt_g.state_dict(),
            "sched_g": sched_g.state_dict(), "ema": ema.state_dict(),
            "best_psnr": best_psnr, "train_losses": train_losses, "val_psnrs": val_psnrs,}, save_path,)
    print(f"[ckpt] saved -> {save_path}")


def load_checkpoint(load_path: str, generator: nn.Module, opt_g: torch.optim.Optimizer,
    sched_g, ema: EMA, device: torch.device,):
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
    opt_g.load_state_dict(ckpt["opt_g"])
    sched_g.load_state_dict(ckpt["sched_g"])
    ema.load_state_dict(ckpt["ema"])

    start_epoch = ckpt.get("epoch", -1) + 1
    best_psnr = ckpt.get("best_psnr", 0.0)
    train_losses = ckpt.get("train_losses", [])
    val_psnrs = ckpt.get("val_psnrs", [])

    print(f"[ckpt] resumed from epoch {start_epoch}, best PSNR = {best_psnr:.4f} dB")
    return start_epoch, best_psnr, train_losses, val_psnrs
