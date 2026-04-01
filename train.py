"""training entry point"""
import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from config import Config, get_config
from models import FrequSpatialGenerator, UNetDiscriminator
from losses import LossManager
from data import get_dataloaders
from utils import (EMA, save_checkpoint, load_checkpoint, compute_metrics, save_result_grid, save_training_curves, save_freq_comparison,)

# LR schedule helpers
def build_schedulers(opt_g, opt_d, cfg: Config):
    """cosine annealing for both optimizers(after warmup)"""
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=cfg.train.num_epochs - cfg.train.warmup_epochs, eta_min=cfg.train.min_lr,)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=cfg.train.num_epochs - cfg.train.warmup_epochs, eta_min=cfg.train.min_lr / 2,)
    return sched_g, sched_d


def apply_warmup_lr(optimizer, base_lr: float, epoch: int, warmup_epochs: int):
    """linear warmup: ramps from base_lr/10 → base_lr over warmup_epochs"""
    if epoch >= warmup_epochs:
        return
    lr = (base_lr / 10) + (base_lr - base_lr / 10) * (epoch / warmup_epochs)
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# one training epoch
def train_epoch(epoch: int, generator: nn.Module, discriminator: nn.Module, opt_g: torch.optim.Optimizer, opt_d: torch.optim.Optimizer, loss_mgr: LossManager,
    ema: EMA, scaler_g: GradScaler, scaler_d: GradScaler, train_loader, device: torch.device, cfg: Config, use_adv: bool,):
    generator.train()
    discriminator.train()

    epoch_losses = {"total": 0.0, "l1": 0.0, "perc": 0.0, "freq": 0.0, "adv": 0.0}
    n_batches = len(train_loader)
    pbar = tqdm(train_loader, desc=f"Train [{epoch+1:03d}]", leave=False, dynamic_ncols=True)

    for batch in pbar:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)

        # discriminator step
        if use_adv:
            opt_d.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=cfg.train.use_amp):
                with torch.no_grad():
                    sr_detach = generator(lr)

                real_g, real_l = discriminator(hr)
                fake_g, fake_l = discriminator(sr_detach.detach())
                d_losses = loss_mgr.discriminator_loss(real_g, real_l, fake_g, fake_l)

            scaler_d.scale(d_losses["total"]).backward()
            scaler_d.unscale_(opt_d)
            nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            scaler_d.step(opt_d)
            scaler_d.update()

        # generator step
        opt_g.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=cfg.train.use_amp):
            sr = generator(lr)

            fake_g, fake_l = discriminator(sr) if use_adv else (None, None)
            g_losses = loss_mgr.generator_loss(sr, hr, fake_g, fake_l)

        scaler_g.scale(g_losses["total"]).backward()
        scaler_g.unscale_(opt_g)
        nn.utils.clip_grad_norm_(generator.parameters(), max_norm=cfg.train.gradient_clip_norm)
        scaler_g.step(opt_g)
        scaler_g.update()

        # update EMA
        if epoch >= cfg.train.ema_start_epoch:
            ema.update(generator)

        # accumulate
        for k in epoch_losses:
            epoch_losses[k] += g_losses.get(k, torch.tensor(0.0)).item()

        pbar.set_postfix({"L": f"{g_losses['total'].item():.3f}", "L1": f"{g_losses['l1'].item():.3f}", "Perc": f"{g_losses['perc'].item():.3f}", "Freq": f"{g_losses['freq'].item():.3f}", "Adv": f"{g_losses['adv'].item():.4f}",})

    return {k: v / n_batches for k, v in epoch_losses.items()}


# validation
@torch.no_grad()
def validate(generator: nn.Module, val_loader, device: torch.device, cfg: Config, epoch: int, save_dir: str, save_vis: bool = False,):
    generator.eval()
    total_psnr, total_ssim, n = 0.0, 0.0, 0

    lr_batch, sr_batch, hr_batch, psnr_list, ssim_list = [], [], [], [], []

    for batch in tqdm(val_loader, desc="  Val", leave=False, dynamic_ncols=True):
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)

        sr = generator(lr).clamp(0, 1)

        psnr, ssim = compute_metrics(sr, hr)
        total_psnr += psnr
        total_ssim += ssim
        n += 1

        if save_vis and len(lr_batch) < 4:
            lr_batch.append(lr.cpu())
            sr_batch.append(sr.cpu())
            hr_batch.append(hr.cpu())
            psnr_list.append(psnr)
            ssim_list.append(ssim)

    avg_psnr = total_psnr / max(n, 1)
    avg_ssim = total_ssim / max(n, 1)

    if save_vis and lr_batch:
        lr_t = torch.cat(lr_batch, dim=0)
        sr_t = torch.cat(sr_batch, dim=0)
        hr_t = torch.cat(hr_batch, dim=0)

        save_result_grid(lr_t, sr_t, hr_t, psnr_list, ssim_list, save_path=os.path.join(save_dir, f"vis_epoch_{epoch:03d}.png"),)
        save_freq_comparison(sr_t[:1], hr_t[:1], save_path=os.path.join(save_dir, f"freq_epoch_{epoch:03d}.png"),)

    return avg_psnr, avg_ssim


# main
def main():
    parser = argparse.ArgumentParser(description="FrequSpatial Revival Training")
    parser.add_argument("--scale", type=int, default=None, help="SR scale factor")
    parser.add_argument("--resume", action="store_true", help="resume from latest checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="override num_epochs")
    parser.add_argument("--bs", type=int, default=None, help="override batch_size")
    args = parser.parse_args()

    # config
    cfg = get_config()
    if args.scale: cfg.model.scale = args.scale
    if args.epochs: cfg.train.num_epochs = args.epochs
    if args.bs: cfg.train.batch_size = args.bs
    cfg.__post_init__()  # re-validate after overrides

    os.makedirs(cfg.train.save_dir, exist_ok=True)
    ckpt_dir = os.path.join(cfg.train.save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"FrequSpatial — scale x{cfg.model.scale} | device: {device}")

    # data
    train_loader, val_loader = get_dataloaders(cfg.train, cfg.model.scale)

    # models
    generator = FrequSpatialGenerator(cfg.model).to(device)
    discriminator = UNetDiscriminator(cfg.model).to(device)

    n_params_g = sum(p.numel() for p in generator.parameters()) / 1e6
    n_params_d = sum(p.numel() for p in discriminator.parameters()) / 1e6
    print(f"generator: {n_params_g:.2f}M params")
    print(f"discriminator: {n_params_d:.2f}M params\n")

    # optimizers
    opt_g = torch.optim.AdamW(generator.parameters(), lr=cfg.train.lr_g, betas=(cfg.train.beta1, cfg.train.beta2), weight_decay=cfg.train.weight_decay,)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=cfg.train.lr_d, betas=(cfg.train.beta1, cfg.train.beta2), weight_decay=0.0,) # no weight decay on discriminator

    # schedulers & AMP
    sched_g, sched_d = build_schedulers(opt_g, opt_d, cfg)
    scaler_g = GradScaler(device.type, enabled=cfg.train.use_amp)
    scaler_d = GradScaler(device.type, enabled=cfg.train.use_amp)

    # EMA & loss
    ema = EMA(generator, decay=cfg.train.ema_decay)
    loss_mgr = LossManager(cfg.train).to(device)

    # checkpoint resume
    latest_ckpt = os.path.join(ckpt_dir, "latest.pth")
    best_ckpt = os.path.join(ckpt_dir, "best.pth")
    start_epoch, best_psnr, train_losses, val_psnrs = 0, 0.0, [], []

    if args.resume or os.path.isfile(latest_ckpt):
        start_epoch, best_psnr, train_losses, val_psnrs = load_checkpoint(latest_ckpt, generator, discriminator, opt_g, opt_d, sched_g, sched_d, ema, device,)

    # wandb
    if cfg.train.use_wandb:
        try:
            import wandb
            wandb.init(project=cfg.train.wandb_project, entity=cfg.train.wandb_entity or None, config={"model": cfg.model.__dict__, "train": cfg.train.__dict__}, resume="allow",)
        except Exception as e:
            print(f"[wandb] failed to init: {e}. disabling")
            cfg.train.use_wandb = False

    # track per-component loss history
    loss_history = {"l1": [], "perc": [], "freq": [], "adv": []}

    # save original loss weights once before the loop(not inside)
    _orig_perceptual_weight = cfg.train.perceptual_weight
    _orig_adv_weight = cfg.train.adv_weight

    # training loop
    print(f"starting from epoch {start_epoch+1} / {cfg.train.num_epochs}\n")

    for epoch in range(start_epoch, cfg.train.num_epochs):
        t0 = time.time()

        # warmup: only L1 + freq for first N epochs(no GAN, no perceptual). this gives the generator a stable starting point before adversarial pressure.
        warmup_phase = epoch < cfg.train.warmup_epochs
        use_adv = not warmup_phase

        if warmup_phase:
            cfg.train.perceptual_weight = 0.0
            cfg.train.adv_weight        = 0.0
        else:
            # restore full weights from the saved originals
            cfg.train.perceptual_weight = _orig_perceptual_weight
            cfg.train.adv_weight = _orig_adv_weight
            if epoch == cfg.train.warmup_epochs:
                print("[train] warmup complete — enabling perceptual + adversarial losses")

        # apply warmup LR
        apply_warmup_lr(opt_g, cfg.train.lr_g, epoch, cfg.train.warmup_epochs)
        apply_warmup_lr(opt_d, cfg.train.lr_d, epoch, cfg.train.warmup_epochs)

        # train
        avg_losses = train_epoch(epoch, generator, discriminator, opt_g, opt_d, loss_mgr, ema, scaler_g, scaler_d, train_loader, device, cfg, use_adv,)
        train_losses.append(avg_losses["total"])
        for k in loss_history:
            loss_history[k].append(avg_losses.get(k, 0.0))

        # step cosine schedulers(only after warmup)
        if epoch >= cfg.train.warmup_epochs:
            sched_g.step()
            sched_d.step()

        # validation with EMA weights
        save_vis = (epoch + 1) % cfg.train.vis_interval == 0
        ema.apply_shadow(generator)
        avg_psnr, avg_ssim = validate(generator, val_loader, device, cfg, epoch=epoch + 1, save_dir=cfg.train.save_dir, save_vis=save_vis,)
        ema.restore(generator)
        val_psnrs.append(avg_psnr)

        elapsed = time.time() - t0
        lr_now  = opt_g.param_groups[0]["lr"]
        print(f"epoch [{epoch+1:03d}/{cfg.train.num_epochs}] " f"loss {avg_losses['total']:.4f} " f"(L1 {avg_losses['l1']:.4f} | "
            f"perc {avg_losses['perc']:.4f} | " f"freq {avg_losses['freq']:.4f} | " f"adv {avg_losses['adv']:.5f}) | "
            f"PSNR {avg_psnr:.3f} dB | SSIM {avg_ssim:.4f} | " f"LR {lr_now:.2e} | {elapsed:.0f}s")

        # Logging
        if cfg.train.use_wandb:
            import wandb
            wandb.log({"epoch": epoch + 1, "loss/total": avg_losses["total"], "loss/l1": avg_losses["l1"], "loss/perc": avg_losses["perc"],
                "loss/freq": avg_losses["freq"], "loss/adv": avg_losses["adv"], "val/psnr": avg_psnr, "val/ssim": avg_ssim, "lr/g": lr_now,})

        # checkpointing
        ckpt_kwargs = dict(generator=generator, discriminator=discriminator, opt_g=opt_g, opt_d=opt_d, sched_g=sched_g, sched_d=sched_d,
            ema=ema, best_psnr=best_psnr, train_losses=train_losses, val_psnrs=val_psnrs,)

        save_checkpoint(latest_ckpt, epoch, **ckpt_kwargs)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            save_checkpoint(best_ckpt, epoch, **ckpt_kwargs)
            print(f"new best PSNR: {best_psnr:.4f} dB")

        if (epoch + 1) % cfg.train.checkpoint_interval == 0:
            periodic = os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pth")
            save_checkpoint(periodic, epoch, **ckpt_kwargs)

        # training curves
        if (epoch + 1) % cfg.train.checkpoint_interval == 0:
            save_training_curves(train_losses, val_psnrs, loss_history, save_path=os.path.join(cfg.train.save_dir, "training_curves.png"),)

    # final curves
    save_training_curves(train_losses, val_psnrs, loss_history, save_path=os.path.join(cfg.train.save_dir, "training_curves_final.png"),)
    print(f"\ntraining complete. best PSNR: {best_psnr:.4f} dB")

    if cfg.train.use_wandb:
        import wandb
        wandb.finish()

if __name__ == "__main__":
    main()