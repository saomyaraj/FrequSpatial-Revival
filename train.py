"""training entry point (PSNR track: pure L1, the lightweight-SR convention)"""
import os
import argparse
import time
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from config import Config, get_config, available_presets
from models import FrequSpatialGenerator
from models.spectral import enable_stats, collect_stats
from data import get_dataloaders
from utils import (EMA, save_checkpoint, load_checkpoint, compute_metrics, save_result_grid,
                   save_training_curves, save_freq_comparison, set_seed, report_complexity, tiled_forward,
                   write_run_manifest, match_size, MetricLogger, format_eta)


# LR schedule helpers
def build_scheduler(opt_g, cfg: Config):
    """cosine annealing for the generator optimizer (after warmup)"""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_g, T_max=cfg.train.num_epochs - cfg.train.warmup_epochs, eta_min=cfg.train.min_lr)


def apply_warmup_lr(optimizer, base_lr: float, epoch: int, warmup_epochs: int):
    """linear warmup: ramps base_lr/warmup → base_lr over warmup_epochs, reaching base_lr on the
    LAST warmup epoch so the first cosine epoch starts cleanly at base_lr."""
    if epoch >= warmup_epochs:
        return
    lr = base_lr * (epoch + 1) / warmup_epochs
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# one training epoch
def train_epoch(epoch, generator, opt_g, criterion, ema, scaler_g, train_loader, device, cfg):
    generator.train()
    total = 0.0
    n_batches = len(train_loader)
    pbar = tqdm(train_loader, desc=f"Train [{epoch+1:03d}]", leave=False, dynamic_ncols=True)
    diag = {}

    for i, batch in enumerate(pbar):
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)

        # SR-MoSO diagnostics on the first batch only (negligible cost, once per epoch)
        if i == 0:
            enable_stats(generator, True)

        opt_g.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=cfg.train.use_amp):
            sr = generator(lr)
            loss = criterion(sr, hr)
        scaler_g.scale(loss).backward()
        scaler_g.unscale_(opt_g)
        nn.utils.clip_grad_norm_(generator.parameters(), max_norm=cfg.train.gradient_clip_norm)
        scaler_g.step(opt_g)
        scaler_g.update()

        if i == 0:
            diag = collect_stats(generator)
            enable_stats(generator, False)

        if epoch >= cfg.train.ema_start_epoch:
            ema.update(generator)

        total += loss.item()
        pbar.set_postfix({"L1": f"{loss.item():.4f}"})

    return total / max(n_batches, 1), diag


# validation
@torch.no_grad()
def validate(generator, val_loader, device, cfg, epoch, save_dir, save_vis=False, max_images=None):
    generator.eval()
    total_psnr, total_ssim, n = 0.0, 0.0, 0
    scale = cfg.model.scale
    tile = cfg.train.val_tile
    vis = []   # list of (lr, sr, hr, psnr, ssim) — kept per-image: full-res val images differ in size

    for batch in tqdm(val_loader, desc="  Val", leave=False, dynamic_ncols=True):
        if max_images is not None and n >= max_images:
            break
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)

        # tiled inference so full-resolution validation images don't OOM
        sr = tiled_forward(generator, lr, scale=scale, tile=tile, overlap=tile // 6).clamp(0, 1)
        sr, hr = match_size(sr, hr)          # guard against scale*LR != HR rounding

        # benchmark protocol: Y-channel + shave `scale` border pixels
        psnr, ssim = compute_metrics(sr, hr, crop_border=scale)
        total_psnr += psnr
        total_ssim += ssim
        n += 1

        if save_vis and len(vis) < 4:
            vis.append((lr.cpu(), sr.cpu(), hr.cpu(), psnr, ssim))

    avg_psnr = total_psnr / max(n, 1)
    avg_ssim = total_ssim / max(n, 1)

    # save each image separately — DIV2K val images have different resolutions, so they cannot
    # be concatenated into one batch (that was a crash).
    for i, (lr_i, sr_i, hr_i, p, s) in enumerate(vis):
        save_result_grid(lr_i, sr_i, hr_i, [p], [s],
                         save_path=os.path.join(save_dir, f"vis_epoch_{epoch:03d}_{i}.png"))
    if vis:
        save_freq_comparison(vis[0][1], vis[0][2],
                             save_path=os.path.join(save_dir, f"freq_epoch_{epoch:03d}.png"))

    return avg_psnr, avg_ssim


# main
def main():
    parser = argparse.ArgumentParser(description="SR-MoSO Training (PSNR track)")
    parser.add_argument("--scale", type=int, default=None, help="SR scale factor")
    parser.add_argument("--preset", type=str, default="proposed", choices=available_presets(), help="ablation preset")
    parser.add_argument("--resume", action="store_true", help="resume from latest checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="override num_epochs")
    parser.add_argument("--bs", type=int, default=None, help="override batch_size")
    parser.add_argument("--save_dir", type=str, default=None, help="override save dir")
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases (remote monitoring)")
    args = parser.parse_args()

    # config
    cfg = get_config(args.preset)
    if args.scale: cfg.model.scale = args.scale
    if args.epochs: cfg.train.num_epochs = args.epochs
    if args.bs: cfg.train.batch_size = args.bs
    if args.save_dir: cfg.train.save_dir = args.save_dir
    if args.wandb: cfg.train.use_wandb = True
    cfg.validate()

    set_seed(cfg.train.seed, cfg.train.deterministic)

    os.makedirs(cfg.train.save_dir, exist_ok=True)
    ckpt_dir = os.path.join(cfg.train.save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    write_run_manifest(cfg.train.save_dir, cfg, cfg.train.seed, cfg.train.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"SR-MoSO — preset '{args.preset}' | scale x{cfg.model.scale} | device: {device}")

    # data
    train_loader, val_loader = get_dataloaders(cfg.train, cfg.model.scale)

    # model
    generator = FrequSpatialGenerator(cfg.model).to(device)
    report_complexity(generator, "generator", scale=cfg.model.scale, device=device)
    print()

    # optimizer, scheduler, AMP
    opt_g = torch.optim.AdamW(generator.parameters(), lr=cfg.train.lr_g,
                              betas=(cfg.train.beta1, cfg.train.beta2), weight_decay=cfg.train.weight_decay)
    sched_g = build_scheduler(opt_g, cfg)
    scaler_g = GradScaler(device.type, enabled=cfg.train.use_amp)

    # EMA & loss
    ema = EMA(generator, decay=cfg.train.ema_decay, update_every=cfg.train.ema_update_every,
              cpu_offload=cfg.train.ema_cpu_offload, start_step_reset=True)
    criterion = nn.L1Loss()

    # checkpoint resume — ONLY when explicitly requested, so a re-run into an existing dir
    # (e.g. run_ablations --force) trains fresh instead of silently inheriting stale state.
    latest_ckpt = os.path.join(ckpt_dir, "latest.pth")
    best_ckpt = os.path.join(ckpt_dir, "best.pth")
    start_epoch, best_psnr, train_losses, val_psnrs = 0, 0.0, [], []

    if args.resume:
        start_epoch, best_psnr, train_losses, val_psnrs = load_checkpoint(latest_ckpt, generator, opt_g, sched_g, ema, device)
    elif os.path.isfile(latest_ckpt):
        print(f"[ckpt] {latest_ckpt} exists but --resume was not passed → training from scratch (will overwrite)")

    # wandb
    if cfg.train.use_wandb:
        try:
            import wandb
            wandb.init(project=cfg.train.wandb_project, entity=cfg.train.wandb_entity or None,
                       config={"model": cfg.model.__dict__, "train": cfg.train.__dict__}, resume="allow")
        except Exception as e:
            print(f"[wandb] failed to init: {e}. disabling")
            cfg.train.use_wandb = False

    logger = MetricLogger(cfg.train.save_dir)
    print(f"starting from epoch {start_epoch+1} / {cfg.train.num_epochs}")
    print(f"[log] per-epoch metrics → {logger.path}\n")

    for epoch in range(start_epoch, cfg.train.num_epochs):
        t0 = time.time()

        apply_warmup_lr(opt_g, cfg.train.lr_g, epoch, cfg.train.warmup_epochs)

        avg_loss, diag = train_epoch(epoch, generator, opt_g, criterion, ema, scaler_g, train_loader, device, cfg)
        train_losses.append(avg_loss)

        # step cosine scheduler (only after warmup)
        if epoch >= cfg.train.warmup_epochs:
            sched_g.step()

        # validation with EMA weights (only meaningful once EMA has started)
        save_vis = (epoch + 1) % cfg.train.vis_interval == 0
        use_ema = epoch >= cfg.train.ema_start_epoch
        # fast fixed-subset validation every epoch drives best-checkpoint selection (consistent metric);
        # the FULL valid set runs periodically + on the last epoch for an honest reported number.
        subset = cfg.train.val_subset if cfg.train.val_subset > 0 else None
        is_last = (epoch + 1) == cfg.train.num_epochs
        full_val = is_last or ((epoch + 1) % cfg.train.full_val_interval == 0)
        if use_ema:
            ema.apply_shadow(generator)
        avg_psnr, avg_ssim = validate(generator, val_loader, device, cfg, epoch=epoch + 1,
                                      save_dir=cfg.train.save_dir, save_vis=save_vis, max_images=subset)
        full_psnr = full_ssim = None
        if full_val and subset is not None:
            full_psnr, full_ssim = validate(generator, val_loader, device, cfg, epoch=epoch + 1,
                                            save_dir=cfg.train.save_dir, save_vis=False, max_images=None)
        if use_ema:
            ema.restore(generator)
        val_psnrs.append(avg_psnr)

        elapsed = time.time() - t0
        lr_now = opt_g.param_groups[0]["lr"]
        eta = format_eta(elapsed * (cfg.train.num_epochs - epoch - 1))
        full_str = f" | FULL {full_psnr:.3f} dB / {full_ssim:.4f}" if full_psnr is not None else ""
        # the two SR-MoSO health numbers: contrib>0 means the spectral path is alive;
        # route_entropy well below 1.0 means routing is specializing rather than staying uniform.
        moso_str = ""
        if "moso/contrib_mean" in diag:
            moso_str = f" | contrib {diag['moso/contrib_mean']:.3f}"
            if "moso/route_entropy_mean" in diag:
                moso_str += f" | H(route) {diag['moso/route_entropy_mean']:.3f}"
        print(f"epoch [{epoch+1:03d}/{cfg.train.num_epochs}] L1 {avg_loss:.4f} | "
              f"PSNR {avg_psnr:.3f} dB | SSIM {avg_ssim:.4f}{full_str}{moso_str} | "
              f"LR {lr_now:.2e} | {elapsed:.0f}s | ETA {eta}")

        logger.log(epoch=epoch + 1, l1=avg_loss, psnr=avg_psnr, ssim=avg_ssim,
                   full_psnr=full_psnr, full_ssim=full_ssim, lr=lr_now,
                   best_psnr=max(best_psnr, avg_psnr), epoch_seconds=elapsed, **diag)

        if cfg.train.use_wandb:
            import wandb
            log = {"epoch": epoch + 1, "loss/l1": avg_loss, "val/psnr": avg_psnr,
                   "val/ssim": avg_ssim, "lr/g": lr_now, **diag}
            if full_psnr is not None:
                log["val/full_psnr"] = full_psnr; log["val/full_ssim"] = full_ssim
            wandb.log(log)

        # checkpointing
        ckpt_kwargs = dict(generator=generator, opt_g=opt_g, sched_g=sched_g,
                           ema=ema, best_psnr=best_psnr, train_losses=train_losses, val_psnrs=val_psnrs)
        save_checkpoint(latest_ckpt, epoch, **ckpt_kwargs)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            ckpt_kwargs["best_psnr"] = best_psnr
            save_checkpoint(best_ckpt, epoch, **ckpt_kwargs)
            print(f"new best PSNR: {best_psnr:.4f} dB")

        if (epoch + 1) % cfg.train.checkpoint_interval == 0:
            save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pth"), epoch, **ckpt_kwargs)
            save_training_curves(train_losses, val_psnrs, save_path=os.path.join(cfg.train.save_dir, "training_curves.png"))

    save_training_curves(train_losses, val_psnrs, save_path=os.path.join(cfg.train.save_dir, "training_curves_final.png"))
    print(f"\ntraining complete. best PSNR: {best_psnr:.4f} dB")

    if cfg.train.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
