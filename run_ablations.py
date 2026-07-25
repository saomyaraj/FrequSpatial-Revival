"""one-shot ablation runner.

Trains the proposed model + every ablation preset sequentially with IDENTICAL settings (scale,
epochs, batch size), each into its own results subdir, then optionally evaluates each on the
benchmark sets. Reuses train.py / evaluate.py via subprocess so the exact training code path is
shared — no logic duplicated here.

Examples
    # ablation sweep, x4, 300 epochs each, evaluate after each
    py -3 run_ablations.py --scale 4 --epochs 300 --evaluate

    # just the headline + the two operator ablations
    py -3 run_ablations.py --presets proposed static_fno no_routing

    # see the exact commands without running them
    py -3 run_ablations.py --dry_run

    # rebuild just the results table from existing runs
    py -3 run_ablations.py --aggregate_only --scale 4

Already-finished presets (a best.pth in their save dir) are skipped unless --force, so the sweep
is resumable after an interruption.
"""
import os
import sys
import csv
import json
import glob
import argparse
import subprocess

from config import available_presets

DEFAULT_SETS = ["Set5", "Set14", "BSD100", "Urban100", "Manga109"]

# default sweep: proposed first, then the ablations that isolate each component
DEFAULT_PRESETS = ["proposed", "no_freq", "conv_block", "static_fno", "no_routing", "ccso_global",
                   "per_expert_mix", "abs_grid", "fixed_grid_fno", "k2", "k8",
                   "edsr_base", "edsr_moso"]


def build_train_cmd(preset, args, save_dir):
    cmd = [sys.executable, "train.py", "--preset", preset, "--save_dir", save_dir,
           "--scale", str(args.scale)]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.bs is not None:
        cmd += ["--bs", str(args.bs)]
    if args.wandb:
        cmd += ["--wandb"]
    if args.resume:
        cmd += ["--resume"]
    return cmd


def build_eval_cmd(preset, args, save_dir):
    ckpt = os.path.join(save_dir, "checkpoints", "best.pth")
    cmd = [sys.executable, "evaluate.py", "--ckpt", ckpt, "--scale", str(args.scale),
           "--preset", preset, "--bench_root", args.bench_root,
           "--out_json", os.path.join(save_dir, "eval.json")]
    if args.self_ensemble:
        cmd += ["--self_ensemble"]
    return cmd


# ── results aggregation (rows = preset, cols = benchmark sets + params/FLOPs/latency) ─────────
def _collect(out_root: str, tag: str):
    """load every <preset>/eval.json under out_root/tag, in the preferred row order."""
    rows = []
    for ej in glob.glob(os.path.join(out_root, tag, "*", "eval.json")):
        with open(ej) as f:
            rows.append(json.load(f))
    order = {p: i for i, p in enumerate(DEFAULT_PRESETS)}
    rows.sort(key=lambda d: (order.get(d.get("preset", ""), len(DEFAULT_PRESETS)), d.get("preset", "")))
    return rows


def _fmt(v, spec, dash="-"):
    return format(v, spec) if isinstance(v, (int, float)) and v >= 0 else dash


def _to_markdown(rows, sets) -> str:
    header = ["preset", "params(M)", "GFLOPs", "ms"] + sets
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for d in rows:
        cells = [d.get("preset", "?"), _fmt(d.get("params_M", -1), ".3f"),
                 _fmt(d.get("gflops", -1), ".1f"), _fmt(d.get("latency_ms", -1), ".1f")]
        for s in sets:
            r = d.get("sets", {}).get(s)
            cells.append(f"{r['psnr']:.2f} / {r['ssim']:.4f}" if r else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _to_csv_rows(rows, sets):
    out = [["preset", "params_M", "gflops", "latency_ms"]
           + [f"{s}_psnr" for s in sets] + [f"{s}_ssim" for s in sets]]
    for d in rows:
        rec = [d.get("preset", "?"), _fmt(d.get("params_M", -1), ".4f", ""),
               _fmt(d.get("gflops", -1), ".2f", ""), _fmt(d.get("latency_ms", -1), ".2f", "")]
        got = d.get("sets", {})
        rec += [f"{got[s]['psnr']:.4f}" if s in got else "" for s in sets]
        rec += [f"{got[s]['ssim']:.4f}" if s in got else "" for s in sets]
        out.append(rec)
    return out


def aggregate(out_root: str, tag: str, sets=None):
    """write results_table.{md,csv} into out_root/tag; return the markdown string (None if empty)."""
    sets = sets or DEFAULT_SETS
    rows = _collect(out_root, tag)
    if not rows:
        print(f"[aggregate] no eval.json found under {os.path.join(out_root, tag)}")
        return None
    dest = os.path.join(out_root, tag)
    md = _to_markdown(rows, sets)
    with open(os.path.join(dest, "results_table.md"), "w") as f:
        f.write(md + "\n")
    with open(os.path.join(dest, "results_table.csv"), "w", newline="") as f:
        csv.writer(f).writerows(_to_csv_rows(rows, sets))
    print(f"[aggregate] wrote results_table.{{md,csv}} ({len(rows)} presets) -> {dest}")
    print("\n" + md + "\n")
    return md


def main():
    p = argparse.ArgumentParser(description="Run the full ablation sweep with consistent settings")
    p.add_argument("--presets", nargs="*", default=DEFAULT_PRESETS,
                   help=f"presets to sweep (default: {DEFAULT_PRESETS}). available: {available_presets()}")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--epochs", type=int, default=None, help="epochs per preset (default: config value)")
    p.add_argument("--bs", type=int, default=None)
    p.add_argument("--wandb", action="store_true", help="log every run to Weights & Biases")
    p.add_argument("--resume", action="store_true", help="pass --resume to each train run")
    p.add_argument("--out_root", type=str, default="results/ablations", help="parent dir for all runs")
    p.add_argument("--evaluate", action="store_true", help="run benchmark eval after each training run")
    p.add_argument("--bench_root", type=str, default="benchmarks")
    p.add_argument("--self_ensemble", action="store_true", help="x8 self-ensemble during evaluation")
    p.add_argument("--force", action="store_true", help="re-run even if best.pth already exists")
    p.add_argument("--dry_run", action="store_true", help="print commands without executing")
    p.add_argument("--aggregate_only", action="store_true",
                   help="skip training/eval; just rebuild results_table.{md,csv} from existing eval.json")
    args = p.parse_args()

    unknown = [x for x in args.presets if x not in available_presets()]
    if unknown:
        p.error(f"unknown preset(s) {unknown}; available: {available_presets()}")

    tag = f"x{args.scale}"
    if args.aggregate_only:
        aggregate(args.out_root, tag)
        return

    print(f"=== ablation sweep | scale x{args.scale} | presets={args.presets} ===\n")

    results = []
    for preset in args.presets:
        save_dir = os.path.join(args.out_root, tag, preset)
        best = os.path.join(save_dir, "checkpoints", "best.pth")
        if os.path.isfile(best) and not args.force and not args.resume:
            print(f"[skip] {preset}: {best} already exists (use --force to re-run)\n")
            results.append((preset, "skipped"))
            continue

        train_cmd = build_train_cmd(preset, args, save_dir)
        print(f"[train] {preset}\n  {' '.join(train_cmd)}")
        if not args.dry_run:
            os.makedirs(save_dir, exist_ok=True)
            rc = subprocess.run(train_cmd).returncode
            if rc != 0:
                print(f"[error] training '{preset}' exited with code {rc}; continuing with next preset\n")
                results.append((preset, f"train_failed({rc})"))
                continue

        if args.evaluate:
            eval_cmd = build_eval_cmd(preset, args, save_dir)
            print(f"[eval]  {preset}\n  {' '.join(eval_cmd)}")
            if not args.dry_run:
                subprocess.run(eval_cmd)
        results.append((preset, "done" if not args.dry_run else "planned"))
        print()

    print("=== sweep summary ===")
    for preset, status in results:
        print(f"  {preset:18s} {status}")

    # aggregate per-preset eval.json into a combined CSV + Markdown table
    if args.evaluate and not args.dry_run:
        print()
        aggregate(args.out_root, tag)


if __name__ == "__main__":
    main()
