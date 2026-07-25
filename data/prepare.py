"""One-time DIV2K download + sub-image preparation (the BasicSR convention).

Training reads one 64x64 LR crop per sample. Reading it from a full 2040x1356 PNG means decoding
~2.8M pixels to keep 4k — and with `iters_per_epoch x batch_size` samples per epoch that decode cost
dominates total wall-clock. Pre-cutting the dataset into small sub-images removes it.

Writes:
    DIV2K/DIV2K_train_HR_sub/*.png                 (HR sub-images, `--size` px)
    DIV2K/DIV2K_train_LR_bicubic/X{s}_sub/*.png    (matching LR sub-images, `--size`/s px)

Usage:
    py -3 -m data.prepare --download            # fetch DIV2K x4 (~4.5 GB) then cut sub-images
    py -3 -m data.prepare                       # cut sub-images from an existing DIV2K/
    py -3 -m data.prepare --scale 2 3 4         # all scales
"""
import os
import glob
import zipfile
import argparse
import urllib.request
from PIL import Image

DIV2K_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K"


def download(data_root: str, scales):
    """fetch the DIV2K zips we need (skips anything already present) and unpack them."""
    names = ["DIV2K_train_HR", "DIV2K_valid_HR"]
    for s in scales:
        names += [f"DIV2K_train_LR_bicubic_X{s}", f"DIV2K_valid_LR_bicubic_X{s}"]

    os.makedirs(data_root, exist_ok=True)
    for name in names:
        # LR zips unpack into DIV2K_{split}_LR_bicubic/X{s}
        marker = name.replace("_LR_bicubic_X", "_LR_bicubic/X") if "_LR_bicubic_X" in name else name
        if os.path.isdir(os.path.join(data_root, marker)):
            print(f"  [have] {marker}")
            continue
        url, zpath = f"{DIV2K_URL}/{name}.zip", os.path.join(data_root, f"{name}.zip")
        if not os.path.isfile(zpath):
            print(f"  [get ] {url}")
            try:
                urllib.request.urlretrieve(url, zpath)
            except Exception as e:
                print(f"  [fail] {name}: {e}\n         download it manually from {DIV2K_URL}/")
                continue
        print(f"  [unzip] {name}.zip")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(data_root)
        os.remove(zpath)


def crop_one(hr_path, lr_path, hr_out, lr_out, scale, size, step):
    """cut one HR/LR pair into aligned sub-images. Returns the number written."""
    hr = Image.open(hr_path).convert("RGB")
    lr = Image.open(lr_path).convert("RGB")
    name = os.path.splitext(os.path.basename(hr_path))[0]
    lr_size, lr_step = size // scale, step // scale

    # LR drives the grid so every HR crop is exactly scale x its LR counterpart
    lw, lh = lr.size
    n = 0
    ys = list(range(0, max(lh - lr_size, 0) + 1, lr_step))
    xs = list(range(0, max(lw - lr_size, 0) + 1, lr_step))
    if ys and ys[-1] + lr_size < lh:
        ys.append(lh - lr_size)
    if xs and xs[-1] + lr_size < lw:
        xs.append(lw - lr_size)
    for y in ys:
        for x in xs:
            lr_c = lr.crop((x, y, x + lr_size, y + lr_size))
            hr_c = hr.crop((x * scale, y * scale, (x + lr_size) * scale, (y + lr_size) * scale))
            if lr_c.size != (lr_size, lr_size) or hr_c.size != (size, size):
                continue
            tag = f"{name}_s{n:03d}.png"
            hr_c.save(os.path.join(hr_out, tag))
            lr_c.save(os.path.join(lr_out, tag))
            n += 1
    return n


def main():
    p = argparse.ArgumentParser(description="Cut DIV2K into training sub-images")
    p.add_argument("--data_root", type=str, default="DIV2K")
    p.add_argument("--scale", type=int, nargs="+", default=[4])
    p.add_argument("--size", type=int, default=480, help="HR sub-image size (must be divisible by every scale)")
    p.add_argument("--step", type=int, default=240, help="HR stride between sub-images")
    p.add_argument("--download", action="store_true", help="download DIV2K first (~4.5 GB for one scale)")
    args = p.parse_args()

    if args.download:
        print(f"downloading DIV2K → {args.data_root}")
        download(args.data_root, args.scale)
        print()

    hr_dir = os.path.join(args.data_root, "DIV2K_train_HR")
    hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
    if not hr_paths:
        raise SystemExit(f"no HR images in {hr_dir} — check --data_root")

    for scale in args.scale:
        if args.size % scale or args.step % scale:
            raise SystemExit(f"--size/--step must be divisible by scale {scale}")
        lr_dir = os.path.join(args.data_root, "DIV2K_train_LR_bicubic", f"X{scale}")
        hr_out = os.path.join(args.data_root, "DIV2K_train_HR_sub")
        lr_out = os.path.join(args.data_root, "DIV2K_train_LR_bicubic", f"X{scale}_sub")
        os.makedirs(hr_out, exist_ok=True)
        os.makedirs(lr_out, exist_ok=True)

        total = 0
        for i, hr_path in enumerate(hr_paths):
            img_id = os.path.splitext(os.path.basename(hr_path))[0]
            lr_path = os.path.join(lr_dir, f"{img_id}x{scale}.png")
            if not os.path.isfile(lr_path):
                print(f"  [skip] missing LR for {img_id}")
                continue
            total += crop_one(hr_path, lr_path, hr_out, lr_out, scale, args.size, args.step)
            if (i + 1) % 100 == 0:
                print(f"  x{scale}: {i+1}/{len(hr_paths)} images → {total} sub-images")
        print(f"x{scale}: wrote {total} sub-images → {hr_out} / {lr_out}")


if __name__ == "__main__":
    main()
