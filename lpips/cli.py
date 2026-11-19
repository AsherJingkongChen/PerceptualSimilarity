"""The lpips command: compare the first image against each of the others, the way dssim does.

Usage: lpips [-n NET] [-d DEVICE] original.png modified.png [modified.png...]
   or: lpips [-n NET] [-d DEVICE] -o difference.png original.png modified.png

Prints one line per modified image, in order: its LPIPS distance to the original
(0 = identical), a tab, its path. Images must have the same size; an alpha channel
is dropped. Images are read as 8-bit sRGB, the way LPIPS was calibrated.
"""

import argparse
import sys

import torch
from torchvision.io import ImageReadMode, decode_image, read_file, write_png

import lpips


def read(path: str, device: torch.device) -> torch.Tensor:
    """An image as a 1×3×H×W tensor in [-1, 1]."""
    img = decode_image(read_file(path), mode=ImageReadMode.RGB)
    return (img.to(device, torch.float32) / 127.5 - 1.0)[None]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="lpips",
        description="Compares the first image against each of the others and prints "
        "their LPIPS distances (0 = identical).",
    )
    p.add_argument("original")
    p.add_argument("modified", nargs="+")
    p.add_argument("-n", "--net", choices=["alex", "vgg", "squeeze"], default="alex",
                   help="the trunk network (default: alex)")
    p.add_argument("-d", "--device", type=torch.device, default=torch.get_default_device(),
                   help="a torch device, such as cpu or cuda:0 (default: %(default)s)")
    p.add_argument("-o", "--output", metavar="difference.png",
                   help="write the per-pixel distance as a grayscale PNG, 1.0 and above "
                   "white; only with one modified image")
    a = p.parse_args(argv)
    if a.output and len(a.modified) != 1:
        p.error("-o takes exactly one modified image")

    metric = lpips.LPIPS(net=a.net, verbose=False).to(a.device)
    with torch.inference_mode():
        x = read(a.original, a.device)
        for path in a.modified:
            y = read(path, a.device)
            if y.shape != x.shape:
                print(f"lpips: {path} is {y.shape[-1]}×{y.shape[-2]}, "
                      f"{a.original} is {x.shape[-1]}×{x.shape[-2]}", file=sys.stderr)
                return 1
            print(f"{metric(x, y).item():.8f}\t{path}")
            if a.output:
                metric.spatial = True  # the same layers, upsampled instead of averaged
                m = (metric(x, y)[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
                metric.spatial = False
                write_png(m, a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
