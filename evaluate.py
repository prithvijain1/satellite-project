"""Evaluation entry point: reports PSNR / SSIM / LPIPS on a test split.

Example:
    python evaluate.py --config configs/default.yaml \
        --ckpt runs/ucmerced_seed42/generator_best.pth \
        --test-split splits/ucmerced_test.txt
"""
import argparse
import torch

from src.utils import load_config
from src.models import Generator
from src.dataset import build_loader
from src import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test-split", required=True)
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = Generator(num_blocks=cfg["model"]["rrdb_blocks"],
                     embed_dim=cfg["model"]["embed_dim"],
                     num_heads=cfg["model"]["num_heads"],
                     num_layers=cfg["model"]["num_layers"]).to(device)
    netG.load_state_dict(torch.load(args.ckpt, map_location=device))
    netG.eval()

    loader = build_loader(args.test_split, 1, cfg["data"]["hr_size"], cfg["data"]["scale"],
                          cfg["data"]["degradation"], augment=False, shuffle=False)
    ps, ss, lp = [], [], []
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = netG(lr)
            ps.append(metrics.psnr(sr, hr))
            ss.append(metrics.ssim(sr, hr))
            if not args.no_lpips:
                lp.append(metrics.lpips(sr, hr, device))
    n = len(ps)
    print(f"PSNR : {sum(ps)/n:.3f} dB")
    print(f"SSIM : {sum(ss)/n:.4f}")
    if lp:
        print(f"LPIPS: {sum(lp)/n:.4f}")


if __name__ == "__main__":
    main()
