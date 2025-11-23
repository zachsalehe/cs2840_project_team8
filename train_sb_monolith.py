from __future__ import annotations
import argparse
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader, Dataset

from sb_monolith import UnifiedSB, UnifiedConfig
from mnist_dataset import MNISTDataset

def default_cfg_for_method(method: str) -> UnifiedConfig:
    ds_defaults = dict(
        digits=tuple(range(10)),
        mask_type="perlin",
        mask_area=0.2,
        mask_scale=1.0,
        usps=False,
    )
    if method == "i2sb":
        return UnifiedConfig(
            method="i2sb",
            sigma=0.5, n_steps=20,
            ref_mode="zero", ref_lam=1.0,
            eps_sink=0.08, tau_sink=0.8, sink_iters=80,
            lambda_X=5.0, lambda_Y=5.0, lambda_cyc=0.0,
            lr_phi=2e-4, lr_theta=2e-4,
            device="mps", log_dir="./runs_i2sb_mnist",
            **ds_defaults,
        )
    if method == "fourterm":
        return UnifiedConfig(
            method="fourterm",
            sigma=0.5, n_steps=20,
            ref_mode="zero", ref_lam=1.0,
            eps_sink=0.08, tau_sink=0.8, sink_iters=80,
            lambda_X=5.0, lambda_Y=5.0, lambda_cyc=1.0,
            lr_phi=2e-4, lr_theta=2e-4,
            device="mps", log_dir="./runs_fourterm_mnist",
            **ds_defaults,
        )
    if method == "unsb":
        return UnifiedConfig(
            method="unsb",
            sigma=0.5, n_steps=30,
            ref_mode="ou", ref_lam=1.0,
            eps_sink=0.35, tau_sink=0.8, sink_iters=25,
            lambda_X=10.0, lambda_Y=10.0, lambda_cyc=0.5,
            lr_phi=2e-4, lr_theta=2e-4,
            device="mps", log_dir="./runs_unsb_mnist",
            **ds_defaults,
        )
    if method == "i3sb":
        return UnifiedConfig(
            method="i3sb",
            sigma=0.5, n_steps=20,
            ref_mode="zero", ref_lam=1.0,
            lambda_forward=1.0, lambda_backward=1.0,
            lr_phi=2e-4, lr_theta=2e-4,
            device="mps", log_dir="./runs_i3sb_mnist",
            **ds_defaults,
        )
    raise ValueError(f"Unknown method: {method}")

class InpaintWrapper(Dataset):
    """Wraps MNISTDataset (img,mask) -> dict(x,y,mask), with y = x outside + noise inside mask."""
    def __init__(self, base: MNISTDataset, sigma: float = 0.5):
        self.base = base
        self.sigma = sigma
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        x, mask = self.base[idx]  # x in [-1,1]
        noise = torch.randn_like(x) * self.sigma
        m = mask.float()
        y = x * (1.0 - m) + noise * m
        return {"x": x, "y": y, "mask": mask}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=str, default="fourterm", choices=["i2sb","fourterm","unsb","i3sb"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--vis_every", type=int, default=100)

    # Dataset overrides (optional)
    ap.add_argument("--digits", type=str, default=None, help="Comma list, e.g. 0,1,2; default is 0-9")
    ap.add_argument("--mask_type", type=str, default=None, choices=["perlin","center"])
    ap.add_argument("--mask_area", type=float, default=None)
    ap.add_argument("--mask_scale", type=float, default=None)
    ap.add_argument("--usps", action="store_true")

    args = ap.parse_args()

    cfg = default_cfg_for_method(args.method)

    # Optional dataset overrides
    if args.digits is not None:
        cfg.digits = tuple(int(d) for d in args.digits.split(",") if d.strip() != "")
    if args.mask_type is not None:
        cfg.mask_type = args.mask_type
    if args.mask_area is not None:
        cfg.mask_area = args.mask_area
    if args.mask_scale is not None:
        cfg.mask_scale = args.mask_scale
    if args.usps:
        cfg.usps = True

    model = UnifiedSB(cfg)

    train_ds = MNISTDataset(
        split="train",
        digits=cfg.digits,
        mask_type=cfg.mask_type,
        mask_area=cfg.mask_area,
        mask_scale=cfg.mask_scale,
        usps=cfg.usps,
    )
    wrapped = InpaintWrapper(train_ds, sigma=cfg.sigma)

    bs = args.batch_size
    dl = DataLoader(wrapped, batch_size=bs, shuffle=True,
                    num_workers=args.num_workers, pin_memory=args.pin_memory)

    print("UnifiedConfig:\n", asdict(cfg))
    print(f"\n[train] method={cfg.method} epochs={args.epochs} batch_size={bs}")

    model.fit(dl, epochs=args.epochs, log_every=args.log_every, vis_every=args.vis_every)

    ckpt_path = f"{cfg.log_dir}/last_{cfg.method}.pt"
    model.save_ckpt(ckpt_path)
    print(f"Saved checkpoint -> {ckpt_path}")
