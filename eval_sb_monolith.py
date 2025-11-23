
"""
eval_sb_monolith_tqdm.py
------------------------
Evaluates all checkpoints in a directory on the same test set/batches, with:
- Per-model tqdm progress bar
- Consistent random example selection across models for saved images
  (same datapoints chosen once; clean/masked saved once; each model saves its recon)
- MNIST/USPS support (test split), digit selection logic (union of untrained digits
  across models; fallback to all digits), SSIM + LPIPS (safe for MNIST)
"""
from __future__ import annotations
import argparse, os, glob, json, random
from typing import Tuple, List, Dict, Set

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
try:
    from torchvision.utils import save_image
except Exception:
    save_image = None

from sb_monolith import UnifiedSB, UnifiedConfig
from mnist_dataset import MNISTDataset

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_digits_arg(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in s.split(',') if x.strip() != '')

# SSIM (global), grayscale in [-1,1]
def ssim_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x01 = (x + 1.0) / 2.0
    y01 = (y + 1.0) / 2.0
    mu_x = x01.mean(dim=[2,3], keepdim=True)
    mu_y = y01.mean(dim=[2,3], keepdim=True)
    sigma_x = ((x01 - mu_x)**2).mean(dim=[2,3], keepdim=True)
    sigma_y = ((y01 - mu_y)**2).mean(dim=[2,3], keepdim=True)
    sigma_xy = ((x01 - mu_x)*(y01 - mu_y)).mean(dim=[2,3], keepdim=True)
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2) + 1e-8)
    return ssim_map.mean()

def try_lpips_device(device: torch.device):
    try:
        import lpips
        net = lpips.LPIPS(net='alex').to(device)
        return net
    except Exception as e:
        print('[warn] lpips not available; skipping LPIPS. Reason:', e)
        return None

def _prep_for_lpips(img: torch.Tensor) -> torch.Tensor:
    # img: (N, C=1 or 3, H, W), values in [-1,1]
    if img.size(1) == 1:
        img = img.repeat(1, 3, 1, 1)
    # Resize to at least 64x64 for alexnet backbone
    if img.size(-1) < 64 or img.size(-2) < 64:
        img = F.interpolate(img, size=(64, 64), mode='bilinear', align_corners=False)
    return img

def compute_lpips(lpips_net, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x_p = _prep_for_lpips(x)
        y_p = _prep_for_lpips(y)
        d = lpips_net(x_p, y_p)
        return d.view(d.size(0), -1).mean()

def load_ckpt_cfg(path: str) -> UnifiedConfig:
    blob = torch.load(path, map_location='cpu')
    cfg_dict = blob.get('cfg', {})
    if not cfg_dict:
        cfg_dict = UnifiedConfig().__dict__
    if 'method' not in cfg_dict:
        m = blob.get('meta', {}).get('method', 'fourterm')
        cfg_dict['method'] = m
    return UnifiedConfig(**cfg_dict)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', type=str, required=True, help='Directory containing model .pt checkpoints')
    ap.add_argument('--dataset', type=str, default='mnist', choices=['mnist','usps'])
    ap.add_argument('--digits', type=str, default=None, help='Optional override: comma list like 0,1,2')
    ap.add_argument('--examples', type=int, default=10, help='Number of examples to save')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--out_dir', type=str, default='./eval_runs')
    ap.add_argument('--seed', type=int, default=0, help='Seed controlling noise and example selection')
    args = ap.parse_args()

    device = torch.device(args.device)
    set_seed(args.seed)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, '*.pt')))
    assert len(ckpts) > 0, f'No .pt files found in {args.ckpt_dir}'
    print(f'[eval] Found {len(ckpts)} checkpoints')

    # Load configs
    cfgs = [load_ckpt_cfg(p) for p in ckpts]

    # Digit set
    if args.digits is not None:
        eval_digits = parse_digits_arg(args.digits)
    else:
        all_digits = set(range(10))
        union_complements = set()
        for c in cfgs:
            trained = set(c.digits if not isinstance(c.digits, range) else list(c.digits))
            comp = all_digits - trained
            union_complements |= comp
        eval_digits = tuple(sorted(union_complements if union_complements else all_digits))
    print('[eval] digits:', eval_digits)

    # Mask params & sigma from first ckpt
    ref_cfg = cfgs[0]
    mask_type = ref_cfg.mask_type
    mask_area = ref_cfg.mask_area
    mask_scale = ref_cfg.mask_scale
    usps = (args.dataset.lower() == 'usps')
    sigma_for_noise = getattr(ref_cfg, 'sigma', 0.5)
    print(f'[eval] Using mask params from first ckpt: type={mask_type}, area={mask_area}, scale={mask_scale}; noise sigma={sigma_for_noise}')

    # Dataset & loader
    ds = MNISTDataset(
        split='test',
        digits=eval_digits,
        mask_type=mask_type,
        mask_area=mask_area,
        mask_scale=mask_scale,
        usps=usps,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Cache batches with fixed noise; also record global positions for each item
    cached_batches = []  # list of tuples: (x, mask, y_noisy, y_white, start_pos)
    torch.manual_seed(args.seed)
    total_count = 0
    for xb, maskb in dl:
        b = xb.size(0)
        xb = xb.to(device)
        maskb = maskb.to(device)
        noise = torch.randn_like(xb) * sigma_for_noise
        mb = maskb.float()
        y_noisy = xb * (1 - mb) + noise * mb
        y_white = xb * (1 - mb) + (1.0) * mb
        cached_batches.append((xb, maskb, y_noisy, y_white, total_count))
        total_count += b
    print(f'[eval] Cached {len(cached_batches)} batches (total {total_count} images)')

    # Choose random example positions once (global indices 0..total_count-1)
    rng = random.Random(args.seed)
    select_k = min(args.examples, total_count)
    example_positions = sorted(rng.sample(range(total_count), select_k))
    example_positions_set: Set[int] = set(example_positions)
    print(f'[eval] Selected example indices (global): {example_positions[:min(10, len(example_positions))]}{" ..." if len(example_positions)>10 else ""}')

    # Output dirs
    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, 'metrics.txt')

    # Track whether we've saved clean/masked for each selected example (only once globally)
    saved_clean_masked: Set[int] = set()

    # LPIPS (optional)
    lpips_net = try_lpips_device(device)

    # Evaluate
    results = []
    for ckpt_path, cfg in zip(ckpts, cfgs):
        tag = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"\n[model] {tag}  method={cfg.method}")
        model = UnifiedSB.load_ckpt(ckpt_path, map_location=device)
        model.cfg.device = str(device)
        model.to(device)
        model.eval()

        ssim_sum, lpips_sum, nimgs = 0.0, 0.0, 0

        bar = tqdm(enumerate(cached_batches), total=len(cached_batches), desc=tag, leave=True)
        for batch_idx, (x, mask, y_noisy, y_white, start_pos) in bar:
            with torch.no_grad():
                x_rec = model.sample_inpaint(y_noisy.to(device), mask.to(device))
            # Metrics
            ssim_val = float(ssim_batch(x_rec, x).item())
            ssim_sum += ssim_val * x.size(0)
            if lpips_net is not None:
                lp = float(compute_lpips(lpips_net, x_rec, x).item())
                lpips_sum += lp * x.size(0)
            nimgs += x.size(0)
            bar.set_postfix({"SSIM": f"{(ssim_sum/max(1,nimgs)):.4f}"})

            # Save examples for this model (but clean/masked saved only once globally)
            if save_image is not None and example_positions_set:
                for local_idx in range(x.size(0)):
                    global_idx = start_pos + local_idx
                    if global_idx in example_positions_set:
                        base = os.path.join(img_dir, f"ex_{global_idx:06d}")
                        if global_idx not in saved_clean_masked:
                            save_image(x[local_idx], base + "_clean.png", normalize=True, value_range=(-1,1))
                            save_image(y_white[local_idx], base + "_masked.png", normalize=True, value_range=(-1,1))
                            saved_clean_masked.add(global_idx)
                        save_image(x_rec[local_idx], base + f"_{tag}_recon.png", normalize=True, value_range=(-1,1))

        ssim_avg = ssim_sum / max(1, nimgs)
        lpips_avg = (lpips_sum / max(1, nimgs)) if lpips_net is not None else None
        print(f"[scores] {tag}: SSIM={ssim_avg:.4f}  LPIPS={'{:.4f}'.format(lpips_avg) if lpips_avg is not None else 'N/A'}")
        results.append({
            "ckpt": os.path.basename(ckpt_path),
            "method": cfg.method,
            "digits_train": list(cfg.digits if not isinstance(cfg.digits, range) else list(cfg.digits)),
            "digits_eval": list(eval_digits),
            "ssim": ssim_avg,
            "lpips": lpips_avg,
        })

    with open(metrics_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved metrics -> {metrics_path}")

if __name__ == '__main__':
    main()
