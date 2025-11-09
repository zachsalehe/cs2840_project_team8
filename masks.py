import math
import torch
import numpy as np
import matplotlib.pyplot as plt


def center_crop_masks(height: int,
                      width: int,
                      mask_area: float,
                      batch_size: int = 1,
                      device: torch.device | str = "cpu") -> torch.Tensor:
    """
    Create (B, H, W) boolean masks with a centered rectangular crop.

    The crop is centered, uses the same aspect ratio as the image, and its
    area equals (approximately exactly) `mask_area * H * W`.

    Args:
        height, width: output size.
        mask_area: target masked fraction in [0, 1] (area of True values).
        batch_size: number of masks.
        device: torch device.

    Returns:
        Bool tensor of shape (B, H, W) on `device`.
    """
    assert height > 0 and width > 0
    assert batch_size >= 1
    mask_area = float(min(max(mask_area, 0.0), 1.0))
    H, W, B = height, width, batch_size

    N = H * W
    k = int(round(mask_area * N))

    if k <= 0:
        return torch.zeros((B, H, W), dtype=torch.bool, device=device)
    if k >= N:
        return torch.ones((B, H, W), dtype=torch.bool, device=device)

    # Start from same-aspect-ratio crop: h≈sqrt(a)*H, w≈sqrt(a)*W
    s = math.sqrt(mask_area)
    h = max(1, min(H, int(round(s * H))))
    w = max(1, min(W, int(round(s * W))))

    # Adjust (h, w) so that h*w is as close as possible to k.
    # Greedy expand/shrink while staying inside [1..H/W].
    area = h * w
    # Prefer adjusting the dimension that keeps aspect ratio closest to H:W
    def step(h, w, inc):
        # inc = +1 to increase area, -1 to decrease
        cand = []
        if 1 <= h + inc <= H:
            cand.append((abs((h + inc) / w - H / W), h + inc, w))
        if 1 <= w + inc <= W:
            cand.append((abs(h / (w + inc) - H / W), h, w + inc))
        if not cand:
            return h, w  # no move possible
        cand.sort(key=lambda t: t[0])
        _, nh, nw = cand[0]
        return nh, nw

    max_iters = H + W + 4
    it = 0
    while area != k and it < max_iters:
        if area < k:
            h, w = step(h, w, +1)
        else:
            h, w = step(h, w, -1)
        area = h * w
        it += 1
    # Center the rectangle
    top = (H - h) // 2
    left = (W - w) // 2
    bottom = top + h
    right = left + w

    mask1 = torch.zeros((H, W), dtype=torch.bool, device=device)
    mask1[top:bottom, left:right] = True
    # Repeat for batch
    return mask1.unsqueeze(0).expand(B, -1, -1)


def perlin_masks(height: int,
                 width: int,
                 scale: float,
                 mask_area: float,
                 batch_size: int = 1,
                 device: torch.device | str = "cpu",
                 seed: int | None = None) -> torch.Tensor:
    """
    Generate B binary 2D masks using Perlin noise (batched).

    Args:
        height, width: mask size.
        scale: frequency of the noise. 1–4 => big blobs, 8–64 => fine texture.
               Interpreted as number of noise cells across the shorter image side.
        mask_area: target masked fraction in [0, 1] for each mask (area of True values).
        batch_size: number of masks to generate in parallel.
        device: torch device.
        seed: optional RNG seed for reproducibility.

    Returns:
        Bool tensor with shape (B, H, W) on `device`.
    """
    assert height > 0 and width > 0
    assert batch_size >= 1
    mask_area = float(min(max(mask_area, 0.0), 1.0))
    scale = max(float(scale), 1.0)

    # RNG (sample parameters on CPU for deterministic seeding, then move to device)
    if seed is not None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        rand = lambda *s: torch.rand(*s, generator=g)
    else:
        rand = torch.rand

    H, W, B = height, width, batch_size
    s_min = min(H, W)

    # Grid resolution (cells) proportional to aspect ratio
    nx = max(1, round((W / s_min) * scale))
    ny = max(1, round((H / s_min) * scale))

    # ----- Build Perlin gradient lattice per batch -----
    # Random gradient unit vectors at lattice points: shape (B, ny+1, nx+1, 2)
    theta = 2 * math.pi * rand(B, ny + 1, nx + 1)
    grads = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1).to(device)

    # Shared pixel grid in [0, nx] × [0, ny]
    ys = torch.linspace(0, ny, H, device=device, dtype=torch.float32)
    xs = torch.linspace(0, nx, W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

    x0 = torch.floor(xx).clamp(max=nx - 1).long()
    y0 = torch.floor(yy).clamp(max=ny - 1).long()
    x1 = (x0 + 1).clamp(max=nx)
    y1 = (y0 + 1).clamp(max=ny)

    dx = (xx - x0.float()).unsqueeze(0)  # (1, H, W)
    dy = (yy - y0.float()).unsqueeze(0)

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    u = fade(dx)  # (1, H, W)
    v = fade(dy)

    # For indexing grads with broadcasted pixel coords, expand to (B, H, W)
    def _g(ix, iy):
        ix = ix.expand(B, H, W)
        iy = iy.expand(B, H, W)
        return grads[torch.arange(B, device=device)[:, None, None], iy, ix]  # (B,H,W,2)

    n00 = (_g(x0, y0) * torch.stack([dx, dy], dim=-1).expand(B, H, W, 2)).sum(-1)
    n10 = (_g(x1, y0) * torch.stack([dx - 1, dy], dim=-1).expand(B, H, W, 2)).sum(-1)
    n01 = (_g(x0, y1) * torch.stack([dx, dy - 1], dim=-1).expand(B, H, W, 2)).sum(-1)
    n11 = (_g(x1, y1) * torch.stack([dx - 1, dy - 1], dim=-1).expand(B, H, W, 2)).sum(-1)

    nx0 = torch.lerp(n00, n10, u)  # (B,H,W)
    nx1 = torch.lerp(n01, n11, u)
    noise = torch.lerp(nx0, nx1, v)  # (B,H,W), ~[-1,1]

    # Per-sample normalization to [0,1] for stable quantiles
    noise_flat = noise.view(B, -1)
    nmin = noise_flat.min(dim=1, keepdim=True).values
    nmax = noise_flat.max(dim=1, keepdim=True).values
    noise_flat = (noise_flat - nmin) / (nmax - nmin + 1e-8)
    noise = noise_flat.view(B, H, W)

    # ----- Threshold each sample to hit target area -----
    N = H * W
    k = int(round(mask_area * N))
    if k <= 0:
        return torch.zeros((B, H, W), dtype=torch.bool, device=device)
    if k >= N:
        return torch.ones((B, H, W), dtype=torch.bool, device=device)

    # For k-th largest across each row: use k_small = N - k + 1 (k-th smallest)
    k_small = N - k + 1
    thresh = torch.kthvalue(noise_flat, k_small, dim=1).values  # (B,)
    mask = (noise >= thresh.view(B, 1, 1))

    # Adjust ties so each sample matches k exactly (rare but possible)
    # This tiny loop is over batch only; pixel ops remain vectorized.
    diffs = mask.view(B, -1).sum(dim=1) - k
    for b in range(B):
        d = int(diffs[b].item())
        if d == 0:
            continue
        tied = (noise[b].reshape(-1) == thresh[b]).nonzero(as_tuple=False).squeeze(1)
        if d > 0:
            mask.view(B, -1)[b, tied[:d]] = False
        else:
            mask.view(B, -1)[b, tied[: -d]] = True

    return mask


