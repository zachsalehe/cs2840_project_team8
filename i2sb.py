# i2sb.py
import math
import os
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

# -------------------------
# Utilities
# -------------------------
EPS = 1e-8


def make_time_plane(b: int, h: int, w: int, t_scalar: float, device: torch.device):
    return torch.full((b, 1, h, w), float(t_scalar), device=device)


def pairwise_mean_sq_cost(x: torch.Tensor, y: torch.Tensor):
    """
    x: [B, C, H, W], y: [B, C, H, W]  (empirical batches)
    Returns C matrix [B, B] with mean squared pixelwise distances (range ~ [0,4]).
    """
    B = x.size(0)
    xf = x.view(B, -1)
    yf = y.view(B, -1)
    # torch.cdist returns Euclidean distance; square and mean over dim
    d2 = torch.cdist(xf, yf, p=2) ** 2  # [B,B]
    d2 = d2 / xf.size(1)                # mean over pixels -> scale ~[0,4]
    return d2.clamp_min(0.0)


# -------------------------
# Balanced Sinkhorn (entropic OT) + debiased Sinkhorn divergence
# -------------------------
def balanced_sinkhorn_ot(a: torch.Tensor,
                         b: torch.Tensor,
                         C: torch.Tensor,
                         eps: float,
                         n_iters: int = 50) -> Dict[str, torch.Tensor]:
    """
    Balanced entropic OT primal:
      OT_eps(a,b) = <C, gamma> + eps * KL(gamma || a ⊗ b)
    using stable multiplicative Sinkhorn iterations in the normal domain.
    Returns dict with:
      - 'ot': primal value (differentiable w.r.t. C)
      - 'gamma': transport plan (detached for inspection)
    Shapes:
      a,b: [B] with a.sum()=b.sum()=1 (empirical masses)
      C: [B,B] mean-sq ground cost (nonnegative)
    """
    device = C.device
    B = C.size(0)
    assert C.size(0) == C.size(1) == a.numel() == b.numel()

    # Kernel
    K = torch.exp(-C / eps).clamp_min(1e-38)   # avoid zeros
    u = torch.ones(B, device=device)
    v = torch.ones(B, device=device)

    for _ in range(n_iters):
        Kv = K @ v
        Kv = Kv.clamp_min(1e-30)
        u = (a / Kv).clamp_min(1e-30)

        Ktu = K.t() @ u
        Ktu = Ktu.clamp_min(1e-30)
        v = (b / Ktu).clamp_min(1e-30)

    gamma = (u[:, None] * K) * v[None, :]      # [B,B]
    gamma_det = gamma.detach()

    # Primal terms
    cost_term = (gamma * C).sum()

    # KL(gamma || a ⊗ b)
    ab = torch.outer(a, b).clamp_min(1e-30)
    kl_g_ab = (gamma * (torch.log(gamma.clamp_min(1e-30)) - torch.log(ab))).sum() - gamma.sum() + ab.sum()

    ot_val = cost_term + eps * kl_g_ab
    return {"ot": ot_val, "gamma": gamma_det}


def sinkhorn_divergence(xA: torch.Tensor,
                        xB: torch.Tensor,
                        eps: float,
                        n_iters: int = 50) -> torch.Tensor:
    """
    Debiased balanced Sinkhorn divergence:
       S_eps(A,B) = OT_eps(A,B) - 0.5 OT_eps(A,A) - 0.5 OT_eps(B,B)
    xA, xB: [B, C, H, W] samples (grad flows into xA and xB)
    """
    B = xA.size(0)
    a = torch.full((B,), 1.0 / B, device=xA.device)
    b = torch.full((B,), 1.0 / B, device=xB.device)

    C_AB = pairwise_mean_sq_cost(xA, xB)
    C_AA = pairwise_mean_sq_cost(xA, xA)
    C_BB = pairwise_mean_sq_cost(xB, xB)

    ot_AB = balanced_sinkhorn_ot(a, b, C_AB, eps, n_iters)["ot"]
    ot_AA = balanced_sinkhorn_ot(a, a, C_AA, eps, n_iters)["ot"]
    ot_BB = balanced_sinkhorn_ot(b, b, C_BB, eps, n_iters)["ot"]

    return ot_AB - 0.5 * (ot_AA + ot_BB)


# -------------------------
# Reference drift g(x,t)
# -------------------------
class ReferenceDrift(nn.Module):
    """
    g(x,t): reference drift. Options:
      - 'zero': g(x,t) = 0
      - 'ou':   g(x,t) = -lambda * x   (Ornstein–Uhlenbeck-like anchor)
    """
    def __init__(self, mode: str = "zero", lam: float = 1.0):
        super().__init__()
        self.mode = mode
        self.lam = lam

    def forward(self, x: torch.Tensor, t_plane: torch.Tensor) -> torch.Tensor:
        if self.mode == "zero":
            return torch.zeros_like(x)
        elif self.mode == "ou":
            return -self.lam * x
        else:
            raise ValueError("ReferenceDrift mode must be 'zero' or 'ou'.")


# -------------------------
# Lightweight U-Net drift (time + condition)
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch): super().__init__(); self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, 4, 2, 1), nn.GroupNorm(8, out_ch), nn.SiLU(True))
    def forward(self, x): return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch): super().__init__(); self.net = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1), nn.GroupNorm(8, out_ch), nn.SiLU(True))
    def forward(self, x): return self.net(x)


class DriftUNet(nn.Module):
    """
    Predicts drift f(x_t, t; cond) with inputs:
      - current state x_t (1 ch)
      - conditioning image cond (1 ch)  (y for forward, x for backward)
      - mask (1 ch)                     (optional, here we include it)
      - time plane t (1 ch)
    Output: drift (1 ch), range unconstrained (we do not tanh here)
    """
    def __init__(self, in_ch=4, base=64, out_ch=1):
        super().__init__()
        self.inc  = DoubleConv(in_ch, base)
        self.d1   = Down(base, base*2)
        self.d2   = Down(base*2, base*4)
        self.bott = DoubleConv(base*4, base*4)
        self.u1   = Up(base*4, base*2)
        self.dc1  = DoubleConv(base*4, base*2)
        self.u2   = Up(base*2, base)
        self.dc2  = DoubleConv(base*2, base)
        self.outc = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x_t, cond, mask, t_plane):
        x = torch.cat([x_t, cond, mask, t_plane], dim=1)
        x0 = self.inc(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        xb = self.bott(x2)
        xu1 = self.u1(xb)
        xcat1 = torch.cat([xu1, x1], dim=1)
        xdc1 = self.dc1(xcat1)
        xu2 = self.u2(xdc1)
        xcat2 = torch.cat([xu2, x0], dim=1)
        xdc2 = self.dc2(xcat2)
        return self.outc(xdc2)


# -------------------------
# Config
# -------------------------
@dataclass
class FourTermConfig:
    # SDE / discretization
    sigma: float = 0.1
    n_steps: int = 10     # EM steps per bridge (>= 5 recommended)
    # SB energy
    ref_mode: str = "zero"  # 'zero' or 'ou'
    ref_lam: float = 1.0
    # Sinkhorn (balanced)
    eps_sink: float = 0.1   # entropic reg (on mean-sq cost ~ [0,4])
    sink_iters: int = 50
    # Cycle loss (unused in I2SB)
    lambda_cyc: float = 0.0
    # Endpoint weights
    lambda_X: float = 10.0
    lambda_Y: float = 10.0
    # Optim
    lr_phi: float = 2e-4
    lr_theta: float = 2e-4
    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = "./runs_fourterm"


# -------------------------
# Trainer implementing I2SB (SB + balanced endpoint OT)
# -------------------------
class FourTermSBTrainer:
    """
    Implements the I2SB-style objective:
      L(phi,theta) = L_SB + lambda_X S_eps(hat p_phi^{(1)}, p_X)
                           + lambda_Y S_eps(hat p_theta^{(0)}, p_Y)
    where S_eps is the debiased balanced Sinkhorn divergence (entropic OT).
    """
    def __init__(self, img_shape: Tuple[int, int], cfg: FourTermConfig):
        h, w = img_shape
        self.H, self.W = h, w
        self.cfg = cfg

        # Drifts
        self.f_phi   = DriftUNet(in_ch=4, out_ch=1).to(cfg.device)   # forward (cond on y)
        self.f_theta = DriftUNet(in_ch=4, out_ch=1).to(cfg.device)   # backward (cond on x)
        self.g_ref   = ReferenceDrift(mode=cfg.ref_mode, lam=cfg.ref_lam).to(cfg.device)

        self.opt_phi = torch.optim.Adam(self.f_phi.parameters(), lr=cfg.lr_phi, betas=(0.5, 0.999))
        self.opt_th  = torch.optim.Adam(self.f_theta.parameters(), lr=cfg.lr_theta, betas=(0.5, 0.999))

        os.makedirs(cfg.log_dir, exist_ok=True)

    # ---------- SDE simulators ----------
    def _em_forward(self, y: torch.Tensor, mask: torch.Tensor):
        """Simulate X^f from t=0..1. Returns (X1_f, path_energy_f)."""
        B = y.size(0); dt = 1.0 / self.cfg.n_steps
        x = y.clone()
        energy = 0.0
        for k in range(self.cfg.n_steps):
            t = k / self.cfg.n_steps
            t_plane = make_time_plane(B, self.H, self.W, t, y.device)
            drift = self.f_phi(x, y, mask, t_plane)
            ref   = self.g_ref(x, t_plane)
            # Energy integrand: mean over pixels per sample, then mean over batch
            diff = drift - ref
            energy += diff.pow(2).mean() * dt
            # Euler–Maruyama step with masking semantics: update only masked pixels
            noise = torch.randn_like(x) * (self.cfg.sigma * math.sqrt(dt))
            new_x = x + drift * dt + noise
            m = mask.bool()                      # ensure boolean semantics once
            x = torch.where(m, new_x, y)         # masked -> new_x, known -> y
        return x, energy  # x is X_1^f

    def _em_backward(self, x_clean: torch.Tensor, mask: torch.Tensor):
        """
        Simulate X^b with terminal condition X_1^b = x_clean using reversed clock.
        Returns (X0_b, path_energy_b).
        """
        B = x_clean.size(0); dt = 1.0 / self.cfg.n_steps
        y = x_clean.clone()  # this 'y' is Y_s in the comment above
        energy = 0.0
        for k in range(self.cfg.n_steps):
            s = k / self.cfg.n_steps
            t = 1.0 - s
            t_plane = make_time_plane(B, self.H, self.W, t, x_clean.device)
            drift_t = self.f_theta(y, x_clean, mask, t_plane)  # f_theta(Y_s, t=1-s; x)
            ref     = self.g_ref(y, t_plane)
            # Energy integrand in ds: ||f_theta(Y_s,1-s) - g(Y_s,1-s)||^2
            diff = drift_t - ref
            energy += diff.pow(2).mean() * dt
            noise = torch.randn_like(y) * (self.cfg.sigma * math.sqrt(dt))
            new_y = y + (-drift_t) * dt + noise
            m = mask.bool()
            y = torch.where(m, new_y, x_clean)
        X0_b = y
        return X0_b, energy

    # ---------- Round-trip maps with fresh noise (kept for diagnostics) ----------
    @torch.no_grad()
    def roundtrip_y(self, y: torch.Tensor, mask: torch.Tensor):
        """tilde y(y) = X_0^b( X_1^f(y) ) with fresh noise in both legs."""
        x1, _ = self._em_forward(y, mask)
        x1 = x1.detach()
        y0, _ = self._em_backward(x1, mask)  # condition on x = X_1^f(y)
        return y0

    @torch.no_grad()
    def roundtrip_x(self, x: torch.Tensor, mask: torch.Tensor):
        """tilde x(x) = X_1^f( X_0^b(x) ) with fresh noise."""
        y0, _ = self._em_backward(x, mask)
        y0 = y0.detach()
        x1, _ = self._em_forward(y0, mask)   # condition on y = X_0^b(x)
        return x1

    # ---------- Public API ----------
    def train_step(self,
                   batch_y_img: torch.Tensor, batch_y_mask: torch.Tensor,
                   batch_x_img: torch.Tensor, batch_x_mask: torch.Tensor,
                   lambda_cyc: Optional[float] = None):
        """
        Inputs:
          - batch_y_img: degraded images y ~ p_Y (we'll create y by masking below)
          - batch_y_mask: corresponding masks for those y (1=hole)
          - batch_x_img: clean images x ~ p_X (independent loader, unpaired)
          - batch_x_mask: masks sampled independently (used only to condition f_theta/f_phi)
        Returns dict with scalar losses and a small preview grid path.
        """
        # lambda_cyc argument is ignored in I2SB (no cycle term)
        # --- Sanitize mask dtypes (bool → float in [0,1]) ---
        my = batch_y_mask
        mx = batch_x_mask
        if my.dtype != batch_y_img.dtype:
            my = my.to(batch_y_img.dtype)
        if mx.dtype != batch_x_img.dtype:
            mx = mx.to(batch_x_img.dtype)
        my = my.clamp(0.0, 1.0)
        mx = mx.clamp(0.0, 1.0)

        # --- Build degraded y from clean images and masks (your p_Y construct) ---
        noise = torch.empty_like(batch_y_img).uniform_(-1.0, 1.0)
        y = batch_y_img * (1.0 - my) + noise * my    # degraded p_Y sample

        x = batch_x_img                                # clean p_X sample

        # --- Simulate both bridges & collect energies (Eq. (2)) ---
        X1_f, energy_f = self._em_forward(y, my)      # terminal forward marginal samples
        X0_b, energy_b = self._em_backward(x, mx)     # initial backward marginal samples

        # L_SB: symmetric path/energy matching (Eq. (3))
        L_sb = (energy_f + energy_b) / (2.0 * (self.cfg.sigma ** 2))

        # --- Endpoint alignment via balanced Sinkhorn divergence (I2SB Terms 2 & 3) ---
        S_forward  = sinkhorn_divergence(X1_f, x.detach(), self.cfg.eps_sink, self.cfg.sink_iters)
        S_backward = sinkhorn_divergence(X0_b, y.detach(), self.cfg.eps_sink, self.cfg.sink_iters)

        # --- Overall objective (I2SB) ---
        loss = L_sb + self.cfg.lambda_X * S_forward + self.cfg.lambda_Y * S_backward

        # --- Optimize both drifts jointly ---
        self.opt_phi.zero_grad(set_to_none=True)
        self.opt_th.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_phi.step()
        self.opt_th.step()

        # Pack metrics
        with torch.no_grad():
            metrics = {
                "loss_total": loss.item(),
                "L_SB": L_sb.item(),
                "S_forward": S_forward.item(),
                "S_backward": S_backward.item(),
            }

        return metrics, {"X1_f": X1_f.detach(), "X0_b": X0_b.detach(), "y": y.detach(), "x": x.detach()}


# -------------------------
# Convenience training loop (MNIST masking)
# -------------------------
def train_four_term_sb(dataset_cls,
                       epochs: int = 3,
                       batch_size: int = 64,
                       out_dir: str = "./runs_fourterm",
                       cfg: "FourTermConfig" = None,
                       *,
                       # I/O & logging
                       log_every: int = 100,
                       vis_every: int = 500,
                       # Checkpointing
                       save_every: int = 1000,
                       keep_last: int = 3,
                       resume_from: str = None,
                       save_best: bool = True,
                       # DataLoader
                       num_workers: int = 0,          # 0 avoids pickling issues on macOS/Windows
                       pin_memory: bool = True):
    """
    Train the I2SB objective on unpaired clean/degraded datasets produced by `dataset_cls`.
    """
    import os
    from pathlib import Path
    from dataclasses import asdict

    import torch
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid, save_image

    # --- Prepare output dirs ---
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Config ---
    if cfg is None:
        cfg = FourTermConfig(log_dir=str(out_dir))

    # --- Datasets / loaders (unpaired) ---
    dsY = dataset_cls(split="train")   # will be transformed into degraded y by the trainer
    dsX = dataset_cls(split="train")   # clean x
    H, W = dsY.h, dsY.w

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(pin_memory and torch.cuda.is_available()),
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    dlY = DataLoader(dsY, **loader_kwargs)
    dlX = DataLoader(dsX, **loader_kwargs)
    iterX = iter(dlX)

    # --- Build trainer ---
    trainer = FourTermSBTrainer(img_shape=(H, W), cfg=cfg)

    # --- Optional resume ---
    start_epoch = 0
    step = 0
    best_score = float("inf")
    saved_paths = []

    if resume_from is not None and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=cfg.device)
        # Load networks & optimizers
        trainer.f_phi.load_state_dict(ckpt["f_phi"])
        trainer.f_theta.load_state_dict(ckpt["f_theta"])
        if "opt_phi" in ckpt:
            trainer.opt_phi.load_state_dict(ckpt["opt_phi"])
        if "opt_th" in ckpt:
            trainer.opt_th.load_state_dict(ckpt["opt_th"])
        start_epoch = int(ckpt.get("epoch", 0))
        step = int(ckpt.get("step", 0))
        # Track best if present
        m = ckpt.get("metrics", {})
        if "S_forward" in m and "S_backward" in m:
            best_score = float(m["S_forward"] + m["S_backward"])
        print(f"[resume] loaded {resume_from} @ epoch {start_epoch}, step {step}")

    # --- Training loop ---
    for epoch in range(start_epoch, epochs):
        for imgY, maskY in dlY:
            try:
                imgX, maskX = next(iterX)
            except StopIteration:
                iterX = iter(dlX)
                imgX, maskX = next(iterX)

            # To device
            imgY = imgY.to(cfg.device, non_blocking=True)
            maskY = maskY.to(cfg.device, non_blocking=True)
            imgX = imgX.to(cfg.device, non_blocking=True)
            maskX = maskX.to(cfg.device, non_blocking=True)

            # One optimization step
            metrics, samples = trainer.train_step(
                batch_y_img=imgY, batch_y_mask=maskY,
                batch_x_img=imgX, batch_x_mask=maskX
            )

            # --- Logging ---
            if (step % log_every) == 0:
                msg = " | ".join([f"{k}:{v:.4f}" for k, v in metrics.items()])
                print(f"[e{epoch:02d} s{step:05d}] {msg}")

            # --- Visualizations ---
            if (step % vis_every) == 0:
                with torch.no_grad():
                    # Backward pair: x -> X0_b(x)
                    grid_bwd = make_grid(
                        torch.cat([samples["x"][:8], samples["X0_b"][:8]], dim=0),
                        nrow=8, padding=2, normalize=True, value_range=(-1, 1)
                    )
                    save_image(grid_bwd, img_dir / f"bwd_e{epoch:03d}_s{step:06d}.png")

                    # Forward pair: y -> X1_f(y)
                    grid_fwd = make_grid(
                        torch.cat([samples["y"][:8], samples["X1_f"][:8]], dim=0),
                        nrow=8, padding=2, normalize=True, value_range=(-1, 1)
                    )
                    save_image(grid_fwd, img_dir / f"fwd_e{epoch:03d}_s{step:06d}.png")

                    # Cycle grids (inference-mode fresh noise) - kept for diagnostics
                    ty = trainer.roundtrip_y(samples["y"][:8].to(cfg.device),
                                             maskY[:8].to(cfg.device))
                    tx = trainer.roundtrip_x(samples["x"][:8].to(cfg.device),
                                             maskX[:8].to(cfg.device))
                    grid_cyc = make_grid(
                        torch.cat([samples["y"][:8], ty, samples["x"][:8], tx], dim=0),
                        nrow=8, padding=2, normalize=True, value_range=(-1, 1)
                    )
                    save_image(grid_cyc, img_dir / f"cycle_e{epoch:03d}_s{step:06d}.png")

            # --- Periodic checkpointing ---
            if (step % save_every == 0) or (step == 0):
                tag = f"e{epoch:03d}_s{step:06d}"
                path = ckpt_dir / f"ckpt_{tag}.pt"
                payload = {
                    "f_phi": trainer.f_phi.state_dict(),
                    "f_theta": trainer.f_theta.state_dict(),
                    "opt_phi": trainer.opt_phi.state_dict(),
                    "opt_th": trainer.opt_th.state_dict(),
                    "cfg": asdict(cfg),
                    "img_shape": (H, W),
                    "epoch": epoch,
                    "step": step,
                    "metrics": metrics,
                }
                torch.save(payload, path)
                saved_paths.append(path)
                # rotate
                if keep_last is not None and keep_last > 0 and len(saved_paths) > keep_last:
                    try:
                        old = saved_paths.pop(0)
                        Path(old).unlink(missing_ok=True)
                    except Exception:
                        pass

            # --- Track & save best ---
            if save_best:
                score = float(metrics["S_forward"] + metrics["S_backward"])
                if score < best_score:
                    best_score = score
                    best_payload = {
                        "f_phi": trainer.f_phi.state_dict(),
                        "f_theta": trainer.f_theta.state_dict(),
                        "opt_phi": trainer.opt_phi.state_dict(),
                        "opt_th": trainer.opt_th.state_dict(),
                        "cfg": asdict(cfg),
                        "img_shape": (H, W),
                        "epoch": epoch,
                        "step": step,
                        "metrics": {**metrics, "score": score},
                    }
                    torch.save(best_payload, ckpt_dir / "best.pt")

            step += 1

    # --- Final save (last + inference) ---
    final_payload = {
        "f_phi": trainer.f_phi.state_dict(),
        "f_theta": trainer.f_theta.state_dict(),
        "opt_phi": trainer.opt_phi.state_dict(),
        "opt_th": trainer.opt_th.state_dict(),
        "cfg": asdict(cfg),
        "img_shape": (H, W),
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
    }
    torch.save(final_payload, ckpt_dir / "last.pt")

    torch.save({
        "f_phi": trainer.f_phi.state_dict(),
        "f_theta": trainer.f_theta.state_dict(),
        "cfg": asdict(cfg),
        "img_shape": (H, W),
    }, ckpt_dir / "inference.pt")

    return trainer