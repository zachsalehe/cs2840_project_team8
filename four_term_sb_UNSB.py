import math
import os
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

# =========================================================
# =============== Patch Discriminator (UNSB) ===============
# =========================================================

class PatchDiscriminator(nn.Module):
    """Simple patch-based discriminator used in UNSB."""
    def __init__(self, in_ch: int = 1, base: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.InstanceNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.InstanceNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, 1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).mean(dim=(2, 3))


# =========================================================
# ===================== Utilities =========================
# =========================================================

EPS = 1e-8


def make_time_plane(b: int, h: int, w: int, t_scalar: float, device: torch.device):
    """Create a constant time-channel plane for input concatenation."""
    return torch.full((b, 1, h, w), float(t_scalar), device=device)


def pairwise_masked_mean_sq_cost(xA, xB, maskA):
    """Pairwise masked squared distance matrix."""
    B = xA.size(0)
    XA, XB = xA.view(B, -1), xB.view(B, -1)
    W = maskA.view(B, -1)
    wsum = W.sum(dim=1).clamp_min(1.0)

    XA2w = (W * XA ** 2).sum(dim=1)[:, None]
    XB2_T = (XB ** 2).t()
    term2 = W @ XB2_T
    term3 = 2.0 * ((W * XA) @ XB.t())
    return (XA2w + term2 - term3).div(wsum[:, None]).clamp_min(0.0)


def unbalanced_sinkhorn_uot(a, b, C, eps, tau, n_iters=50):
    """Stable unbalanced Sinkhorn in the normal domain."""
    B = C.size(0)
    K = torch.exp(-C / eps).clamp_min(1e-38)
    alpha = tau / (tau + eps + 1e-12)

    u = torch.ones(B, device=C.device)
    v = torch.ones(B, device=C.device)

    for _ in range(n_iters):
        Kv = (K @ v).clamp_min(1e-30)
        u = (a / Kv).clamp_min(1e-30).pow(alpha)
        Ktu = (K.t() @ u).clamp_min(1e-30)
        v = (b / Ktu).clamp_min(1e-30).pow(alpha)

    gamma = (u[:, None] * K) * v[None, :]
    gamma_det = gamma.detach()

    m = gamma.sum(dim=1)
    n = gamma.sum(dim=0)

    cost_term = (gamma * C).sum()
    ab = torch.outer(a, b).clamp_min(1e-30)
    kl_g_ab = (gamma * (torch.log(gamma.clamp_min(1e-30)) - torch.log(ab))).sum() - gamma.sum() + ab.sum()
    kl_m_a = (m * (torch.log(m.clamp_min(1e-30)) - torch.log(a.clamp_min(1e-30)))).sum() - m.sum() + a.sum()
    kl_n_b = (n * (torch.log(n.clamp_min(1e-30)) - torch.log(b.clamp_min(1e-30)))).sum() - n.sum() + b.sum()

    uot_val = cost_term + eps * kl_g_ab + tau * (kl_m_a + kl_n_b)
    return {"uot": uot_val, "gamma": gamma_det}


def uot_only_masked(xA, xB, maskA, eps, tau, n_iters=50):
    """Compute UOT only for masked pixels."""
    B = xA.size(0)
    a = b = torch.full((B,), 1.0 / B, device=xA.device)
    C = pairwise_masked_mean_sq_cost(xA, xB, maskA)
    return unbalanced_sinkhorn_uot(a, b, C, eps, tau, n_iters)["uot"]


# =========================================================
# ============== Reference Drift & U-Net ==================
# =========================================================

class ReferenceDrift(nn.Module):
    """Reference drift g(x, t) = 0 or OU drift."""
    def __init__(self, mode: str = "zero", lam: float = 1.0):
        super().__init__()
        self.mode = mode
        self.lam = lam

    def forward(self, x, t_plane):
        if self.mode == "zero":
            return torch.zeros_like(x)
        elif self.mode == "ou":
            return -self.lam * x
        else:
            raise ValueError("ReferenceDrift mode must be 'zero' or 'ou'.")


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
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(True)
        )

    def forward(self, x): return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(True)
        )

    def forward(self, x): return self.net(x)


class DriftUNet(nn.Module):
    """Predicts drift f(x_t, t; cond)."""
    def __init__(self, in_ch=4, base=64, out_ch=1):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.bott = DoubleConv(base * 4, base * 4)
        self.u1 = Up(base * 4, base * 2)
        self.dc1 = DoubleConv(base * 4, base * 2)
        self.u2 = Up(base * 2, base)
        self.dc2 = DoubleConv(base * 2, base)
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


# =========================================================
# =================== Config Class ========================
# =========================================================

@dataclass
class FourTermConfig:
    sigma: float = 0.1
    n_steps: int = 10
    ref_mode: str = "zero"
    ref_lam: float = 1.0
    eps_sink: float = 0.1
    tau_sink: float = 0.5
    sink_iters: int = 50
    lambda_cyc: float = 1.0
    lambda_X: float = 10.0
    lambda_Y: float = 10.0
    lr_phi: float = 2e-4
    lr_theta: float = 2e-4
    lr_d: float = 2e-4
    d_steps: int = 1
    adv_weight: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = "./runs_unsb"


# =========================================================
# =============== Main UNSB Trainer =======================
# =========================================================

class FourTermSBTrainer:
    """Implements Unbalanced Neural Schrödinger Bridge (UNSB)."""

    def __init__(self, img_shape: Tuple[int, int], cfg: FourTermConfig):
        h, w = img_shape
        self.H, self.W = h, w
        self.cfg = cfg

        # Discriminators
        self.D_X = PatchDiscriminator(in_ch=1).to(cfg.device)
        self.D_Y = PatchDiscriminator(in_ch=1).to(cfg.device)
        self.opt_DX = torch.optim.Adam(self.D_X.parameters(), lr=cfg.lr_d, betas=(0.5, 0.999))
        self.opt_DY = torch.optim.Adam(self.D_Y.parameters(), lr=cfg.lr_d, betas=(0.5, 0.999))

        # Drifts
        self.f_phi = DriftUNet(in_ch=4, out_ch=1).to(cfg.device)
        self.f_theta = DriftUNet(in_ch=4, out_ch=1).to(cfg.device)
        self.g_ref = ReferenceDrift(mode=cfg.ref_mode, lam=cfg.ref_lam).to(cfg.device)

        self.opt_phi = torch.optim.Adam(self.f_phi.parameters(), lr=cfg.lr_phi, betas=(0.5, 0.999))
        self.opt_th = torch.optim.Adam(self.f_theta.parameters(), lr=cfg.lr_theta, betas=(0.5, 0.999))

        os.makedirs(cfg.log_dir, exist_ok=True)

    # ---------- Euler–Maruyama simulations ----------
    def _em_forward(self, y, mask):
        """Forward SDE simulation."""
        B = y.size(0)
        dt = 1.0 / self.cfg.n_steps
        x = y.clone()
        energy = 0.0
        for k in range(self.cfg.n_steps):
            t = k / self.cfg.n_steps
            t_plane = make_time_plane(B, self.H, self.W, t, y.device)
            drift = self.f_phi(x, y, mask, t_plane)
            ref = self.g_ref(x, t_plane)
            diff = drift - ref
            energy += diff.pow(2).mean() * dt
            noise = torch.randn_like(x) * (self.cfg.sigma * math.sqrt(dt))
            new_x = x + drift * dt + noise
            x = torch.where(mask.bool(), new_x, y)
        return x, energy

    def _em_backward(self, x_clean, mask):
        """Backward SDE simulation."""
        B = x_clean.size(0)
        dt = 1.0 / self.cfg.n_steps
        y = x_clean.clone()
        energy = 0.0
        for k in range(self.cfg.n_steps):
            s = k / self.cfg.n_steps
            t = 1.0 - s
            t_plane = make_time_plane(B, self.H, self.W, t, x_clean.device)
            drift_t = self.f_theta(y, x_clean, mask, t_plane)
            ref = self.g_ref(y, t_plane)
            diff = drift_t - ref
            energy += diff.pow(2).mean() * dt
            noise = torch.randn_like(y) * (self.cfg.sigma * math.sqrt(dt))
            new_y = y + (-drift_t) * dt + noise
            y = torch.where(mask.bool(), new_y, x_clean)
        return y, energy

    # ---------- Cycle ----------
    @torch.no_grad()
    def roundtrip_y(self, y, mask):
        x1, _ = self._em_forward(y, mask)
        y0, _ = self._em_backward(x1, mask)
        return y0

    @torch.no_grad()
    def roundtrip_x(self, x, mask):
        y0, _ = self._em_backward(x, mask)
        x1, _ = self._em_forward(y0, mask)
        return x1

    # ---------- Train Step ----------
    def train_step(self, batch_y_img, batch_y_mask, batch_x_img, batch_x_mask):
        """Single training iteration."""
        my = batch_y_mask.float().clamp(0.0, 1.0)
        mx = batch_x_mask.float().clamp(0.0, 1.0)

        # Construct degraded Y
        noise = torch.empty_like(batch_y_img).uniform_(-1.0, 1.0)
        y = batch_y_img * (1.0 - my) + noise * my
        x = batch_x_img

        # Forward/backward bridges
        X1_f, _ = self._em_forward(y, my)
        X0_b, _ = self._em_backward(x, mx)

        # Cycle consistency
        tilde_y, _ = self._em_backward(X1_f.detach(), my)
        tilde_x, _ = self._em_forward(X0_b.detach(), mx)
        rho = 0.7
        L_cyc_y = (F.l1_loss(tilde_y * (1 - my), y * (1 - my)) +
                   rho * F.l1_loss(tilde_y * my, y * my))
        L_cyc_x = F.l1_loss(tilde_x, x)
        L_cyc = L_cyc_y + L_cyc_x

        # Unbalanced Sinkhorn endpoint terms
        S_forward = uot_only_masked(X1_f, x.detach(), my,
                                    self.cfg.eps_sink, self.cfg.tau_sink, self.cfg.sink_iters)
        S_backward = uot_only_masked(X0_b, y.detach(), mx,
                                     self.cfg.eps_sink, self.cfg.tau_sink, self.cfg.sink_iters)

        # ========== UNSB adversarial losses ==========
        for _ in range(self.cfg.d_steps):
            # D_X
            self.opt_DX.zero_grad(set_to_none=True)
            dx_real = self.D_X(x)
            dx_fake = self.D_X(X1_f.detach())
            dX_loss = (torch.relu(1.0 - dx_real).mean() +
                       torch.relu(1.0 + dx_fake).mean())
            dX_loss.backward()
            self.opt_DX.step()

            # D_Y
            self.opt_DY.zero_grad(set_to_none=True)
            dy_real = self.D_Y(y)
            dy_fake = self.D_Y(X0_b.detach())
            dY_loss = (torch.relu(1.0 - dy_real).mean() +
                       torch.relu(1.0 + dy_fake).mean())
            dY_loss.backward()
            self.opt_DY.step()

        # Generator adversarial loss
        L_adv = self.cfg.adv_weight * (-self.D_X(X1_f).mean() - self.D_Y(X0_b).mean())

        # Total loss
        loss = L_adv + self.cfg.lambda_cyc * L_cyc

        self.opt_phi.zero_grad(set_to_none=True)
        self.opt_th.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_phi.step()
        self.opt_th.step()

        # Metrics
        with torch.no_grad():
            metrics = {
                "L_cyc": float(L_cyc.item()),
                "L_adv": float(L_adv.item()),
                "D_X": float(dX_loss.item()),
                "D_Y": float(dY_loss.item()),
                "S_forward": float(S_forward.item()),
                "S_backward": float(S_backward.item()),
            }

        return metrics, {"X1_f": X1_f.detach(), "X0_b": X0_b.detach(), "y": y.detach(), "x": x.detach()}
    

# =========================================================
# =============== Convenience Training Loop ===============
# =========================================================
import os
from pathlib import Path
from dataclasses import asdict
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import torch

def train_four_term_sb(dataset_cls,
                       epochs: int = 5,
                       batch_size: int = 64,
                       out_dir: str = "./runs_unsb",
                       cfg: Optional[FourTermConfig] = None,
                       *,
                       log_every: int = 100,
                       vis_every: int = 500,
                       save_every: int = 1000,
                       keep_last: int = 3,
                       resume_from: Optional[str] = None,
                       save_best: bool = True,
                       num_workers: int = 0,
                       pin_memory: bool = True):
    """
    High-level UNSB training loop with checkpointing and visualization.
    """
    # ---------------- Directories ----------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    ckpt_dir = out_dir / "ckpts"
    img_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if cfg is None:
        cfg = FourTermConfig(log_dir=str(out_dir))

    # ---------------- Data ----------------
    dsY = dataset_cls(split="train")  # degraded (y)
    dsX = dataset_cls(split="train")  # clean (x)
    H, W = dsY.h, dsY.w

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(pin_memory and torch.cuda.is_available())
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    dlY = DataLoader(dsY, **loader_kwargs)
    dlX = DataLoader(dsX, **loader_kwargs)
    iterX = iter(dlX)

    trainer = FourTermSBTrainer(img_shape=(H, W), cfg=cfg)
    step = 0
    start_epoch = 0
    best_score = float("inf")
    saved_paths = []

    # ---------------- Resume ----------------
    if resume_from is not None and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=cfg.device)
        trainer.f_phi.load_state_dict(ckpt["f_phi"])
        trainer.f_theta.load_state_dict(ckpt["f_theta"])
        if "opt_phi" in ckpt:
            trainer.opt_phi.load_state_dict(ckpt["opt_phi"])
        if "opt_th" in ckpt:
            trainer.opt_th.load_state_dict(ckpt["opt_th"])
        start_epoch = int(ckpt.get("epoch", 0))
        step = int(ckpt.get("step", 0))
        best_score = float(ckpt.get("metrics", {}).get("S_forward", 0)
                           + ckpt.get("metrics", {}).get("S_backward", 0))
        print(f"[resume] Loaded checkpoint from {resume_from} @ epoch {start_epoch}, step {step}")

    # ---------------- Training Loop ----------------
    for epoch in range(start_epoch, epochs):
        for imgY, maskY in dlY:
            try:
                imgX, maskX = next(iterX)
            except StopIteration:
                iterX = iter(dlX)
                imgX, maskX = next(iterX)

            imgY, maskY = imgY.to(cfg.device), maskY.to(cfg.device)
            imgX, maskX = imgX.to(cfg.device), maskX.to(cfg.device)

            metrics, samples = trainer.train_step(
                batch_y_img=imgY, batch_y_mask=maskY,
                batch_x_img=imgX, batch_x_mask=maskX
            )

            # ----- Logging -----
            if step % log_every == 0:
                msg = " | ".join([f"{k}:{v:.4f}" for k, v in metrics.items()])
                print(f"[e{epoch:02d} s{step:05d}] {msg}")

            # ----- Visualization -----
            if step % vis_every == 0:
                with torch.no_grad():
                    grid = make_grid(torch.cat([
                        samples["x"][:8],
                        samples["X0_b"][:8],
                        samples["y"][:8],
                        samples["X1_f"][:8]
                    ], dim=0), nrow=8, padding=2, normalize=True, value_range=(-1, 1))
                    save_image(grid, img_dir / f"e{epoch:03d}_s{step:06d}.png")

            # ----- Checkpoint -----
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
                    "metrics": metrics
                }
                torch.save(payload, path)
                saved_paths.append(path)
                # keep only last N
                if keep_last and len(saved_paths) > keep_last:
                    old = saved_paths.pop(0)
                    try:
                        Path(old).unlink(missing_ok=True)
                    except Exception:
                        pass

            # ----- Best checkpoint -----
            if save_best:
                score = metrics["S_forward"] + metrics["S_backward"]
                if score < best_score:
                    best_score = score
                    best_path = ckpt_dir / "best.pt"
                    torch.save({
                        "f_phi": trainer.f_phi.state_dict(),
                        "f_theta": trainer.f_theta.state_dict(),
                        "opt_phi": trainer.opt_phi.state_dict(),
                        "opt_th": trainer.opt_th.state_dict(),
                        "cfg": asdict(cfg),
                        "img_shape": (H, W),
                        "epoch": epoch,
                        "step": step,
                        "metrics": {**metrics, "score": score}
                    }, best_path)

            step += 1

    # ---------------- Final Save ----------------
    final_path = ckpt_dir / "last.pt"
    torch.save({
        "f_phi": trainer.f_phi.state_dict(),
        "f_theta": trainer.f_theta.state_dict(),
        "opt_phi": trainer.opt_phi.state_dict(),
        "opt_th": trainer.opt_th.state_dict(),
        "cfg": asdict(cfg),
        "img_shape": (H, W),
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
    }, final_path)

    print(f"[done] Training complete. Final checkpoint saved to {final_path}")
    return trainer

