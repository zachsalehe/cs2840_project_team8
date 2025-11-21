# i3sb.py
import math
import os
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from pathlib import Path

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

# -------------------------
# Reference drift g(x,t)
# -------------------------
class ReferenceDrift(nn.Module):
    """
    g(x,t): reference drift. Options:
      - 'zero': g(x,t) = 0
      - 'ou':   g(x,t) = -lambda * x
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
# Lightweight U-Net drift (I3SB: non-Markovian conditioning)
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
    """
    I³SB drift with non-Markovian conditioning.
    Inputs:
      - current state x_t (1 ch)
      - conditioning image cond (1 ch) - for I³SB, this is x_0 (initial)
      - mask (1 ch)
      - time plane t (1 ch)
    Output: drift (1 ch)
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
        out = self.outc(xdc2)
        # Clamp output to prevent explosions
        return out.clamp(-5.0, 5.0)

# -------------------------
# Config
# -------------------------
@dataclass
class I3SBConfig:
    # SDE / discretization
    sigma: float = 0.1
    n_steps: int = 10
    
    # SB energy
    ref_mode: str = "zero"
    ref_lam: float = 1.0
    
    # I3SB endpoint matching (balanced, MSE-based)
    lambda_forward: float = 1.0
    lambda_backward: float = 1.0
    
    # Optim
    lr_phi: float = 2e-4
    lr_theta: float = 2e-4
    
    # Misc
    device: str = "mps"
    log_dir: str = "./runs_i3sb"

# -------------------------
# I3SB Trainer
# -------------------------
class I3SBTrainer:
    """
    Implements I³SB (Implicit Image-to-Image Schrödinger Bridge).
    
    Key difference from I²SB: Non-Markovian conditioning where the drift
    always sees the initial condition x_0, allowing for fewer inference steps
    while maintaining quality.
    
    L(phi,theta) = L_SB(phi,theta) + lambda_fwd * ||X1_f - x||^2 
                                    + lambda_bwd * ||X0_b - y||^2
    
    No unbalanced Sinkhorn (balanced endpoints).
    No cycle consistency term.
    """
    def __init__(self, img_shape: Tuple[int, int], cfg: I3SBConfig):
        h, w = img_shape
        self.H, self.W = h, w
        self.cfg = cfg

        # Drifts
        self.f_phi   = DriftUNet(in_ch=4, out_ch=1).to(cfg.device)
        self.f_theta = DriftUNet(in_ch=4, out_ch=1).to(cfg.device)
        self.g_ref   = ReferenceDrift(mode=cfg.ref_mode, lam=cfg.ref_lam).to(cfg.device)

        self.opt_phi = torch.optim.Adam(self.f_phi.parameters(), lr=cfg.lr_phi, betas=(0.5, 0.999))
        self.opt_th  = torch.optim.Adam(self.f_theta.parameters(), lr=cfg.lr_theta, betas=(0.5, 0.999))

        os.makedirs(cfg.log_dir, exist_ok=True)

    # ---------- SDE simulators (non-Markovian: always condition on x_0) ----------
    def _em_forward(self, y: torch.Tensor, mask: torch.Tensor):
        """
        I³SB forward: non-Markovian conditioning on initial y (x_0).
        Returns (X1_f, path_energy_f).
        """
        B = y.size(0)
        dt = 1.0 / self.cfg.n_steps
        x = y.clone()
        x_0 = y.clone()  # Store initial for non-Markovian conditioning
        energy = 0.0
        
        for k in range(self.cfg.n_steps):
            t = k / self.cfg.n_steps
            t_plane = make_time_plane(B, self.H, self.W, t, y.device)
            
            # Non-Markovian: always pass x_0
            drift = self.f_phi(x, x_0, mask, t_plane)
            ref = self.g_ref(x, t_plane)
            
            diff = drift - ref
            energy += diff.pow(2).mean() * dt
            
            noise = torch.randn_like(x) * (self.cfg.sigma * math.sqrt(dt))
            new_x = x + drift * dt + noise
            
            # Clamp to prevent explosions
            new_x = new_x.clamp(-10.0, 10.0)
            
            m = mask.bool()
            x = torch.where(m, new_x, x_0)
            
        return x, energy

    def _em_backward(self, x_clean: torch.Tensor, mask: torch.Tensor):
        """
        I³SB backward: non-Markovian conditioning on terminal x_clean (x_1).
        Returns (X0_b, path_energy_b).
        """
        B = x_clean.size(0)
        dt = 1.0 / self.cfg.n_steps
        y = x_clean.clone()
        x_1 = x_clean.clone()  # Store terminal for non-Markovian conditioning
        energy = 0.0
        
        for k in range(self.cfg.n_steps):
            s = k / self.cfg.n_steps
            t = 1.0 - s
            t_plane = make_time_plane(B, self.H, self.W, t, x_clean.device)
            
            # Non-Markovian: always pass x_1
            drift_t = self.f_theta(y, x_1, mask, t_plane)
            ref = self.g_ref(y, t_plane)
            
            diff = drift_t - ref
            energy += diff.pow(2).mean() * dt
            
            noise = torch.randn_like(y) * (self.cfg.sigma * math.sqrt(dt))
            new_y = y + (-drift_t) * dt + noise
            
            # Clamp to prevent explosions
            new_y = new_y.clamp(-10.0, 10.0)
            
            m = mask.bool()
            y = torch.where(m, new_y, x_1)
            
        return y, energy

    # ---------- Round-trip maps with fresh noise ----------
    @torch.no_grad()
    def roundtrip_y(self, y: torch.Tensor, mask: torch.Tensor):
        """tilde y(y) = X_0^b( X_1^f(y) ) with fresh noise."""
        x1, _ = self._em_forward(y, mask)
        x1 = x1.detach()
        y0, _ = self._em_backward(x1, mask)
        return y0

    @torch.no_grad()
    def roundtrip_x(self, x: torch.Tensor, mask: torch.Tensor):
        """tilde x(x) = X_1^f( X_0^b(x) ) with fresh noise."""
        y0, _ = self._em_backward(x, mask)
        y0 = y0.detach()
        x1, _ = self._em_forward(y0, mask)
        return x1

    # ---------- Public API ----------
    def train_step(self,
                   batch_y_img: torch.Tensor, 
                   batch_y_mask: torch.Tensor,
                   batch_x_img: torch.Tensor, 
                   batch_x_mask: torch.Tensor,
                   profile: bool = False):
        """
        Train one step of I³SB.
        Returns dict with scalar losses and samples.
        """
        import time
        times = {} if profile else None
        t0 = time.time() if profile else None
        
        # Sanitize masks
        my = batch_y_mask.to(batch_y_img.dtype).clamp(0.0, 1.0)
        mx = batch_x_mask.to(batch_x_img.dtype).clamp(0.0, 1.0)
        
        # Build degraded y
        noise = torch.empty_like(batch_y_img).uniform_(-1.0, 1.0)
        y = batch_y_img * (1.0 - my) + noise * my
        x = batch_x_img
        
        if profile:
            times['prep'] = time.time() - t0
            t0 = time.time()
        
        # Simulate both bridges
        X1_f, energy_f = self._em_forward(y, my)
        
        if profile:
            times['forward_bridge'] = time.time() - t0
            t0 = time.time()
            
        X0_b, energy_b = self._em_backward(x, mx)
        
        if profile:
            times['backward_bridge'] = time.time() - t0
            t0 = time.time()
        
        # L_SB: path/energy matching
        L_sb = (energy_f + energy_b) / (2.0 * (self.cfg.sigma ** 2))
        
        # Balanced endpoint matching
        loss_fwd = F.mse_loss(X1_f, x.detach())
        loss_bwd = F.mse_loss(X0_b, y.detach())
        
        # Total loss (no cycle term in I³SB)
        loss = L_sb + self.cfg.lambda_forward * loss_fwd + self.cfg.lambda_backward * loss_bwd
        
        if profile:
            times['loss_calc'] = time.time() - t0
            t0 = time.time()
        
        # Optimize
        self.opt_phi.zero_grad(set_to_none=True)
        self.opt_th.zero_grad(set_to_none=True)
        loss.backward()
        
        if profile:
            times['backward_pass'] = time.time() - t0
            t0 = time.time()
        
        # Gradient clipping to prevent explosions
        torch.nn.utils.clip_grad_norm_(self.f_phi.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.f_theta.parameters(), max_norm=1.0)
        
        self.opt_phi.step()
        self.opt_th.step()
        
        if profile:
            times['optimizer_step'] = time.time() - t0
        
        # Metrics
        with torch.no_grad():
            metrics = {
                "loss_total": loss.item(),
                "L_SB": L_sb.item(),
                "loss_forward": loss_fwd.item(),
                "loss_backward": loss_bwd.item(),
            }
            if profile:
                metrics['times'] = times
        
        return metrics, {
            "X1_f": X1_f.detach(), 
            "X0_b": X0_b.detach(), 
            "y": y.detach(), 
            "x": x.detach()
        }

# -------------------------
# Training loop
# -------------------------
def train_i3sb(dataset_cls,
               epochs: int = 3,
               batch_size: int = 64,
               out_dir: str = "./runs_i3sb",
               cfg: Optional[I3SBConfig] = None,
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
    Train I³SB on unpaired clean/degraded datasets.
    Structure matches four_term_sb.py for easy comparison.
    """
    from dataclasses import asdict
    
    # Prepare output dirs
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Config
    if cfg is None:
        cfg = I3SBConfig(log_dir=str(out_dir))
    
    # Datasets (unpaired)
    dsY = dataset_cls(split="train")
    dsX = dataset_cls(split="train")
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
    
    # Build trainer
    trainer = I3SBTrainer(img_shape=(H, W), cfg=cfg)
    
    # Optional resume
    start_epoch = 0
    step = 0
    best_score = float("inf")
    saved_paths = []
    
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
        m = ckpt.get("metrics", {})
        if "loss_forward" in m and "loss_backward" in m:
            best_score = float(m["loss_forward"] + m["loss_backward"])
        print(f"[resume] loaded {resume_from} @ epoch {start_epoch}, step {step}")
    
    # Training loop
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
            
            # Train step
            metrics, samples = trainer.train_step(
                batch_y_img=imgY, batch_y_mask=maskY,
                batch_x_img=imgX, batch_x_mask=maskX,
                profile=(step == 0)  # Profile first step only
            )
            
            # MPS memory fix: clear cache after every step to prevent fragmentation
            if cfg.device == "mps":
                torch.mps.empty_cache()
                torch.mps.synchronize()  # Ensure operations complete
            
            # Logging
            if (step % log_every) == 0:
                msg = " | ".join([f"{k}:{v:.4f}" for k, v in metrics.items() if k != 'times'])
                print(f"[e{epoch:02d} s{step:05d}] {msg}")
                
                # Print timing breakdown on first step
                if step == 0 and 'times' in metrics:
                    print("  Timing breakdown (seconds):")
                    for k, v in metrics['times'].items():
                        print(f"    {k}: {v:.3f}s")
            
            # Visualizations (same structure as four_term_sb)
            if (step % vis_every) == 0 and step > 0:  # Skip step 0 to save time
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
                    
                    # Cycle grids
                    ty = trainer.roundtrip_y(samples["y"][:8].to(cfg.device),
                                           maskY[:8].to(cfg.device))
                    tx = trainer.roundtrip_x(samples["x"][:8].to(cfg.device),
                                           maskX[:8].to(cfg.device))
                    grid_cyc = make_grid(
                        torch.cat([samples["y"][:8], ty, samples["x"][:8], tx], dim=0),
                        nrow=8, padding=2, normalize=True, value_range=(-1, 1)
                    )
                    save_image(grid_cyc, img_dir / f"cycle_e{epoch:03d}_s{step:06d}.png")
            
            # Periodic checkpointing
            if (step % save_every == 0) and step > 0:  # Skip step 0
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
                
                # Rotate old checkpoints
                if keep_last is not None and keep_last > 0 and len(saved_paths) > keep_last:
                    try:
                        old = saved_paths.pop(0)
                        Path(old).unlink(missing_ok=True)
                    except Exception:
                        pass
            
            # Track best
            if save_best:
                score = float(metrics["loss_forward"] + metrics["loss_backward"])
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
    
    # Final saves
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