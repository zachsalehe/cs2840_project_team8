from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Dict, Any

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

Method = Literal["i2sb", "fourterm", "unsb", "i3sb"]

@dataclass
class UnifiedConfig:
    # Identity & bookkeeping
    method: Method = "fourterm"
    version: int = 1
    arch: str = "drift_unet_tiny"
    dataset: str = "MNIST"

    # Dataset parameters (from MNISTDataset)
    digits: tuple | list | range = tuple(range(10))
    mask_type: str = "perlin"
    mask_area: float = 0.2
    mask_scale: float = 1.0
    usps: bool = False

    # SDE & simulator
    sigma: float = 0.5
    n_steps: int = 20
    ref_mode: str = "zero"      # "zero" or "ou"
    ref_lam: float = 1.0

    # Endpoint objectives (cover all variants)
    eps_sink: float = 0.08
    tau_sink: float = 0.8
    sink_iters: int = 80
    lambda_X: float = 5.0
    lambda_Y: float = 5.0
    lambda_cyc: float = 0.0  # =0 for I²SB; >0 for four-term/UNSB

    # I³SB (MSE endpoints)
    lambda_forward: float = 1.0
    lambda_backward: float = 1.0

    # Optim (for training in this file)
    lr_phi: float = 2e-4
    lr_theta: float = 2e-4
    device: str = "cpu"
    log_dir: str = "./runs_unified"


EPS = 1e-8

def make_time_plane(b: int, h: int, w: int, t: float, device: torch.device):
    return torch.full((b, 1, h, w), float(t), device=device)

class ReferenceDrift(nn.Module):
    def __init__(self, mode: str = "zero", lam: float = 1.0):
        super().__init__()
        self.mode = mode
        self.lam = lam
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "zero":
            return torch.zeros_like(x)
        if self.mode == "ou":
            return -self.lam * x
        raise ValueError("ref_mode must be 'zero' or 'ou'")

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class DriftUNet(nn.Module):
    """Tiny U-Net that consumes [x_t, cond, mask, t] -> drift."""
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        ch = base
        self.down1 = DoubleConv(in_ch * 2 + 1 + 1, ch)   # x, cond, mask, t
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(ch, ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.mid = DoubleConv(ch * 2, ch * 4)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
        self.dec2 = DoubleConv(ch * 4, ch * 2)
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.dec1 = DoubleConv(ch * 2, ch)
        self.out = nn.Conv2d(ch, 1, 1)
    def forward(self, x, cond, mask, t_plane):
        z = torch.cat([x, cond, mask, t_plane], dim=1)
        d1 = self.down1(z)
        d2 = self.down2(self.pool1(d1))
        m = self.mid(self.pool2(d2))
        u2 = self.up2(m)
        c2 = self.dec2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(c2)
        c1 = self.dec1(torch.cat([u1, d1], dim=1))
        return self.out(c1)

@torch.no_grad()
def em_path(
    x0: torch.Tensor,
    cond: torch.Tensor,
    mask: torch.Tensor,
    n_steps: int,
    sigma: float,
    drift: nn.Module,
    ref: nn.Module,
    non_markov: bool,
    device: torch.device,
):
    """Generic masked EM path used for both forward/backward traces."""
    b, c, h, w = x0.shape
    x = x0.clone().to(device)
    mask = mask.to(device)
    tgrid = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    for i in range(n_steps):
        t = float(tgrid[i])
        dt = float(tgrid[i + 1] - tgrid[i])
        t_plane = make_time_plane(b, h, w, t, device)
        cimg = x0 if non_markov else cond
        drift_term = drift(x, cimg, mask, t_plane)
        ref_term = ref(x)
        noise = torch.randn_like(x)
        x = x + (drift_term + ref_term) * dt + sigma * math.sqrt(dt) * noise
        # clamp outside hole
        m = mask.float()
        x = x * m + cond * (1 - m)
    return x

class BalancedOT(nn.Module):
    def __init__(self, eps: float, iters: int):
        super().__init__()
        self.eps, self.iters = eps, iters
    def forward(self, xA: torch.Tensor, xB: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.expand_as(xA).float()
        diff = (xA - xB) * m
        denom = m.sum() + EPS
        return (diff.pow(2).sum() / denom).clamp_min(0)

class UnbalancedOT(nn.Module):
    def __init__(self, eps: float, tau: float, iters: int):
        super().__init__()
        self.eps, self.tau, self.iters = eps, tau, iters
    def forward(self, xA: torch.Tensor, xB: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.expand_as(xA).float()
        diff = (xA - xB) * m
        denom = m.sum() + EPS
        l2 = diff.pow(2).sum() / denom
        l1 = diff.abs().sum() / denom
        return 0.5 * l2 + 0.5 * (self.tau * l1)

class UnifiedSB(nn.Module):
    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.phi = DriftUNet(1, 32)
        self.theta = DriftUNet(1, 32)
        self.ref = ReferenceDrift(cfg.ref_mode, cfg.ref_lam)
        self.to(self.device)

        if cfg.method == "i3sb":
            self.endpoint = None
            self.non_markov = True
        else:
            self.non_markov = False
            if cfg.method in ("unsb", "fourterm"):
                self.endpoint = UnbalancedOT(cfg.eps_sink, cfg.tau_sink, cfg.sink_iters)
            elif cfg.method == "i2sb":
                self.endpoint = BalancedOT(cfg.eps_sink, cfg.sink_iters)
            else:
                raise ValueError(f"Unknown method {cfg.method}")

    @torch.no_grad()
    def sample_inpaint(self, y: torch.Tensor, mask: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        s = num_steps or self.cfg.n_steps
        x0 = y.clone()
        return em_path(x0, y, mask, s, self.cfg.sigma, self.phi, self.ref, self.non_markov, self.device)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, mask: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        s = num_steps or self.cfg.n_steps
        return em_path(x.clone(), x, mask, s, self.cfg.sigma, self.theta, self.ref, self.non_markov, self.device)

    @torch.no_grad()
    def round_trip(self, y: torch.Tensor, mask: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        s = num_steps or self.cfg.n_steps
        x1 = self.sample_inpaint(y, mask, s)
        return em_path(x1.clone(), y, mask, s, self.cfg.sigma, self.theta, self.ref, self.non_markov, self.device)

    def save_ckpt(self, path: str, extra: Optional[Dict[str, Any]] = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "state": {
                "phi": self.phi.state_dict(),
                "theta": self.theta.state_dict(),
            },
            "cfg": asdict(self.cfg),
            "meta": {
                "method": self.cfg.method,
                "version": self.cfg.version,
                "arch": self.cfg.arch,
                "dataset": self.cfg.dataset,
                "digits": tuple(self.cfg.digits) if not isinstance(self.cfg.digits, range) else tuple(self.cfg.digits),
                "mask_type": self.cfg.mask_type,
                "mask_area": self.cfg.mask_area,
                "mask_scale": self.cfg.mask_scale,
                "usps": self.cfg.usps,
            },
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)

    @staticmethod
    def load_ckpt(path: str, map_location: str | torch.device = "cpu") -> "UnifiedSB":
        blob = torch.load(path, map_location=map_location)
        method = (
            blob.get("meta", {}).get("method")
            or blob.get("cfg", {}).get("method")
            or "fourterm"
        )
        cfg_dict = {**blob.get("cfg", {})}
        if "method" not in cfg_dict:
            cfg_dict["method"] = method
        cfg = UnifiedConfig(**cfg_dict)
        model = UnifiedSB(cfg)
        state = blob.get("state", {})
        if "phi" in state: model.phi.load_state_dict(state["phi"], strict=False)
        if "theta" in state: model.theta.load_state_dict(state["theta"], strict=False)
        return model

    def train_step(self, batch, optim_phi, optim_theta):
        # set modules to train mode without shadowing fit()
        super().train()
        x, y, mask = batch["x"], batch["y"], batch["mask"]
        device = self.device
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        # Forward path (y -> x)
        b, c, h, w = y.shape
        tgrid = torch.linspace(0.0, 1.0, self.cfg.n_steps + 1, device=device)
        x_t = y.clone()
        loss_energy = 0.0
        for i in range(self.cfg.n_steps):
            t = float(tgrid[i])
            dt = float(tgrid[i + 1] - tgrid[i])
            t_plane = make_time_plane(b, h, w, t, device)
            cond = y
            drift_term = self.phi(x_t, cond, mask, t_plane)
            ref_term = self.ref(x_t)
            noise = torch.randn_like(x_t)
            x_next = x_t + (drift_term + ref_term) * dt + self.cfg.sigma * math.sqrt(dt) * noise
            m = mask.float()
            x_next = x_next * m + y * (1 - m)
            loss_energy = loss_energy + (drift_term.pow(2).mean()) * dt
            x_t = x_next

        if self.cfg.method == "i3sb":
            loss_end = self.cfg.lambda_forward * F.mse_loss(x_t * mask.float(), x * mask.float())
        else:
            loss_end = self.cfg.lambda_X * self.endpoint(x_t, x, mask)

        loss = loss_energy + loss_end
        optim_phi.zero_grad(set_to_none=True)
        loss.backward()
        optim_phi.step()

        # Backward path (x -> y)
        x_t = x.clone()
        loss_energy_b = 0.0
        for i in range(self.cfg.n_steps):
            t = float(tgrid[i])
            dt = float(tgrid[i + 1] - tgrid[i])
            t_plane = make_time_plane(b, h, w, t, device)
            cond = x
            drift_term = self.theta(x_t, cond, mask, t_plane)
            ref_term = self.ref(x_t)
            noise = torch.randn_like(x_t)
            x_next = x_t + (drift_term + ref_term) * dt + self.cfg.sigma * math.sqrt(dt) * noise
            m = mask.float()
            x_next = x_next * m + x * (1 - m)
            loss_energy_b = loss_energy_b + (drift_term.pow(2).mean()) * dt
            x_t = x_next

        if self.cfg.method == "i3sb":
            loss_end_b = self.cfg.lambda_backward * F.mse_loss(x_t * mask.float(), y * mask.float())
        else:
            loss_end_b = self.cfg.lambda_Y * self.endpoint(x_t, y, mask)

        loss_cyc = torch.tensor(0.0, device=device)
        if self.cfg.lambda_cyc > 0:
            with torch.no_grad():
                x1 = self.sample_inpaint(y, mask)
            y_rt = em_path(x1.clone(), y, mask, self.cfg.n_steps, self.cfg.sigma, self.theta, self.ref, self.non_markov, device)
            loss_cyc = F.l1_loss(y_rt * mask.float(), y * mask.float())

        loss_b = loss_energy_b + loss_end_b + self.cfg.lambda_cyc * loss_cyc
        optim_theta.zero_grad(set_to_none=True)
        loss_b.backward()
        optim_theta.step()

        return {
            "loss_f": float(loss.detach().cpu()),
            "loss_b": float(loss_b.detach().cpu()),
            "loss_cyc": float(loss_cyc.detach().cpu()),
        }

    def fit(self, dl, epochs: int = 5, *, log_every: int = 100, vis_every: int = 100):
        """High-level training with tqdm + mid-training visualizations."""
        from tqdm.auto import tqdm
        try:
            from torchvision.utils import make_grid, save_image
        except Exception as e:
            make_grid = save_image = None
            print("[warn] torchvision not available; mid-training visuals disabled:", e)

        os.makedirs(self.cfg.log_dir, exist_ok=True)
        img_dir = os.path.join(self.cfg.log_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        optim_phi = torch.optim.Adam(self.phi.parameters(), lr=self.cfg.lr_phi)
        optim_theta = torch.optim.Adam(self.theta.parameters(), lr=self.cfg.lr_theta)

        step = 0
        for ep in range(1, epochs + 1):
            print(f"\n===== Epoch {ep}/{epochs} =====")
            bar = tqdm(dl, desc=f"epoch {ep}", leave=True)
            for it, batch in enumerate(bar, 1):
                stats = self.train_step(batch, optim_phi, optim_theta)
                if (step % log_every) == 0:
                    msg = {k: f"{v:.4f}" for k, v in stats.items()}
                    bar.set_postfix(msg)

                if make_grid is not None and save_image is not None and (step % vis_every) == 0:
                    with torch.no_grad():
                        x = batch["x"].to(self.device)
                        y = batch["y"].to(self.device)
                        mask = batch["mask"].to(self.device)
                        x1_f = self.sample_inpaint(y, mask)
                        x0_b = self.reconstruct(x, mask)
                        grid = make_grid(
                            torch.cat([x[:8], x0_b[:8], y[:8], x1_f[:8]], dim=0),
                            nrow=8, padding=2, normalize=True, value_range=(-1, 1)
                        )
                        save_image(grid, os.path.join(img_dir, f"e{ep:03d}_s{step:06d}.png"))
                step += 1
                bar.update(0)

class DictWrapper(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.base = base_ds
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        x, y, mask = self.base[idx]
        return {"x": x, "y": y, "mask": mask}
