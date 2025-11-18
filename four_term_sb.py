# four_term_sb.py
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

# def pairwise_mean_sq_cost(x: torch.Tensor, y: torch.Tensor):
#     """
#     x: [B, C, H, W], y: [B, C, H, W]  (empirical batches)
#     Returns C matrix [B, B] with mean squared pixelwise distances (range ~ [0,4]).
#     """
#     B = x.size(0)
#     xf = x.view(B, -1)
#     yf = y.view(B, -1)
#     # torch.cdist returns Euclidean distance; square and mean over dim
#     d2 = torch.cdist(xf, yf, p=2) ** 2  # [B,B]
#     d2 = d2 / xf.size(1)                # mean over pixels -> scale ~[0,4]
#     return d2
# def pairwise_mean_sq_cost_masked(xA, xB, mask):
#     # xA,xB: [B,C,H,W], mask: [B,1,H,W] with 1 = hole
#     B = xA.size(0)
#     m = mask.expand_as(xA).reshape(B, -1).float()
#     a = (xA.reshape(B, -1) * m)
#     b = (xB.reshape(B, -1) * m)
#     # normalize per-sample to get a *mean* over active pixels
#     denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
#     a = a / denom.sqrt()
#     b = b / denom.sqrt()
#     # squared Euclidean pairwise distances
#     # (||a||^2 + ||b||^2 - 2 a b^T), but torch.cdist is fine
#     return torch.cdist(a, b, p=2).pow(2)
# -------------------------
# Unbalanced Sinkhorn (UOT) + debiased Sinkhorn divergence
# -------------------------
# def unbalanced_sinkhorn_uot(a: torch.Tensor,
#                             b: torch.Tensor,
#                             C: torch.Tensor,
#                             eps: float,
#                             tau: float,
#                             n_iters: int = 50) -> Dict[str, torch.Tensor]:
#     """
#     Compute the *unbalanced* entropic OT objective:
#       UOT_{eps, tau}(a,b) = <C, gamma> + eps KL(gamma || a ⊗ b) + tau[ KL(gamma 1 || a) + KL(gamma^T 1 || b) ]
#     using log-domain Sinkhorn with exponent alpha = tau / (tau + eps).
#     Returns dict with:
#       - 'uot': primal value (differentiable w.r.t. C)
#       - 'gamma': transport plan (for debugging/visualization)
#     Shapes:
#       a,b: [B] with a.sum()=b.sum()=1 (empirical masses)
#       C: [B,B] mean-sq ground cost (nonnegative)
#     """
#     device = C.device
#     B = C.size(0)
#     assert C.size(0) == C.size(1) == a.numel() == b.numel()

#     alpha = tau / (tau + eps)
#     log_a = torch.log(a + EPS)
#     log_b = torch.log(b + EPS)
#     logK = -C / eps  # [B,B]

#     # Initialize in log domain
#     logu = torch.zeros(B, device=device)
#     logv = torch.zeros(B, device=device)

#     for _ in range(n_iters):
#         # logu <- alpha * (log a - logsumexp(logK + logv))
#         logu = alpha * (log_a - torch.logsumexp(logK + logv[None, :], dim=1))
#         # logv <- alpha * (log b - logsumexp(logK^T + logu))
#         logv = alpha * (log_b - torch.logsumexp(logK.t() + logu[None, :], dim=1))

#     # Recover gamma in log domain then exponentiate
#     logGamma = logu[:, None] + logK + logv[None, :]
#     gamma = torch.exp(logGamma)  # [B,B], may have total mass != 1 (unbalanced)

#     # Marginals
#     m = gamma.sum(dim=1)  # [B]
#     n = gamma.sum(dim=0)  # [B]

#     # KL( gamma || a ⊗ b )
#     ab = torch.outer(a, b) + EPS
#     kl_g_ab = (gamma * (torch.log(gamma + EPS) - torch.log(ab))).sum() - gamma.sum() + ab.sum()

#     # KL( m || a ) + KL( n || b )
#     kl_m_a = (m * (torch.log(m + EPS) - log_a)).sum() - m.sum() + a.sum()
#     kl_n_b = (n * (torch.log(n + EPS) - log_b)).sum() - n.sum() + b.sum()

#     # Primal cost
#     cost_term = (gamma * C).sum()
#     uot_val = cost_term + eps * kl_g_ab + tau * (kl_m_a + kl_n_b)

#     return {"uot": uot_val, "gamma": gamma.detach()}
def pairwise_masked_mean_sq_cost(xA, xB, maskA):
    B = xA.size(0)
    XA, XB = xA.view(B,-1), xB.view(B,-1)
    W = maskA.view(B,-1)
    wsum = W.sum(dim=1).clamp_min(1.0)

    XA2w  = (W * XA**2).sum(dim=1)[:, None]
    XB2_T = (XB**2).t()
    term2 = W @ XB2_T
    term3 = 2.0 * ((W * XA) @ XB.t())
    return (XA2w + term2 - term3).div(wsum[:, None]).clamp_min(0.0)

def uot_only_masked(xA, xB, maskA, eps, tau, n_iters=50):
    B = xA.size(0)
    a = b = torch.full((B,), 1.0/B, device=xA.device)
    C = pairwise_masked_mean_sq_cost(xA, xB, maskA)
    return unbalanced_sinkhorn_uot(a, b, C, eps, tau, n_iters)["uot"]

def unbalanced_sinkhorn_uot(a: torch.Tensor,
                            b: torch.Tensor,
                            C: torch.Tensor,
                            eps: float,
                            tau: float,
                            n_iters: int = 50) -> Dict[str, torch.Tensor]:
    """
    Stable unbalanced Sinkhorn in the *normal* domain:
      u <- (a / (K v))^alpha
      v <- (b / (K^T u))^alpha
    with alpha = tau / (tau + eps), K = exp(-C/eps).
    Returns {'uot': value, 'gamma': plan (detached)}.
    """
    device = C.device
    B = C.size(0)
    assert C.size(1) == B and a.numel() == B and b.numel() == B

    # Kernel
    K = torch.exp(-C / eps).clamp_min(1e-38)   # avoid exact 0 in fp32
    alpha = tau / (tau + eps + 1e-12)

    # Initialize positive scalings
    u = torch.ones(B, device=device)
    v = torch.ones(B, device=device)

    for _ in range(n_iters):
        Kv  = K @ v
        Kv  = Kv.clamp_min(1e-30)              # avoid divide-by-zero
        u   = (a / Kv).clamp_min(1e-30).pow(alpha)

        Ktu = K.t() @ u
        Ktu = Ktu.clamp_min(1e-30)
        v   = (b / Ktu).clamp_min(1e-30).pow(alpha)

    # Transport plan
    gamma = (u[:, None] * K) * v[None, :]      # [B,B]
    gamma_det = gamma.detach()

    # Marginals
    m = gamma.sum(dim=1)                       # [B]
    n = gamma.sum(dim=0)                       # [B]

    # Primal value terms
    # <C, gamma>
    cost_term = (gamma * C).sum()

    # KL(gamma || a ⊗ b)
    ab = torch.outer(a, b).clamp_min(1e-30)
    kl_g_ab = (gamma * (torch.log(gamma.clamp_min(1e-30)) - torch.log(ab))).sum() - gamma.sum() + ab.sum()

    # KL(m || a) + KL(n || b)
    kl_m_a = (m * (torch.log(m.clamp_min(1e-30)) - torch.log(a.clamp_min(1e-30)))).sum() - m.sum() + a.sum()
    kl_n_b = (n * (torch.log(n.clamp_min(1e-30)) - torch.log(b.clamp_min(1e-30)))).sum() - n.sum() + b.sum()

    uot_val = cost_term + eps * kl_g_ab + tau * (kl_m_a + kl_n_b)
    return {"uot": uot_val, "gamma": gamma_det}
def sinkhorn_divergence_unbalanced(xA: torch.Tensor,
                                   xB: torch.Tensor,
                                   eps: float,
                                   tau: float,
                                   n_iters: int = 50) -> torch.Tensor:
    """
    Debiased unbalanced Sinkhorn divergence:
       S_{eps,tau}(A,B) = UOT(A,B) - 0.5 UOT(A,A) - 0.5 UOT(B,B)
    xA, xB: [B,1,H,W] samples (grad flows into xA and xB)
    """
    B = xA.size(0)
    a = torch.full((B,), 1.0 / B, device=xA.device)
    b = torch.full((B,), 1.0 / B, device=xB.device)

    C_AB = pairwise_mean_sq_cost_masked(xA, xB)
    C_AA = pairwise_mean_sq_cost_masked(xA, xA)
    C_BB = pairwise_mean_sq_cost_masked(xB, xB)

    uot_AB = unbalanced_sinkhorn_uot(a, b, C_AB, eps, tau, n_iters)["uot"]
    uot_AA = unbalanced_sinkhorn_uot(a, a, C_AA, eps, tau, n_iters)["uot"]
    uot_BB = unbalanced_sinkhorn_uot(b, b, C_BB, eps, tau, n_iters)["uot"]

    return uot_AB - 0.5 * (uot_AA + uot_BB)

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
    # Sinkhorn (unbalanced)
    eps_sink: float = 0.1   # entropic reg (on mean-sq cost ~ [0,4])
    tau_sink: float = 0.5   # marginal relaxation; tau -> inf recovers balanced
    sink_iters: int = 50
    # Cycle loss
    lambda_cyc: float = 1.0
    # Endpoint weights
    lambda_X: float = 10.0
    lambda_Y: float = 10.0
    # Optim
    lr_phi: float = 2e-4
    lr_theta: float = 2e-4
    # Misc
    device: str = "mps" #if torch.cuda.is_available() else "cpu"
    log_dir: str = "./runs_fourterm"

# -------------------------
# Trainer implementing Eq. (1)
# -------------------------
class FourTermSBTrainer:
    """
    Implements:
      L(phi,theta) = L_SB + lambda_X S_{eps,tau}(hat p_phi^{(1)}, p_X)
                           + lambda_Y S_{eps,tau}(hat p_theta^{(0)}, p_Y)
                           + lambda_cyc L_cycle
    with:
      - forward bridge:   dX_t^f = f_phi(X_t^f, t; y) dt + sigma dW_t, X_0^f = y
      - backward bridge:  dX_t^b = f_theta^b(X_t^b, t; x) dt + sigma dW_t, X_1^b = x
        (simulated via s = 0..1 with t=1-s, i.e. X_0^b recovered at s=1)
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
            # Euler–Maruyama step
            # noise = torch.randn_like(x) * (self.cfg.sigma * math.sqrt(dt))
            # x = x + drift * dt + noise
            noise = torch.randn_like(x) * (self.cfg.sigma * math.sqrt(dt))
            new_x = x + drift * dt + noise
            #x = new_x * mask + y * (1 - mask) 
            m = mask.bool()                      # ensure boolean semantics once
            x = torch.where(m, new_x, y)         # masked -> new_x, known -> y
        return x, energy  # x is X_1^f

    def _em_backward(self, x_clean: torch.Tensor, mask: torch.Tensor):
        """
        Simulate X^b with terminal condition X_1^b = x_clean using s in [0,1]:
          Y_s := X^b_{1-s},  dY_s = -f_theta(Y_s, 1-s; x) ds + sigma dW_s,  Y_0 = x
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
            #noise = torch.randn_like(y) * (self.cfg.sigma * math.sqrt(dt))
            # Note the minus sign for reversed clock
            noise = torch.randn_like(y) * (self.cfg.sigma * math.sqrt(dt))
            new_y = y + (-drift_t) * dt + noise
            #y = new_y * mask + x_clean * (1 - mask)
            m = mask.bool()
            y = torch.where(m, new_y, x_clean)
            #y = y + (-drift_t) * dt + noise
        X0_b = y
        return X0_b, energy

    # ---------- Round-trip maps with fresh noise ----------
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
        if lambda_cyc is None:
            lambda_cyc = self.cfg.lambda_cyc

        # --- Build degraded y from clean images and masks (your p_Y construct) ---
        # For training, we treat 'batch_y_img' as the *clean* image source to build y.
        # Create y by inserting uniform noise in the masked region; keep unmasked pixels.
        # noise = torch.empty_like(batch_y_img).uniform_(-1.0, 1.0)
        # y = batch_y_img * (1.0 - batch_y_mask) + noise * batch_y_mask    # degraded p_Y sample

        # x = batch_x_img                                                     # clean p_X sample
        # my = batch_y_mask
        # mx = batch_x_mask  # used as conditioning; you can share the same mask dist for both branches
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

        # --- Endpoint alignment via *unbalanced* Sinkhorn (Terms 2 & 3) ---
        #S_forward  = sinkhorn_divergence_unbalanced(X1_f, x.detach(), self.cfg.eps_sink, self.cfg.tau_sink, self.cfg.sink_iters)
        #S_backward = sinkhorn_divergence_unbalanced(X0_b, y.detach(), self.cfg.eps_sink, self.cfg.tau_sink, self.cfg.sink_iters)
        S_forward  = uot_only_masked(X1_f, x.detach(),  my, self.cfg.eps_sink, self.cfg.tau_sink, self.cfg.sink_iters)
        S_backward = uot_only_masked(X0_b, y.detach(),  mx, self.cfg.eps_sink, self.cfg.tau_sink, self.cfg.sink_iters)

        # --- Cycle maps with fresh noise in each leg (Eq. (4)) ---
        # with torch.no_grad():
        #     # We explicitly prevent gradient from flowing across legs (fresh noise + stopgrad),
        #     # to implement the stochastic round-trip as stated.
        #     tilde_y = self.roundtrip_y(y, my)   # ~y_{\phi,theta}(y)
        #     tilde_x = self.roundtrip_x(x, mx)   # ~x_{\phi,theta}(x)
        # # simple L1 cycle; you can add perceptual loss on top if desired
        # L_cyc = F.l1_loss(tilde_y, y) + F.l1_loss(tilde_x, x)
         # --- Cycle with per‑leg gradients (theta trained by y‑cycle, phi by x‑cycle) ---
        # NOTE: reuse X1_f and X0_b we already computed above to save time.
        # y‑cycle: stop grad on forward terminal -> train only theta with backward leg.

        tilde_y, _ = self._em_backward(X1_f.detach(), my)   # grads -> f_theta only
        # L_cyc_y = F.l1_loss(tilde_y, y)
        # # x‑cycle: stop grad on backward initial -> train only phi with forward leg.
        tilde_x, _ = self._em_forward(X0_b.detach(), mx)    # grads -> f_phi only
        # L_cyc_x = F.l1_loss(tilde_x, x)
        # L_cyc = L_cyc_y + L_cyc_x

        rho = 0.7  # small weight on masked region
        L_cyc_y = (F.l1_loss(tilde_y*(1-my), y*(1-my)) +
                rho * F.l1_loss(tilde_y*my,   y*my))
        L_cyc_x = F.l1_loss(tilde_x, x)
        L_cyc   = L_cyc_y + L_cyc_x

        # --- Overall objective (Eq. (1)) ---
        loss = L_sb + self.cfg.lambda_X * S_forward + self.cfg.lambda_Y * S_backward + lambda_cyc * L_cyc

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
                "L_cyc": L_cyc.item(),
            }

        return metrics, {"X1_f": X1_f.detach(), "X0_b": X0_b.detach(), "y": y.detach(), "x": x.detach()}

# -------------------------
# Convenience training loop (MNIST masking)
# # -------------------------
# def train_four_term_sb(dataset_cls,
#                        epochs: int = 3,
#                        batch_size: int = 64,
#                        out_dir: str = "./runs_fourterm",
#                        cfg: Optional[FourTermConfig] = None):
#     """
#     dataset_cls: your MNISTDataset. We build two unpaired loaders from it.
#     """
#     if cfg is None:
#         cfg = FourTermConfig(log_dir=out_dir)
#     os.makedirs(out_dir, exist_ok=True)

#     # Build two *independent* loaders (unpaired sampling of x and y)
#     dsY = dataset_cls(split="train")  # we'll construct y by masking dsY.img with dsY.masks
#     dsX = dataset_cls(split="train")  # for clean x

#     H, W = dsY.h, dsY.w
#     dlY = DataLoader(dsY, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
#     dlX = DataLoader(dsX, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
#     iterX = iter(dlX)

#     trainer = FourTermSBTrainer(img_shape=(H, W), cfg=cfg)

#     step = 0
#     for epoch in range(epochs):
#         for imgY, maskY in dlY:
#             try:
#                 imgX, maskX = next(iterX)
#             except StopIteration:
#                 iterX = iter(dlX)
#                 imgX, maskX = next(iterX)

#             # To device
#             imgY = imgY.to(cfg.device)
#             imgX = imgX.to(cfg.device)
#             maskY = maskY.to(cfg.device)
#             maskX = maskX.to(cfg.device)

#             metrics, samples = trainer.train_step(
#                 batch_y_img=imgY, batch_y_mask=maskY,
#                 batch_x_img=imgX, batch_x_mask=maskX
#             )

#             if step % 100 == 0:
#                 msg = " | ".join([f"{k}:{v:.4f}" for k, v in metrics.items()])
#                 print(f"[e{epoch:02d} s{step:05d}] {msg}")

#             if step % 500 == 0:
#                 with torch.no_grad():
#                     grid = make_grid(torch.cat([
#                         samples["x"][:8],                 # real clean
#                         samples["X0_b"][:8],              # backward initial (should look degraded)
#                         samples["y"][:8],                 # real degraded
#                         samples["X1_f"][:8],              # forward terminal (should look clean)
#                     ], dim=0), nrow=8, padding=2, normalize=True, value_range=(-1, 1))
#                     save_image(grid, os.path.join(out_dir, f"e{epoch:03d}_s{step:06d}.png"))
#             step += 1

#     return trainer

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
    Train the four-term SB objective on unpaired clean/degraded datasets produced by `dataset_cls`.

    dataset_cls: a class whose __getitem__ returns (img, mask) with img in [-1,1] and mask in {0,1}
    epochs:      number of passes over the Y-loader (X-loader is cycled independently)
    batch_size:  mini-batch size (beware: Sinkhorn is O(B^2))
    out_dir:     root directory for logs, images, and checkpoints

    Logging / I/O:
      log_every  : print metrics every N steps
      vis_every  : save forward/backward/cycle grids every N steps

    Checkpointing:
      save_every : save a checkpoint every N steps (also at step 0)
      keep_last  : keep the most recent K periodic checkpoints (rotating)
      resume_from: path to a previous checkpoint (.pt) to resume from
      save_best  : track and save a 'best.pt' with smallest (S_forward + S_backward)

    DataLoader:
      num_workers: set to >0 for speed once your Dataset is picklable
      pin_memory : recommended True when training on GPU
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

                    # Cycle grids (inference-mode fresh noise)
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