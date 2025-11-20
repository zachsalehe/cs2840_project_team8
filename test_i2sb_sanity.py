# i2sb_sanity_check.py
import torch
import torch.nn.functional as F

# Your file exposes these names (per the snippet you shared)
from i2sb import FourTermSBTrainer as I2SBTrainer
from i2sb import FourTermConfig as I2SBConfig

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_dummy_batch(batch=8, h=28, w=28, device=DEVICE):
    imgs = torch.randn(batch, 1, h, w, device=device)
    masks = (torch.rand(batch, 1, h, w, device=device) > 0.5).float()
    return imgs, masks


@torch.no_grad()
def check_energy_symmetry(trainer):
    """Forward/backward path energies should be same order of magnitude."""
    x, m = make_dummy_batch()
    _, ef = trainer._em_forward(x, m)
    _, eb = trainer._em_backward(x, m)
    ratio = float(ef / (eb + 1e-8))
    print(f"[Energy symmetry] Forward/Backward ratio: {ratio:.3f} (ideal ~1.0)")


def grad_norms(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm().item()
    return total


def main():
    print("\n=== I2SB Sanity Diagnostics ===\n")

    cfg = I2SBConfig(
        n_steps=5,
        sigma=0.1,
        ref_mode="ou", ref_lam=1.0,   # OU anchor helps early stability
        eps_sink=0.3,                 # entropic reg for balanced OT
        sink_iters=20,
        lambda_X=5.0, lambda_Y=5.0,   # endpoint weights
        lambda_cyc=0.0,               # I2SB has no cycle term
        lr_phi=2e-4, lr_theta=2e-4,
        device=DEVICE,
    )
    trainer = I2SBTrainer((28, 28), cfg)

    # 1) Energy symmetry
    check_energy_symmetry(trainer)

    # 2) Short training dynamics with gradient stats
    print("\n[Training dynamics]")
    y_img, y_mask = make_dummy_batch()
    x_img, x_mask = make_dummy_batch()
    prev_total = None

    for step in range(5):
        metrics, samples = trainer.train_step(
            batch_y_img=y_img, batch_y_mask=y_mask,
            batch_x_img=x_img, batch_x_mask=x_mask
        )

        # I2SB total = L_SB + λ_X S_fwd + λ_Y S_bwd
        total = (
            metrics["L_SB"]
            + cfg.lambda_X * metrics["S_forward"]
            + cfg.lambda_Y * metrics["S_backward"]
        )

        # After the step, inspect gradient norms from last backward()
        g_phi = grad_norms(trainer.f_phi)
        g_theta = grad_norms(trainer.f_theta)

        print(
            f"Step {step:02d} | "
            f"L_SB={metrics['L_SB']:.3f} | "
            f"Sf={metrics['S_forward']:.3f} | "
            f"Sb={metrics['S_backward']:.3f} | "
            f"Total={total:.3f} | "
            f"‖∇φ‖={g_phi:.2e} | ‖∇θ‖={g_theta:.2e}"
        )

        # Numerical sanity
        assert torch.isfinite(torch.tensor(total)), "NaN/Inf in total loss"
        if prev_total is not None:
            ratio = total / (prev_total + 1e-8)
            print(f"  Δloss ratio: {ratio:.3f}")
        prev_total = total

    # 3) Optional round-trip diagnostic (I2SB keeps these helpers)
    with torch.no_grad():
        y0 = samples["y"]
        y_recon = trainer.roundtrip_y(y0, y_mask)
        l1_cycle = F.l1_loss(y_recon, y0).item()
        print(f"\n[Round-trip diagnostic] Mean L1(ŷ, y): {l1_cycle:.4f}")

    print("\n=== Diagnostics complete ===\n")


if __name__ == "__main__":
    # Use -u when running if your terminal buffers output: python -u test_i2sb_sanity.py
    main()
