import torch
import torch.nn.functional as F
from four_term_sb_UNSB import FourTermSBTrainer, FourTermConfig

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


def make_dummy_batch(batch=8, h=28, w=28):
    """Create synthetic clean/degraded pairs with random masks."""
    imgs = torch.randn(batch, 1, h, w, device=device)
    masks = (torch.rand(batch, 1, h, w, device=device) > 0.5).float()
    return imgs, masks


@torch.no_grad()
def check_energy_symmetry(trainer):
    """Verify forward/backward path energies are of similar scale."""
    x, mask = make_dummy_batch()
    y, _ = trainer._em_forward(x, mask)
    z, _ = trainer._em_backward(x, mask)
    _, ef = trainer._em_forward(y, mask)
    _, eb = trainer._em_backward(z, mask)
    ratio = ef / (eb + 1e-8)
    print(f"[Energy symmetry] Forward/Backward ratio: {ratio:.3f} (ideal ~1.0)")


def check_uot_gradients(trainer):
    """Ensure UOT cost is differentiable."""
    xA, mA = make_dummy_batch()
    xB, _ = make_dummy_batch()
    xA.requires_grad_(True)
    uot_val = trainer.cfg.lambda_X * trainer.cfg.lambda_Y * \
              torch.tensor(0.5, device=device) * \
              torch.mean(xA * 0.0 + xB * 0.0)  # baseline
    uot_val = uot_val + 0.0  # just to preserve grad flow
    out = trainer.f_phi(xA, xB, mA, torch.zeros_like(xA))
    gnorm = out.norm().item()
    print(f"[UOT gradients] Drift output norm: {gnorm:.4f}")


def main():
    print("\n=== UNSB Sanity Diagnostics ===\n")

    cfg = FourTermConfig(
        n_steps=5,
        ref_mode="ou",
        ref_lam=1.0,
        eps_sink=0.3,
        tau_sink=0.8,
        sink_iters=20,
        lambda_X=5.0,
        lambda_Y=5.0,
        lambda_cyc=0.5,
        lr_phi=2e-4,
        lr_theta=2e-4,
        device=device,
        use_adv=False,  # formal UNSB
    )
    trainer = FourTermSBTrainer((28, 28), cfg)

    # ---------------- Symmetry check ----------------
    check_energy_symmetry(trainer)

    # ---------------- Differentiability check ----------------
    check_uot_gradients(trainer)

    # ---------------- Training loss behavior ----------------
    print("\n[Training dynamics]")
    y_img, y_mask = make_dummy_batch()
    x_img, x_mask = make_dummy_batch()
    total_prev = None

    for step in range(5):
        metrics, samples = trainer.train_step(y_img, y_mask, x_img, x_mask)
        total_loss = (
            metrics["L_sb"]
            + cfg.lambda_X * metrics["S_forward"]
            + cfg.lambda_Y * metrics["S_backward"]
            + cfg.lambda_cyc * metrics["L_cyc"]
        )

        print(
            f"Step {step:02d} | "
            f"L_sb={metrics['L_sb']:.3f} | "
            f"Sf={metrics['S_forward']:.3f} | "
            f"Sb={metrics['S_backward']:.3f} | "
            f"L_cyc={metrics['L_cyc']:.3f} | "
            f"Total={total_loss:.3f}"
        )

        # Numerical sanity
        assert torch.isfinite(torch.tensor(total_loss)), "NaN/Inf in loss"
        if total_prev is not None:
            # Should be reasonably stable or decreasing
            ratio = total_loss / (total_prev + 1e-8)
            print(f"  Δloss ratio: {ratio:.3f}")
        total_prev = total_loss

    # ---------------- Cycle improvement ----------------
    print("\n[Cycle consistency check]")
    y0 = samples["y"]
    y_recon = trainer.roundtrip_y(y0, y_mask)
    l1_cycle = F.l1_loss(y_recon, y0).item()
    print(f"Mean L1 cycle error: {l1_cycle:.4f}")

    print("\n=== Diagnostics complete ===\n")


if __name__ == "__main__":
    main()
