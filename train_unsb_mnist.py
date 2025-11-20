import torch
from four_term_sb_UNSB import train_four_term_sb, FourTermConfig
from mnist_dataset import MNISTDataset  # <-- where your class lives

if __name__ == "__main__":
    cfg = FourTermConfig(
        n_steps=30,          # slightly finer EM
        ref_mode="ou", ref_lam=1.0,   # OU anchor helps conditioning (optional)
        eps_sink=0.35,       # smoother kernel
        tau_sink=0.8,
        sink_iters=25,       # fewer iters once eps is larger
        lambda_X=10.0,       # less endpoint weight until dynamics move
        lambda_Y=10.0,
        lambda_cyc=0.5,      # lighter cycle early
        lr_phi=2e-4, lr_theta=2e-4,
        log_dir="./runs_unsb_mnist"
    )

    trainer = train_four_term_sb(
        MNISTDataset,
        epochs=5,
        batch_size=64,
        out_dir=cfg.log_dir,
        cfg=cfg
    )
