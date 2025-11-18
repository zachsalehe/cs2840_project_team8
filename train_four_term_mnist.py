# train_four_term_mnist.py
import torch
from four_term_sb import train_four_term_sb, FourTermConfig
from mnist_dataset import MNISTDataset  # <-- where your class lives

if __name__ == "__main__":
    # cfg = FourTermConfig(
    #     sigma=0.08, n_steps=20,
    #     ref_mode="zero", ref_lam=1.0,
    #     eps_sink=0.1, tau_sink=0.8, sink_iters=50,
    #     lambda_cyc=1.0, lambda_X=5.0, lambda_Y=5.0,
    #     lr_phi=2e-4, lr_theta=2e-4,
    #     log_dir="./runs_fourterm_mnist"
    # )
#     cfg = FourTermConfig(
#     sigma=0.15,          # ↑ reduces 1/(2σ²) to ~22.2 (from 78.1)
#     n_steps=30,          # slightly finer EM
#     ref_mode="ou", ref_lam=1.0,   # OU anchor helps conditioning (optional but useful)

#     eps_sink=0.35,       # smoother kernel
#     tau_sink=0.8,
#     sink_iters=25,       # fewer iters once eps is larger

#     lambda_X=10.0,       # less endpoint weight until dynamics move
#     lambda_Y=10.0,
#     lambda_cyc=0.5,      # lighter cycle early

#     lr_phi=2e-4, lr_theta=2e-4,
#     log_dir="./runs_fourterm_mnist"
# )
    cfg = FourTermConfig(
        sigma=0.5,            # weaker SB weight
        n_steps=20,

        ref_mode="zero",      # OU later if you like, but start zero
        #lambda_SB=0.1,        # or 0.0 for the first 1–2 epochs

        eps_sink=0.08,        # sharper transport on masked cost
        tau_sink=0.8,
        sink_iters=80,

        lambda_X=5.0,         # endpoint signals balanced with SB
        lambda_Y=5.0,
        lambda_cyc=1.0,       # give the cycle some say
    )
    trainer = train_four_term_sb(MNISTDataset,
                                 epochs=5,
                                 batch_size=64,
                                 out_dir=cfg.log_dir,
                                 cfg=cfg)