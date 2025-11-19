import torch
from four_term_sb import train_four_term_sb, FourTermConfig
from mnist_dataset import MNISTDataset

if __name__ == "__main__":
    cfg = FourTermConfig(
        sigma=0.5,
        n_steps=20,

        ref_mode="zero",

        eps_sink=0.08,
        sink_iters=80,

        lambda_X=5.0,
        lambda_Y=5.0,
        lambda_cyc=0.0,     # I2SB: NO cycle loss
    )

    trainer = train_four_term_sb(
        MNISTDataset,
        epochs=5,
        batch_size=64,
        out_dir=cfg.log_dir,
        cfg=cfg
    )