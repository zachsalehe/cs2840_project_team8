# train_i3sb_mnist.py
import torch
from i3sb import train_i3sb, I3SBConfig
from mnist_dataset import MNISTDataset

if __name__ == "__main__":
    cfg = I3SBConfig(
        # SDE parameters
        sigma=0.1,
        n_steps=20,
        
        # Reference drift
        ref_mode="zero",
        ref_lam=1.0,
        
        # Endpoint matching weights (balanced MSE, not unbalanced Sinkhorn)
        lambda_forward=1.0,
        lambda_backward=1.0,
        
        # Optimization
        lr_phi=2e-4,
        lr_theta=2e-4,
        
        # Logging
        log_dir="./runs_i3sb_mnist"
    )
    
    trainer = train_i3sb(
        MNISTDataset,
        epochs=5,
        batch_size=64,
        out_dir=cfg.log_dir,
        cfg=cfg,
        log_every=100,
        vis_every=500,
        save_every=1000,
        keep_last=3,
        save_best=True,
        num_workers=0,
        pin_memory=True
    )