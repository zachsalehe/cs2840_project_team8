# train_i3sb_mnist.py
import torch
import time
from i3sb import train_i3sb, I3SBConfig
from mnist_dataset import MNISTDataset

if __name__ == "__main__":
    # Force CPU for MPS issues - will be slower but stable
    device = "cpu"
    
    cfg = I3SBConfig(
        # SDE parameters - reduced for CPU
        sigma=0.5,           
        n_steps=10,          # Reduce to 10 for CPU speed
        
        # Reference drift
        ref_mode="zero",
        ref_lam=1.0,
        
        # Endpoint matching weights
        lambda_forward=1.0,
        lambda_backward=1.0,
        
        # Optimization
        lr_phi=2e-4,
        lr_theta=2e-4,
        
        # Logging
        device=device,
        log_dir="./runs_i3sb_mnist"
    )
    
    start_time = time.time()
    trainer = train_i3sb(
        MNISTDataset,
        epochs=5,
        batch_size=32,  # Smaller batch for CPU
        out_dir=cfg.log_dir,
        cfg=cfg,
        log_every=10,  # More frequent logging
        vis_every=500,
        save_every=1000,
        keep_last=3,
        save_best=True,
        num_workers=0,
        pin_memory=False,  # No pinning for CPU
    )
    
    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed/3600:.2f} hours")