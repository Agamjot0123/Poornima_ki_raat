"""
Trainer — Training loop for AsteroMesh with logging and checkpointing.

Config-driven via YAML. Supports cosine annealing LR scheduling,
gradient clipping, and checkpoint saving/loading.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import yaml
import os
import time
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from ..models.asteromesh import AsteroMesh
from ..data.dataset import AsteroidDataset
from .losses import CompositeLoss


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class Trainer:
    """Training manager for AsteroMesh.
    
    Handles:
        - Model instantiation from config
        - Training loop with loss computation
        - Learning rate scheduling
        - Checkpoint saving/loading
        - Logging
    """
    
    def __init__(self, config: dict, device: str = None):
        self.config = config
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Model
        model_cfg = config.get('model', {})
        self.model = AsteroMesh(
            light_curve_length=model_cfg.get('light_curve_length', 512),
            radar_image_size=model_cfg.get('radar_image_size', 224),
            encoder_dim=model_cfg.get('encoder_dim', 256),
            fused_dim=model_cfg.get('fused_dim', 512),
            max_degree=model_cfg.get('spharm_degree', 25),
            mesh_resolution=model_cfg.get('mesh_resolution', 100),
        ).to(self.device)
        
        # Print parameter counts
        param_counts = self.model.count_parameters()
        print(f"Model parameters:")
        for name, count in param_counts.items():
            print(f"  {name}: {count:,}")
        
        # Loss
        loss_cfg = config.get('loss', {})
        self.criterion = CompositeLoss(
            mse_weight=loss_cfg.get('mse_weight', 1.0),
            chamfer_weight=loss_cfg.get('chamfer_weight', 0.0),
            spharm_reg_weight=loss_cfg.get('spharm_reg_weight', 0.01),
            max_degree=model_cfg.get('spharm_degree', 25),
        ).to(self.device)
        
        # Optimiser
        train_cfg = config.get('training', {})
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_cfg.get('learning_rate', 0.001),
            weight_decay=train_cfg.get('weight_decay', 0.0001),
        )
        
        # Scheduler
        self.epochs = train_cfg.get('epochs', 100)
        scheduler_type = train_cfg.get('lr_scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=1e-6
            )
        else:
            self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        
        # Checkpointing
        self.checkpoint_dir = Path(train_cfg.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float('inf')
        
        # Logging
        self.log_interval = train_cfg.get('log_interval', 10)
        self.history = {'train_loss': [], 'val_loss': []}
    
    def create_dataloaders(self) -> tuple:
        """Create train and validation dataloaders from config."""
        data_cfg = self.config.get('data', {})
        model_cfg = self.config.get('model', {})
        train_cfg = self.config.get('training', {})
        
        train_dataset = AsteroidDataset(
            data_dir=data_cfg.get('train_dir', 'data/synthetic/train'),
            mode='both',
            light_curve_length=model_cfg.get('light_curve_length', 512),
            radar_image_size=model_cfg.get('radar_image_size', 224),
            num_coefficients=model_cfg.get('num_coefficients', 1352),
            augment=True,
        )
        
        val_dataset = AsteroidDataset(
            data_dir=data_cfg.get('val_dir', 'data/synthetic/val'),
            mode='both',
            light_curve_length=model_cfg.get('light_curve_length', 512),
            radar_image_size=model_cfg.get('radar_image_size', 224),
            num_coefficients=model_cfg.get('num_coefficients', 1352),
            augment=False,
        )
        
        batch_size = train_cfg.get('batch_size', 16)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Run one training epoch."""
        self.model.train()
        epoch_losses = {'total': 0, 'mse': 0, 'regularisation': 0}
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training", leave=False):
            light_curve = batch['light_curve'].to(self.device)
            radar_image = batch['radar_image'].to(self.device)
            target_coeffs = batch['spharm_coefficients'].to(self.device)
            
            # Forward pass
            pred_coeffs = self.model(light_curve, radar_image)
            
            # Compute loss
            losses = self.criterion(pred_coeffs, target_coeffs)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            num_batches += 1
        
        # Average
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        """Run validation."""
        self.model.eval()
        epoch_losses = {'total': 0, 'mse': 0, 'regularisation': 0}
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            light_curve = batch['light_curve'].to(self.device)
            radar_image = batch['radar_image'].to(self.device)
            target_coeffs = batch['spharm_coefficients'].to(self.device)
            
            pred_coeffs = self.model(light_curve, radar_image)
            losses = self.criterion(pred_coeffs, target_coeffs)
            
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            num_batches += 1
        
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        return epoch_losses
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config,
        }
        
        # Save latest
        path = self.checkpoint_dir / 'latest_model.pth'
        torch.save(checkpoint, str(path))
        
        # Save best
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, str(path))
            print(f"  ★ Saved best model (loss: {loss:.6f})")
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', '?')})")
        return checkpoint.get('epoch', 0)
    
    def train(self, resume_from: str = None):
        """Full training loop."""
        train_loader, val_loader = self.create_dataloaders()
        
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from)
        
        print(f"\nStarting training for {self.epochs} epochs...")
        print(f"{'='*60}")
        
        for epoch in range(start_epoch, self.epochs):
            start_time = time.time()
            
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            val_losses = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Track history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            
            # Check for best model
            is_best = val_losses['total'] < self.best_loss
            if is_best:
                self.best_loss = val_losses['total']
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_losses['total'], is_best)
            
            # Log
            elapsed = time.time() - start_time
            lr = self.optimizer.param_groups[0]['lr']
            
            if (epoch + 1) % self.log_interval == 0 or epoch == 0 or is_best:
                print(f"Epoch {epoch+1}/{self.epochs} "
                      f"| Train: {train_losses['total']:.6f} "
                      f"| Val: {val_losses['total']:.6f} "
                      f"| LR: {lr:.2e} "
                      f"| Time: {elapsed:.1f}s"
                      f"{' ★' if is_best else ''}")
        
        print(f"{'='*60}")
        print(f"Training complete. Best val loss: {self.best_loss:.6f}")
        
        return self.history


def main():
    """Entry point for training via command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AsteroMesh')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    config = load_config(args.config)
    trainer = Trainer(config, device=args.device)
    trainer.train(resume_from=args.resume)


if __name__ == '__main__':
    main()
