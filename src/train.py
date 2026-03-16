import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import logging
from pathlib import Path

from data.dataset import PhysicalAsteroidDataset
from models.asteromesh import AsteroMesh
from training.losses import CompositeLoss

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def train_model(data_dir: str = "data/simulated",
                epochs: int = 50,
                batch_size: int = 16,
                lr: float = 1e-4,
                val_split: float = 0.2):
    """
    Trains the Dual-Stream AsteroMesh Pipeline on physical simulated dataset.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Dataset & DataLoaders
    full_dataset = PhysicalAsteroidDataset(data_dir=data_dir, augment=True)
    
    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Disable augmentations on validation set
    val_dataset.dataset.augment = False
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Model Initialization
    # Disable pretrained ImageNet weights for speed/demonstration if needed
    model = AsteroMesh(pretrained_backbone=True).to(device)
    logger.info(f"Model Initialized. Total parameters: {model.count_parameters()['total']:,}")
    
    # 3. Optimizer, Scheduler & Loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = CompositeLoss(mse_weight=1.0, spharm_reg_weight=0.01).to(device)
    
    best_val_loss = float('inf')
    Path('checkpoints').mkdir(exist_ok=True)
    
    # 4. Training Loop
    logger.info(f"Starting Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in train_pbar:
            # Move to device
            lc = batch['light_curve'].to(device)
            radar = batch['radar_image'].to(device)
            coeffs = batch['spharm_coefficients'].to(device)
            radar_coords = batch['radar_coords'].to(device)
            lc_phases = batch['lc_phases'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            preds = model(lc, radar, lc_phases, radar_coords)
            
            # Loss
            losses = criterion(preds, coeffs)
            loss = losses['total']
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss_total += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss_total / len(train_loader)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                lc = batch['light_curve'].to(device)
                radar = batch['radar_image'].to(device)
                coeffs = batch['spharm_coefficients'].to(device)
                radar_coords = batch['radar_coords'].to(device)
                lc_phases = batch['lc_phases'].to(device)
                
                preds = model(lc, radar, lc_phases, radar_coords)
                losses = criterion(preds, coeffs)
                val_loss_total += losses['total'].item()
                
        avg_val_loss = val_loss_total / len(val_loader)
        
        logger.info(f"Epoch {epoch+1} Complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = "checkpoints/best_model.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f" -> New Best Model Saved to {save_path}!")

if __name__ == "__main__":
    # Configured for full GPU training runs on Colab/Kaggle
    train_model(epochs=50, batch_size=32)
