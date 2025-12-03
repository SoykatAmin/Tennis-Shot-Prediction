"""
Training script for tennis shot prediction models.

This script handles the complete training pipeline including:
- Dataset loading and preprocessing
- Model initialization
- Training loop with validation
- Checkpointing and model saving
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import os
import argparse
from pathlib import Path
import json
import time
from typing import Dict, Any

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data import MCPTennisDataset, create_data_loaders, compute_class_weights
from src.models import create_model, FocalLoss
from src.utils import setup_logging, save_checkpoint, load_checkpoint


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train tennis shot prediction model')
    
    # Data paths
    parser.add_argument('--points_path', type=str, required=True,
                        help='Path to tennis points CSV file')
    parser.add_argument('--matches_path', type=str, required=True,
                        help='Path to tennis matches CSV file')
    parser.add_argument('--atp_path', type=str, required=True,
                        help='Path to ATP players CSV file')
    parser.add_argument('--wta_path', type=str, required=True,
                        help='Path to WTA players CSV file')
    
    # Model configuration
    parser.add_argument('--model_type', type=str, default='player_aware',
                        choices=['player_aware', 'context_only'],
                        help='Type of model to train')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--n_head', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--n_cycles', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--seq_len', type=int, default=30,
                        help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay for optimizer')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss')
    parser.add_argument('--class_weight_power', type=float, default=0.3,
                        help='Smoothing power for class weights')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    return parser.parse_args()


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    model_type: str
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use
        model_type: Type of model being trained
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        x_z = batch['x_zone'].to(device)
        x_t = batch['x_type'].to(device)
        x_c = batch['context'].to(device)
        y = batch['y_target'].to(device)
        
        optimizer.zero_grad()
        
        if model_type == 'player_aware':
            x_s = batch['x_s_id'].to(device)
            x_r = batch['x_r_id'].to(device)
            logits = model(x_z, x_t, x_c, x_s, x_r)
        else:
            logits = model(x_z, x_t, x_c)
        
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'lr': scheduler.get_last_lr()[0]
    }


def validate_epoch(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: str,
    model_type: str,
    zone_vocab_size: int
) -> Dict[str, float]:
    """
    Validate model for one epoch.
    
    Args:
        model: The neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        model_type: Type of model being validated
        zone_vocab_size: Size of zone vocabulary
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x_z = batch['x_zone'].to(device)
            x_t = batch['x_type'].to(device)
            x_c = batch['context'].to(device)
            y = batch['y_target'].to(device)
            
            if model_type == 'player_aware':
                x_s = batch['x_s_id'].to(device)
                x_r = batch['x_r_id'].to(device)
                logits = model(x_z, x_t, x_c, x_s, x_r)
            else:
                logits = model(x_z, x_t, x_c)
            
            # Calculate loss only on valid targets
            if (y != 0).sum() > 0:
                loss = criterion(logits.view(-1, zone_vocab_size), y.view(-1))
                total_loss += loss.item()
                num_batches += 1
            
            # Calculate accuracy
            mask = (y != 0)
            preds = logits.argmax(dim=-1)
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()
    
    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'accuracy': correct / total if total > 0 else 0.0
    }


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir / 'train.log')
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Check if dataset files exist
    if not all(os.path.exists(path) for path in [args.points_path, args.matches_path]):
        logger.error("Dataset files not found. Please check the paths.")
        return
    
    # Initialize dataset
    logger.info("Initializing dataset...")
    dataset = MCPTennisDataset(
        points_path=args.points_path,
        matches_path=args.matches_path,
        atp_path=args.atp_path,
        wta_path=args.wta_path,
        max_seq_len=args.seq_len
    )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Compute class weights
    logger.info("Computing class weights...")
    class_weights = compute_class_weights(
        dataset,
        power=args.class_weight_power,
        device=device
    )
    
    # Initialize model
    logger.info(f"Initializing {args.model_type} model...")
    model_kwargs = {
        'context_dim': 6,
        'embed_dim': args.embed_dim,
        'n_head': args.n_head,
        'n_cycles': args.n_cycles,
        'seq_len': args.seq_len,
        'dropout': args.dropout
    }
    
    if args.model_type == 'player_aware':
        model_kwargs['num_players'] = len(dataset.player_vocab)
    
    model = create_model(
        model_type=args.model_type,
        zone_vocab_size=len(dataset.zone_vocab),
        type_vocab_size=len(dataset.shot_vocab),
        **model_kwargs
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr / 25,  # Start low for OneCycleLR
        weight_decay=args.weight_decay
    )
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )
    
    # Initialize loss function
    criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, args.model_type
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, args.model_type, len(dataset.zone_vocab)
        )
        
        epoch_time = time.time() - start_time
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} "
            f"| Train Loss: {train_metrics['loss']:.4f} "
            f"| Val Loss: {val_metrics['loss']:.4f} "
            f"| Val Acc: {val_metrics['accuracy']*100:.2f}% "
            f"| LR: {train_metrics['lr']:.2e} "
            f"| Time: {epoch_time:.1f}s"
        )
        
        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
        
        save_checkpoint(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'best_val_acc': best_val_acc,
                'args': vars(args)
            },
            is_best=is_best,
            checkpoint_dir=output_dir
        )
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc*100:.2f}%")


if __name__ == '__main__':
    main()