#!/usr/bin/env python3
"""
MSS-GAN B1 Training Script for Tennis Shot Prediction

This script demonstrates how to train the MSS-GAN B1 baseline model on tennis data.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

import torch
from src.data import MCPTennisDataset
from src.models import MSSGAN_B1_Trainer

def main():
    """Main training function for MSS-GAN B1 baseline."""
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Data paths - update these to match your data location
    DATA_PATHS = {
        'points_path': '/path/to/your/charting-m-points.csv',
        'matches_path': '/path/to/your/charting-m-matches.csv',
        'atp_players': '/path/to/your/atp_players.csv',
        'wta_players': '/path/to/your/wta_players.csv'
    }
    
    # Check if data files exist
    for name, path in DATA_PATHS.items():
        if not os.path.exists(path):
            print(f"⚠️ Data file not found: {path}")
            print("Please update DATA_PATHS in this script with your actual file locations.")
            return
    
    # Load dataset
    print("📊 Loading tennis dataset...")
    dataset = MCPTennisDataset(
        points_path=DATA_PATHS['points_path'],
        matches_path=DATA_PATHS['matches_path'],
        atp_players_path=DATA_PATHS['atp_players'], 
        wta_players_path=DATA_PATHS['wta_players'],
        max_seq_len=30
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Zone vocabulary size: {len(dataset.zone_vocab)}")
    print(f"Shot vocabulary size: {len(dataset.shot_vocab)}")
    
    # Initialize MSS-GAN B1 Trainer
    print("🤖 Initializing MSS-GAN B1 trainer...")
    trainer = MSSGAN_B1_Trainer(
        dataset=dataset,
        device=device,
        n_zones=len(dataset.zone_vocab),
        n_types=len(dataset.shot_vocab),
        embed_dim=256,
        latent_dim=64,
        em_N=1100,  # Episodic memory size
        em_l=3,     # Extract depth for episodic memory
        sm_b=80     # Semantic memory basis size
    )
    
    # Training parameters
    epochs = 20
    batch_size = 32
    dataloader_workers = 2
    
    print(f"Training configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {dataloader_workers}")
    
    # Start training
    print("🎾 Starting MSS-GAN B1 training...")
    trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        dataloader_workers=dataloader_workers
    )
    
    # Evaluate the trained model
    print("📈 Evaluating trained model...")
    trainer.evaluate(batch_size=64)
    
    print("✅ Training complete!")

if __name__ == "__main__":
    main()