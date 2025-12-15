"""
Data utilities and helper functions for tennis shot prediction.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, List, Dict
from pathlib import Path
import sys

# Ensure src/models is on sys.path (expects this file at src/data/utils.py)
_models_path = Path(__file__).resolve().parent.parent / "models"
_models_path_str = str(_models_path)
if _models_path_str not in sys.path:
    sys.path.insert(0, _models_path_str)

from models import UnifiedShotLSTM, RichInputLSTM, SimpleMultiHeadBaseline, HierarchicalCristianGPT, SimpleUnifiedBaseline

def create_data_loaders(
    dataset, 
    batch_size: int = 64, 
    test_size: float = 0.2, 
    random_state: int = 42,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders with match-based splitting.
    
    Args:
        dataset: Tennis dataset instance
        batch_size: Batch size for data loaders
        test_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    print("Starting train/validation split (grouped by match)...")
    
    # Get unique match IDs
    all_matches = list(set(dataset.sample_match_ids))
    
    # Split matches
    train_matches, val_matches = train_test_split(
        all_matches, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Convert to sets for faster lookup
    train_matches_set = set(train_matches)
    val_matches_set = set(val_matches)
    
    print(f"Total Matches: {len(all_matches)}")
    print(f"Train Matches: {len(train_matches)} | Val Matches: {len(val_matches)}")

    # Create indices based on match ID
    train_indices = []
    val_indices = []
    
    for idx, m_id in enumerate(dataset.sample_match_ids):
        if m_id in train_matches_set:
            train_indices.append(idx)
        else:
            val_indices.append(idx)
            
    print(f"Train Samples: {len(train_indices)} | Val Samples: {len(val_indices)}")

    # Create subsets & loaders
    train_sub = Subset(dataset, train_indices)
    val_sub = Subset(dataset, val_indices)
    
    train_loader = DataLoader(
        train_sub, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_sub, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader


def compute_class_weights(
    dataset, 
    power: float = 0.3, 
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Compute balanced class weights for handling class imbalance.
    
    Args:
        dataset: Tennis dataset instance
        power: Smoothing power for weights (0.0 = no weighting, 1.0 = full balanced)
        device: Device to place the weights tensor on
        
    Returns:
        Tensor of class weights
    """
    print("Computing balanced class weights...")
    
    # Flatten all targets to count them
    all_targets = [s['y_target'] for s in dataset.samples]
    all_targets_flat = torch.cat(all_targets).numpy()
    
    # Remove padding (0) for calculation
    valid_targets = all_targets_flat[all_targets_flat != 0]
    
    # Compute sklearn weights
    unique_classes = np.unique(valid_targets)
    raw_weights = compute_class_weight(
        class_weight='balanced', 
        classes=unique_classes, 
        y=valid_targets
    )
    
    # Apply smoothing
    smoothed_weights = raw_weights ** power
    
    # Create the weight tensor
    # Initialize with 0.0 for padding (index 0)
    weights_list = [0.0]
    for i in range(1, 11):  # Zones 1 to 10
        if i in unique_classes:
            # Map class i to its weight index
            idx = np.where(unique_classes == i)[0][0]
            w = float(smoothed_weights[idx])
            weights_list.append(w)
        else:
            weights_list.append(1.0)  # Default if class missing
    
    weights_tensor = torch.tensor(weights_list, dtype=torch.float32).to(device)
    
    print("Computed Balanced Weights:")
    for i, w in enumerate(weights_list):
        print(f"Zone {i}: {w:.4f}")
        
    return weights_tensor


def create_zone_mappings() -> Dict[str, Dict]:
    """
    Create useful zone mappings for analysis.
    
    Returns:
        Dictionary containing various zone mappings
    """
    # Directional mapping: Left/Center/Right
    directional_map = {
        0: 0,  # padding
        1: 1, 4: 1, 7: 1,  # left
        2: 2, 5: 2, 8: 2,  # center
        3: 3, 6: 3, 9: 3,  # right
        10: 0  # other
    }
    
    # Court depth mapping: Net/Mid/Back
    depth_map = {
        0: 0,  # padding
        1: 3, 2: 3, 3: 3,  # back court
        4: 2, 5: 2, 6: 2,  # mid court
        7: 1, 8: 1, 9: 1,  # net/winners
        10: 0  # other
    }
    
    # Winner zones
    winner_zones = {7, 8, 9}
    
    # Zone flip mapping for data augmentation
    flip_zone_map = {1: 3, 3: 1, 4: 6, 6: 4, 7: 9, 9: 7}
    
    return {
        'directional': directional_map,
        'depth': depth_map,
        'winner_zones': winner_zones,
        'flip_zones': flip_zone_map
    }


def calculate_directional_accuracy(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    mask: torch.Tensor = None
) -> float:
    """
    Calculate accuracy based on directional grouping (Left/Center/Right).
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        mask: Optional mask for valid positions
        
    Returns:
        Directional accuracy as float
    """
    mappings = create_zone_mappings()
    dir_map = mappings['directional']
    
    if mask is None:
        mask = (targets != 0)
    
    # Apply mask
    y_real = targets[mask].cpu().numpy()
    p_real = predictions[mask].cpu().numpy()
    
    # Map to directions
    y_dir = [dir_map.get(int(i), 0) for i in y_real]
    p_dir = [dir_map.get(int(i), 0) for i in p_real]
    
    # Compare
    correct = np.sum(np.array(y_dir) == np.array(p_dir))
    total = len(y_dir)
    
    return correct / total if total > 0 else 0.0


def calculate_winner_detection_rate(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    mask: torch.Tensor = None
) -> float:
    """
    Calculate how well the model detects winner shots (zones 7, 8, 9).
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        mask: Optional mask for valid positions
        
    Returns:
        Winner detection rate as float
    """
    mappings = create_zone_mappings()
    winner_zones = mappings['winner_zones']
    
    if mask is None:
        mask = (targets != 0)
    
    winner_correct = 0
    winner_total = 0
    
    # Check only positions where the true target was a winner zone
    targets_masked = targets[mask]
    preds_masked = predictions[mask]
    
    for true_val, pred_val in zip(targets_masked, preds_masked):
        if true_val.item() in winner_zones:
            winner_total += 1
            if pred_val.item() in winner_zones:
                winner_correct += 1
    
    return winner_correct / winner_total if winner_total > 0 else 0.0


def calculate_top_k_accuracy(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    k: int = 3, 
    mask: torch.Tensor = None
) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        logits: Model logits
        targets: Ground truth targets
        k: Number of top predictions to consider
        mask: Optional mask for valid positions
        
    Returns:
        Top-k accuracy as float
    """
    if mask is None:
        mask = (targets != 0)
    
    _, topk = logits.topk(k, dim=-1)
    hits = topk.eq(targets.unsqueeze(-1)).any(dim=-1)
    correct = hits[mask].sum().item()
    total = mask.sum().item()
    
    return correct / total if total > 0 else 0.0
