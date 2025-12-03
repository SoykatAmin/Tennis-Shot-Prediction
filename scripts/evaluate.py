"""
Evaluation script for tennis shot prediction models.

This script provides comprehensive evaluation of trained models including:
- Accuracy metrics
- Confusion matrices
- Tactical analysis
- Directional accuracy
- Winner detection rates
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import argparse
from pathlib import Path
import json
from typing import Dict, List, Tuple

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data import MCPTennisDataset, create_data_loaders
from src.data.utils import (
    calculate_directional_accuracy, 
    calculate_winner_detection_rate,
    calculate_top_k_accuracy,
    create_zone_mappings
)
from src.models import create_model
from src.utils import load_checkpoint, setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate tennis shot prediction model')
    
    # Data paths (same as training)
    parser.add_argument('--points_path', type=str, required=True,
                        help='Path to tennis points CSV file')
    parser.add_argument('--matches_path', type=str, required=True,
                        help='Path to tennis matches CSV file')
    parser.add_argument('--atp_path', type=str, required=True,
                        help='Path to ATP players CSV file')
    parser.add_argument('--wta_path', type=str, required=True,
                        help='Path to WTA players CSV file')
    
    # Model and checkpoint
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to training config (args.json)')
    
    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save evaluation plots')
    
    return parser.parse_args()


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
    model_type: str,
    zone_vocab_size: int
) -> Dict[str, any]:
    """
    Evaluate model on given data loader.
    
    Args:
        model: The neural network model
        data_loader: Data loader for evaluation
        device: Device to use
        model_type: Type of model being evaluated
        zone_vocab_size: Size of zone vocabulary
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_logits = []
    
    with torch.no_grad():
        for batch in data_loader:
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
            
            # Filter out padding
            mask = (y != 0).view(-1)
            
            if mask.sum() > 0:
                preds = logits.argmax(dim=-1).view(-1)[mask]
                targets = y.view(-1)[mask]
                logits_masked = logits.view(-1, zone_vocab_size)[mask]
                
                all_predictions.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_logits.append(logits_masked.cpu())
    
    # Convert to tensors
    all_predictions = torch.tensor(all_predictions)
    all_targets = torch.tensor(all_targets)
    all_logits = torch.cat(all_logits, dim=0)
    
    # Calculate metrics
    accuracy = (all_predictions == all_targets).float().mean().item()
    
    # Top-k accuracies
    top3_acc = calculate_top_k_accuracy(all_logits, all_targets, k=3)
    top5_acc = calculate_top_k_accuracy(all_logits, all_targets, k=5)
    
    # Directional accuracy
    directional_acc = calculate_directional_accuracy(
        all_predictions, all_targets
    )
    
    # Winner detection rate
    winner_detection = calculate_winner_detection_rate(
        all_predictions, all_targets
    )
    
    return {
        'accuracy': accuracy,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'directional_accuracy': directional_acc,
        'winner_detection_rate': winner_detection,
        'predictions': all_predictions.numpy(),
        'targets': all_targets.numpy(),
        'logits': all_logits
    }


def create_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Path = None,
    show_plot: bool = True
) -> np.ndarray:
    """
    Create and display confusion matrix.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        save_path: Path to save plot (optional)
        show_plot: Whether to display plot
        
    Returns:
        Confusion matrix array
    """
    # Use only zones that appear in the data
    expected_labels = sorted(list(set(targets.tolist()) | set(predictions.tolist())))
    expected_labels = [x for x in expected_labels if x > 0]  # Remove padding
    
    target_names = [str(i) for i in expected_labels]
    
    # Create confusion matrix
    cm = confusion_matrix(targets, predictions, labels=expected_labels)
    
    if show_plot or save_path:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=target_names, 
            yticklabels=target_names
        )
        plt.xlabel('Model Prediction')
        plt.ylabel('Ground Truth')
        plt.title('Tennis Shot Prediction - Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    return cm


def generate_classification_report(
    predictions: np.ndarray,
    targets: np.ndarray
) -> str:
    """
    Generate detailed classification report.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        Classification report string
    """
    expected_labels = sorted(list(set(targets.tolist()) | set(predictions.tolist())))
    expected_labels = [x for x in expected_labels if x > 0]  # Remove padding
    target_names = [f"Zone {i}" for i in expected_labels]
    
    return classification_report(
        targets, 
        predictions, 
        labels=expected_labels, 
        target_names=target_names, 
        zero_division=0
    )


def analyze_zone_performance(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[int, Dict[str, float]]:
    """
    Analyze performance by zone.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        Dictionary with per-zone performance metrics
    """
    zones = sorted(list(set(targets.tolist())))
    zones = [z for z in zones if z > 0]  # Remove padding
    
    zone_stats = {}
    
    for zone in zones:
        # Get predictions for this zone
        zone_mask = (targets == zone)
        zone_predictions = predictions[zone_mask]
        zone_targets = targets[zone_mask]
        
        if len(zone_targets) > 0:
            accuracy = (zone_predictions == zone_targets).mean()
            support = len(zone_targets)
            
            zone_stats[zone] = {
                'accuracy': float(accuracy),
                'support': int(support),
                'frequency': float(support) / len(targets)
            }
    
    return zone_stats


def create_performance_plots(
    zone_stats: Dict[int, Dict[str, float]],
    save_dir: Path = None
):
    """
    Create performance visualization plots.
    
    Args:
        zone_stats: Per-zone performance statistics
        save_dir: Directory to save plots (optional)
    """
    zones = sorted(zone_stats.keys())
    accuracies = [zone_stats[z]['accuracy'] for z in zones]
    frequencies = [zone_stats[z]['frequency'] for z in zones]
    supports = [zone_stats[z]['support'] for z in zones]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy by zone
    ax1.bar(zones, accuracies, alpha=0.7)
    ax1.set_xlabel('Zone')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy by Zone')
    ax1.set_xticks(zones)
    
    # Frequency by zone
    ax2.bar(zones, frequencies, alpha=0.7, color='orange')
    ax2.set_xlabel('Zone')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Zone Frequency in Dataset')
    ax2.set_xticks(zones)
    
    # Support by zone (log scale)
    ax3.bar(zones, supports, alpha=0.7, color='green')
    ax3.set_xlabel('Zone')
    ax3.set_ylabel('Number of Samples (log scale)')
    ax3.set_title('Sample Count by Zone')
    ax3.set_yscale('log')
    ax3.set_xticks(zones)
    
    # Accuracy vs Frequency scatter
    ax4.scatter(frequencies, accuracies, s=100, alpha=0.7)
    for i, zone in enumerate(zones):
        ax4.annotate(f'Z{zone}', (frequencies[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points')
    ax4.set_xlabel('Zone Frequency')
    ax4.set_ylabel('Zone Accuracy')
    ax4.set_title('Accuracy vs Frequency')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / 'zone_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
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
    logger = setup_logging(output_dir / 'evaluation.log')
    
    # Load training configuration
    if args.config_path:
        with open(args.config_path, 'r') as f:
            train_args = json.load(f)
    else:
        # Try to find config in same directory as checkpoint
        config_path = Path(args.checkpoint_path).parent / 'args.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                train_args = json.load(f)
        else:
            logger.error("Training config not found. Please specify --config_path")
            return
    
    # Initialize dataset (same as training)
    logger.info("Initializing dataset...")
    dataset = MCPTennisDataset(
        points_path=args.points_path,
        matches_path=args.matches_path,
        atp_path=args.atp_path,
        wta_path=args.wta_path,
        max_seq_len=train_args.get('seq_len', 30)
    )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=2
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model_type = train_args.get('model_type', 'player_aware')
    
    model_kwargs = {
        'context_dim': 6,
        'embed_dim': train_args.get('embed_dim', 64),
        'n_head': train_args.get('n_head', 4),
        'n_cycles': train_args.get('n_cycles', 3),
        'seq_len': train_args.get('seq_len', 30),
        'dropout': train_args.get('dropout', 0.1)
    }
    
    if model_type == 'player_aware':
        model_kwargs['num_players'] = len(dataset.player_vocab)
    
    model = create_model(
        model_type=model_type,
        zone_vocab_size=len(dataset.zone_vocab),
        type_vocab_size=len(dataset.shot_vocab),
        **model_kwargs
    ).to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = load_checkpoint(args.checkpoint_path, model)
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_metrics = evaluate_model(
        model, val_loader, device, model_type, len(dataset.zone_vocab)
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy: {val_metrics['accuracy']*100:.2f}%")
    print(f"Top-3 Accuracy: {val_metrics['top3_accuracy']*100:.2f}%")
    print(f"Top-5 Accuracy: {val_metrics['top5_accuracy']*100:.2f}%")
    print(f"Directional Accuracy (L/C/R): {val_metrics['directional_accuracy']*100:.2f}%")
    print(f"Winner Detection Rate: {val_metrics['winner_detection_rate']*100:.2f}%")
    print("="*60)
    
    # Generate classification report
    print("\nCLASSIFICATION REPORT:")
    print("-"*40)
    report = generate_classification_report(
        val_metrics['predictions'], 
        val_metrics['targets']
    )
    print(report)
    
    # Analyze zone performance
    zone_stats = analyze_zone_performance(
        val_metrics['predictions'],
        val_metrics['targets']
    )
    
    print("\nZONE PERFORMANCE:")
    print("-"*40)
    for zone, stats in sorted(zone_stats.items()):
        print(f"Zone {zone}: {stats['accuracy']*100:.1f}% accuracy "
              f"({stats['support']:,} samples, {stats['frequency']*100:.1f}% of data)")
    
    # Create and save visualizations
    if args.save_plots:
        logger.info("Creating visualization plots...")
        
        # Confusion matrix
        create_confusion_matrix(
            val_metrics['predictions'],
            val_metrics['targets'],
            save_path=output_dir / 'confusion_matrix.png',
            show_plot=False
        )
        
        # Performance plots
        create_performance_plots(
            zone_stats,
            save_dir=output_dir
        )
    
    # Save results to JSON
    results = {
        'accuracy': val_metrics['accuracy'],
        'top3_accuracy': val_metrics['top3_accuracy'],
        'top5_accuracy': val_metrics['top5_accuracy'],
        'directional_accuracy': val_metrics['directional_accuracy'],
        'winner_detection_rate': val_metrics['winner_detection_rate'],
        'zone_performance': zone_stats,
        'classification_report': report,
        'checkpoint_epoch': checkpoint.get('epoch', 'unknown'),
        'model_type': model_type
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {output_dir}")


if __name__ == '__main__':
    main()