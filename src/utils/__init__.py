"""
Utility functions for the tennis shot prediction project.
"""

import logging
import torch
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

# Import evaluation classes and functions
from .evaluation import (
    ModelAdapter,
    UnifiedAdapter,
    MultiHeadAdapter,
    TennisEvaluator,
    get_universal_decoder_map
)


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger('tennis_shot_prediction')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool = False,
    checkpoint_dir: Path = Path('./checkpoints'),
    filename: str = 'checkpoint.pth'
):
    """
    Save model checkpoint.
    
    Args:
        state: Dictionary containing model state and metadata
        is_best: Whether this is the best model so far
        checkpoint_dir: Directory to save checkpoints
        filename: Checkpoint filename
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = checkpoint_dir / filename
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = checkpoint_dir / 'best_model.pth'
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        
    Returns:
        Checkpoint dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Get the size of a model in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def create_run_name(args: Dict[str, Any]) -> str:
    """
    Create a descriptive run name from arguments.
    
    Args:
        args: Dictionary of training arguments
        
    Returns:
        Run name string
    """
    model_type = args.get('model_type', 'unknown')
    embed_dim = args.get('embed_dim', 64)
    epochs = args.get('epochs', 10)
    lr = args.get('lr', 1e-3)
    
    return f"{model_type}_embed{embed_dim}_ep{epochs}_lr{lr:.0e}"


def format_time(seconds: float) -> str:
    """
    Format time duration in a human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_model_summary(model: torch.nn.Module, model_name: str = "Model"):
    """
    Print a summary of the model architecture and parameters.
    
    Args:
        model: PyTorch model
        model_name: Name of the model for display
    """
    print(f"\n{model_name} Summary:")
    print("=" * 50)
    
    # Count parameters by layer type
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if module_params > 0:
                print(f"{name}: {module_params:,} parameters")
                
            total_params += module_params
            trainable_params += module_trainable
    
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {get_model_size_mb(model):.2f} MB")
    print("=" * 50)


def ensure_dir(directory: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to directory
        
    Returns:
        Path to directory
    """
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_device(device_name: str = 'auto') -> str:
    """
    Get the appropriate device for computation.
    
    Args:
        device_name: Device specification ('auto', 'cpu', 'cuda')
        
    Returns:
        Device string
    """
    if device_name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device_name == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        return 'cpu'
    else:
        return device_name


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: numpy and random seeds would be set here too if needed


__all__ = [
    # Evaluation classes and functions
    'ModelAdapter',
    'UnifiedAdapter', 
    'MultiHeadAdapter',
    'TennisEvaluator',
    'get_universal_decoder_map',
    
    # Utility functions
    'setup_logging',
    'save_checkpoint',
    'load_checkpoint',
    'get_device',
    'set_seed'
]