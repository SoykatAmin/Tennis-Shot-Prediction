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

# Import models directly from their files to avoid circular imports
from .models import UnifiedShotLSTM, RichInputLSTM, SimpleMultiHeadBaseline, HierarchicalCristianGPT, SimpleUnifiedBaseline, HybridRichLSTM, UnifiedCristianGPT


# === Load UnifiedShotLSTM Model [GOOD MODEL 1]===

def load_unified_lstm(dataset, checkpoint_path="unified_lstm_model.pth", device="cuda"):
    vocab_size = len(dataset.unified_vocab)
    context_dim = dataset.context_tensor.size(1)

    model = UnifiedShotLSTM(
        vocab_size=vocab_size,
        context_dim=context_dim,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"✅ Unified LSTM loaded from {checkpoint_path}")
    return model


def load_rich_lstm(
    checkpoint_path,
    dataset=None,
    device="cuda",
    load_optimizer=False
):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Rebuild model using saved vocab sizes
    model = RichInputLSTM(
        unified_vocab_size=len(checkpoint["unified_vocab"]),
        num_players=len(checkpoint["player_vocab"]),
        type_vocab_size=len(checkpoint["type_vocab"]),
        dir_vocab_size=len(checkpoint["dir_vocab"]),
        depth_vocab_size=len(checkpoint["depth_vocab"]),
        context_dim=10
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✅ RichInputLSTM loaded from {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']} | Val Acc: {checkpoint['val_accuracy']:.2f}%")

    # Optional: restore vocabs into dataset (for inference)
    if dataset is not None:
        dataset.unified_vocab = checkpoint["unified_vocab"]
        dataset.player_vocab  = checkpoint["player_vocab"]
        dataset.type_vocab    = checkpoint["type_vocab"]
        dataset.dir_vocab     = checkpoint["dir_vocab"]
        dataset.depth_vocab   = checkpoint["depth_vocab"]
        print("📦 Vocabularies restored into dataset")

    optimizer = None
    if load_optimizer:
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("🔁 Optimizer state restored")

    return model, optimizer, checkpoint

# Example usage:
# device = "cuda" if torch.cuda.is_available() else "cpu"

# model, optimizer, ckpt = load_rich_lstm(
#     checkpoint_path="checkpoints/rich_lstm_best.pt",
#     dataset=dataset,        # optional but recommended
#     device=device,
#     load_optimizer=False    # True only if resuming training
# )

def load_multi_baseline_model(
    checkpoint_path,
    dataset=None,
    device="cuda",
    load_optimizer=False
):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = SimpleMultiHeadBaseline(
        unified_vocab_size=len(checkpoint["unified_vocab"]),
        type_vocab_size=len(checkpoint["type_vocab"]),
        dir_vocab_size=len(checkpoint["dir_vocab"]),
        depth_vocab_size=len(checkpoint["depth_vocab"]),
        context_dim=10
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✅ Baseline model loaded from {checkpoint_path}")
    print(
        f"   Epoch: {checkpoint['epoch']} | "
        f"Type: {checkpoint['type_acc']:.2f}% | "
        f"Dir: {checkpoint['dir_acc']:.2f}% | "
        f"Depth: {checkpoint['depth_acc']:.2f}%"
    )

    # Optional: restore vocabularies into dataset
    if dataset is not None:
        dataset.unified_vocab = checkpoint["unified_vocab"]
        dataset.type_vocab    = checkpoint["type_vocab"]
        dataset.dir_vocab     = checkpoint["dir_vocab"]
        dataset.depth_vocab   = checkpoint["depth_vocab"]
        print("📦 Vocabularies restored into dataset")

    optimizer = None
    if load_optimizer:
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("🔁 Optimizer state restored")

    return model, optimizer, checkpoint

# device = "cuda" if torch.cuda.is_available() else "cpu"

# model, optimizer, ckpt = load_multi_baseline_model(
#     checkpoint_path="checkpoints_baseline/baseline_multi_best.pt",
#     dataset=dataset,       # optional but recommended for inference
#     device=device,
#     load_optimizer=False
# )

import torch

def load_hierarchical_checkpoint(
    checkpoint_path,
    dataset=None,
    device="cuda",
    load_optimizer=False,
    load_scheduler=False
):
    """
    Loads a hierarchical model checkpoint, restoring architecture, weights, and vocabularies.
    """
    # Safety check for device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, switching to CPU")
        device = 'cpu'

    print(f"📂 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 1. Extract Config & Vocabs
    cfg = checkpoint.get("config")
    vocabs = checkpoint.get("vocabs")
    
    # Basic validation
    if cfg is None or vocabs is None:
        raise KeyError(f"Checkpoint '{checkpoint_path}' is missing 'config' or 'vocabs' keys. "
                       "Ensure it was saved with the updated 'train_hierarchical_model' function.")

    # 2. Re-Initialize Model Architecture
    # We must init the model with the EXACT same dimensions as trained
    model = HierarchicalCristianGPT(
        dir_vocab_size=len(vocabs["dir_vocab"]),
        depth_vocab_size=len(vocabs["depth_vocab"]),
        type_vocab_size=len(vocabs["type_vocab"]),
        num_players=len(vocabs["player_vocab"]),
        context_dim=cfg["context_dim"],
        embed_dim=cfg["embed_dim"],
        n_head=cfg["n_head"],
        n_cycles=cfg["n_cycles"]
    ).to(device)

    # 3. Load State Dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval() # Default to eval mode

    # 4. Restore Vocabs to Dataset (Critical for Inference/Validation consistency)
    if dataset is not None:
        dataset.type_vocab   = vocabs["type_vocab"]
        dataset.dir_vocab    = vocabs["dir_vocab"]
        dataset.depth_vocab  = vocabs["depth_vocab"]
        dataset.player_vocab = vocabs["player_vocab"]
        
        # Optional: Re-create inverse vocabs if your dataset class uses them
        # (Useful for converting IDs back to strings during inference)
        if hasattr(dataset, 'inv_type_vocab'):
             dataset.inv_type_vocab = {v: k for k, v in dataset.type_vocab.items()}
             dataset.inv_dir_vocab  = {v: k for k, v in dataset.dir_vocab.items()}
             dataset.inv_depth_vocab= {v: k for k, v in dataset.depth_vocab.items()}
        
        print("✅ Vocabularies successfully restored into dataset object.")

    # 5. Optional: Restore Optimizer & Scheduler
    optimizer = None
    scheduler = None

    if load_optimizer and "optimizer_state_dict" in checkpoint:
        # Re-init optimizer structure
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("✅ Optimizer state restored.")

    if load_scheduler and "scheduler_state_dict" in checkpoint:
        # Re-init scheduler structure (Assuming ReduceLROnPlateau)
        if optimizer:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("✅ Scheduler state restored.")

    # 6. Extract Metadata for easy access
    extras = {
        "epoch": checkpoint.get("epoch"),
        "val_metrics": checkpoint.get("val_metrics"),
        "serve_type_id": checkpoint.get("serve_type_id"),
        "unk_depth_id": checkpoint.get("unk_depth_id"),
        "unk_dir_id": checkpoint.get("unk_dir_id"),
        "config": cfg,
        "vocabs": vocabs,
        "optimizer": optimizer,
        "scheduler": scheduler
    }

    print(f"✔ Model loaded successfully.")
    if extras["epoch"] is not None:
        print(f"   ↳ Checkpoint Epoch: {extras['epoch']}")
    if extras["val_metrics"]:
        print(f"   ↳ Best Val Acc (Avg): {((extras['val_metrics'].get('type',0) + extras['val_metrics'].get('dir',0) + extras['val_metrics'].get('depth',0))/3):.2f}%")

    return model, extras

# device = "cuda" if torch.cuda.is_available() else "cpu"

# model, optimizer, scheduler, ckpt = load_hierarchical_model(
#     checkpoint_path="checkpoints_hierarchical/hierarchical_best.pt",
#     dataset=dataset,       # optional, but recommended for inference
#     device=device,
#     load_optimizer=False,
#     load_scheduler=False
# )

def load_checkpoint_single_baseline(filepath, dataset=None, device='cpu'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No checkpoint found at {filepath}")

    print(f"Loading checkpoint from '{filepath}'...")
    checkpoint = torch.load(filepath, map_location=device)
    
    # 1. Try to get config
    config = checkpoint.get('config')

    # 2. Fallback logic: If 'config' is missing (e.g. old checkpoint), try to reconstruct it
    if config is None:
        print("Warning: 'config' dict not found. Attempting to reconstruct...")
        
        # We need vocab_size to init the model. 
        # Try to find it in 'unified_vocab' or 'vocab' keys.
        saved_vocab = checkpoint.get('unified_vocab', checkpoint.get('vocab'))
        
        if saved_vocab is None:
            raise ValueError("Cannot determine vocab_size. Checkpoint has no 'config' and no 'unified_vocab'.")
            
        config = {
            'vocab_size': len(saved_vocab),
            'context_dim': checkpoint.get('context_dim', 10), # Default to 10 if missing
            'embed_dim': 64,   # ASSUMPTION: Using default class values
            'hidden_dim': 128  # ASSUMPTION: Using default class values
        }

    # 3. Initialize Model
    model = SimpleUnifiedBaseline(
        vocab_size=config['vocab_size'],
        context_dim=config['context_dim'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim']
    )

    # 4. Load Weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"ERROR: Model shape mismatch. The saved model might have different embed/hidden dims than defaults.")
        print(f"Expected: {config}")
        raise e

    model.to(device)
    model.eval()

    # 5. Restore Dataset Vocab (Crucial for inference mapping)
    if dataset is not None:
        # Check specific keys you used in the saver
        vocab_data = checkpoint.get('unified_vocab', checkpoint.get('vocab'))
        if vocab_data:
            dataset.unified_vocab = vocab_data
            dataset.inv_unified_vocab = {v: k for k, v in dataset.unified_vocab.items()}
            print(f"📦 Vocabularies restored. Size: {len(dataset.unified_vocab)}")

    print(f"Model loaded successfully (Epoch {checkpoint.get('epoch', '?')})")
    return model

def load_singlehead_baseline(
    dataset,
    checkpoint_path,
    device="cuda"
):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = SimpleUnifiedBaseline(
        vocab_size=len(dataset.unified_vocab),
        context_dim=10
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✅ Single-head baseline loaded from {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")

    return model

def load_hybrid_model_checkpoint(checkpoint_path, dataset, device='cuda'):
    """
    Loads the HybridRichLSTM model from a saved .pth checkpoint.
    """
    # 1. Re-initialize the model architecture
    # Ensure these parameters match your training configuration exactly
    model = HybridRichLSTM(
        num_players=len(dataset.player_vocab),
        type_vocab_size=len(dataset.type_vocab),
        dir_vocab_size=len(dataset.dir_vocab),
        depth_vocab_size=len(dataset.depth_vocab),
        context_dim=10
    )
    
    # 2. Load the state dictionary
    # map_location ensures it loads correctly even if trained on GPU but loaded on CPU
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # 3. Move to target device
    model.to(device)
    
    # 4. Set to evaluation mode (important for Dropout/BatchNorm layers)
    model.eval()
    
    print(f"--- Model loaded successfully from {checkpoint_path} ---")
    return model

# Usage Example:
# model = load_hybrid_model_checkpoint('hybrid_rich_lstm.pth', dataset, device='cuda')

def load_UnifiedCristianGPT_checkpoint(checkpoint_path, dataset, embed_dim=128, device="cpu"):
    """
    Loads a saved UnifiedCristianGPT model from a state dict.
    
    Args:
        model_class: The class of the model (UnifiedCristianGPT).
        checkpoint_path: Path to the .pth or .pt file.
        dataset: The dataset instance (needed for vocab sizes).
        embed_dim: Must match the dimension used during training.
        device: The torch device to load onto.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # 1. Initialize the model architecture with identical parameters
    model = UnifiedCristianGPT(
        unified_vocab_size=len(dataset.unified_vocab),
        num_players=len(dataset.player_vocab),
        context_dim=10, 
        seq_len=dataset.max_seq_len,
        embed_dim=embed_dim
    ).to(device)
    
    # 2. Load the state dictionary
    try:
        # map_location ensures we don't crash if loading a GPU model onto a CPU
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ Weights loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

    # 3. Set to evaluation mode (turns off Dropout/Batchnorm)
    model.eval()
    return model

# --- USAGE ---
# singleModel1 = load_tennis_checkpoint(UnifiedCristianGPT, save_path, dataset)