# MSS-GAN B1 Baseline for Tennis Shot Prediction

This document describes the MSS-GAN B1 baseline model implementation for tennis shot prediction.

## Overview

MSS-GAN (Memory-based Sequential Shot GAN) B1 is a baseline model that combines:
- **Episodic Memory**: Binary tree-based LSTM for storing and retrieving historical shot patterns
- **Semantic Memory**: Learnable memory matrix for capturing general tennis knowledge
- **GAN Architecture**: Generator-discriminator setup for realistic shot sequence generation

## Architecture Components

### 1. Memory Systems

#### Episodic Memory (EM)
- Uses binary tree LSTM to organize historical shot contexts
- Maintains a queue of recent shot embeddings (default: 1100 items)
- Retrieves relevant memories via attention mechanism

#### Semantic Memory (SM)
- Learnable parameter matrix (256 x 80 by default)
- Captures general tennis patterns and strategies
- Updated via moving average from episodic memories

### 2. Core Networks

#### Step Encoder
- Encodes individual shots: zone + shot type + context
- Uses embedding layers for categorical features
- Outputs contextual representations via GRU cell

#### Generator
- Takes current context + memories + noise
- Outputs distributions over zones and shot types
- Uses multi-layer perceptron with categorical heads

#### Discriminator
- Distinguishes real vs generated shot sequences
- Also performs auxiliary classification on shot types
- Helps ensure realistic and diverse generation

## Usage

### Training

```python
from src.data import MCPTennisDataset
from src.models import MSSGAN_B1_Trainer

# Load your tennis dataset
dataset = MCPTennisDataset(
    points_path='path/to/points.csv',
    matches_path='path/to/matches.csv',
    atp_players_path='path/to/atp_players.csv',
    wta_players_path='path/to/wta_players.csv',
    max_seq_len=30
)

# Initialize trainer
trainer = MSSGAN_B1_Trainer(
    dataset=dataset,
    device='cuda',
    n_zones=len(dataset.zone_vocab),
    n_types=len(dataset.shot_vocab)
)

# Train with strict joint objective (recommended)
trainer.train_full_strict(epochs=20, batch_size=32)

# Or use basic training
trainer.train_full(epochs=20, batch_size=32)

# Evaluate
trainer.evaluate()
```

## Training Methods

### 1. Basic Training (`train_full`)
- Standard GAN training with classification loss
- 80/20 train/validation split
- Real-time validation monitoring

### 2. Strict Joint Training (`train_full_strict`) **[Recommended]**
- Implements Fernando et al. (2019) joint objective (Equation 21)
- Balanced adversarial and classification objectives
- Label smoothing for improved stability
- Better convergence properties

### 3. Legacy Training (`train`)
- Maintained for backward compatibility
- Calls `train_full` internally

```bash
# Update data paths in scripts/train_mssgan_b1.py
python scripts/train_mssgan_b1.py
```

## Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| embed_dim | 256 | Embedding dimension for all components |
| latent_dim | 64 | Noise dimension for generator |
| em_N | 1100 | Episodic memory queue size |
| em_l | 3 | Extraction depth for binary tree |
| sm_b | 80 | Semantic memory basis vectors |

## Key Features

1. **Memory-Augmented Learning**: Combines episodic and semantic memory for better shot prediction
2. **Adversarial Training**: GAN setup helps generate realistic shot sequences
3. **Categorical Output**: Directly predicts zone and shot type distributions
4. **Context Awareness**: Incorporates match context (surface, score, etc.)
5. **Improved Target Labeling**: Fixed next-shot prediction logic
6. **Proper Validation**: Real-time monitoring with comprehensive metrics
7. **Joint Training Objective**: Strict implementation of SS-GAN methodology
8. **Label Smoothing**: Enhanced training stability

## Performance Metrics

The model is evaluated on:
- **Zone Prediction Accuracy**: Correctness of predicted landing zones
- **Shot Type Accuracy**: Correctness of predicted shot types  
- **Precision/Recall/F1**: Detailed performance breakdown
- **Confusion Matrices**: Error analysis for both zones and shot types

## Files

- `src/models/mssgan_b1.py`: Main implementation
- `scripts/train_mssgan_b1.py`: Training script
- `notebooks/`: Example usage notebooks (coming soon)

## Citation

Based on memory-augmented sequential modeling principles for sports analytics.