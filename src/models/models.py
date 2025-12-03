"""
Neural network models for tennis shot prediction.

This module contains the transformer-based models for predicting tennis shots.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SymbolicTinyRM_PlayerAware(nn.Module):
    """
    Symbolic Tiny RM with Player Awareness and Causal Masking.
    
    This model uses transformer layers to predict tennis shot targets based on:
    - Shot zones and types
    - Player embeddings (server and receiver)
    - Match context (surface, score, handedness, etc.)
    """
    
    def __init__(
        self, 
        zone_vocab_size: int, 
        type_vocab_size: int, 
        num_players: int, 
        context_dim: int = 6, 
        embed_dim: int = 64, 
        player_dim: int = 64,
        n_head: int = 4, 
        n_cycles: int = 3, 
        seq_len: int = 30,
        dropout: float = 0.1
    ):
        """
        Initialize the player-aware model.
        
        Args:
            zone_vocab_size: Number of zone types
            type_vocab_size: Number of shot types
            num_players: Number of unique players
            context_dim: Dimension of context vector
            embed_dim: Embedding dimension
            player_dim: Player embedding dimension
            n_head: Number of attention heads
            n_cycles: Number of transformer layers
            seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.n_cycles = n_cycles
        self.seq_len = seq_len
        
        # Embeddings
        self.zone_emb = nn.Embedding(zone_vocab_size, embed_dim)
        self.type_emb = nn.Embedding(type_vocab_size, embed_dim)
        self.player_emb = nn.Embedding(num_players, player_dim)
        
        # Context fusion
        self.context_fusion = nn.Sequential(
            nn.Linear(context_dim + (player_dim * 2), embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Input dimension: Zone + Type + Context
        self.input_dim = embed_dim * 3
        
        # Position embedding
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, self.input_dim))
        
        # Transformer layers
        self.shared_block = nn.TransformerEncoderLayer(
            d_model=self.input_dim, 
            nhead=n_head, 
            dim_feedforward=embed_dim * 8,
            batch_first=True, 
            norm_first=True, 
            dropout=dropout
        )
        
        # Output layers
        self.norm_f = nn.LayerNorm(self.input_dim)
        self.head = nn.Linear(self.input_dim, zone_vocab_size)

    def generate_causal_mask(self, sz: int) -> torch.Tensor:
        """
        Generate a causal mask to prevent looking at future tokens.
        
        Args:
            sz: Sequence length
            
        Returns:
            Causal mask tensor
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    
    def forward(
        self, 
        x_z: torch.Tensor, 
        x_t: torch.Tensor, 
        x_c: torch.Tensor, 
        x_s: torch.Tensor, 
        x_r: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_z: Zone indices [batch_size, seq_len]
            x_t: Shot type indices [batch_size, seq_len]
            x_c: Context vector [batch_size, context_dim]
            x_s: Server player ID [batch_size]
            x_r: Receiver player ID [batch_size]
            
        Returns:
            Logits for next shot prediction [batch_size, seq_len, zone_vocab_size]
        """
        # Embeddings
        z = self.zone_emb(x_z)  # [batch_size, seq_len, embed_dim]
        t = self.type_emb(x_t)  # [batch_size, seq_len, embed_dim]
        s_emb = self.player_emb(x_s)  # [batch_size, player_dim]
        r_emb = self.player_emb(x_r)  # [batch_size, player_dim]
        
        # Fuse context with player embeddings
        c_combined = torch.cat([x_c, s_emb, r_emb], dim=1)  # [batch_size, context_dim + 2*player_dim]
        c = self.context_fusion(c_combined)  # [batch_size, embed_dim]
        c = c.unsqueeze(1).expand(-1, z.size(1), -1)  # [batch_size, seq_len, embed_dim]
        
        # Combine embeddings
        x = torch.cat([z, t, c], dim=-1)  # [batch_size, seq_len, input_dim]
        x = x + self.pos_emb[:, :x.size(1), :]
        
        # Apply causal mask
        causal_mask = self.generate_causal_mask(x.size(1)).to(x.device)
        
        # Transformer layers
        memory = x.clone()
        for _ in range(self.n_cycles):
            memory = self.shared_block(memory, src_mask=causal_mask)
            
        return self.head(self.norm_f(memory))


class SymbolicTinyRM_Context(nn.Module):
    """
    Simplified version without player embeddings.
    
    This model focuses on shot zones, types, and match context without
    player-specific information.
    """
    
    def __init__(
        self, 
        zone_vocab_size: int, 
        type_vocab_size: int, 
        context_dim: int = 6, 
        embed_dim: int = 64, 
        n_head: int = 4, 
        n_cycles: int = 3, 
        seq_len: int = 30,
        dropout: float = 0.1
    ):
        """
        Initialize the context-only model.
        
        Args:
            zone_vocab_size: Number of zone types
            type_vocab_size: Number of shot types
            context_dim: Dimension of context vector
            embed_dim: Embedding dimension
            n_head: Number of attention heads
            n_cycles: Number of transformer layers
            seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.n_cycles = n_cycles
        self.seq_len = seq_len
        
        # Embeddings
        self.zone_emb = nn.Embedding(zone_vocab_size, embed_dim)
        self.type_emb = nn.Embedding(type_vocab_size, embed_dim)
        
        # Context processing
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, embed_dim), 
            nn.ReLU(), 
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Input dimension
        self.input_dim = embed_dim * 3
        
        # Position embedding
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, self.input_dim))
        
        # Transformer layer
        self.shared_block = nn.TransformerEncoderLayer(
            d_model=self.input_dim, 
            nhead=n_head, 
            dim_feedforward=embed_dim * 8,
            batch_first=True, 
            norm_first=True, 
            dropout=dropout
        )
        
        # Output layers
        self.norm_f = nn.LayerNorm(self.input_dim)
        self.head = nn.Linear(self.input_dim, zone_vocab_size)

    def generate_causal_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask."""
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(
        self, 
        x_z: torch.Tensor, 
        x_t: torch.Tensor, 
        x_c: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_z: Zone indices [batch_size, seq_len]
            x_t: Shot type indices [batch_size, seq_len]
            x_c: Context vector [batch_size, context_dim]
            
        Returns:
            Logits for next shot prediction [batch_size, seq_len, zone_vocab_size]
        """
        # Embeddings
        z = self.zone_emb(x_z)
        t = self.type_emb(x_t)
        
        # Process context
        c = self.context_mlp(x_c)
        c = c.unsqueeze(1).expand(-1, z.size(1), -1)
        
        # Combine embeddings
        x = torch.cat([z, t, c], dim=-1)
        x = x + self.pos_emb[:, :x.size(1), :]
        
        # Apply causal mask
        causal_mask = self.generate_causal_mask(x.size(1)).to(x.device)
        
        # Transformer layers
        memory = x.clone()
        for _ in range(self.n_cycles):
            memory = self.shared_block(memory, src_mask=causal_mask)
            
        return self.head(self.norm_f(memory))


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss applies a modulating term to the cross entropy loss in order to
    focus learning on hard negative examples and down-weight easy examples.
    """
    
    def __init__(
        self, 
        alpha: Optional[torch.Tensor] = None, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weights tensor
            gamma: Focusing parameter (higher gamma = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Model logits [N, C]
            targets: Target class indices [N]
            
        Returns:
            Focal loss value
        """
        # Standard cross entropy (no reduction yet)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=0)
        
        # Calculate probabilities
        pt = torch.exp(-ce_loss)
        
        # Calculate focal term: (1 - pt)^gamma
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # Get weight for each target
            at = self.alpha.gather(0, targets.view(-1))
            focal_loss = focal_loss * at
            
        # Apply reduction
        if self.reduction == 'mean':
            mask = targets != 0
            if mask.sum() > 0:
                return focal_loss[mask].mean()
            else:
                return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_model(
    model_type: str,
    zone_vocab_size: int,
    type_vocab_size: int,
    num_players: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('player_aware' or 'context_only')
        zone_vocab_size: Number of zone types
        type_vocab_size: Number of shot types
        num_players: Number of unique players (required for player_aware)
        **kwargs: Additional model parameters
        
    Returns:
        Initialized model
    """
    if model_type == 'player_aware':
        if num_players is None:
            raise ValueError("num_players is required for player_aware model")
        return SymbolicTinyRM_PlayerAware(
            zone_vocab_size=zone_vocab_size,
            type_vocab_size=type_vocab_size,
            num_players=num_players,
            **kwargs
        )
    elif model_type == 'context_only':
        return SymbolicTinyRM_Context(
            zone_vocab_size=zone_vocab_size,
            type_vocab_size=type_vocab_size,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")