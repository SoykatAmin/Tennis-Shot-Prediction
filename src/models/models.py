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
    
# === UnifiedShotLSTM Model Definition [GOOD MODEL 1] === 
    
class UnifiedShotLSTM(nn.Module):
    def __init__(self, vocab_size, context_dim=10,  # match context vector size
                 embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Context embedding
        self.context_fc = nn.Linear(context_dim, embed_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim * 2,  # token + context
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Output: Unified token prediction
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_tokens, x_context):
        """
        x_tokens: [batch, seq_len] tensor of token IDs
        x_context: [batch, context_dim] tensor
        """
        token_emb = self.token_emb(x_tokens)  # [B, L, embed_dim]
        context_emb = self.context_fc(x_context).unsqueeze(1)  # [B, 1, embed_dim]
        context_emb = context_emb.expand(-1, token_emb.size(1), -1)  # [B, L, embed_dim]

        lstm_input = torch.cat([token_emb, context_emb], dim=-1)  # [B, L, embed*2]
        lstm_out, _ = self.lstm(lstm_input)  # [B, L, hidden_dim]

        logits = self.fc_out(lstm_out)  # [B, L, vocab_size]
        return logits
    
class RichInputLSTM(nn.Module):
    def __init__(self, 
                 unified_vocab_size, # Output size
                 num_players,        # For Player Embeddings
                 type_vocab_size,    # Input sizes...
                 dir_vocab_size,
                 depth_vocab_size,
                 context_dim=10, 
                 hidden_dim=256, 
                 num_layers=2, 
                 dropout=0.2):
        super().__init__()

        # --- 1. Decomposed Input Embeddings ---
        # Instead of one big embedding, we learn smaller, specific ones
        self.emb_type = nn.Embedding(type_vocab_size, 64, padding_idx=0)
        self.emb_dir  = nn.Embedding(dir_vocab_size, 16, padding_idx=0)
        self.emb_depth= nn.Embedding(depth_vocab_size, 16, padding_idx=0)
        
        # --- 2. Player Style Embeddings ---
        # Learn a vector for every player (Nadal, Federer, etc.)
        self.emb_player = nn.Embedding(num_players, 32, padding_idx=0)
        
        # --- 3. Context Projection ---
        self.context_fc = nn.Linear(context_dim, 32)

        # Calculate total input dimension for LSTM
        # Type(64) + Dir(16) + Depth(16) + P1(32) + P2(32) + Context(32)
        lstm_input_dim = 64 + 16 + 16 + 32 + 32 + 32

        # --- 4. LSTM Backbone ---
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # --- 5. Output Head (Still Unified for now) ---
        self.fc_out = nn.Linear(hidden_dim, unified_vocab_size)

    def forward(self, x_type, x_dir, x_depth, x_s_id, x_r_id, x_context):
        # 1. Embed Sequences [Batch, Seq_Len, Emb_Dim]
        e_t = self.emb_type(x_type)
        e_d = self.emb_dir(x_dir)
        e_p = self.emb_depth(x_depth)
        
        # 2. Embed Players [Batch, Emb_Dim] -> Expand to [Batch, Seq_Len, Emb_Dim]
        # Server ID
        e_s = self.emb_player(x_s_id).unsqueeze(1).expand(-1, e_t.size(1), -1)
        # Returner ID
        e_r = self.emb_player(x_r_id).unsqueeze(1).expand(-1, e_t.size(1), -1)
        
        # 3. Embed Context [Batch, Emb_Dim] -> Expand
        e_ctx = self.context_fc(x_context).unsqueeze(1).expand(-1, e_t.size(1), -1)
        
        # 4. Concatenate everything
        # This gives the LSTM a massive amount of specific detail for every timestep
        full_input = torch.cat([e_t, e_d, e_p, e_s, e_r, e_ctx], dim=-1)
        
        # 5. Pass through LSTM
        lstm_out, _ = self.lstm(full_input)
        
        # 6. Predict next Unified Token
        logits = self.fc_out(lstm_out)
        return logits
    
class SimpleMultiHeadBaseline(nn.Module):
    def __init__(self, 
                 unified_vocab_size, 
                 type_vocab_size, 
                 dir_vocab_size, 
                 depth_vocab_size,
                 context_dim=10, 
                 embed_dim=64,   # Smaller embedding than LSTM
                 hidden_dim=128): # Smaller hidden layer
        super().__init__()
        
        # 1. Simple Embeddings (No sequence processing)
        self.token_emb = nn.Embedding(unified_vocab_size, embed_dim, padding_idx=0)
        self.context_fc = nn.Linear(context_dim, embed_dim)
        
        # 2. Feed-Forward Network (Replacing LSTM)
        # This acts as a "Markov" predictor: P(Next | Current)
        self.fc_shared = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim), # Takes (Token + Context)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 3. Independent Prediction Heads
        self.head_type = nn.Linear(hidden_dim, type_vocab_size)
        self.head_dir  = nn.Linear(hidden_dim, dir_vocab_size)
        self.head_depth = nn.Linear(hidden_dim, depth_vocab_size)

    def forward(self, x_tokens, x_context):
        # x_tokens: [Batch, Seq_Len]
        # x_context: [Batch, Context_Dim]
        
        # Embed inputs
        token_emb = self.token_emb(x_tokens) # [B, L, E]
        
        # Expand context to match sequence length
        ctx_emb = self.context_fc(x_context).unsqueeze(1) # [B, 1, E]
        ctx_emb = ctx_emb.expand(-1, token_emb.size(1), -1) # [B, L, E]
        
        # Concatenate: Input is just (Current Token + Match Context)
        x_in = torch.cat([token_emb, ctx_emb], dim=-1) # [B, L, E*2]
        
        # Extract features (Independent for every timestep, no history)
        features = self.fc_shared(x_in) # [B, L, Hidden]
        
        # Predict
        logits_type  = self.head_type(features)
        logits_dir   = self.head_dir(features)
        logits_depth = self.head_depth(features)
        
        return logits_type, logits_dir, logits_depth
    
class HierarchicalCristianGPT(nn.Module):
    def __init__(self, dir_vocab_size, depth_vocab_size, type_vocab_size, num_players, 
                 context_dim=6, embed_dim=64, n_head=4, n_cycles=3, seq_len=30):        
        super().__init__()
        
        # --- Shared Transformer Body (Same as before) ---
        self.input_dim = embed_dim * 3 
        self.zone_emb = nn.Embedding(dir_vocab_size, embed_dim)
        self.type_emb = nn.Embedding(type_vocab_size, embed_dim)
        self.player_emb = nn.Embedding(num_players, 64)
        
        self.context_fusion = nn.Sequential(
            nn.Linear(context_dim + 128, embed_dim), 
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, self.input_dim))
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=n_head, 
                                       dim_feedforward=embed_dim*4, batch_first=True),
            num_layers=n_cycles
        )
        
        # --- HIERARCHICAL HEADS ---
        
        # 1. Type Head (First prediction)
        self.head_type = nn.Linear(self.input_dim, type_vocab_size)
        
        # 2. Direction Head (Takes Transformer Output + Predicted Type)
        self.type_proj = nn.Linear(type_vocab_size, embed_dim) # Project logits back to embed size
        self.head_dir = nn.Linear(self.input_dim + embed_dim, dir_vocab_size)
        
        # 3. Depth Head (Takes Transformer Output + Predicted Type + Predicted Direction)
        self.dir_proj = nn.Linear(dir_vocab_size, embed_dim)
        self.head_depth = nn.Linear(self.input_dim + embed_dim + embed_dim, depth_vocab_size)

    def forward(self, x_z, x_t, x_c, x_s, x_r):
        # ... (Embeddings and Transformer Pass - same as before) ...
        # Embeddings
        z = self.zone_emb(x_z)
        t = self.type_emb(x_t)
        s_emb = self.player_emb(x_s)
        r_emb = self.player_emb(x_r)
        
        c = self.context_fusion(torch.cat([x_c, s_emb, r_emb], dim=1))
        c = c.unsqueeze(1).expand(-1, z.size(1), -1)
        
        x = torch.cat([z, t, c], dim=-1) + self.pos_emb[:, :z.size(1), :]
        
        # Causal Mask
        mask = torch.triu(torch.full((x.size(1), x.size(1)), float('-inf')), diagonal=1).to(x.device)
        memory = self.encoder(x, mask=mask)
        
        # --- CHAINED PREDICTIONS ---
        
        # 1. Predict Type
        logits_type = self.head_type(memory) # [B, T, V_type]
        
        # 2. Predict Direction (Conditioned on Type)
        # We use the raw logits as a "soft" embedding of the prediction
        type_feat = self.type_proj(torch.softmax(logits_type, dim=-1)) 
        cat_dir = torch.cat([memory, type_feat], dim=-1)
        logits_dir = self.head_dir(cat_dir)
        
        # 3. Predict Depth (Conditioned on Type + Direction)
        dir_feat = self.dir_proj(torch.softmax(logits_dir, dim=-1))
        cat_depth = torch.cat([memory, type_feat, dir_feat], dim=-1)
        logits_depth = self.head_depth(cat_depth)
        
        return logits_dir, logits_depth, logits_type

class SimpleUnifiedBaseline(nn.Module):
    def __init__(self, 
                 vocab_size,       # Size of unified_vocab (e.g., ~3000 unique shot combos)
                 context_dim=10, 
                 embed_dim=64, 
                 hidden_dim=128):
        super().__init__()
        
        # 1. Embedding
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.context_fc = nn.Linear(context_dim, embed_dim)
        
        # 2. Feed-Forward Network (No LSTM)
        # Predicts Next Token directly from Current Token + Context
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size) # Output is one giant list of all shot combos
        )

    def forward(self, x_tokens, x_context):
        # Embed
        tok_emb = self.token_emb(x_tokens) # [B, L, E]
        
        # Expand context
        ctx_emb = self.context_fc(x_context).unsqueeze(1) # [B, 1, E]
        ctx_emb = ctx_emb.expand(-1, tok_emb.size(1), -1)
        
        # Concat
        x_in = torch.cat([tok_emb, ctx_emb], dim=-1) # [B, L, E*2]
        
        # Predict
        logits = self.net(x_in) # [B, L, Vocab_Size]
        return logits
    

class HybridRichLSTM(nn.Module):
    def __init__(self, 
                 num_players,        # For Player Embeddings
                 type_vocab_size,    # Size of type vocab (e.g. ~20: f, b, s, ...)
                 dir_vocab_size,     # Size of direction vocab (e.g. 4: 1, 2, 3)
                 depth_vocab_size,   # Size of depth vocab (e.g. 4: 7, 8, 9)
                 context_dim=10, 
                 hidden_dim=256, 
                 num_layers=2, 
                 dropout=0.2):
        super().__init__()

        # --- 1. Rich Input Embeddings (Same as RichInputLSTM) ---
        self.emb_type = nn.Embedding(type_vocab_size, 64, padding_idx=0)
        self.emb_dir  = nn.Embedding(dir_vocab_size, 16, padding_idx=0)
        self.emb_depth= nn.Embedding(depth_vocab_size, 16, padding_idx=0)
        
        # Player Style Embeddings
        self.emb_player = nn.Embedding(num_players, 32, padding_idx=0)
        
        # Context Projection
        self.context_fc = nn.Linear(context_dim, 32)

        # Total Input Dimension: 64+16+16 + 32+32 + 32 = 192
        lstm_input_dim = 64 + 16 + 16 + 32 + 32 + 32

        # --- 2. Shared Backbone ---
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # --- 3. Parallel Output Heads ---
        # Instead of predicting one Unified ID, we predict the components separately
        self.head_type  = nn.Linear(hidden_dim, type_vocab_size)
        self.head_dir   = nn.Linear(hidden_dim, dir_vocab_size)
        self.head_depth = nn.Linear(hidden_dim, depth_vocab_size)

    def forward(self, x_type, x_dir, x_depth, x_s_id, x_r_id, x_context):
        # A. Create Rich Embeddings
        e_t = self.emb_type(x_type)
        e_d = self.emb_dir(x_dir)
        e_p = self.emb_depth(x_depth)
        
        # Expand Player & Context vectors to sequence length
        # [Batch, 1, Dim] -> [Batch, Seq, Dim]
        e_s = self.emb_player(x_s_id).unsqueeze(1).expand(-1, e_t.size(1), -1)
        e_r = self.emb_player(x_r_id).unsqueeze(1).expand(-1, e_t.size(1), -1)
        e_ctx = self.context_fc(x_context).unsqueeze(1).expand(-1, e_t.size(1), -1)
        
        # B. Concatenate All Features
        full_input = torch.cat([e_t, e_d, e_p, e_s, e_r, e_ctx], dim=-1)
        
        # C. Pass through Shared LSTM
        lstm_out, _ = self.lstm(full_input)
        
        # D. Split into Parallel Heads
        logits_type  = self.head_type(lstm_out)
        logits_dir   = self.head_dir(lstm_out)
        logits_depth = self.head_depth(lstm_out)
        
        return logits_type, logits_dir, logits_depth

class UnifiedCristianGPT(nn.Module):
    def __init__(self, unified_vocab_size, num_players, context_dim=10, 
                 embed_dim=128, n_head=4, n_cycles=3, seq_len=30):        
        super().__init__()
        
        # 1. Embeddings
        # We embed the Unified Token directly
        self.token_emb = nn.Embedding(unified_vocab_size, embed_dim)
        
        # Player Embeddings (Static Style)
        self.player_emb = nn.Embedding(num_players, 64)
        
        # Context Fusion
        self.context_fusion = nn.Sequential(
            nn.Linear(context_dim + (64 * 2), embed_dim), 
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Input = Token_Emb + Context_Emb
        self.input_dim = embed_dim # We sum them or concat? Let's Add.
        
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        # Transformer
        self.blocks = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_head, dim_feedforward=embed_dim*4,
            batch_first=True, norm_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(self.blocks, num_layers=n_cycles)
        self.norm_f = nn.LayerNorm(embed_dim)
        
        # Single Output Head
        self.head = nn.Linear(embed_dim, unified_vocab_size)

    def generate_causal_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(DEVICE)
    
    def forward(self, x_seq, x_c, x_s, x_r):
        # x_seq: [Batch, Seq] -> [Batch, Seq, Dim]
        tok = self.token_emb(x_seq)
        
        # Players
        s = self.player_emb(x_s)
        r = self.player_emb(x_r)
        
        # Context: [Batch, CtxDim] -> [Batch, Dim]
        c_vec = torch.cat([x_c, s, r], dim=1)
        c_emb = self.context_fusion(c_vec)
        
        # Expand Context to Sequence Length
        # [Batch, 1, Dim] -> [Batch, Seq, Dim]
        c_emb = c_emb.unsqueeze(1).expand(-1, tok.size(1), -1)
        
        # Combine: Token + Context + Position
        # We ADD context to the token embedding (ResNet style) rather than concat
        x = tok + c_emb + self.pos_emb[:, :tok.size(1), :]
        
        # Causal Mask
        mask = torch.triu(torch.full((x.size(1), x.size(1)), float('-inf')), diagonal=1).to(x.device)
        
        # Transformer
        x = self.transformer(x, mask=mask)
        x = self.norm_f(x)
        
        return self.head(x)