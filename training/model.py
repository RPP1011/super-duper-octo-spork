"""Ability Transformer model.

Small transformer encoder (2 layers, 4 heads, d_model=64) for learning
compositional structure from ability DSL token sequences.

Supports three modes:
  1. Pre-training: masked token prediction (MLM)
  2. Fine-tuning: urgency + target prediction from [CLS] embedding
  3. Generation: autoregressive token prediction (future)

Architecture follows the grokking paper's setup: no dropout, designed for
high weight decay (AdamW lambda=1.0).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbilityTransformer(nn.Module):
    """Transformer encoder for ability DSL tokens.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in vocabulary.
    d_model : int
        Embedding / hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer encoder layers.
    d_ff : int
        Feed-forward inner dimension.
    max_seq_len : int
        Maximum sequence length (for positional embeddings).
    pad_id : int
        Token ID used for padding (masked out in attention).
    cls_id : int
        Token ID for [CLS] — initialized to zero per grokking plan.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        max_seq_len: int = 256,
        pad_id: int = 0,
        cls_id: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.cls_id = cls_id

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # Learned positional embedding
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder layers (no dropout — per grokking plan §3.3)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for stability with high weight decay
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer norm on output
        self.out_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with reduced scale per Kumar et al. (2310.06110).

        3x smaller init starts closer to the Goldilocks weight norm zone,
        reducing grokking delay. Zero-init [CLS] embedding per grokking plan §3.2.
        """
        nn.init.normal_(self.token_emb.weight, std=0.007)
        nn.init.normal_(self.pos_emb.weight, std=0.007)

        # Zero-init [CLS] token embedding
        with torch.no_grad():
            self.token_emb.weight[self.cls_id].zero_()
            # Keep padding at zero
            self.token_emb.weight[self.pad_id].zero_()

        # Initialize linear layers with reduced scale
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.007)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode token sequence.

        Parameters
        ----------
        input_ids : (batch, seq_len) int tensor
        attention_mask : (batch, seq_len) float tensor, 1=attend 0=ignore

        Returns
        -------
        hidden : (batch, seq_len, d_model) float tensor
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # Build key_padding_mask: True = ignore
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = input_ids == self.pad_id

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.out_norm(x)

        return x

    def cls_embedding(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Extract [CLS] embedding (position 0).

        Returns
        -------
        cls : (batch, d_model) float tensor
        """
        hidden = self.forward(input_ids, attention_mask)
        return hidden[:, 0, :]


class MLMHead(nn.Module):
    """Masked language model head for pre-training."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Predict masked tokens.

        Parameters
        ----------
        hidden : (batch, seq_len, d_model)

        Returns
        -------
        logits : (batch, seq_len, vocab_size)
        """
        x = self.dense(hidden)
        x = F.gelu(x)
        x = self.norm(x)
        x = self.proj(x)
        return x


class DecisionHead(nn.Module):
    """Fine-tuning head: predict urgency + target from [CLS] embedding.

    Used in Phase 2 to replace the per-category MLP ability evaluators.
    """

    def __init__(self, d_model: int, n_targets: int = 3):
        super().__init__()
        self.urgency = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.target = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_targets),
        )

    def forward(self, cls_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict urgency and target scores.

        Parameters
        ----------
        cls_emb : (batch, d_model) — [CLS] embedding

        Returns
        -------
        urgency : (batch, 1) — sigmoid output in [0, 1]
        target_logits : (batch, n_targets)
        """
        return self.urgency(cls_emb), self.target(cls_emb)


class AbilityTransformerMLM(nn.Module):
    """Pre-training model: transformer + MLM head."""

    def __init__(self, vocab_size: int, **kwargs):
        super().__init__()
        self.transformer = AbilityTransformer(vocab_size=vocab_size, **kwargs)
        self.mlm_head = MLMHead(self.transformer.d_model, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden = self.transformer(input_ids, attention_mask)
        return self.mlm_head(hidden)


class EntityEncoder(nn.Module):
    """Encode game state entities into d_model tokens for cross-attention.

    Input: flat game_state (batch, 70) = 7 entities × 10 features.
    Output: entity tokens (batch, 7, d_model) with learned type embeddings.

    Entity order: [self, enemy0, enemy1, enemy2, ally0, ally1, ally2]
    Entity types: 0=self, 1=enemy, 2=ally
    """

    ENTITY_DIM = 30
    NUM_ENTITIES = 7
    NUM_TYPES = 3  # self, enemy, ally
    # Type assignment for each slot
    TYPE_IDS = [0, 1, 1, 1, 2, 2, 2]  # self, 3×enemy, 3×ally

    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(self.ENTITY_DIM, d_model)
        self.type_emb = nn.Embedding(self.NUM_TYPES, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Self-attention over entities (matches pretrain_entity.py architecture)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, game_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode game state entities.

        Parameters
        ----------
        game_state : (batch, 210)

        Returns
        -------
        entity_tokens : (batch, 7, d_model)
        entity_mask : (batch, 7) — True where entity doesn't exist (for key_padding_mask)
        """
        B = game_state.shape[0]
        device = game_state.device

        entities = game_state.view(B, self.NUM_ENTITIES, self.ENTITY_DIM)

        tokens = self.proj(entities)
        type_ids = torch.tensor(self.TYPE_IDS, device=device, dtype=torch.long)
        tokens = tokens + self.type_emb(type_ids).unsqueeze(0)
        tokens = self.input_norm(tokens)

        # Entity mask: exists feature is index 29
        exists = entities[:, :, 29]
        entity_mask = exists < 0.5  # True = ignore (padding)

        tokens = self.encoder(tokens, src_key_padding_mask=entity_mask)
        tokens = self.out_norm(tokens)

        return tokens, entity_mask


class CrossAttentionBlock(nn.Module):
    """Cross-attention: ability [CLS] attends to game state entity tokens.

    Query: [CLS] embedding from ability transformer (1 token)
    Key/Value: entity tokens from EntityEncoder (7 tokens)
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.0, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_ff = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query : (batch, d_model) — [CLS] embedding
        kv : (batch, n_entities, d_model) — entity tokens
        kv_mask : (batch, n_entities) — True where entity is padding

        Returns
        -------
        out : (batch, d_model)
        """
        # Expand query to (batch, 1, d_model) for attention
        q = self.norm_q(query).unsqueeze(1)
        k = v = self.norm_kv(kv)

        attn_out, _ = self.cross_attn(q, k, v, key_padding_mask=kv_mask)
        attn_out = attn_out.squeeze(1)  # (batch, d_model)

        # Residual + FF
        x = query + attn_out
        x = x + self.ff(self.norm_ff(x))

        return x


class AbilityTransformerDecision(nn.Module):
    """Fine-tuning model: transformer + cross-attention + decision head.

    Ability tokens go through the transformer encoder to produce a [CLS] embedding.
    Game state entities are encoded into tokens. The [CLS] embedding cross-attends
    to entity tokens, producing a context-aware representation for the decision head.
    """

    def __init__(
        self,
        vocab_size: int,
        game_state_dim: int = 0,
        n_targets: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.transformer = AbilityTransformer(vocab_size=vocab_size, **kwargs)
        d = self.transformer.d_model

        self.has_cross_attn = game_state_dim > 0
        if self.has_cross_attn:
            n_heads = kwargs.get("n_heads", 4)
            n_layers = kwargs.get("n_layers", 2)
            self.entity_encoder = EntityEncoder(d, n_heads=n_heads, n_layers=n_layers)
            self.cross_attn = CrossAttentionBlock(d, n_heads=n_heads)

        self.decision_head = DecisionHead(d, n_targets)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        game_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls_emb = self.transformer.cls_embedding(input_ids, attention_mask)

        if self.has_cross_attn and game_state is not None:
            entity_tokens, entity_mask = self.entity_encoder(game_state)
            cls_emb = self.cross_attn(cls_emb, entity_tokens, entity_mask)

        return self.decision_head(cls_emb)
