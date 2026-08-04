"""Lightweight hierarchical latent memory for long-video prototypes.

The module operates on already encoded clip tokens. It first compresses every
clip into a small fixed number of learned latent slots, then applies causal
cross-clip attention and reads the resulting persistent memory with a question
embedding. The Qwen vision/language backbone can remain frozen.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def _clip_causal_mask(num_clips: int, slots_per_clip: int, device: torch.device) -> torch.Tensor:
    """Block future clips while allowing full attention within the same clip."""
    clip_ids = torch.arange(num_clips, device=device).repeat_interleave(slots_per_clip)
    return clip_ids.unsqueeze(0) > clip_ids.unsqueeze(1)


class _MemoryReadout(nn.Module):
    def __init__(self, input_dim: int, memory_dim: int, num_classes: int):
        super().__init__()
        self.question_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, memory_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(memory_dim * 3),
            nn.Linear(memory_dim * 3, memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, num_classes),
        )

    def forward(
        self,
        memory: torch.Tensor,
        question_embedding: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        question = self.question_projection(question_embedding)
        scores = torch.einsum("bd,bld->bl", question, memory) / math.sqrt(memory.shape[-1])
        if memory_mask is not None:
            scores = scores.masked_fill(~memory_mask, torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        retrieved = torch.einsum("bl,bld->bd", attention, memory)
        logits = self.classifier(torch.cat((question, retrieved, question * retrieved), dim=-1))
        return logits, attention


class TemporalLatentMemory(nn.Module):
    """Compress local clip tokens and maintain ordered latent memory slots."""

    def __init__(
        self,
        input_dim: int,
        memory_dim: int = 256,
        slots_per_clip: int = 2,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 4,
        max_clips: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        if memory_dim % num_heads:
            raise ValueError("memory_dim must be divisible by num_heads")
        if slots_per_clip < 1:
            raise ValueError("slots_per_clip must be positive")
        self.slots_per_clip = slots_per_clip
        self.max_clips = max_clips
        self.token_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, memory_dim))
        self.latent_queries = nn.Parameter(torch.randn(slots_per_clip, memory_dim) / math.sqrt(memory_dim))
        self.local_cross_attention = nn.MultiheadAttention(
            memory_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.local_norm = nn.LayerNorm(memory_dim)
        self.local_ffn = nn.Sequential(
            nn.LayerNorm(memory_dim),
            nn.Linear(memory_dim, memory_dim * 4),
            nn.GELU(),
            nn.Linear(memory_dim * 4, memory_dim),
        )
        self.clip_position = nn.Parameter(torch.randn(max_clips, memory_dim) / math.sqrt(memory_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=memory_dim,
            nhead=num_heads,
            dim_feedforward=memory_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(memory_dim))
        self.readout = _MemoryReadout(input_dim, memory_dim, num_classes)

    def forward(
        self,
        clip_tokens: torch.Tensor,
        question_embedding: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        clip_mask: Optional[torch.Tensor] = None,
        shuffle_memory: bool = False,
    ) -> dict[str, torch.Tensor]:
        if clip_tokens.ndim != 4:
            raise ValueError("clip_tokens must have shape [batch, clips, tokens, input_dim]")
        batch, clips, tokens, _ = clip_tokens.shape
        if clips > self.max_clips:
            raise ValueError(f"received {clips} clips but max_clips={self.max_clips}")

        projected = self.token_projection(clip_tokens).reshape(batch * clips, tokens, -1)
        queries = self.latent_queries.unsqueeze(0).expand(batch * clips, -1, -1)
        key_padding_mask = None
        if token_mask is not None:
            key_padding_mask = ~token_mask.reshape(batch * clips, tokens).bool()
        compressed, local_attention = self.local_cross_attention(
            queries,
            projected,
            projected,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        compressed = self.local_norm(compressed + queries)
        compressed = compressed + self.local_ffn(compressed)
        compressed = compressed.reshape(batch, clips, self.slots_per_clip, -1)

        if shuffle_memory:
            # One shared permutation keeps the ablation deterministic within a
            # batch while destroying chronological clip order.
            order = torch.randperm(clips, device=compressed.device)
            compressed = compressed[:, order]
            if clip_mask is not None:
                clip_mask = clip_mask[:, order]

        # Positions describe the sequence presented to global memory. For the
        # shuffled ablation they must be assigned after permutation; otherwise
        # the original position embedding would leak the correct chronology.
        compressed = compressed + self.clip_position[:clips].view(1, clips, 1, -1)

        memory = compressed.reshape(batch, clips * self.slots_per_clip, -1)
        causal_mask = _clip_causal_mask(clips, self.slots_per_clip, memory.device)
        memory_mask = None
        padding_mask = None
        if clip_mask is not None:
            memory_mask = clip_mask.bool().repeat_interleave(self.slots_per_clip, dim=1)
            padding_mask = ~memory_mask
        memory = self.temporal_encoder(memory, mask=causal_mask, src_key_padding_mask=padding_mask)
        logits, read_attention = self.readout(memory, question_embedding, memory_mask)
        return {
            "logits": logits,
            "memory": memory,
            "local_attention": local_attention.reshape(batch, clips, *local_attention.shape[1:]),
            "read_attention": read_attention,
        }


class UniformTokenBaseline(nn.Module):
    """Equal-global-token-budget control using uniformly selected raw tokens."""

    def __init__(
        self,
        input_dim: int,
        memory_dim: int = 256,
        global_tokens: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if memory_dim % num_heads:
            raise ValueError("memory_dim must be divisible by num_heads")
        self.global_tokens = global_tokens
        self.token_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, memory_dim))
        self.position = nn.Parameter(torch.randn(global_tokens, memory_dim) / math.sqrt(memory_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=memory_dim,
            nhead=num_heads,
            dim_feedforward=memory_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(memory_dim))
        self.readout = _MemoryReadout(input_dim, memory_dim, num_classes)

    def forward(self, clip_tokens: torch.Tensor, question_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        if clip_tokens.ndim != 4:
            raise ValueError("clip_tokens must have shape [batch, clips, tokens, input_dim]")
        flat = clip_tokens.flatten(1, 2)
        total_tokens = flat.shape[1]
        if self.global_tokens > total_tokens:
            raise ValueError("global_tokens cannot exceed available clip tokens")
        indices = torch.linspace(0, total_tokens - 1, self.global_tokens, device=flat.device).round().long()
        selected = flat.index_select(1, indices)
        memory = self.token_projection(selected) + self.position.unsqueeze(0)
        memory = self.encoder(memory)
        logits, read_attention = self.readout(memory, question_embedding)
        return {"logits": logits, "memory": memory, "read_attention": read_attention}
