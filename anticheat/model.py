"""The detector: one model, transformer encoder blocks applied on an LSTM backbone.

The LSTM and the transformer are not alternatives here. The LSTM reads the window tick
by tick, and the encoder works on what it produces, inside a single network trained
end to end.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig
from .data import HEADS


class AttentionPool(nn.Module):
    """Learned query over time.

    Mean pooling dilutes a cheat that occupies twenty of eighty frames; the snap or
    the trigger pull is a moment, not an average, so the pool learns where to look.
    The weights come back with the result, one per tick, so they can be checked
    against the ticks in activations.csv where a cheat actually fired.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(h), dim=1)
        return (weights * h).sum(dim=1), weights.squeeze(-1)


class HybridDetector(nn.Module):
    """LSTM backbone, transformer encoder on its hidden states, attention pooling, four heads.

    The LSTM carries the short-range dynamics: a snap is a few ticks of angular velocity,
    a trigger pull is the tick a target crosses the cone. Being recurrent, its outputs
    already encode order, so no positional encoding is added before the encoder.

    The encoder then lets every tick attend to every other, which is what relates events
    far apart in the window: line of sight opening and the shot that follows it, or aim
    that kept tracking a target the agent had no way of seeing.

    One shared trunk feeds a binary head per entry in HEADS. They are independent
    sigmoids, not a softmax, because one agent can run several cheats at once.
    """

    def __init__(self, n_features: int, cfg: ModelConfig):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=cfg.hidden,
            num_layers=cfg.layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.layers > 1 else 0.0,
        )

        dim = cfg.hidden * (2 if cfg.bidirectional else 1)
        if dim % cfg.attn_heads:
            raise ValueError(f"LSTM output width {dim} does not divide across {cfg.attn_heads} attention heads")

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=cfg.attn_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        # Nested tensors only help with padding masks, which fixed-length windows never have.
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.encoder_layers, enable_nested_tensor=False)

        self.pool = AttentionPool(dim)

        self.trunk = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.heads = nn.Linear(cfg.hidden, len(HEADS))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        h, _ = self.lstm(x)
        h = self.encoder(h)
        pooled, weights = self.pool(h)
        logits = self.heads(self.trunk(pooled))
        return (logits, weights) if return_attention else logits


def build_model(n_features: int, cfg: ModelConfig) -> nn.Module:
    return HybridDetector(n_features, cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
