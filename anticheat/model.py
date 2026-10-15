"""Sequence classifiers over a window of server-observed behaviour.

The LSTM is the project's committed architecture. The transformer is here as a
matched baseline, because the closest prior work (AntiCheatPT, IEEE CoG 2025) never
ran one and so never showed that attention was the thing doing the work.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import ModelConfig


class AttentionPool(nn.Module):
    """Learned query over time.

    Mean pooling dilutes a cheat that occupies twenty of eighty frames; the snap or
    the trigger pull is a moment, not an average, so the pool learns where to look.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(h), dim=1)
        return (weights * h).sum(dim=1)


def _pool(kind: str, dim: int) -> tuple[nn.Module, int]:
    if kind == "attention":
        return AttentionPool(dim), dim
    if kind in ("mean", "max", "last"):
        return nn.Identity(), dim
    raise ValueError(f"unknown pooling {kind!r}")


def _apply_pool(kind: str, module: nn.Module, h: torch.Tensor) -> torch.Tensor:
    if kind == "attention":
        return module(h)
    if kind == "mean":
        return h.mean(dim=1)
    if kind == "max":
        return h.max(dim=1).values
    return h[:, -1]


class LSTMDetector(nn.Module):
    def __init__(self, n_features: int, cfg: ModelConfig):
        super().__init__()
        self.pooling = cfg.pooling

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=cfg.hidden,
            num_layers=cfg.layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.layers > 1 else 0.0,
        )

        dim = cfg.hidden * (2 if cfg.bidirectional else 1)
        self.pool, pooled = _pool(cfg.pooling, dim)

        self.head = nn.Sequential(
            nn.LayerNorm(pooled),
            nn.Linear(pooled, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.head(_apply_pool(self.pooling, self.pool, h)).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerDetector(nn.Module):
    """Encoder baseline with a CLS token.

    The input is projected up to d_model first. AntiCheatPT instead set d_model to the
    raw feature count (44) with a single attention head, and then had to rescale its
    positional encodings into [0, 0.1] to stop them drowning the features - a symptom
    of that choice, not a technique worth copying.
    """

    def __init__(self, n_features: int, cfg: ModelConfig):
        super().__init__()
        self.project = nn.Linear(n_features, cfg.d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos = PositionalEncoding(cfg.d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.layers)

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, 1),
        )

        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.project(x)
        h = torch.cat([self.cls.expand(h.size(0), -1, -1), h], dim=1)
        h = self.pos(h)
        h = self.encoder(h)
        return self.head(h[:, 0]).squeeze(-1)


def build_model(n_features: int, cfg: ModelConfig) -> nn.Module:
    if cfg.arch == "lstm":
        return LSTMDetector(n_features, cfg)
    if cfg.arch == "transformer":
        return TransformerDetector(n_features, cfg)
    raise ValueError(f"unknown arch {cfg.arch!r}, expected lstm|transformer")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
