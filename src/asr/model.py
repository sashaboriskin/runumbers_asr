"""Compact Conformer-CTC for Russian spoken-numbers ASR (≤ 5M parameters).

Input: log-mel [B, n_mels=80, T] and lengths [B]
Output: logits [B, T', vocab_size], out_lens [B]  where T' = T // 4

The encoder:
    1. Conv2D subsampling by 4x (time and freq).
    2. Linear projection to d_model.
    3. N Conformer-lite blocks (MHSA + depth-wise conv + FFN with macaron).
    4. Linear CTC head.

No pretrained weights are used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_mels: int = 80
    d_model: int = 144
    n_heads: int = 4
    n_blocks: int = 8
    ff_mult: int = 4
    conv_kernel: int = 15
    dropout: float = 0.1
    vocab_size: int = 34  # see asr.text.VOCAB_SIZE
    subsample: int = 4


class ConvSubsampling(nn.Module):
    """2x (Conv2d stride 2) => 4x time reduction, 4x freq reduction.

    Input: [B, n_mels, T]  -> [B, d_model, T/4]
    """

    def __init__(self, n_mels: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # after two stride-2 convs with padding=1 and k=3, freq dim halves twice
        freq_out = ((n_mels + 1) // 2 + 1) // 2
        self.proj = nn.Linear(32 * freq_out, d_model)

    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, n_mels, T]  -> [B, 1, n_mels, T]
        x = x.unsqueeze(1)
        x = self.conv(x)  # [B, 32, F', T']
        B, C, Fm, Tm = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, Tm, C * Fm)  # [B, T', C*F']
        x = self.proj(x)  # [B, T', d_model]
        # lengths: each stride-2 conv halves T (with rounding up due to padding)
        new_lens = ((lens + 1) // 2 + 1) // 2
        return x, new_lens


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mult: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * mult)
        self.fc2 = nn.Linear(d_model * mult, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        y = F.silu(self.fc1(y))
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return y


class MHSA(nn.Module):
    """Standard MHSA with absolute sinusoidal positional encoding added once outside.

    Mask-aware: respects padding via `key_padding_mask`.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        y = self.ln(x)
        y, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        return self.drop(y)


class ConvModule(nn.Module):
    """Conformer-style conv module: GLU pointwise + depthwise + pointwise."""

    def __init__(self, d_model: int, kernel: int, dropout: float):
        super().__init__()
        assert kernel % 2 == 1
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.dw = nn.Conv1d(
            d_model, d_model, kernel_size=kernel, padding=kernel // 2, groups=d_model
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        y = self.ln(x).transpose(1, 2)  # [B, D, T]
        y = self.pw1(y)
        y = F.glu(y, dim=1)
        y = self.dw(y)
        y = self.bn(y)
        y = F.silu(y)
        y = self.pw2(y)
        y = self.drop(y)
        return y.transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ff1 = FeedForward(cfg.d_model, cfg.ff_mult, cfg.dropout)
        self.attn = MHSA(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.conv = ConvModule(cfg.d_model, cfg.conv_kernel, cfg.dropout)
        self.ff2 = FeedForward(cfg.d_model, cfg.ff_mult, cfg.dropout)
        self.ln_out = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x, key_padding_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.ln_out(x)


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T]


class ConformerCTC(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.sub = ConvSubsampling(cfg.n_mels, cfg.d_model)
        self.pe = SinusoidalPE(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([ConformerBlock(cfg) for _ in range(cfg.n_blocks)])
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, mel: torch.Tensor, mel_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """mel: [B, n_mels, T],  mel_lens: [B] (in time frames)"""
        x, lens = self.sub(mel, mel_lens)
        x = self.pe(x)
        x = self.drop(x)
        T = x.size(1)
        idx = torch.arange(T, device=x.device).unsqueeze(0)
        key_pad = idx >= lens.unsqueeze(1)  # True where padded
        for blk in self.blocks:
            x = blk(x, key_pad)
        logits = self.head(x)  # [B, T', vocab]
        return logits, lens


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
