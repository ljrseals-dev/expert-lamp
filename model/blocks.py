"""Model definition for the neural compressor.

Contract with training/train.py
-------------------------------
forward(x) -> (p_pred, g_pred, aux)

    x       : (B, 1, T) float32 mono audio in [-1, 1]
    p_pred  : (B, 4)    normalized [threshold, ratio, attack, release], all in [0, 1]
    g_pred  : (B, 1, F) predicted gain curve in dB at frame rate (F = ceil(T / frame))
    aux     : dict with
                "y"      : (B, 1, T) compressed audio (full rate)
                "g_full" : (B, 1, T) gain curve upsampled to full rate, in dB

The band head present in earlier drafts is intentionally absent: the teacher is a
single-band compressor, so every band's target gain is identical and a per-band
head has no target to learn. See README §3.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CompressorNet", "rms_envelope", "apply_gain_db"]


def rms_envelope(x: torch.Tensor, win: int = 512) -> torch.Tensor:
    """Symmetric RMS envelope. x: (B, 1, T) -> (B, 1, T). Differentiable."""
    win = max(2, int(win))
    pad = win // 2
    xp = F.pad(x, (pad, pad), mode="replicate")
    return torch.sqrt(F.avg_pool1d(xp * xp, win, stride=1).clamp_min(1e-12))


def apply_gain_db(x: torch.Tensor, g_db: torch.Tensor) -> torch.Tensor:
    """x, g_db: (B, 1, T) -> (B, 1, T). g_db is the gain *applied* (<= 0 reduces)."""
    return x * torch.pow(10.0, g_db / 20.0)


class _ConvBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel, padding=dilation * (kernel // 2), dilation=dilation),
            nn.GroupNorm(1, ch),
            nn.SiLU(),
            nn.Conv1d(ch, ch, kernel, padding=dilation * (kernel // 2), dilation=dilation),
            nn.GroupNorm(1, ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))


class CompressorNet(nn.Module):
    """Small dilated CNN over waveform + envelope. Predicts a gain curve and 4 params."""

    def __init__(
        self,
        channels: int = 32,
        n_blocks: int = 4,
        kernel: int = 15,
        frame: int = 256,
        env_win: int = 512,
    ) -> None:
        super().__init__()
        self.frame = int(frame)
        self.env_win = int(env_win)

        self.stem = nn.Conv1d(2, channels, kernel, padding=kernel // 2)
        self.blocks = nn.ModuleList(
            _ConvBlock(channels, kernel, dilation=2 ** i) for i in range(n_blocks)
        )
        self.curve_head = nn.Conv1d(channels, 1, 1)

        self.param_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, 64),
            nn.SiLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

        nn.init.zeros_(self.curve_head.weight)
        nn.init.zeros_(self.curve_head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        B, C, T = x.shape
        if C != 1:
            raise ValueError(f"expected mono input (B, 1, T), got {tuple(x.shape)}")

        env = rms_envelope(x, self.env_win)
        feat = torch.cat([x, torch.log10(env + 1e-9)], dim=1)

        h = self.stem(feat)
        for blk in self.blocks:
            h = blk(h)

        g_pred = self.curve_head(h).clamp(-60.0, 0.0)
        g_pred = F.avg_pool1d(g_pred, self.frame, stride=self.frame, ceil_mode=True)

        g_full = F.interpolate(g_pred, size=T, mode="linear", align_corners=False)
        y = apply_gain_db(x, g_full)

        p_pred = self.param_head(h)
        return p_pred, g_pred, {"y": y, "g_full": g_full}
