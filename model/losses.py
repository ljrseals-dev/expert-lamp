"""Objective for the neural compressor.

`CompressorLoss.total` is the ONLY place `total` is computed. training/train.py
must never recompute it from the individual terms — that was the defect that made
the epoch-average `total` inconsistent between STFT and non-STFT steps.

`total` is the objective actually optimised on this step. When `use_stft=False`
the STFT term is absent, so `total` is a different (smaller) objective. The
harness therefore accumulates per-term sums with per-term counts and reports
`stft` as a mean over STFT steps only. See training/train.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CompressorLoss", "W_STFT", "W_CURVE", "W_PARAM", "W_SMOOTH"]

W_STFT = 1.0
W_CURVE = 1.0
W_PARAM = 0.5
W_SMOOTH = 0.01


class _SpectralLoss(nn.Module):
    def __init__(self, n_fft: int, hop: int, win: int | None = None) -> None:
        super().__init__()
        self.n_fft, self.hop = int(n_fft), int(hop)
        self.win = int(win or n_fft)
        self.register_buffer("window", torch.hann_window(self.win), persistent=False)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        n = min(a.shape[-1], b.shape[-1])
        a, b = a[..., :n], b[..., :n]
        A = torch.stft(
            a.squeeze(1), self.n_fft, self.hop, self.win,
            window=self.window, center=True, return_complex=True,
        ).abs()
        Bm = torch.stft(
            b.squeeze(1), self.n_fft, self.hop, self.win,
            window=self.window, center=True, return_complex=True,
        ).abs()
        eps = 1e-7
        sc = torch.linalg.vector_norm(A - Bm, dim=-1) / (
            torch.linalg.vector_norm(Bm, dim=-1) + eps
        )
        logl1 = F.l1_loss(torch.log(A + eps), torch.log(Bm + eps))
        return sc.mean() + logl1


class CompressorLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int = 44100,
        *,
        w_stft: float = W_STFT,
        w_curve: float = W_CURVE,
        w_param: float = W_PARAM,
        w_smooth: float = W_SMOOTH,
    ) -> None:
        super().__init__()
        self.w_stft, self.w_curve = float(w_stft), float(w_curve)
        self.w_param, self.w_smooth = float(w_param), float(w_smooth)
        self.spec_a = _SpectralLoss(1024, 256)
        self.spec_b = _SpectralLoss(512, 128)

    def stft_loss(self, y_pred: torch.Tensor, y_tgt: torch.Tensor) -> torch.Tensor:
        return self.spec_a(y_pred, y_tgt) + self.spec_b(y_pred, y_tgt)

    def forward(
        self,
        p_pred: torch.Tensor,
        g_pred: torch.Tensor,
        p_tgt: torch.Tensor,
        g_tgt: torch.Tensor,
        *,
        y_pred: torch.Tensor | None = None,
        y_tgt: torch.Tensor | None = None,
        use_stft: bool = True,
    ) -> dict[str, torch.Tensor]:
        p_pred, g_pred = p_pred.float(), g_pred.float()
        p_tgt, g_tgt = p_tgt.float(), g_tgt.float()

        n = min(g_pred.shape[-1], g_tgt.shape[-1])
        g_pred_c, g_tgt_c = g_pred[..., :n], g_tgt[..., :n]

        curve = F.l1_loss(g_pred_c, g_tgt_c)
        param = F.l1_loss(p_pred, p_tgt)
        smooth = F.l1_loss(
            g_pred_c[..., 1:] - g_pred_c[..., :-1],
            torch.zeros_like(g_pred_c[..., 1:]),
        )

        total = self.w_curve * curve + self.w_param * param + self.w_smooth * smooth
        stft = torch.zeros((), device=curve.device, dtype=curve.dtype)

        if use_stft:
            if y_pred is None or y_tgt is None:
                raise ValueError("use_stft=True requires y_pred and y_tgt")
            stft = self.stft_loss(y_pred.float(), y_tgt.float())
            total = total + self.w_stft * stft

        return {
            "total": total,
            "stft": stft.detach(),
            "curve": curve.detach(),
            "param": param.detach(),
            "smooth": smooth.detach(),
        }
