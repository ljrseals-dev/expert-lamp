"""Synthetic dataset + classical teacher compressor."""

from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset

__all__ = ["CompressorDataset", "teacher_compress", "denormalize_params"]

PARAM_RANGES = {
    "threshold_db": (-60.0, 0.0),
    "ratio": (1.0, 20.0),
    "attack_ms": (0.1, 200.0),
    "release_ms": (10.0, 2000.0),
}


def denormalize_params(p: torch.Tensor) -> dict[str, torch.Tensor]:
    thr = p[..., 0] * (PARAM_RANGES["threshold_db"][1] - PARAM_RANGES["threshold_db"][0])
    thr = thr + PARAM_RANGES["threshold_db"][0]
    ratio = torch.exp(
        torch.log(torch.tensor(PARAM_RANGES["ratio"][0]))
        + p[..., 1] * (math.log(PARAM_RANGES["ratio"][1]) - math.log(PARAM_RANGES["ratio"][0]))
    )
    att = torch.exp(
        torch.log(torch.tensor(PARAM_RANGES["attack_ms"][0]))
        + p[..., 2] * (math.log(PARAM_RANGES["attack_ms"][1]) - math.log(PARAM_RANGES["attack_ms"][0]))
    )
    rel = torch.exp(
        torch.log(torch.tensor(PARAM_RANGES["release_ms"][0]))
        + p[..., 3] * (math.log(PARAM_RANGES["release_ms"][1]) - math.log(PARAM_RANGES["release_ms"][0]))
    )
    return {"threshold_db": thr, "ratio": ratio, "attack_ms": att, "release_ms": rel}


@torch.no_grad()
def teacher_compress(
    x: torch.Tensor,
    sample_rate: int,
    params: torch.Tensor,
    *,
    detect_sr: int = 1000,
    knee_db: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T = x.shape
    p = denormalize_params(params)
    thr = p["threshold_db"].to(x.dtype)
    ratio = p["ratio"].to(x.dtype)
    a_att = torch.exp(-1.0 / (p["attack_ms"] * 1e-3 * detect_sr)).to(x.dtype)
    a_rel = torch.exp(-1.0 / (p["release_ms"] * 1e-3 * detect_sr)).to(x.dtype)

    hop = max(1, sample_rate // detect_sr)
    n = T // hop
    xr = x[:, : n * hop].reshape(B, n, hop)
    env = torch.sqrt((xr * xr).mean(-1).clamp_min(1e-12))
    env_db = 20.0 * torch.log10(env + 1e-9)

    over = env_db - thr.unsqueeze(1)
    slope = 1.0 - 1.0 / ratio.unsqueeze(1)
    knee = knee_db
    gr = torch.where(
        over <= -knee / 2,
        torch.zeros_like(over),
        torch.where(
            over >= knee / 2,
            -slope * over,
            -slope * (over + knee / 2) ** 2 / (2.0 * knee),
        ),
    )

    out = torch.empty_like(gr)
    prev = gr[:, 0]
    for i in range(n):
        tgt = gr[:, i]
        a = torch.where(tgt < prev, a_att, a_rel)
        prev = a * prev + (1.0 - a) * tgt
        out[:, i] = prev

    g_full = torch.nn.functional.interpolate(
        out.unsqueeze(1), size=T, mode="linear", align_corners=False
    ).squeeze(1)
    y = x * torch.pow(10.0, g_full / 20.0)
    return y, out


class CompressorDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 256,
        duration_s: float = 1.5,
        sample_rate: int = 44100,
        seed: int = 0,
        detect_sr: int = 1000,
    ) -> None:
        self.n = int(n_samples)
        self.T = int(round(duration_s * sample_rate))
        self.sr = int(sample_rate)
        self.detect_sr = int(detect_sr)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.n

    def _signal(self, g: torch.Generator) -> torch.Tensor:
        T, sr = self.T, self.sr
        t = torch.arange(T, dtype=torch.float32) / sr

        f0 = 60.0 * torch.pow(2.0, torch.rand((), generator=g) * 4.0)
        tone = torch.sin(2 * math.pi * f0 * t) * torch.exp(
            -t * torch.rand((), generator=g) * 3.0
        )
        tone = tone * (0.3 + 0.7 * torch.rand((), generator=g))

        noise_burst = torch.randn(T, generator=g) * torch.exp(
            -t * (5.0 + 15.0 * torch.rand((), generator=g))
        ) * 0.5

        bed = torch.randn(T, generator=g) * 0.05 * torch.exp(
            -t * 0.5 * torch.rand((), generator=g)
        )

        gesture = 0.5 + 0.5 * torch.sin(
            2 * math.pi * (0.2 + 2.0 * torch.rand((), generator=g)) * t
            + torch.rand((), generator=g) * 6.28
        )

        x = (tone + noise_burst) * gesture + bed
        return (x / (x.abs().max() + 1e-6) * 0.9).clamp(-1.0, 1.0)

    def _params(self, g: torch.Generator) -> torch.Tensor:
        thr = torch.rand((), generator=g) * 0.8 + 0.1
        ratio = torch.rand((), generator=g) * 0.7 + 0.1
        att = torch.rand((), generator=g) * 0.6 + 0.1
        rel = torch.rand((), generator=g) * 0.8 + 0.1
        return torch.stack([thr, ratio, att, rel])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)
        x = self._signal(g).unsqueeze(0)
        p = self._params(g)
        y, gr = teacher_compress(x, self.sr, p.unsqueeze(0), detect_sr=self.detect_sr)
        return {
            "x": x,
            "y": y,
            "g": gr.squeeze(0).unsqueeze(0),
            "params": p,
            "idx": torch.tensor(idx),
        }
