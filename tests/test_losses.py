import torch

import model.losses as L


def test_weights_single_sourced():
    loss = L.CompressorLoss()
    assert (loss.w_stft, loss.w_curve, loss.w_param, loss.w_smooth) == (
        L.W_STFT, L.W_CURVE, L.W_PARAM, L.W_SMOOTH,
    )


def test_total_is_recomputed_from_terms():
    crit = L.CompressorLoss()
    p = torch.rand(2, 4)
    g = torch.randn(2, 1, 32).clamp(-60, 0)
    y = torch.randn(2, 1, 1024) * 0.1
    logs = crit(p, g, p.clone(), g.clone(), y_pred=y, y_tgt=y, use_stft=True)
    expect = (
        L.W_CURVE * logs["curve"] + L.W_PARAM * logs["param"]
        + L.W_SMOOTH * logs["smooth"] + L.W_STFT * logs["stft"]
    )
    assert torch.allclose(logs["total"], expect, atol=1e-5)


def test_no_stft_omits_the_term():
    crit = L.CompressorLoss()
    p = torch.rand(2, 4)
    g = torch.randn(2, 1, 32).clamp(-60, 0)
    logs = crit(p, g, p.clone(), g.clone(), use_stft=False)
    assert float(logs["stft"]) == 0.0
    expect = L.W_CURVE * logs["curve"] + L.W_PARAM * logs["param"] + L.W_SMOOTH * logs["smooth"]
    assert torch.allclose(logs["total"], expect, atol=1e-5)


def test_identity_target_gives_zero_curve_loss():
    crit = L.CompressorLoss()
    g = torch.full((1, 1, 16), -3.0)
    logs = crit(torch.rand(1, 4), g, torch.rand(1, 4), g, use_stft=False)
    assert float(logs["curve"]) == 0.0
