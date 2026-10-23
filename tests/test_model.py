import torch

from model.blocks import CompressorNet, apply_gain_db, rms_envelope


def test_shapes_and_finiteness():
    m = CompressorNet()
    x = torch.randn(2, 1, 4410)
    p, g, aux = m(x)
    assert p.shape == (2, 4)
    assert g.dim() == 3 and g.shape[0] == 2 and g.shape[1] == 1
    assert aux["y"].shape == x.shape
    assert aux["g_full"].shape == x.shape
    assert torch.isfinite(p).all() and torch.isfinite(g).all()
    assert (p >= 0).all() and (p <= 1).all()


def test_zero_gain_is_identity_at_init():
    m = CompressorNet()
    x = torch.randn(1, 1, 2048)
    _, _, aux = m(x)
    assert torch.allclose(aux["y"], x, atol=1e-3)


def test_gain_and_envelope_helpers():
    x = torch.ones(1, 1, 64)
    assert torch.allclose(apply_gain_db(x, torch.zeros_like(x)), x)
    assert torch.allclose(apply_gain_db(x, torch.full_like(x, -6.0206)), x * 0.5, atol=1e-3)
    env = rms_envelope(x, 16)
    assert env.shape == x.shape and (env > 0).all()
