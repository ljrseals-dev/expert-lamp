import torch

from training.dataset import CompressorDataset, denormalize_params, teacher_compress


def test_determinism():
    a = CompressorDataset(n_samples=4, duration_s=0.25, seed=3)
    b = CompressorDataset(n_samples=4, duration_s=0.25, seed=3)
    for i in range(4):
        assert torch.equal(a[i]["x"], b[i]["x"])
        assert torch.equal(a[i]["g"], b[i]["g"])


def test_seeds_differ():
    a = CompressorDataset(n_samples=2, duration_s=0.25, seed=1)
    b = CompressorDataset(n_samples=2, duration_s=0.25, seed=2)
    assert not torch.equal(a[0]["x"], b[0]["x"])


def test_shapes_and_ranges():
    ds = CompressorDataset(n_samples=2, duration_s=0.5, seed=0)
    item = ds[0]
    assert item["x"].shape == (1, 22050)
    assert item["g"].shape[0] == 1
    assert item["params"].shape == (4,)
    assert (item["params"] >= 0).all() and (item["params"] <= 1).all()
    assert item["x"].abs().max() <= 1.0
    assert (item["g"] <= 0.0).all()


def test_teacher_is_gain_reduction_only():
    x = torch.full((1, 2048), 0.5)
    p = torch.tensor([[0.9, 0.5, 0.2, 0.5]])
    y, g = teacher_compress(x, 44100, p)
    assert (g <= 1e-6).all()
    assert y.abs().max() <= x.abs().max() + 1e-6


def test_denormalize_roundtrip_edges():
    p = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    d = denormalize_params(p)
    assert abs(float(d["threshold_db"][0]) - (-60.0)) < 1e-4
    assert abs(float(d["ratio"][0]) - 1.0) < 1e-4
    assert abs(float(d["ratio"][1]) - 20.0) < 1e-2
