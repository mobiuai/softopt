"""Tests for SoftOptTorch. Skipped when torch is not installed."""
import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F
from softopt import SoftOptTorch


def _cnn():
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.MaxPool2d(2), nn.Flatten(), nn.Linear(8 * 4 * 4, 10)).double()


def _loss(model, batch):
    x, y = batch
    return F.cross_entropy(model(x), y)


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(16, 3, 8, 8, generator=g, dtype=torch.float64),
            torch.randint(0, 10, (16,), generator=g))


def _smooth_cnn():
    # tanh, no pooling: smooth, so finite differences are a fair reference
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.Tanh(),
        nn.Flatten(), nn.Linear(8 * 8 * 8, 10)).double()


def test_D1_D2_match_finite_differences():
    torch.manual_seed(0)
    model = _smooth_cnn().train()
    opt = SoftOptTorch(model.parameters(), model, _loss, seed=0)
    opt._names_params = list(model.named_parameters())
    batch = _batch()
    delta = [torch.randint(0, 2, p.shape).double() * 2 - 1 for p in model.parameters()]
    D1, D2 = opt._directional_D1_D2(batch, delta)

    def L_at(t):
        with torch.no_grad():
            for p, d in zip(model.parameters(), delta): p.add_(d, alpha=t)
            bn = model[1]; bn.track_running_stats = False
            v = float(_loss(model, batch))
            bn.track_running_stats = True
            for p, d in zip(model.parameters(), delta): p.add_(d, alpha=-t)
        return v
    h = 1e-4
    fd1 = (L_at(h) - L_at(-h)) / (2 * h)
    fd2 = (L_at(h) - 2 * L_at(0) + L_at(-h)) / h ** 2
    assert abs(D1 - fd1) < 1e-6 * max(1, abs(fd1))
    # PyTorch's forward-over-forward rule for batch_norm is ~1% off in
    # the second derivative; D1 is exact.
    assert abs(D2 - fd2) < 2e-2 * max(1, abs(fd2))


def test_correction_does_not_touch_bn_running_stats():
    torch.manual_seed(0)
    model = _cnn().train()
    opt = SoftOptTorch(model.parameters(), model, _loss, seed=0)
    opt._names_params = list(model.named_parameters())
    before = model[1].running_mean.clone(), int(model[1].num_batches_tracked)
    delta = [torch.ones_like(p) for p in model.parameters()]
    opt._directional_D1_D2(_batch(), delta)
    assert torch.equal(model[1].running_mean, before[0])
    assert int(model[1].num_batches_tracked) == before[1]


def test_correction_bounded_by_newton_bound():
    torch.manual_seed(0)
    model = _cnn().train()
    opt = SoftOptTorch(model.parameters(), model, _loss, lr=1e-3, seed=0)
    for s in range(5):
        opt.step(_batch(s))
        assert abs(opt.last['t_star']) <= 1e-3 + 1e-15


def test_trains_on_fixed_batch():
    torch.manual_seed(0)
    model = _cnn().train()
    opt = SoftOptTorch(model.parameters(), model, _loss, lr=1e-2, seed=0)
    b = _batch()
    first = opt.step(b)
    for _ in range(60):
        last = opt.step(b)
    assert last < 0.3 * first


def test_plain_hook_matches_torch_adam():
    torch.manual_seed(0)
    m1 = _cnn().eval(); m2 = _cnn().eval(); m2.load_state_dict(m1.state_dict())
    o1 = SoftOptTorch(m1.parameters(), m1, _loss, lr=1e-3)
    o2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
    for s in range(5):
        b = _batch(s)
        o1.step(b, _test_magnitude_source='plain')
        o2.zero_grad(); _loss(m2, b).backward(); o2.step()
    for p, q in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p, q, atol=1e-10)


def test_dropout_model_runs():
    model = nn.Sequential(nn.Flatten(), nn.Linear(192, 32), nn.ReLU(),
                          nn.Dropout(0.5), nn.Linear(32, 10)).double().train()
    opt = SoftOptTorch(model.parameters(), model, _loss, seed=0)
    opt.step(_batch())
    assert opt.last['mode'] in ('newton', 'fallback')


def test_maxpool_correction_avoids_plain_max_pool_kernel(monkeypatch):
    """Apple MPS has no forward-AD rule for plain max_pool2d. The correction
    pass must not call it, and must give the same D1/D2 as before."""
    torch.manual_seed(0)
    model = _cnn().train()                      # contains MaxPool2d
    opt = SoftOptTorch(model.parameters(), model, _loss, seed=0)
    opt._names_params = list(model.named_parameters())
    delta = [torch.randint(0, 2, p.shape).double() * 2 - 1 for p in model.parameters()]
    ref = opt._directional_D1_D2(_batch(), delta)

    def boom(*a, **k):
        raise NotImplementedError("forward AD with max_pool2d (simulated MPS)")
    monkeypatch.setattr(torch, "max_pool2d", boom)
    got = opt._directional_D1_D2(_batch(), delta)
    assert got == pytest.approx(ref, rel=1e-12)
    assert 'forward' not in model[3].__dict__   # patch removed afterwards


def test_readme_example_fits_known_model():
    """The README's PyTorch example: a fixed objective on a known model."""
    class HeatModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.k = nn.Parameter(torch.tensor([0.5]))
            self.T0 = nn.Parameter(torch.tensor([20.0]))

        def forward(self, t):
            return self.T0 * torch.exp(-self.k * t)

    def loss_fn(model, batch):
        t, measured = batch
        return ((model(t) - measured) ** 2).mean()

    g = torch.Generator().manual_seed(0)
    t = torch.linspace(0, 5, 50)
    measured = 30 * torch.exp(-0.8 * t) + 0.3 * torch.randn(50, generator=g)
    model = HeatModel()
    opt = SoftOptTorch(model.parameters(), model, loss_fn, lr=1e-2, seed=0)
    for _ in range(3000):
        loss = opt.step((t, measured))
    assert loss < 0.2
    assert abs(model.k.item() - 0.8) < 0.05
    assert abs(model.T0.item() - 30.0) < 1.0
