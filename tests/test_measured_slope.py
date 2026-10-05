"""slope="measured": slope from the real system, curvature from the model."""
import numpy as np
import pytest
from softopt import SoftOpt, soft_compile, spsa_gradient

N = 6
SHIFT = np.array([0.8, -0.6, 0.5, -0.4, 0.7, -0.5])   # the model's optimum is off by this


def truth(x):
    return float(np.sum(np.asarray(x) ** 2))


def biased_model(th):
    return sum((t - s) ** 2 for t, s in zip(th, SHIFT))


def _run(slope, steps=400, seed=0):
    g_delta = soft_compile(biased_model, n_params=N)
    opt = SoftOpt(N, g_delta, lr=0.05, seed=seed, slope=slope)
    rng = np.random.default_rng(seed)
    x = np.full(N, 2.0)
    for _ in range(steps):
        g, d, s = spsa_gradient(truth, x, c=0.1, rng=rng)
        x = opt.step(x, g, direction=d, measured_slope=s)
    return x


def test_measured_slope_finds_the_real_optimum_despite_a_biased_model():
    x = _run("measured")
    assert np.allclose(x, 0.0, atol=0.05)          # the system's optimum


def test_model_slope_follows_the_model():
    x = _run("model")
    # documented behaviour: the correction pulls towards the model's optimum
    assert np.linalg.norm(x - SHIFT) < np.linalg.norm(x - 0.0)


def test_measured_mode_requires_direction_and_slope():
    opt = SoftOpt(N, soft_compile(biased_model, n_params=N), slope="measured")
    with pytest.raises(ValueError, match="direction"):
        opt.step(np.ones(N), np.ones(N))
    with pytest.raises(ValueError, match="shape"):
        opt.step(np.ones(N), np.ones(N), direction=np.ones(N + 1), measured_slope=1.0)


def test_unknown_slope_mode_rejected():
    with pytest.raises(ValueError, match="slope"):
        SoftOpt(N, lambda t, d: 0.0, slope="both")


def test_model_mode_ignores_direction_arguments():
    g_delta = soft_compile(biased_model, n_params=N)
    a = SoftOpt(N, g_delta, lr=0.05, seed=3)
    b = SoftOpt(N, g_delta, lr=0.05, seed=3)
    x = np.ones(N)
    xa = a.step(x, 2 * x)
    xb = b.step(x, 2 * x, direction=-np.ones(N), measured_slope=123.0)
    assert np.array_equal(xa, xb)


def test_spsa_gradient_on_a_linear_function_is_exact():
    w = np.arange(1.0, N + 1)
    g, d, s = spsa_gradient(lambda x: float(w @ x), np.zeros(N), c=0.3, rng=np.random.default_rng(0))
    assert s == pytest.approx(w @ d)
    assert np.allclose(g, s * d)
    assert set(np.unique(d)) <= {-1.0, 1.0}


def test_frozen_ablation_hook_works_in_measured_mode():
    g_delta = soft_compile(biased_model, n_params=N)
    opt = SoftOpt(N, g_delta, lr=0.05, seed=0, slope="measured")
    x = np.ones(N)
    g, d, s = spsa_gradient(truth, x, rng=np.random.default_rng(1))
    out = opt.step(x, g, direction=d, measured_slope=s, _test_magnitude_source="frozen")
    assert out.shape == x.shape and np.all(np.isfinite(out))
