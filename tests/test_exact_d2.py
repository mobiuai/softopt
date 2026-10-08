"""
Tests for the exact-curvature path (0.6.4): nested soft numbers give D2 with no step size, and a
hand-written g_delta without d1_d2 keeps the 0.6.3 finite-difference behaviour bit for bit.
"""
import numpy as np
from softopt import soft_compile, SoftOpt


def _model(p):
    x, y, z = p
    return np.cos(x) * np.sin(y) + x * x * z + np.exp(0.3 * y) * z


def test_soft_compile_attaches_exact_d1_d2():
    g = soft_compile(_model, n_params=3)
    assert hasattr(g, 'd1_d2')
    rng = np.random.default_rng(0)
    for _ in range(10):
        t, d = rng.normal(0, 1, 3), rng.choice([-1.0, 1.0], 3)
        D1, D2 = g.d1_d2(t, d)
        assert np.isclose(D1, g(t, d), rtol=1e-12)
        h = 1e-5
        D2_fd = (g(t + h * d, d) - g(t - h * d, d)) / (2 * h)
        assert np.isclose(D2, D2_fd, rtol=1e-6, atol=1e-8)


def test_hand_written_g_delta_unchanged():
    g = soft_compile(_model, n_params=3)
    hand = lambda t, d: g(t, d)                      # no d1_d2 attribute -> finite difference
    a = SoftOpt(3, hand, lr=0.02, seed=1)
    b = SoftOpt(3, g, lr=0.02, seed=1, exact_d2=False)
    ta = tb = np.array([0.3, -0.2, 0.5])
    for _ in range(20):
        gr = np.array([g(ta, e) for e in np.eye(3)])
        ta, tb = a.step(ta, gr), b.step(tb, gr)
    assert np.array_equal(ta, tb)


def test_exact_and_fd_paths_agree():
    g = soft_compile(_model, n_params=3)
    a = SoftOpt(3, g, lr=0.02, seed=2)               # exact (auto)
    b = SoftOpt(3, g, lr=0.02, seed=2, exact_d2=False)
    ta = tb = np.array([0.3, -0.2, 0.5])
    for _ in range(50):
        ta = a.step(ta, np.array([g(ta, e) for e in np.eye(3)]))
        tb = b.step(tb, np.array([g(tb, e) for e in np.eye(3)]))
    assert np.allclose(ta, tb, atol=1e-6)
