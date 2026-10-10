"""calibrate=True: the soft Jacobian is exact, the default path is unchanged, and calibration helps on a
biased model."""
import importlib.util
import os
import numpy as np
import pytest
from softopt import SoftOpt, soft_compile

N = 4
A = np.array([1.0, 2.0, 0.5, 1.5]); CC = np.array([0.3, -0.2, 0.5, 0.1])


def model(th, be):
    return sum(A[i] * (th[i] - CC[i] - be[i]) ** 2 + 0.3 * np.cos(th[i] + be[i]) for i in range(N))


def test_soft_jacobian_matches_finite_differences():
    cm = soft_compile(model, n_params=N, n_bias=N).calibration
    rng = np.random.default_rng(0)
    x, d, b = rng.normal(size=N), rng.choice([-1.0, 1.0], N), rng.normal(0, 0.2, N)
    D1, jac = cm.d1_and_jac(x, d, b); h = 1e-6
    for j in range(N):
        e = np.zeros(N); e[j] = h
        fd = (cm.d1_d2(x, d, b + e)[0] - cm.d1_d2(x, d, b - e)[0]) / (2 * h)
        assert abs(jac[j] - fd) < 1e-6 * max(1, abs(fd))
    fd1 = (cm.value(x + h * d, b) - cm.value(x - h * d, b)) / (2 * h)
    assert abs(D1 - fd1) < 1e-6 * max(1, abs(fd1))
    D1b, D2 = cm.d1_d2(x, d, b)
    fd2 = (cm.d1_d2(x + h * d, d, b)[0] - cm.d1_d2(x - h * d, d, b)[0]) / (2 * h)
    assert abs(D2 - fd2) < 1e-5 * max(1, abs(fd2))


def test_nominal_model_is_beta_zero():
    g = soft_compile(model, n_params=N, n_bias=N)
    x, d = np.full(N, 0.2), np.ones(N)
    assert abs(g(x, d) - g.calibration.d1_d2(x, d, np.zeros(N))[0]) < 1e-12


def test_default_path_unchanged():
    """calibrate=False must be bit-identical to the same SoftOpt without the calibrate code."""
    g = soft_compile(lambda th: model(th, [0.0] * N), n_params=N)
    a = SoftOpt(N, g, lr=0.05, seed=3); b = SoftOpt(N, g, lr=0.05, seed=3, calibrate=False)
    x1 = x2 = np.zeros(N); rng = np.random.default_rng(4)
    for _ in range(30):
        gr = rng.normal(size=N)
        x1, x2 = a.step(x1, gr), b.step(x2, gr)
    assert np.array_equal(x1, x2)


def _run(s, mode, budget=200):
    rng_b = np.random.default_rng(500 + s); btrue = rng_b.normal(0, 0.3, N); noise = np.random.default_rng(900 + s)
    meas = lambda x: model(x, btrue) + 0.05 * noise.normal()                               # noqa: E731
    g = soft_compile(model, n_params=N, n_bias=N)
    kw = dict(calibrate=True, bias_scale=0.3) if mode == 'cal' else dict(slope='measured')
    opt = SoftOpt(N, g, lr=0.05, seed=s * 7 + 2, **kw)
    x = np.zeros(N); rng = np.random.default_rng(s * 7 + 1)
    for _ in range(budget // 2):
        d = rng.choice([-1.0, 1.0], N); sl = (meas(x + 0.1 * d) - meas(x - 0.1 * d)) / 0.2
        x = opt.step(x, sl * d, direction=d, measured_slope=sl)
    from scipy.optimize import minimize
    return model(x, btrue) - minimize(lambda z: model(z, btrue), CC + btrue).fun


def test_calibration_helps_on_a_biased_model():
    pytest.importorskip('scipy')
    cal = np.array([_run(s, 'cal') for s in range(8)]); mea = np.array([_run(s, 'measured') for s in range(8)])
    assert np.median(cal) < np.median(mea) and (cal < mea).sum() >= 6


def test_beta_free_calibration_runs():
    g = soft_compile(lambda th: model(th, [0.0] * N), n_params=N)
    opt = SoftOpt(N, g, lr=0.05, seed=0, calibrate=True)
    x = np.zeros(N); rng = np.random.default_rng(1)
    for _ in range(20):
        d = rng.choice([-1.0, 1.0], N); sl = float(rng.normal())
        x = opt.step(x, sl * d, direction=d, measured_slope=sl)
    assert np.all(np.isfinite(x)) and opt.calibration['beta'].size == 0


def test_calibrate_requires_direction_and_scale():
    g = soft_compile(model, n_params=N, n_bias=N)
    with pytest.raises(ValueError):
        SoftOpt(N, g, calibrate=True)                         # bias parameters need bias_scale
    opt = SoftOpt(N, g, calibrate=True, bias_scale=0.1)
    with pytest.raises(ValueError):
        opt.step(np.zeros(N), np.zeros(N))
