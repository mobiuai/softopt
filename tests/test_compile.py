"""
Tests for soft_compile -- generating g_delta from a plain-Python model.
"""
import numpy as np
import pytest
from softopt import soft_compile, SoftOpt


def test_polynomial_matches_finite_difference():
    def model(p):
        x, y, z = p
        return x * x + y * y * y + x * y * z

    g = soft_compile(model, n_params=3)
    theta = np.array([1.3, -0.7, 2.1])
    direction = np.array([1.0, -1.0, 1.0])
    h = 1e-6
    approx = (model(list(theta + h * direction)) - model(list(theta - h * direction))) / (2 * h)
    assert np.isclose(g(theta, direction), approx, rtol=1e-6)


def test_elementary_functions_work():
    def model(p):
        x, y = p
        return np.exp(x) * np.sin(y) + np.sqrt(x * x + 1.0)

    g = soft_compile(model, n_params=2)
    theta = np.array([0.4, 1.1])
    direction = np.array([1.0, -1.0])
    h = 1e-6
    approx = (model(list(theta + h * direction)) - model(list(theta - h * direction))) / (2 * h)
    assert np.isclose(g(theta, direction), approx, rtol=1e-5)


def test_fractional_and_negative_powers():
    def model(p):
        x, = p
        return x ** 2.5 + x ** -1

    g = soft_compile(model, n_params=1)
    theta = np.array([1.7])
    direction = np.array([1.0])
    h = 1e-6
    approx = (model(list(theta + h * direction)) - model(list(theta - h * direction))) / (2 * h)
    assert np.isclose(g(theta, direction), approx, rtol=1e-5)


def test_model_ignoring_params_is_rejected():
    def model(p):
        return 5.0

    with pytest.raises(TypeError, match="doesn't depend on the traced parameters"):
        soft_compile(model, n_params=2)


def test_verify_requires_n_params():
    with pytest.raises(ValueError, match="pass n_params"):
        soft_compile(lambda p: p[0] * p[0])


def test_verify_can_be_skipped():
    g = soft_compile(lambda p: p[0] * p[0], verify=False)
    assert np.isclose(g(np.array([3.0]), np.array([1.0])), 6.0)


def test_compiled_model_drives_the_optimizer():
    """End-to-end: compile a model, optimize with it, reach the minimum."""
    def model(p):
        x, y = p
        return (x - 2.0) ** 2 + (y + 1.0) ** 2

    g = soft_compile(model, n_params=2)
    opt = SoftOpt(2, g, lr=0.1, seed=0)
    theta = np.array([0.0, 0.0])
    for _ in range(300):
        grad = np.array([2 * (theta[0] - 2.0), 2 * (theta[1] + 1.0)])
        theta = opt.step(theta, grad)
    assert np.allclose(theta, [2.0, -1.0], atol=1e-2)


def _fd_check(model, n, seed=77):
    from softopt import soft_compile
    g = soft_compile(model, n_params=n)
    rng = np.random.default_rng(seed)
    th = rng.normal(0.4, 0.9, n)
    dl = rng.normal(0, 1, n)
    exact = g(th, dl)
    h = 1e-6
    fd = (float(model(list(th + h * dl))) - float(model(list(th - h * dl)))) / (2 * h)
    scale = max(abs(fd), abs(exact), 1.0)
    assert abs(exact - fd) / scale < 1e-4, f"exact={exact} fd={fd}"


def test_loops_over_parameters():
    def model(p):
        x, = p
        total = x * 0.0
        for i in range(5):
            total = total + x ** (i + 1) / (i + 1)
        return total
    _fd_check(model, 1)


def test_branching_on_parameter_value():
    def model(p):
        x, = p
        return x * x if x > 0 else -x * x * x
    _fd_check(model, 1)


def test_abs_and_max():
    def model(p):
        x, y = p
        return abs(x) + max(y, 0.0)
    _fd_check(model, 2)


def test_numpy_log_and_tanh():
    def model(p):
        x, = p
        return np.log(np.exp(x) + 2.0) + np.tanh(x)
    _fd_check(model, 1)


def test_base_to_the_power_of_a_parameter():
    def model(p):
        x, = p
        return 2.0 ** x + np.e ** x
    _fd_check(model, 1)


def test_object_array_quadratic_form():
    def model(p):
        v = np.array(p, dtype=object)
        M = np.array([[1.0, 0.5], [0.5, 2.0]])
        return sum(v[i] * M[i, j] * v[j] for i in range(2) for j in range(2))
    _fd_check(model, 2)


def test_model_that_casts_to_float_is_caught():
    """np.array(p) on a list of SoftNumbers silently converts to floats,
    dropping the derivative -- verification must catch this."""
    def model(p):
        v = np.array(p)      # drops the soft components
        return float(v @ v)
    with pytest.raises(TypeError, match="doesn't depend on the traced parameters"):
        soft_compile(model, n_params=2)
