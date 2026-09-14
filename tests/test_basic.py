"""
Basic sanity tests for SoftOpt. These are NOT the domain-validation
benchmarks (see benchmarks/) -- just fast checks that the optimizer
runs correctly and does what it's supposed to on a trivial problem.
"""
import numpy as np
import pytest
from softopt import SoftOpt


def quadratic_g_delta(theta, delta):
    """g_delta for f(theta) = |theta|^2 -- exact directional derivative
    is 2*theta . delta."""
    return float(2 * theta @ delta)


def test_converges_on_convex_quadratic():
    opt = SoftOpt(5, quadratic_g_delta, lr=0.1, seed=0)
    theta = np.ones(5)
    for _ in range(200):
        grad = 2 * theta  # exact gradient, standing in for a real gradient estimate
        theta = opt.step(theta, grad)
    assert np.allclose(theta, 0.0, atol=1e-2)


def test_step_is_deterministic_given_seed():
    opt1 = SoftOpt(3, quadratic_g_delta, lr=0.05, seed=42)
    opt2 = SoftOpt(3, quadratic_g_delta, lr=0.05, seed=42)
    theta1 = theta2 = np.array([1.0, 2.0, 3.0])
    for _ in range(10):
        theta1 = opt1.step(theta1, 2 * theta1)
        theta2 = opt2.step(theta2, 2 * theta2)
    assert np.allclose(theta1, theta2)


def test_plain_arm_skips_correction():
    """The _test_magnitude_source='plain' hook should return exactly the
    Adam-only update, with no correction applied."""
    opt = SoftOpt(3, quadratic_g_delta, lr=0.1, seed=0)
    theta = np.array([1.0, 1.0, 1.0])
    result = opt.step(theta, 2 * theta, _test_magnitude_source="plain")
    assert np.all(np.isfinite(result))
