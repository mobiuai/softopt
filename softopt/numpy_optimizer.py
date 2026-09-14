"""
SoftOpt: each step() call performs an update followed by a Newton correction
computed from your own model.

Requirement: you supply g_delta(theta, delta) -> the exact directional
derivative of your true objective along delta, computed from your own
known model (circuit, projection, physical law). `soft_compile` generates
this for you from a model written in ordinary Python.

The correction is a bounded Newton step. The first directional derivative is
exact, propagated through your model by soft-number arithmetic. The second is
a central finite difference OF that exact derivative -- an approximation, with
the usual step-size trade-off, controlled by h_fd.
"""
import numpy as np


class SoftOpt:
    def __init__(self, n_params, g_delta, lr=0.02, betas=(0.9, 0.999), eps=1e-8,
                 newton_lo=-0.5, newton_hi=0.5, eta_fallback=0.3,
                 h_fd=1e-4, seed=None, _test_frozen_constant=1.0):
        self.g_delta = g_delta
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.newton_lo = newton_lo
        self.newton_hi = newton_hi
        self.eta_fallback = eta_fallback
        self.h_fd = h_fd
        self.m = None
        self.v = None
        self.t = 0
        self.rng = np.random.default_rng(seed)
        self._ablation_rng = None
        self._test_frozen_constant = _test_frozen_constant

    def step(self, theta, grad_estimate, _test_magnitude_source=None, _test_foreign_sampler=None, _test_rng=None):
        """One complete optimizer step: Adam base update + Newton-style
        correction from the known model, in a single call.
        theta: current parameters (np.ndarray)
        grad_estimate: a gradient estimate of the loss w.r.t. theta (e.g.
            from SPSA or backprop) -- exactly what you'd normally hand
            to Adam.
        Returns: theta for the next step.

        The _test_* arguments are TEST-ONLY hooks used exclusively by the
        causal-validation ablation harness to deliberately corrupt the
        correction (feed a frozen constant or a foreign point's curvature
        instead of the genuine one) so that "does real content matter"
        can be checked against the EXACT SAME code path real usage takes.
        Normal usage never sets these; leaving them unset gives the
        genuine, real behavior.
        """
        theta = np.asarray(theta, dtype=float)
        grad_estimate = np.asarray(grad_estimate, dtype=float)

        # --- Adam base update ---
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
        self.m = self.b1 * self.m + (1 - self.b1) * grad_estimate
        self.v = self.b2 * self.v + (1 - self.b2) * grad_estimate ** 2
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        theta_after_adam = theta - self.lr * mh / (np.sqrt(vh) + self.eps)

        if _test_magnitude_source == 'plain':
            return theta_after_adam

        # --- correction from the known, exact model ---
        n = theta.shape[0]
        gen_rng = _test_rng if _test_rng is not None else self.rng
        delta = gen_rng.choice([-1.0, 1.0], size=n)

        # The ablation arms must not disturb the probe-direction stream, or
        # a corrupted arm would also change every direction that follows and
        # the comparison would no longer be like-for-like.
        if self._ablation_rng is None:
            self._ablation_rng = np.random.default_rng(
                self.rng.integers(0, 2**63 - 1))
        D1 = self.g_delta(theta_after_adam, delta)
        gp = self.g_delta(theta_after_adam + self.h_fd * delta, delta)
        gm = self.g_delta(theta_after_adam - self.h_fd * delta, delta)
        D2_real = (gp - gm) / (2 * self.h_fd)

        if _test_magnitude_source in (None, 'real'):
            D2_used = D2_real
        elif _test_magnitude_source == 'frozen':
            D2_used = self._test_frozen_constant
        elif _test_magnitude_source in ('foreign_point_and_dir', 'foreign_point_same_dir'):
            theta_f = _test_foreign_sampler(self._ablation_rng)
            delta_f = (self._ablation_rng.choice([-1.0, 1.0], size=n)
                       if _test_magnitude_source == 'foreign_point_and_dir' else delta)
            gp_f = self.g_delta(theta_f + self.h_fd * delta_f, delta_f)
            gm_f = self.g_delta(theta_f - self.h_fd * delta_f, delta_f)
            D2_used = abs((gp_f - gm_f) / (2 * self.h_fd))
        else:
            raise ValueError(_test_magnitude_source)

        if D2_real > 1e-8 and D2_used > 1e-9:
            t_star = float(np.clip(-D1 / D2_used, self.newton_lo, self.newton_hi))
        else:
            t_star = float(np.clip(-self.eta_fallback * D1, self.newton_lo, self.newton_hi))

        return theta_after_adam + t_star * delta
