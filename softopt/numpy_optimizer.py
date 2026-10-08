"""
SoftOpt: each step() call performs an update followed by a Newton correction
computed from your own model.

Requirement: you supply g_delta(theta, delta) -> the exact directional
derivative of your true objective along delta, computed from your own
known model (circuit, projection, physical law). `soft_compile` generates
this for you from a model written in ordinary Python.

The correction is a bounded Newton step, t* = -D1 / D2 along a direction delta.
D2 (the curvature) always comes from your model. When g_delta comes from
`soft_compile` (or carries a `d1_d2(theta, delta) -> (D1, D2)` attribute), D2 is
EXACT: one pass of the model on nested soft numbers (two soft axes, each with
eps^2 = 0; the mixed term is D2) -- no step size. For a hand-written g_delta
without that attribute, D2 is a central finite difference OF the exact
soft-number derivative, controlled by h_fd (the 0.6.3 behaviour).

D1 (the slope) has two possible sources, set by `slope`:

  slope="model"     (default) D1 is the model's exact directional derivative
                    along a fresh random direction. Robust to very noisy
                    readings, but the correction pulls towards the MODEL's
                    optimum: if the model is biased, so is the answer.

  slope="measured"  D1 is the slope you measured on the real system, along the
                    direction you measured it in (an SPSA difference gives you
                    both for free). The model only supplies the curvature, so
                    the fixed point is where the MEASURED slope vanishes: the
                    real system's optimum, even when the model is off. Needs
                    readings clean enough that the measured slope is informative.

`spsa_gradient` returns the gradient estimate, the direction and the measured
slope in one call, ready for either mode.
"""
import numpy as np


class SoftOpt:
    def __init__(self, n_params, g_delta, lr=0.02, betas=(0.9, 0.999), eps=1e-8,
                 newton_lo=-0.5, newton_hi=0.5, eta_fallback=0.3,
                 h_fd=1e-4, seed=None, slope="model", exact_d2="auto", _test_frozen_constant=1.0):
        if slope not in ("model", "measured"):
            raise ValueError(f'slope must be "model" or "measured", got {slope!r}')
        self.slope = slope
        self.g_delta = g_delta
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.newton_lo = newton_lo
        self.newton_hi = newton_hi
        self.eta_fallback = eta_fallback
        self.h_fd = h_fd
        if exact_d2 not in ("auto", True, False):
            raise ValueError('exact_d2 must be "auto", True or False')
        self._d1_d2 = getattr(g_delta, "d1_d2", None) if exact_d2 else None
        if exact_d2 is True and self._d1_d2 is None:
            raise ValueError("exact_d2=True needs a g_delta with a d1_d2 attribute (e.g. from soft_compile)")
        self.m = None
        self.v = None
        self.t = 0
        self.rng = np.random.default_rng(seed)
        self._ablation_rng = None
        self._test_frozen_constant = _test_frozen_constant

    def step(self, theta, grad_estimate, direction=None, measured_slope=None,
             _test_magnitude_source=None, _test_foreign_sampler=None, _test_rng=None):
        """One complete optimizer step: Adam base update + Newton-style
        correction from the known model, in a single call.
        theta: current parameters (np.ndarray)
        grad_estimate: a gradient estimate of the loss w.r.t. theta (e.g.
            from SPSA or backprop) -- exactly what you'd normally hand
            to Adam.
        direction, measured_slope: required when slope="measured", ignored
            otherwise. The direction your readings probed and the slope you
            measured along it at theta, e.g. for SPSA
            (f(theta + c*d) - f(theta - c*d)) / (2c) along d.
            spsa_gradient() returns both.
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
        if self.slope == "measured":
            if direction is None or measured_slope is None:
                raise ValueError('slope="measured" needs direction= and measured_slope= on every '
                                 'step (spsa_gradient() returns both)')
            direction = np.asarray(direction, dtype=float)
            if direction.shape != theta.shape:
                raise ValueError(f"direction has shape {direction.shape}, theta has {theta.shape}")

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
        if self.slope == "measured":
            delta = direction
        else:
            gen_rng = _test_rng if _test_rng is not None else self.rng
            delta = gen_rng.choice([-1.0, 1.0], size=n)

        # The ablation arms must not disturb the probe-direction stream, or
        # a corrupted arm would also change every direction that follows and
        # the comparison would no longer be like-for-like.
        if self._ablation_rng is None:
            self._ablation_rng = np.random.default_rng(
                self.rng.integers(0, 2**63 - 1))
        if self._d1_d2 is not None:          # exact: nested soft numbers, one pass
            D1_model, D2_real = self._d1_d2(theta_after_adam, delta)
        else:                                # finite difference of the exact g_delta
            D1_model = None
            gp = self.g_delta(theta_after_adam + self.h_fd * delta, delta)
            gm = self.g_delta(theta_after_adam - self.h_fd * delta, delta)
            D2_real = (gp - gm) / (2 * self.h_fd)
        if self.slope == "measured":
            D1 = float(measured_slope)       # from the real system
        else:
            D1 = D1_model if D1_model is not None else self.g_delta(theta_after_adam, delta)

        if _test_magnitude_source in (None, 'real'):
            D2_used = D2_real
        elif _test_magnitude_source == 'frozen':
            D2_used = self._test_frozen_constant
        elif _test_magnitude_source in ('foreign_point_and_dir', 'foreign_point_same_dir'):
            theta_f = _test_foreign_sampler(self._ablation_rng)
            delta_f = (self._ablation_rng.choice([-1.0, 1.0], size=n)
                       if _test_magnitude_source == 'foreign_point_and_dir' else delta)
            if self._d1_d2 is not None:
                D2_used = abs(self._d1_d2(theta_f, delta_f)[1])
            else:
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


def spsa_gradient(measure, theta, c=0.1, rng=None):
    """One SPSA probe of a noisy objective.

    measure: callable theta -> one (noisy) reading of the real system.
    Returns (grad_estimate, direction, measured_slope), where measured_slope
    is (measure(theta + c*d) - measure(theta - c*d)) / (2c) along the random
    +-1 direction d, and grad_estimate = measured_slope * d. Costs 2 readings.
    """
    theta = np.asarray(theta, dtype=float)
    rng = np.random.default_rng() if rng is None else rng
    d = rng.choice([-1.0, 1.0], size=theta.shape)
    slope = (measure(theta + c * d) - measure(theta - c * d)) / (2 * c)
    return slope * d, d, float(slope)
