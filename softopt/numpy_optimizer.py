"""
SoftOpt: a COMPLETE, standalone optimizer -- not a wrapper around Adam.
Exactly one step() call does both the Adam-style base update AND the
validated Newton-style correction. You don't bring your own optimizer;
this IS the optimizer, in the same sense that torch.optim.Adam IS an
optimizer (internally momentum + RMS-scaling), not a wrapper around SGD.

Requirement (the "known computation graph" criterion, validated across
nine domains today): you must supply g_delta(theta, delta) -> the exact
directional derivative of your TRUE, differentiable objective along
delta, computed via Klein-Maimon soft-number propagation through your
own known model (circuit, projection, force field, ...). If you cannot
write g_delta without sampling the world, SoftOpt is not the right tool
-- use plain Adam/SGD instead.

Three correction modes (see README for details):
- mode="newton" (default): bounded Newton step, best for one-signed
  curvature (most physics/circuit problems).
- mode="mobius": bounded, sign-safe Mobius-map step (book Ch.5.3), for
  landscapes whose curvature isn't reliably one-signed (e.g. QAOA).
- mode="auto" (EXPERIMENTAL, not yet reliable -- see README): attempts
  to diagnose which of the above fits your problem by sampling D2's
  sign near the starting point. Tested against GRAPE (a domain that
  needs "newton" with high confidence) it only picked "newton" 40% of
  the time -- a single starting region's local curvature sign is not
  a reliable predictor of the right mode for the whole optimization.
  Shipped anyway, clearly labeled, so you can inspect `opt.detected_mode`
  and help characterize when it does/doesn't work, but do not rely on
  it yet -- the empirically-validated way to choose is still: try both
  modes on a short run and keep whichever beats plain Adam.
"""
import numpy as np


def _mobius_B(x, y):
    """Book Ch.5.3's Möbius map, bounded to [-1,1] -- a sign-safe
    alternative to raw Newton division for landscapes whose curvature
    isn't reliably one-signed (e.g. QAOA, empirically)."""
    denom = abs(x) + abs(y)
    if denom < 1e-12:
        return 0.0
    sgn = 1.0 if x >= 0 else -1.0
    return y * sgn / denom


class SoftOpt:
    def __init__(self, n_params, g_delta, lr=0.02, betas=(0.9, 0.999), eps=1e-8,
                 mode="newton", auto_detect_steps=30, auto_detect_threshold=0.8,
                 newton_lo=-0.5, newton_hi=0.5, eta_fallback=0.3,
                 eta_mobius=0.3, h_fd=1e-4, seed=None,
                 _test_frozen_constant=1.0):
        self.g_delta = g_delta
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        if mode not in ("newton", "mobius", "auto"):
            raise ValueError("mode must be 'newton', 'mobius', or 'auto'")
        self.mode = mode
        self.auto_detect_steps = auto_detect_steps
        self.auto_detect_threshold = auto_detect_threshold
        self._auto_pending = (mode == "auto")
        self.detected_mode = None if mode == "auto" else mode
        self.newton_lo = newton_lo
        self.newton_hi = newton_hi
        self.eta_fallback = eta_fallback
        self.eta_mobius = eta_mobius
        self.h_fd = h_fd
        self.m = None
        self.v = None
        self.t = 0
        self.rng = np.random.default_rng(seed)
        self._test_frozen_constant = _test_frozen_constant

    def _diagnose_mode(self, theta0):
        """Run once, before any optimization step: sample D2's sign
        across several nearby points (not just theta0 itself -- a single
        point's local curvature sign is not always representative) and
        several random directions per point, matching the methodology
        that correctly distinguished GRAPE/VQE's one-signed curvature
        from QAOA's sign-indefinite curvature."""
        n = theta0.shape[0]
        n_points = 5
        dirs_per_point = max(1, self.auto_detect_steps // n_points)
        signs = []
        for p in range(n_points):
            point = theta0 if p == 0 else theta0 + self.rng.normal(scale=0.1, size=n)
            for _ in range(dirs_per_point):
                delta = self.rng.choice([-1.0, 1.0], size=n)
                gp = self.g_delta(point + self.h_fd * delta, delta)
                gm = self.g_delta(point - self.h_fd * delta, delta)
                D2 = (gp - gm) / (2 * self.h_fd)
                signs.append(1.0 if D2 >= 0 else -1.0)
        signs = np.array(signs)
        agreement = max(np.mean(signs > 0), np.mean(signs < 0))
        self.detected_mode = "newton" if agreement >= self.auto_detect_threshold else "mobius"
        self.mode = self.detected_mode

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

        if self._auto_pending:
            self._diagnose_mode(theta)
            self._auto_pending = False

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
        D1 = self.g_delta(theta_after_adam, delta)
        gp = self.g_delta(theta_after_adam + self.h_fd * delta, delta)
        gm = self.g_delta(theta_after_adam - self.h_fd * delta, delta)
        D2_real = (gp - gm) / (2 * self.h_fd)

        if _test_magnitude_source in (None, 'real'):
            D2_used = D2_real
        elif _test_magnitude_source == 'frozen':
            D2_used = self._test_frozen_constant
        elif _test_magnitude_source in ('foreign_point_and_dir', 'foreign_point_same_dir'):
            theta_f = _test_foreign_sampler(gen_rng)
            delta_f = gen_rng.choice([-1.0, 1.0], size=n) if _test_magnitude_source == 'foreign_point_and_dir' else delta
            gp_f = self.g_delta(theta_f + self.h_fd * delta_f, delta_f)
            gm_f = self.g_delta(theta_f - self.h_fd * delta_f, delta_f)
            D2_used = abs((gp_f - gm_f) / (2 * self.h_fd))
        else:
            raise ValueError(_test_magnitude_source)

        if self.mode == "mobius":
            B = _mobius_B(D1, D2_used)
            t_star = self.eta_mobius * B
        else:
            activate = D2_real > 1e-8
            if activate and D2_used > 1e-9:
                t_star = float(np.clip(-D1 / D2_used, self.newton_lo, self.newton_hi))
            else:
                t_star = float(np.clip(-self.eta_fallback * D1, self.newton_lo, self.newton_hi))

        return theta_after_adam + t_star * delta
