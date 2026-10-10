"""
Online calibration of the model against the real system (calibrate=True).

The model you give SoftOpt is rarely exactly the system you are optimising: a simulator with approximate
parameters, an ideal circuit on a device with gate errors, a nominal plant. With calibrate=True every SPSA
pair you already measure is also used to correct the model while the optimisation runs -- no extra readings.

What the soft algebra does here
  The model is evaluated on nested soft numbers: one axis (eps1) along the probe direction d, the other
  (eps2) along a bias parameter beta_j. The mixed eps1*eps2 part of the result is d D1 / d beta_j exactly --
  how the model's predicted slope responds to each uncertain parameter. That is the measurement Jacobian of a
  Kalman filter on beta: each measured slope s along d is explained as
        s  ~  kappa * ( D1_model(x, d; beta) + dg . d )
  kappa is the contrast the device loses (noise), and dg is the part of the slope the model cannot explain at
  all (its eps-part is d itself, so it costs nothing). Two filters run on the same readings -- dg local
  (forgotten at 0.95) and dg settled (not forgotten) -- and the one that has predicted the readings better so
  far (log-likelihood) supplies the correction.

The correction is SoftOpt's bounded Newton step on the CALIBRATED model, scaled by the filter's own
confidence D1^2 / (D1^2 + var D1), and checked before it is taken: the calibrated model is evaluated at the
target, and if it predicts no improvement the step is shortened to the minimum of the parabola through
f(0), D1 and f(step).

Two ways to use it
  * calibrate=True with a plain model(theta): no bias parameters; only kappa and dg are learned.
  * soft_compile(model(theta, beta), n_params, n_bias=k) and SoftOpt(..., calibrate=True, bias_scale=s):
    beta are the model's uncertain parameters (offsets from their nominal values, 0 = nominal); bias_scale is
    their expected size (a number or one per parameter), on the PHYSICAL scale -- a prior far too wide lets the
    filter fit noise.

Validated (development + held-out seeds) where the bias structure is right: gate calibration, adaptive optics,
H2 with coherent gate errors, robot sim-to-real. When the model lacks a major error type the median is still at
or below Adam but single runs can be worse; when the model is already exact, calibrate=False is better.
"""
import numpy as np

from .soft_number import SoftNumber


def _sn(e, a, c, v):
    """v + c*eps_in + a*eps_out + e*eps_in*eps_out as a nested SoftNumber."""
    return SoftNumber(SoftNumber(e, a), SoftNumber(c, v))


def _parts(r):
    if isinstance(r, SoftNumber):
        outer_a, outer_b = r.a, r.b
        e, a = (outer_a.a, outer_a.b) if isinstance(outer_a, SoftNumber) else (0.0, outer_a)
        c, v = (outer_b.a, outer_b.b) if isinstance(outer_b, SoftNumber) else (0.0, outer_b)
        return float(v), float(c), float(a), float(e)
    return float(r), 0.0, 0.0, 0.0


class CompiledModel:
    """model_fn(theta) or model_fn(theta, beta) on nested soft numbers: D1/D2 along d, d D1 / d beta, value."""

    def __init__(self, model_fn, n_bias=0):
        self.f, self.nb = model_fn, int(n_bias)

    def _call(self, th, bs):
        return self.f(th, bs) if self.nb else self.f(th)

    def d1_d2(self, x, d, beta):
        th = [_sn(0.0, float(di), float(di), float(xi)) for xi, di in zip(x, d)]
        bs = [_sn(0.0, 0.0, 0.0, float(b)) for b in beta]
        _, D1, _, D2 = _parts(self._call(th, bs))
        return D1, D2

    def d1_and_jac(self, x, d, beta):
        th = [_sn(0.0, 0.0, float(di), float(xi)) for xi, di in zip(x, d)]          # eps_in along d
        jac, D1 = np.zeros(self.nb), None
        for j in range(self.nb):
            bs = [_sn(0.0, 1.0 if i == j else 0.0, 0.0, float(b)) for i, b in enumerate(beta)]  # eps_out along beta_j
            _, D1, _, jac[j] = _parts(self._call(th, bs))
        if D1 is None:
            _, D1, _, _ = _parts(self._call(th, []))
        return D1, jac

    def value(self, x, beta):
        th = [_sn(0.0, 0.0, 0.0, float(xi)) for xi in x]
        bs = [_sn(0.0, 0.0, 0.0, float(b)) for b in beta]
        return _parts(self._call(th, bs))[0]


class Calibrator:
    FORGETS = (0.95, 1.0)

    def __init__(self, n, model, bias_scale=None):
        self.n, self.M = n, model
        nb = model.nb
        if nb and bias_scale is None:
            raise ValueError('calibrate=True with bias parameters needs bias_scale= (their expected size, '
                             'on the physical scale)')
        self.ib, self.ik, self.ig = slice(0, nb), nb, slice(nb + 1, nb + 1 + n)
        self.F = []
        for fg in self.FORGETS:
            st = np.zeros(nb + 1 + n); st[nb] = 1.0
            P = np.zeros((nb + 1 + n,) * 2)
            if nb:
                P[self.ib, self.ib] = np.diag(np.broadcast_to(np.asarray(bias_scale, float) ** 2, (nb,)))
            P[nb, nb] = 0.2 ** 2
            self.F.append(dict(forget=fg, state=st, P=P, r2=None, ll=0.0))
        self.t = 0

    def update(self, x, d, slope):
        """One measured slope along d at x: update both filters and their evidence."""
        ib, ik, ig, n = self.ib, self.ik, self.ig, self.n
        for f in self.F:
            st, P = f['state'], f['P']
            D1, jac = self.M.d1_and_jac(x, d, st[ib])
            pred = D1 + st[ig] @ d
            innov = slope - st[ik] * pred
            if f['r2'] is None:
                f['r2'] = innov ** 2
                P[ig, ig] = np.eye(n) * (f['r2'] / n)
            else:
                f['r2'] = 0.95 * f['r2'] + 0.05 * innov ** 2
            P[ig, :] /= np.sqrt(f['forget']); P[:, ig] /= np.sqrt(f['forget'])
            H = np.r_[st[ik] * jac, pred, st[ik] * d]
            S = H @ P @ H + f['r2']
            if S <= 0 or not np.isfinite(S):
                continue
            if self.t > 0:
                f['ll'] += -0.5 * (np.log(S) + innov ** 2 / S)
            K = P @ H / S
            f['state'] = st + K * innov
            f['P'] = P - np.outer(K, H @ P)
        self.t += 1

    def best(self):
        return self.F[int(np.argmax([f['ll'] for f in self.F]))]

    def correction(self, x, e, lo, hi, eta_fallback):
        f = self.best(); st, P = f['state'], f['P']
        beta, dg = st[self.ib], st[self.ig]
        D1m, D2m = self.M.d1_d2(x, e, beta)
        _, jac_e = self.M.d1_and_jac(x, e, beta)
        D1 = D1m + dg @ e
        h = np.r_[jac_e, 0.0, e]
        var = max(float(h @ P @ h), 0.0)
        t = -D1 / D2m if D2m > 1e-8 else -eta_fallback * D1
        w = D1 ** 2 / (D1 ** 2 + var) if (D1 ** 2 + var) > 0 else 0.0
        step = float(np.clip(w * t, lo, hi))
        if step != 0.0:
            f0 = self.M.value(x, beta)
            for _ in range(20):
                df = self.M.value(x + step * e, beta) - f0 + step * (dg @ e)
                if df < 0:
                    break
                c = (df - D1 * step) / step ** 2
                new = -D1 / (2 * c) if c > 0 else 0.5 * step
                step = new if 0 < new / step < 1 else 0.5 * step
        return step

    @property
    def state(self):
        f = self.best(); st = f['state']
        return dict(beta=st[self.ib].copy(), kappa=float(st[self.ik]), discrepancy=st[self.ig].copy(),
                    local_discrepancy=f['forget'] < 1.0)
