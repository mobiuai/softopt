#!/usr/bin/env python3
"""
SoftBench: one SMALL problem per domain, one fixed protocol for all (pre-registered before any run).

Every problem has the same shape, the one SoftOpt's calibration is for:
  model(theta, beta)   the user's simulator: plain Python, soft-compiled. beta = offsets of its uncertain
                       parameters from nominal (0 = nominal). beta_true ~ N(0, bias_sd) per seed.
  truth(theta)         the real system = model(theta, beta_true) + an UNMODELLED effect the model does not have
                       (named per problem) -- so no arm ever gets a perfect model.
  measure(theta)       one noisy reading of truth.
Protocol (identical everywhere): start theta = 0 (the nominal setting), budget 200 readings, 10 seeds,
SPSA probe c = 0.1, score = fraction of the start gap remaining, (truth(x) - f*) / (truth(0) - f*), with f*
from multistart L-BFGS on the noiseless truth.
Arms: adam (best of lr 0.01 / 0.03 / 0.1 / 0.3 / 1.0 on these seeds; first run used 3 points and Adam sat at the edge in 8 of 11 domains, so the sweep was widened and everything rerun), spsa (Qiskit-style gains), cobyla (rhobeg 0.5),
softopt (slope=model, as built, default), softopt_measured, cal_free (calibrate=True, no bias parameters),
cal_beta (calibrate=True with the model's bias parameters, bias_scale = bias_sd). Every SoftOpt arm uses Adam's lr.
Success criterion (declared in advance): no domain where the calibrated arms are catastrophically worse than the
best baseline (more than 2x its median gap), and win-or-tie with the best baseline in most domains.
    python3 softbench.py --problem vqe     (or --list)
"""
import argparse, json, sys, time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import wilcoxon

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))   # the repo root
from softopt import SoftOpt, soft_compile                       # noqa: E402
from softopt.soft_number import SoftNumber                      # noqa: E402


def _f(name):
    def g(x):
        return getattr(x, name)() if isinstance(x, SoftNumber) else getattr(np, name)(x)
    return g


EXP, SIN, COS, TANH = _f('exp'), _f('sin'), _f('cos'), _f('tanh')


class Problem:
    n = nb = 0
    bias_sd = 0.2
    noise = 0.01
    budget = 200

    def __init__(self, seed):
        r = np.random.default_rng(31000 + seed)
        self.beta_true = r.normal(0, 1, self.nb) * self.bias_sd
        self.setup(r)
        self.nrng = np.random.default_rng(32000 + seed)
        self.readings = 0

    def setup(self, r):
        pass

    def unmodelled(self, th):
        return 0.0

    def truth(self, th):
        return float(self.model(list(th), list(self.beta_true))) + self.unmodelled(np.asarray(th, float))

    def measure(self, th):
        self.readings += 1
        return self.truth(th) + self.noise * self.nrng.normal()


# ------------------------------------------------------------------ 1. quantum chemistry: H2, real 2-qubit ansatz
class VQE(Problem):
    """6 RY angles, 2 CNOTs; beta = RY over-rotations. Unmodelled: depolarisation (energy shrinks toward
    Tr(H)/4 by 5%)."""
    n = nb = 6
    bias_sd, noise = 0.08, 0.01
    I2 = np.eye(2); X = np.array([[0, 1], [1, 0.]]); Z = np.diag([1, -1.]); Y = np.array([[0, -1], [1, 0.]])  # iY: real
    H = (-0.4804 * np.kron(I2, I2) + 0.3435 * np.kron(Z, Z) - 0.4347 * np.kron(I2, Z) + 0.5716 * np.kron(Z, I2)
         + 0.0910 * np.kron(X, X) - 0.0910 * np.kron(Y, Y))                    # (iY)x(iY) = -YxY

    def model(self, th, be):
        psi = [1.0, 0.0, 0.0, 0.0]                  # index = b0 + 2 b1
        k = 0
        for layer in range(3):
            for q in range(2):
                a = th[k] * (1 + be[k]); k += 1
                c, s = COS(a * 0.5), SIN(a * 0.5)
                st = 1 << q
                for i in range(4):
                    if not i & st:
                        u, v = psi[i], psi[i | st]
                        psi[i], psi[i | st] = c * u - s * v, s * u + c * v
            if layer < 2:
                psi[1], psi[3] = psi[3], psi[1]     # CNOT control q0 -> target q1
        E = 0.0
        for i in range(4):
            for j in range(4):
                if self.H[i, j] != 0:
                    E = E + psi[i] * self.H[i, j] * psi[j]
        return E

    def unmodelled(self, th):
        E = float(self.model(list(th), list(self.beta_true)))
        return -0.05 * E + 0.05 * np.trace(self.H) / 4


# ------------------------------------------------------------------ 2. RF beam steering (phased array)
class Beam(Problem):
    """6 element phases steering to 0.3 rad; beta = per-element phase errors. Unmodelled: element gain errors."""
    n = nb = 6
    bias_sd, noise = 0.3, 0.01

    def setup(self, r):
        self.amp = 1 + 0.1 * r.normal(size=self.n)
        self.geo = np.arange(self.n) * np.pi * np.sin(0.3)

    def _gain(self, th, be, amp):
        cs = ss = 0.0
        for k in range(self.n):
            p = th[k] + be[k] - self.geo[k] + 0.5 * k        # the nominal array needs re-phasing to steer
            cs = cs + amp[k] * COS(p); ss = ss + amp[k] * SIN(p)
        return 1 - (cs * cs + ss * ss) / float(np.sum(amp)) ** 2

    def model(self, th, be):
        return self._gain(th, be, np.ones(self.n))

    def truth(self, th):
        return float(self._gain(list(th), list(self.beta_true), self.amp))


# ------------------------------------------------------------------ 3. robotics: 2-link arm PD tuning
class Arm(Problem):
    """log-gains (Kp, Kd) for two joints; beta = relative mass and damping offsets. Unmodelled: Coulomb friction."""
    n, nb = 4, 4
    bias_sd, noise = 0.2, 0.002

    def _sim(self, th, be, coulomb=0.0):
        Kp = [20 * EXP(th[0]), 20 * EXP(th[2])]; Kd = [4 * EXP(th[1]), 4 * EXP(th[3])]
        m = [1.0 * (1 + be[0]), 0.6 * (1 + be[1])]; b = [0.3 * (1 + be[2]), 0.2 * (1 + be[3])]
        q, w = [0.0, 0.0], [0.0, 0.0]; cost = 0.0; dt = 0.04
        for k in range(25):
            ref = [0.8 * np.sin(0.25 * k), 0.5 * np.cos(0.2 * k) - 0.5]
            tau = [Kp[i] * (ref[i] - q[i]) - Kd[i] * w[i] for i in range(2)]
            acc = [(tau[0] - b[0] * w[0] - 9.81 * (m[0] + m[1]) * 0.5 * SIN(q[0]) - 0.3 * m[1] * SIN(q[0] - q[1])
                    - coulomb * np.tanh(20 * float(w[0]))) / (m[0] + m[1]),
                   (tau[1] - b[1] * w[1] - 9.81 * m[1] * 0.5 * SIN(q[1]) + 0.3 * m[1] * SIN(q[0] - q[1])
                    - coulomb * np.tanh(20 * float(w[1]))) / m[1]]
            for i in range(2):
                w[i] = w[i] + dt * acc[i]; q[i] = q[i] + dt * w[i]
                e = ref[i] - q[i]
                cost = cost + e * e + 1e-4 * tau[i] * tau[i]
        return cost * (1.0 / 25)

    def model(self, th, be):
        return self._sim(th, be)

    def truth(self, th):
        return float(self._sim(list(th), list(self.beta_true), coulomb=0.4))


# ------------------------------------------------------------------ 4. chemical engineering: batch reactor
class Reactor(Problem):
    """Temperature profile (4 segments) maximising intermediate B in A->B->C; beta = activation-energy and
    pre-factor offsets. Unmodelled: a small side reaction A->D."""
    n, nb = 4, 3
    bias_sd, noise = 0.3, 0.003

    def _sim(self, th, be, side=0.0):
        A, B = 1.0, 0.0
        for k in range(20):
            T = 350 + 30 * TANH(th[k // 5])
            x = 350.0 / T - 1
            k1 = 0.5 * EXP(be[2]) * EXP(-(20 + 5 * be[0]) * x)
            k2 = 0.2 * EXP(-(30 + 5 * be[1]) * x)
            dA = -k1 * A - side * A; dB = k1 * A - k2 * B
            A = A + 0.25 * dA; B = B + 0.25 * dB
        return -B

    def model(self, th, be):
        return self._sim(th, be)

    def truth(self, th):
        return float(self._sim(list(th), list(self.beta_true), side=0.03))


# ------------------------------------------------------------------ 5. pharmacology: dosing
class PK(Problem):
    """4 log-doses (every 6 h) to hold concentration at 1; beta = log ka, log CL, log V offsets (the patient).
    Unmodelled: saturable clearance."""
    n, nb = 4, 3
    bias_sd, noise = 0.25, 0.005

    def _sim(self, th, be, sat=0.0):
        ka, CL, V = 1.0 * EXP(be[0]), 0.2 * EXP(be[1]), 1.0 * EXP(be[2])
        G, C, cost = 0.0, 0.0, 0.0
        for h in range(24):
            if h % 6 == 0:
                G = G + 0.25 * EXP(th[h // 6])
            dG = -ka * G
            dC = ka * G / V - CL / V * C - sat * C * C
            G = G + dG; C = C + dC
            if h >= 3:
                cost = cost + (C - 1) * (C - 1)
        return cost * (1.0 / 21)

    def model(self, th, be):
        return self._sim(th, be)

    def truth(self, th):
        return float(self._sim(list(th), list(self.beta_true), sat=0.02))


# ------------------------------------------------------------------ 6. finance: portfolio
class Portfolio(Problem):
    """softmax weights of 6 assets, mean-variance utility; beta = errors of the estimated mean returns.
    Unmodelled: extra correlation in the true covariance."""
    n = nb = 6
    bias_sd, noise = 0.02, 0.0005

    def setup(self, r):
        self.mu = np.array([0.05, 0.07, 0.04, 0.09, 0.06, 0.03])
        s = np.array([0.15, 0.2, 0.1, 0.3, 0.18, 0.08])
        self.S = np.diag(s ** 2)
        self.S_true = self.S + 0.3 * np.outer(s, s) * (1 - np.eye(6))

    def _u(self, th, be, S):
        e = [EXP(t) for t in th]; tot = sum(e); w = [x / tot for x in e]
        ret = sum(w[i] * (self.mu[i] + be[i]) for i in range(6))
        var = 0.0
        for i in range(6):
            for j in range(6):
                if S[i, j] != 0:
                    var = var + w[i] * S[i, j] * w[j]
        return -ret + 3.0 * var

    def model(self, th, be):
        return self._u(th, be, self.S)

    def truth(self, th):
        return float(self._u(list(th), list(self.beta_true), self.S_true))


# ------------------------------------------------------------------ 7. building thermal control
class Thermal(Problem):
    """6-segment heater schedule; beta = log R, log C, outdoor-temperature offset. Unmodelled: solar gain."""
    n, nb = 6, 3
    bias_sd, noise = 0.25, 0.005

    def _sim(self, th, be, solar=0.0):
        R, Cc = 2.0 * EXP(be[0]), 5.0 * EXP(be[1])
        T, cost = 18.0, 0.0
        for h in range(24):
            Tout = 8 + 5 * np.sin(2 * np.pi * (h - 9) / 24) + 4 * be[2]
            p = 2.0 * (1 + TANH(th[h // 4]))
            T = T + ((Tout - T) / (R * Cc) + p / Cc + solar * max(0.0, np.sin(np.pi * (h - 6) / 12)))
            cost = cost + (T - 21) * (T - 21) * 0.02 + 0.05 * p
        return cost * (1.0 / 24)

    def model(self, th, be):
        return self._sim(th, be)

    def truth(self, th):
        return float(self._sim(list(th), list(self.beta_true), solar=0.3))


# ------------------------------------------------------------------ 8. energy: battery fast charging
class Battery(Problem):
    """5-segment charging current; beta = log capacity, log resistance, OCV-slope offset.
    Unmodelled: resistance rising with state of charge."""
    n, nb = 5, 3
    bias_sd, noise = 0.2, 0.002

    def _sim(self, th, be, rsoc=0.0):
        Q, R0, slope = 1.0 * EXP(be[0]), 0.1 * EXP(be[1]), 0.8 * (1 + be[2])
        soc, heat, pen = 0.1, 0.0, 0.0
        for k in range(20):
            I = 1 + TANH(th[k // 4])
            R = R0 * (1 + rsoc * soc)
            V = 3.4 + slope * soc + I * R
            soc = soc + 0.05 * I / Q
            heat = heat + I * I * R
            pen = pen + EXP(10 * (V - 4.25))
        return -soc + 0.2 * heat * (1.0 / 20) + 0.02 * pen

    def model(self, th, be):
        return self._sim(th, be)

    def truth(self, th):
        return float(self._sim(list(th), list(self.beta_true), rsoc=1.0))


# ------------------------------------------------------------------ 9. epidemiology: vaccination
class Epidemic(Problem):
    """5-period vaccination rate; beta = log transmission, log recovery. Unmodelled: waning immunity."""
    n, nb = 5, 2
    bias_sd, noise = 0.15, 0.002

    def _sim(self, th, be, wane=0.0):
        b, g = 0.35 * EXP(be[0]), 0.1 * EXP(be[1])
        S, I, R, inf, cost = 0.99, 0.01, 0.0, 0.0, 0.0
        for t in range(40):
            v = 0.02 * (1 + TANH(th[t // 8]))
            new = b * S * I
            S, I, R = S - new - v * S + wane * R, I + new - g * I, R + g * I + v * S - wane * R
            inf = inf + new; cost = cost + 0.5 * v
        return inf + cost

    def model(self, th, be):
        return self._sim(th, be)

    def truth(self, th):
        return float(self._sim(list(th), list(self.beta_true), wane=0.01))


# ------------------------------------------------------------------ 10. machine learning: covariate shift
class Shift(Problem):
    """Logistic regression (4 weights + bias) trained on a small source sample; beta = the mean shift of the
    4 features in deployment. Truth = loss on the deployment distribution (which is also wider -- unmodelled);
    readings = loss on fresh deployment minibatches."""
    n, nb = 5, 4
    bias_sd, noise = 0.5, 0.0

    def setup(self, r):
        self.Xs = r.normal(size=(40, 4)); self.wt = np.array([1.5, -1.0, 0.8, 0.0])
        self.ys = (self.Xs @ self.wt + 0.3 > 0).astype(float)
        Xd = r.normal(size=(4000, 4)) * 1.3 + self.beta_true; self.Xd = Xd
        self.yd = (Xd @ self.wt + 0.3 > 0).astype(float)

    def _loss(self, th, X, y, be):
        tot = 0.0
        for i in range(len(y)):
            z = th[4] + sum(th[j] * (X[i, j] + be[j]) for j in range(4))
            p = 1 / (1 + EXP(-z))
            tot = tot + (p - y[i]) * (p - y[i])            # Brier loss: smooth, bounded
        return tot * (1.0 / len(y))

    def model(self, th, be):                     # the source sample, moved by the assumed shift (labels kept)
        return self._loss(th, self.Xs, self.ys, be)

    def truth(self, th):
        z = self.Xd @ np.asarray(th[:4]) + th[4]
        p = 1 / (1 + np.exp(-z))
        return float(np.mean((p - self.yd) ** 2))

    def measure(self, th):
        self.readings += 1
        idx = self.nrng.integers(0, len(self.yd), 32)
        z = self.Xd[idx] @ np.asarray(th[:4]) + th[4]
        p = 1 / (1 + np.exp(-z))
        return float(np.mean((p - self.yd[idx]) ** 2))


# ------------------------------------------------------------------ 11. control: the model is exact
class Exact(Problem):
    """A smooth coupled objective whose model is exactly right (beta_true = 0, nothing unmodelled):
    calibration must not hurt here."""
    n, nb = 6, 2
    bias_sd, noise = 0.0, 0.01

    def model(self, th, be):
        f = 0.0
        for i in range(6):
            f = f + (th[i] - 0.5 - be[i % 2]) ** 2 * (1 + 0.5 * i)
        for i in range(5):
            f = f + 0.5 * (th[i + 1] - th[i] * th[i]) ** 2
        return f


LRS = (0.01, 0.03, 0.1, 0.3, 1.0)
BIAS_MULT = 1.0
PROBLEMS = dict(vqe=VQE, beam=Beam, arm=Arm, reactor=Reactor, pk=PK, portfolio=Portfolio, thermal=Thermal,
                battery=Battery, epidemic=Epidemic, shift=Shift, exact=Exact)


# ------------------------------------------------------------------ runner
def optimum(P):
    best = np.inf
    rng = np.random.default_rng(7)
    for x0 in [np.zeros(P.n)] + [rng.normal(0, 1, P.n) for _ in range(6)]:
        r = minimize(P.truth, x0, method='L-BFGS-B')
        best = min(best, r.fun)
    return best


def spsa_pair(P, x, d, c=0.1):
    return (P.measure(x + c * d) - P.measure(x - c * d)) / (2 * c)


def run(P, arm, seed, lr, g, hist=None):
    rng = np.random.default_rng(seed * 7 + 1); x = np.zeros(P.n)
    rec = (lambda z: hist.append((P.readings, np.array(z, float)))) if hist is not None else (lambda z: None)   # noqa: E731
    if arm == 'cobyla':
        f = lambda z: P.measure(z) if P.readings < P.budget else 1e9      # noqa: E731
        best = {'f': np.inf, 'x': x}
        def ff(z):
            v = f(z)
            if v < best['f']:
                best.update(f=v, x=np.array(z))
            rec(best['x'])
            return v
        minimize(ff, x, method='COBYLA', options=dict(maxiter=P.budget, rhobeg=0.5))
        return best['x']
    if arm == 'spsa':
        mags = [abs(spsa_pair(P, x, rng.choice([-1.0, 1.0], P.n))) for _ in range(5)]
        a = (2 * np.pi / 10) * 0.3 / max(np.mean(mags), 1e-12); k = 0
        while P.readings + 2 <= P.budget:
            d = rng.choice([-1.0, 1.0], P.n); k += 1
            ck = 0.1 / k ** 0.101
            s = (P.measure(x + ck * d) - P.measure(x - ck * d)) / (2 * ck)
            x = x - a / k ** 0.602 * s * d
            rec(x)
        return x
    if arm == 'adam':
        m = v = np.zeros(P.n); t = 0
        while P.readings + 2 <= P.budget:
            d = rng.choice([-1.0, 1.0], P.n); s = spsa_pair(P, x, d); gr = s * d; t += 1
            m = 0.9 * m + 0.1 * gr; v = 0.999 * v + 0.001 * gr * gr
            x = x - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            rec(x)
        return x
    kw = dict(softopt=dict(), softopt_measured=dict(slope='measured'), cal_free=dict(calibrate=True),
              cal_beta=dict(calibrate=True, bias_scale=max(P.bias_sd, 1e-3) * BIAS_MULT))[arm]
    gg = g['free'] if arm != 'cal_beta' else g['beta']
    opt = SoftOpt(P.n, gg, lr=lr, seed=seed * 7 + 2, **kw)
    while P.readings + 2 <= P.budget:
        d = rng.choice([-1.0, 1.0], P.n); s = spsa_pair(P, x, d)
        x = opt.step(x, s * d, direction=d, measured_slope=s)
        rec(x)
    return x


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--problem'); ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--list', action='store_true'); ap.add_argument('--bias_mult', type=float, default=None)
    a = ap.parse_args()
    if a.list:
        for k, c in PROBLEMS.items():
            print(f'{k:10s} n={c.n} nb={c.nb}  {c.__doc__.strip().splitlines()[0]}')
        return
    say = lambda *x: print(*x, flush=True)                                  # noqa: E731
    C = PROBLEMS[a.problem]; seeds = range(a.seeds)
    if a.bias_mult is not None:
        global BIAS_MULT
        BIAS_MULT = a.bias_mult
        d = json.load(open(f'softbench_{a.problem}.json')); best = d['best_lr']; gaps = d['gaps']
        info = []
        for s in seeds:
            P = C(s); info.append((optimum(P), P.truth(np.zeros(P.n))))
        v = []
        for s in seeds:
            P = C(s)
            g = dict(beta=soft_compile(P.model, n_params=P.n, n_bias=P.nb, verify=False), free=None)
            x = run(P, 'cal_beta', s, best, g); fs, f0 = info[s]; v.append((P.truth(x) - fs) / max(f0 - fs, 1e-12))
        base = d['base']; B = np.array(gaps[base]); v = np.array(v)
        say(f'{a.problem:10s} bias_scale x{a.bias_mult:g}: cal_beta median {np.median(v):.3g} worst {v.max():.3g}  '
            f'(x1: {np.median(gaps["cal_beta"]):.3g})  best baseline {base} {np.median(B):.3g}  ratio {np.median(v) / np.median(B):.2f}  '
            f'better {int((v < B).sum())}/{a.seeds}')
        json.dump(dict(cal_beta=list(v)), open(f'softbench_{a.problem}_bias{a.bias_mult:g}.json', 'w'))
        return
    gaps, t0 = {}, time.time()
    info = []
    for s in seeds:
        P = C(s); fstar = optimum(P); f0 = P.truth(np.zeros(P.n))
        info.append((fstar, f0))
    def gap(P, x, s):
        fstar, f0 = info[s]
        return (P.truth(x) - fstar) / max(f0 - fstar, 1e-12)
    def compiled(P):
        free = soft_compile(lambda th: P.model(th, [0.0] * P.nb), n_params=P.n, verify=False)
        beta = soft_compile(P.model, n_params=P.n, n_bias=P.nb, verify=False)
        return dict(free=free, beta=beta)
    for lr in LRS:
        gaps[f'adam@{lr}'] = []
        for s in seeds:
            P = C(s); x = run(P, 'adam', s, lr, None); gaps[f'adam@{lr}'].append(gap(P, x, s))
    best = min(LRS, key=lambda lr: np.median(gaps[f'adam@{lr}']))
    gaps['adam'] = gaps[f'adam@{best}']
    say(f'[{a.problem}] start gap median {np.median([f0 - fs for fs, f0 in info]):.4g}; Adam lr medians '
        + ', '.join(f'{lr}: {np.median(gaps[f"adam@{lr}"]):.3g}' for lr in LRS) + f' -> {best}'
        + (' (EDGE)' if best in (LRS[0], LRS[-1]) else '') + f'  ({time.time() - t0:.0f}s)')
    for arm in ('spsa', 'cobyla', 'softopt', 'softopt_measured', 'cal_free', 'cal_beta'):
        gaps[arm] = []
        for s in seeds:
            P = C(s); g = compiled(P)
            try:
                x = run(P, arm, s, best, g); gaps[arm].append(gap(P, x, s))
            except Exception as ex:                                       # report, never hide
                say(f'   {arm} seed {s} FAILED: {type(ex).__name__}: {ex}'); gaps[arm].append(float('nan'))
        say(f'  {arm:17s} median gap {np.nanmedian(gaps[arm]):.3g}  ({time.time() - t0:.0f}s)')
    arms = ['adam', 'spsa', 'cobyla', 'softopt', 'softopt_measured', 'cal_free', 'cal_beta']
    base = min(['adam', 'spsa', 'cobyla'], key=lambda k: np.median(gaps[k]))
    say(f'\n=== SoftBench {a.problem}: fraction of the start gap remaining (lower is better), {a.seeds} seeds; '
        f'best baseline = {base}')
    B = np.array(gaps[base])
    for k in arms:
        v = np.array(gaps[k]); dd = v - B
        c = '' if k == base else f'better than {base} {int((dd < 0).sum())}/{a.seeds}  p={wilcoxon(dd).pvalue if np.any(dd) and not np.any(np.isnan(dd)) else 1:.2g}'
        say(f'  {k:17s}{np.median(v):10.4g}{np.max(v):10.4g}   {c}')
    json.dump(dict(gaps=gaps, best_lr=best, base=base), open(f'softbench_{a.problem}.json', 'w'))


if __name__ == '__main__':
    main()
