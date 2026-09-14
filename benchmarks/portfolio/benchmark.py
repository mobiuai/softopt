#!/usr/bin/env python3
"""
================================================================================
Portfolio optimization — CUSTOMER VIEW TEST (realistic backtest noise, no ablation)
================================================================================
Realistic customer scenario: you have a risk model (expected returns mu,
covariance Sigma -- from your own estimation process, treated here as the
KNOWN, FIXED model, exactly like a known quantum circuit or camera model).
But you never observe the TRUE expected utility of a candidate portfolio
directly -- you observe REALIZED performance from a finite backtest
sample (a batch of simulated/historical trading days), which is noisy
by construction. This is the direct financial analog of quantum shot
noise: a fresh, finite-sample measurement every time you evaluate.

Baseline:  plain Adam, driven by noisy backtest-sample gradient estimates.
Test:      SoftOpt, whose correction uses the EXACT analytic
           mean-variance formula (portfolio_lemma61.py, the validated
           engine) computed from the KNOWN mu/Sigma -- independent of
           the noisy realized-sample measurements.

No ablation shown here on purpose -- just Adam vs SoftOpt.

USAGE:
    pip install numpy scipy
    python3 script8_portfolio_realistic_backtest_noise.py
================================================================================
"""
import numpy as np, sys, time
sys.path.insert(0, '.')
from portfolio_lemma61 import g_delta, energy_exact, N_ASSETS, MU, SIGMA as COV
from softopt import SoftOpt
from scipy.stats import wilcoxon

# ── Config ──────────────────────────────────────────────────────────────────
NUM_STEPS = 100
NUM_SEEDS = 20
BACKTEST_DAYS = 40      # size of each "realized returns" sample -- a realistic
                        # backtest window (about two trading months), not a
                        # huge dataset -- this IS the source of the noise
LR = 0.05
C_SHIFT = 0.05

def softmax(theta):
    e = np.exp(theta - theta.max())
    return e / e.sum()

def realized_utility_noisy(theta, rng):
    """A FRESH finite sample of realized daily returns every call -- the
    customer's actual backtest measurement, not the true expectation."""
    w = softmax(theta)
    daily_returns = rng.multivariate_normal(MU, COV, size=BACKTEST_DAYS)  # (days, assets)
    portfolio_daily = daily_returns @ w                                    # (days,)
    realized_mean = portfolio_daily.mean()
    realized_var = portfolio_daily.var()
    LAMBDA_RISK = 3.0
    return -(realized_mean - 0.5*LAMBDA_RISK*realized_var)  # negative utility, to minimize

class Adam:
    def __init__(self, lr=LR):
        self.lr = lr; self.m = self.v = None; self.t = 0
    def step(self, x, g):
        self.t += 1
        if self.m is None: self.m = np.zeros_like(g); self.v = np.zeros_like(g)
        self.m = 0.9*self.m + 0.1*g; self.v = 0.999*self.v + 0.001*g*g
        mh = self.m/(1-0.9**self.t); vh = self.v/(1-0.999**self.t)
        return x - self.lr*mh/(np.sqrt(vh)+1e-8)

def spsa_grad_noisy(theta, delta, rng):
    Ep = realized_utility_noisy(theta + C_SHIFT*delta, rng)
    Em = realized_utility_noisy(theta - C_SHIFT*delta, rng)
    return (Ep - Em) / (2*C_SHIFT) * delta

def run_plain_adam(seed):
    rng = np.random.default_rng(seed*1000)
    theta = np.random.default_rng(seed).normal(0, 0.5, N_ASSETS)
    opt = Adam()
    for _ in range(NUM_STEPS):
        delta = rng.choice([-1.,1.], size=N_ASSETS)
        g = spsa_grad_noisy(theta, delta, rng)
        theta = opt.step(theta, g)
    return energy_exact(theta)  # scored on the TRUE objective, not a noisy sample

def run_softopt(seed):
    rng = np.random.default_rng(seed*1000)
    theta = np.random.default_rng(seed).normal(0, 0.5, N_ASSETS)
    opt = SoftOpt(N_ASSETS, g_delta, lr=LR, seed=seed*1000+555)
    for _ in range(NUM_STEPS):
        delta = rng.choice([-1.,1.], size=N_ASSETS)
        g = spsa_grad_noisy(theta, delta, rng)
        theta = opt.step(theta, g)
    return energy_exact(theta)

if __name__ == "__main__":
    print(f"Portfolio optimization, {N_ASSETS} assets, realistic {BACKTEST_DAYS}-day "
          f"backtest-sample noise, {NUM_STEPS} steps, {NUM_SEEDS} seeds\n")
    t0 = time.time()
    adam_scores = np.array([run_plain_adam(s) for s in range(NUM_SEEDS)])
    softopt_scores = np.array([run_softopt(s) for s in range(NUM_SEEDS)])
    print(f"Adam mean (true objective):    {adam_scores.mean():.5f}")
    print(f"SoftOpt mean (true objective): {softopt_scores.mean():.5f}")
    wins = int(np.sum(softopt_scores < adam_scores))
    print(f"SoftOpt beats Adam: {wins}/{NUM_SEEDS} seeds")
    p = wilcoxon(softopt_scores, adam_scores).pvalue
    print(f"Wilcoxon p={p:.4g}  ({time.time()-t0:.0f}s)")
