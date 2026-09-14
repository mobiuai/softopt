#!/usr/bin/env python3
"""
================================================================================
Camera calibration / bundle adjustment — CUSTOMER VIEW TEST
(realistic per-measurement pixel noise + outlier mismatches, no ablation)
================================================================================
Realistic customer scenario: a real feature detector/matcher, not a
clean synthetic dataset. Two realistic noise sources, both regenerated
FRESH every time the objective is evaluated (like quantum shot noise):
  1. Sub-pixel detection noise (Gaussian, ~1.0 px std -- typical for a
     decent corner/feature detector).
  2. A small fraction of OUTLIER correspondences (mismatched features,
     very common in real SLAM/photogrammetry pipelines) -- these get a
     large random offset instead of the true noisy position.

Baseline:  plain Adam, driven by these noisy/outlier-corrupted measurements.
Test:      SoftOpt, whose correction uses the EXACT analytic projection
           model (camera_calib_lemma61.py, the validated engine) --
           independent of the noisy/outlier measurements. Since L2 loss
           is not naturally robust to outliers, this is a genuinely
           harder, more realistic test than pure Gaussian noise.

No ablation shown here on purpose -- just Adam vs SoftOpt.

USAGE:
    pip install numpy scipy
    python3 script9_camera_calib_realistic_noise.py
================================================================================
"""
import numpy as np, sys, time
sys.path.insert(0, '.')
from camera_calib_lemma61 import g_delta, N_PARAMS, TRUE_THETA, WORLD_PTS, FOCAL, project_exact
from softopt import SoftOpt
from scipy.stats import wilcoxon

# ── Config ──────────────────────────────────────────────────────────────────
NUM_STEPS = 100
NUM_SEEDS = 20
PIXEL_NOISE_STD = 1.0
OUTLIER_FRACTION = 0.08      # ~8% of correspondences mismatched -- realistic for a real matcher
OUTLIER_MAGNITUDE = 50.0     # pixels -- a gross mismatch, not a small error
LR = 0.02
C_SHIFT = 0.05

TRUE_PROJ = project_exact(TRUE_THETA)   # what a perfect, noiseless sensor would see

def noisy_observed(rng):
    """A FRESH noisy+outlier-corrupted measurement every call."""
    obs = TRUE_PROJ + rng.normal(0, PIXEL_NOISE_STD, size=TRUE_PROJ.shape)
    n_pts = TRUE_PROJ.shape[0]
    n_outliers = int(round(OUTLIER_FRACTION * n_pts))
    if n_outliers > 0:
        outlier_idx = rng.choice(n_pts, size=n_outliers, replace=False)
        obs[outlier_idx] += rng.normal(0, OUTLIER_MAGNITUDE, size=(n_outliers, 2))
    return obs

def true_gap(theta):
    """Scoring uses the TRUE noise-free target, not one noisy snapshot."""
    pred = project_exact(theta)
    return float(np.mean((pred - TRUE_PROJ)**2))

def noisy_energy(theta, rng):
    pred = project_exact(theta)
    obs = noisy_observed(rng)
    return float(np.mean((pred - obs)**2))

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
    Ep = noisy_energy(theta + C_SHIFT*delta, rng)
    Em = noisy_energy(theta - C_SHIFT*delta, rng)
    return (Ep - Em) / (2*C_SHIFT) * delta

def run_plain_adam(seed):
    rng = np.random.default_rng(seed*1000)
    theta = TRUE_THETA + np.random.default_rng(seed).normal(0, 0.3, N_PARAMS)
    opt = Adam()
    for _ in range(NUM_STEPS):
        delta = rng.choice([-1.,1.], size=N_PARAMS)
        g = spsa_grad_noisy(theta, delta, rng)
        theta = opt.step(theta, g)
    return true_gap(theta)

def run_softopt(seed):
    rng = np.random.default_rng(seed*1000)
    theta = TRUE_THETA + np.random.default_rng(seed).normal(0, 0.3, N_PARAMS)
    opt = SoftOpt(N_PARAMS, g_delta, lr=LR, seed=seed*1000+555)
    for _ in range(NUM_STEPS):
        delta = rng.choice([-1.,1.], size=N_PARAMS)
        g = spsa_grad_noisy(theta, delta, rng)
        theta = opt.step(theta, g)
    return true_gap(theta)

if __name__ == "__main__":
    print(f"Camera calibration, {WORLD_PTS.shape[0]} points, {PIXEL_NOISE_STD}px noise + "
          f"{OUTLIER_FRACTION*100:.0f}% outliers (fresh every measurement), "
          f"{NUM_STEPS} steps, {NUM_SEEDS} seeds\n")
    t0 = time.time()
    adam_scores = np.array([run_plain_adam(s) for s in range(NUM_SEEDS)])
    softopt_scores = np.array([run_softopt(s) for s in range(NUM_SEEDS)])
    print(f"Adam mean (true reprojection error):    {adam_scores.mean():.4f}")
    print(f"SoftOpt mean (true reprojection error): {softopt_scores.mean():.4f}")
    wins = int(np.sum(softopt_scores < adam_scores))
    print(f"SoftOpt beats Adam: {wins}/{NUM_SEEDS} seeds")
    p = wilcoxon(softopt_scores, adam_scores).pvalue
    print(f"Wilcoxon p={p:.4g}  ({time.time()-t0:.0f}s)")
