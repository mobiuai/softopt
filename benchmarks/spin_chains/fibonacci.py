#!/usr/bin/env python3
"""
================================================================================
Fibonacci Chain XY Model -- SoftOpt vs real torch.optim.Adam
================================================================================
Matches fibonacci_benchmark.py's exact setup: N=16 antiferromagnetic
XY spins on a Fibonacci chain (J_S=1.0 short bonds, J_L=1/phi long
bonds, substitution rule L->LS,S->L), fair SPSA-gradient comparison,
exact ground state via many-restart L-BFGS-B. D1 verified exact vs
finite differences (diff~4.9e-10) during construction.

Only change from the original reference script: real torch.optim.Adam
(not a hand-rolled clone) as the baseline, and SoftOpt (standalone,
via fibonacci_lemma61.py's genuine Lemma-6.1 engine) as the test arm --
no Mobiu-Q dependency.

USAGE:
    pip install numpy scipy torch
    python3 script12_fibonacci_xy_softopt.py
================================================================================
"""
import numpy as np, sys, torch
from scipy.optimize import minimize
from scipy.stats import wilcoxon
sys.path.insert(0, '.')
from fibonacci_lemma61 import fibonacci_bonds, energy_exact, g_delta, PHI
from softopt import SoftOpt

N = 16
NUM_SEEDS = 10
NUM_STEPS = 500
LR = 0.02
SPSA_C = 0.1

BONDS = fibonacci_bonds(N)


def compute_exact_ground_state(n_restarts=2000, seed=42):
    rng = np.random.default_rng(seed)
    best_e, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = rng.uniform(-np.pi, np.pi, N)
        res = minimize(lambda p: energy_exact(p, BONDS), p0, method="L-BFGS-B",
                        options={"maxiter": 500, "ftol": 1e-14})
        if res.fun < best_e:
            best_e, best_p = res.fun, res.x
    return best_e, best_p


def spsa_step(params, delta, c=SPSA_C):
    e_p = energy_exact(params + c * delta, BONDS)
    e_m = energy_exact(params - c * delta, BONDS)
    grad = (e_p - e_m) / (2 * c) * delta
    e = energy_exact(params, BONDS)
    return grad, e


def run_adam(init_params, spsa_deltas):
    params = init_params.copy()
    theta_torch = torch.tensor(params.copy(), requires_grad=True)
    opt_torch = torch.optim.Adam([theta_torch], lr=LR)
    best = np.inf
    for step in range(NUM_STEPS):
        grad, e = spsa_step(params, spsa_deltas[step])
        best = min(best, e)
        opt_torch.zero_grad()
        theta_torch.grad = torch.tensor(grad, dtype=theta_torch.dtype)
        opt_torch.step()
        params = theta_torch.detach().numpy().copy()
    return best


def run_softopt(init_params, spsa_deltas, seed):
    params = init_params.copy()
    opt = SoftOpt(N, lambda th, d: g_delta(th, d, BONDS), lr=LR, seed=seed + 999)
    best = np.inf
    for step in range(NUM_STEPS):
        grad, e = spsa_step(params, spsa_deltas[step])
        best = min(best, e)
        params = opt.step(params, grad)
    return best


def main():
    print(f"Computing exact ground state for N={N} Fibonacci chain...")
    print(f"  Bonds: {len(BONDS)}  |  Bond pattern: "
          + "".join("S" if abs(J-1.0) < 1e-9 else "L" for _, _, J in BONDS))
    gs_energy, _ = compute_exact_ground_state()
    print(f"  Exact GS energy: {gs_energy:.6f}")
    print()
    print("=" * 70)
    print(f"FIBONACCI CHAIN XY -- SoftOpt vs real torch.optim.Adam")
    print(f"N={N} spins | {len(BONDS)} bonds | Steps={NUM_STEPS} | Seeds={NUM_SEEDS}")
    print("=" * 70)

    adam_gaps, soft_gaps = [], []
    for seed in range(NUM_SEEDS):
        np.random.seed(seed * 1000 + 7)
        init_params = np.random.uniform(-np.pi, np.pi, N)
        spsa_deltas = [np.random.choice([-1.0, 1.0], size=N) for _ in range(NUM_STEPS)]

        adam_e = run_adam(init_params, spsa_deltas)
        soft_e = run_softopt(init_params, spsa_deltas, seed)

        adam_gap = adam_e - gs_energy
        soft_gap = soft_e - gs_energy
        adam_gaps.append(adam_gap); soft_gaps.append(soft_gap)

        winner = "SoftOpt" if soft_gap < adam_gap else "Adam"
        print(f"Seed {seed+1:2d} | Adam gap: {adam_gap:+.4f} | SoftOpt gap: {soft_gap:+.4f} | {winner}")

    adam_arr = np.array(adam_gaps); soft_arr = np.array(soft_gaps)
    wins = int(np.sum(soft_arr < adam_arr))
    gap_reduction = (adam_arr.mean() - soft_arr.mean()) / (abs(adam_arr.mean()) + 1e-10) * 100
    try:
        _, p_val = wilcoxon(adam_arr, soft_arr)
    except Exception:
        p_val = 1.0

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print(f"Exact GS energy:    {gs_energy:.6f}")
    print(f"Adam mean gap:      {adam_arr.mean():.4f}")
    print(f"SoftOpt mean gap:   {soft_arr.mean():.4f}")
    print(f"Gap reduction:      {gap_reduction:+.1f}%")
    print(f"Win rate:           {wins}/{NUM_SEEDS}")
    print(f"p-value (Wilcoxon): {p_val:.6f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
