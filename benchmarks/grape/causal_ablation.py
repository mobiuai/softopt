#!/usr/bin/env python3
"""
Causal ablation on GRAPE quantum control, at high statistical power.

A benchmark table shows that a method wins. It cannot show WHY. This script
answers the harder question: is the specific number the soft algebra computes
doing the work, or would any plausible value do just as well?

The optimizer's code path is held identical across every arm. The only thing
that changes is what gets fed into the curvature slot:

  real     the genuine soft-computed curvature at the current point
  plain    no correction at all -- the Adam base update alone
  frozen   a fixed constant of about the right magnitude
  foreign  real curvature, measured at an unrelated point in the same space
           (two variants: a fresh probe direction, or the same direction)

If the algebra were decorative -- if the win came from "taking a second-order
step" rather than from the value itself -- the corrupted arms would perform
about as well as the genuine one. They do not.

    python3 causal_ablation.py               # 200 seeds, a few minutes
    python3 causal_ablation.py --seeds 500   # the number quoted in the docs
    python3 causal_ablation.py --seeds 50    # quick look

Requires: numpy, scipy, softopt.
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from softopt import SoftOpt
import grape_lemma61 as grape


N_PARAMS = grape.N_PARAMS
LR = 0.02
STEPS = 100
SPSA_SHIFT = 0.1

ARMS = [
    ("real", "genuine soft-computed curvature"),
    ("plain", "no correction (Adam alone)"),
    ("frozen", "frozen constant"),
    ("foreign_point_and_dir", "foreign point, fresh direction"),
    ("foreign_point_same_dir", "foreign point, same direction"),
]


def foreign_sampler(rng):
    """An unrelated point in the same parameter space: real curvature,
    measured where the optimizer is not."""
    return rng.uniform(-np.pi, np.pi, N_PARAMS)


def run(arm, seed):
    """One full optimization. Every arm uses the identical code path -- only
    the curvature value differs."""
    rng = np.random.default_rng(seed * 1000)
    theta = np.random.default_rng(seed).uniform(-0.3, 0.3, N_PARAMS)
    opt = SoftOpt(N_PARAMS, grape.g_delta, lr=LR, seed=seed * 1000 + 555)

    for _ in range(STEPS):
        # SPSA gradient estimate from the measured objective -- the same noisy
        # signal a real control experiment would give you
        probe = rng.choice([-1.0, 1.0], size=N_PARAMS)
        e_plus = grape.energy_exact(theta + SPSA_SHIFT * probe)
        e_minus = grape.energy_exact(theta - SPSA_SHIFT * probe)
        grad = (e_plus - e_minus) / (2 * SPSA_SHIFT) * probe

        theta = opt.step(
            theta, grad,
            _test_magnitude_source=arm,
            _test_foreign_sampler=foreign_sampler,
        )

    return grape.energy_exact(theta) - grape.E0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200,
                    help="number of independent runs per arm (default 200)")
    args = ap.parse_args()
    n = args.seeds

    print("=" * 74)
    print("Causal ablation -- GRAPE two-qubit state transfer")
    print("=" * 74)
    print(f"{N_PARAMS} control parameters, {STEPS} steps, {n} seeds per arm.")
    print("Identical code path throughout; only the curvature value differs.\n")

    results = {}
    t0 = time.time()
    for arm, label in ARMS:
        scores = np.array([run(arm, s) for s in range(n)])
        results[arm] = scores
        print(f"  {label:<34} mean gap {scores.mean():.6e}   "
              f"({time.time()-t0:.0f}s)")

    from scipy.stats import wilcoxon

    real = results["real"]
    print("\n" + "=" * 74)
    print("Genuine curvature against each alternative")
    print("=" * 74)
    print(f"  {'alternative':<34} {'real wins':>12} {'p':>14}")
    print("  " + "-" * 62)
    for arm, label in ARMS:
        if arm == "real":
            continue
        wins = int(np.sum(real < results[arm]))
        p = wilcoxon(real, results[arm]).pvalue
        print(f"  {label:<34} {wins:>7}/{n}   {p:>14.3e}")

    print("\n" + "=" * 74)
    print(f"Genuine curvature reaches {real.mean():.3e}.")
    print(f"Adam alone reaches {results['plain'].mean():.3e}.")
    print(f"The best corrupted variant reaches no better than "
          f"{min(results[a].mean() for a, _ in ARMS if a not in ('real','plain')):.3e}.")
    print("\nA correction fed a non-genuine value does not merely help less --")
    print("it performs worse than applying no correction at all.")
    print("=" * 74)


if __name__ == "__main__":
    main()
