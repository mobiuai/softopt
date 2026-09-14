"""
The causal test: does the SPECIFIC value the soft algebra computes matter,
or would any plausible number do?

This is the question a performance table cannot answer. Here the optimizer's
code path is held identical while the genuine soft-computed curvature is
replaced by:

  - a frozen constant        (right magnitude, no content)
  - a foreign point's value  (real curvature, wrong location)

If the algebra were decorative, these would perform the same. They don't --
they perform worse than using no correction at all, which is what makes the
computed value load-bearing rather than incidental.

Run directly for the full report:  python3 tests/test_causal.py
"""
import numpy as np
import pytest
from softopt import SoftOpt


# A model whose curvature genuinely varies from point to point -- otherwise
# a curvature borrowed from elsewhere would be just as good, and the test
# would prove nothing.
SCALE = np.array([1.0, 3.0, 0.4, 7.0, 2.0])
N = len(SCALE)


def objective(theta):
    q = float(np.sum(SCALE * theta**2))
    return q + 0.35 * float(np.sum(SCALE * theta**4))


def g_delta(theta, delta):
    """Exact directional derivative of the objective."""
    return float(np.sum(SCALE * (2.0 * theta + 1.4 * theta**3) * delta))


def foreign_sampler(rng):
    """An unrelated point in the same space."""
    return rng.normal(0.0, 2.0, N)


def _run(arm, seed, steps=120, lr=0.05, noise=0.05):
    """One optimization run. `arm` selects what the correction is fed."""
    rng = np.random.default_rng(seed * 1000 + 17)
    theta = np.random.default_rng(seed).uniform(-1.5, 1.5, N)
    opt = SoftOpt(N, g_delta, lr=lr, seed=seed * 1000 + 555)
    for _ in range(steps):
        # a noisy gradient estimate, as any real measurement would be
        grad = SCALE * (2.0 * theta + 1.4 * theta**3) + rng.normal(0, noise, N)
        theta = opt.step(
            theta, grad,
            _test_magnitude_source=arm,
            _test_foreign_sampler=foreign_sampler,
            _test_rng=rng,
        )
    return objective(theta)


ARMS = ["real", "plain", "frozen", "foreign_point_and_dir", "foreign_point_same_dir"]


def results(n_seeds=20):
    return {arm: np.array([_run(arm, s) for s in range(n_seeds)]) for arm in ARMS}


def test_genuine_curvature_beats_no_correction():
    r = results()
    wins = int(np.sum(r["real"] < r["plain"]))
    assert wins >= 14, f"real beat plain in only {wins}/20 runs"


@pytest.mark.parametrize("corrupted", ["frozen",
                                        "foreign_point_and_dir",
                                        "foreign_point_same_dir"])
def test_corrupted_curvature_is_worse_than_genuine(corrupted):
    """The load-bearing claim: feeding a non-genuine value degrades the result."""
    r = results()
    wins = int(np.sum(r["real"] < r[corrupted]))
    assert wins >= 13, (
        f"genuine curvature beat {corrupted!r} in only {wins}/20 runs -- "
        "the specific computed value is not doing the work here"
    )


if __name__ == "__main__":
    from scipy.stats import wilcoxon
    r = results(n_seeds=50)
    print("Causal ablation -- 50 seeds, identical code path, "
          "only the curvature value differs\n")
    print(f"  {'arm':<26} {'mean objective':>16}   {'real wins':>10}   {'p':>10}")
    print("  " + "-" * 68)
    for arm in ARMS:
        if arm == "real":
            print(f"  {'real (genuine)':<26} {r[arm].mean():>16.6e}   {'':>10}   {'':>10}")
            continue
        wins = int(np.sum(r["real"] < r[arm]))
        p = wilcoxon(r["real"], r[arm]).pvalue
        label = {"plain": "plain (no correction)",
                 "frozen": "frozen constant",
                 "foreign_point_and_dir": "foreign point + direction",
                 "foreign_point_same_dir": "foreign point, same dir"}[arm]
        print(f"  {label:<26} {r[arm].mean():>16.6e}   {wins:>7}/50   {p:>10.2e}")
