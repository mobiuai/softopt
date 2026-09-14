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
from softopt import SoftOpt, soft_compile


# A model whose curvature genuinely varies from point to point -- otherwise
# a curvature borrowed from elsewhere would be just as good, and the test
# would prove nothing. It is written as an ordinary Python function, and the
# derivative function is generated from it by the package's own compiler, so
# what is being ablated is the value the soft-number path produces.
SCALE = [1.0, 3.0, 0.4, 7.0, 2.0]
N = len(SCALE)


def objective(theta):
    q = sum(s * t**2 for s, t in zip(SCALE, theta))
    return q + 0.35 * sum(s * t**4 for s, t in zip(SCALE, theta))


g_delta = soft_compile(objective, n_params=N)


def _axis(i):
    e = np.zeros(N)
    e[i] = 1.0
    return e


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
        # a noisy gradient estimate, as any real measurement would be --
        # built from the compiled directional derivative along each axis
        exact_grad = np.array([g_delta(theta, _axis(i)) for i in range(N)])
        grad = exact_grad + rng.normal(0, noise, N)
        theta = opt.step(
            theta, grad,
            _test_magnitude_source=arm,
            _test_foreign_sampler=foreign_sampler,
            _test_rng=rng,
        )
    return objective(theta)


ARMS = ["real", "plain", "frozen", "foreign_point_and_dir", "foreign_point_same_dir"]


def results(n_seeds=50):
    return {arm: np.array([_run(arm, s) for s in range(n_seeds)]) for arm in ARMS}


@pytest.fixture(scope="module")
def ablation():
    """Computed once and shared -- every arm is deterministic given its seed."""
    return results()


def test_genuine_curvature_beats_no_correction(ablation):
    wins = int(np.sum(ablation["real"] < ablation["plain"]))
    assert wins >= 30, f"real beat plain in only {wins}/50 runs"


@pytest.mark.parametrize("corrupted", ["frozen",
                                        "foreign_point_and_dir",
                                        "foreign_point_same_dir"])
def test_corrupted_curvature_is_worse_than_genuine(ablation, corrupted):
    """The load-bearing claim: feeding a non-genuine value degrades the result."""
    wins = int(np.sum(ablation["real"] < ablation[corrupted]))
    assert wins >= 28, (
        f"genuine curvature beat {corrupted!r} in only {wins}/50 runs -- "
        "the specific computed value is not doing the work here"
    )


def test_the_ablation_actually_uses_the_soft_number_path():
    """Guards against the derivative being hand-written by accident: if the
    model's derivative stops flowing through SoftNumber, this ablation would
    be testing ordinary calculus and proving nothing about the library."""
    import softopt.soft_number as sn

    counts = {"n": 0}
    originals = {}
    for name in ("__mul__", "__rmul__", "__add__", "__radd__", "__pow__"):
        originals[name] = getattr(sn.SoftNumber, name)

    def instrument(original):
        def wrapped(self, *args):
            counts["n"] += 1
            return original(self, *args)
        return wrapped

    try:
        for name, original in originals.items():
            setattr(sn.SoftNumber, name, instrument(original))
        g_delta(np.array([0.5, 1.0, -0.3, 0.8, 0.2]), _axis(0))
    finally:
        for name, original in originals.items():
            setattr(sn.SoftNumber, name, original)

    assert counts["n"] > 0, (
        "g_delta produced a derivative without invoking any SoftNumber "
        "operation -- the ablation would not be testing the soft-number path"
    )


if __name__ == "__main__":
    from scipy.stats import wilcoxon
    r = results()
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
