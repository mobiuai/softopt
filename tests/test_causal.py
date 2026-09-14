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


def test_both_interfaces_agree():
    """The package exposes the same algebra two ways: a SoftNumber class, and
    plain tuple functions. The class is for models a user writes by hand; the
    tuple functions vectorise over whole numpy arrays, which is what the
    quantum and vision engines need. They must produce identical numbers.

    This guards the claim made in the README, so that "the benchmarks use the
    tuple functions" is a documented design choice rather than a discrepancy
    someone finds later."""
    from softopt import SoftNumber, smul, sadd, ssub, sdiv, sinv, sexp, ssin, scos

    rng = np.random.default_rng(4)
    for _ in range(50):
        a1, b1, a2, b2 = rng.normal(0, 2, 4)
        b1, b2 = abs(b1) + 0.3, abs(b2) + 0.3      # keep divisions well defined
        x, y = SoftNumber(a1, b1), SoftNumber(a2, b2)
        tx, ty = (a1, b1), (a2, b2)

        for via_class, via_tuple in (
            ((x * y).as_tuple(), smul(tx, ty)),
            ((x + y).as_tuple(), sadd(tx, ty)),
            ((x - y).as_tuple(), ssub(tx, ty)),
            ((x / y).as_tuple(), sdiv(tx, ty)),
            (x.inverse().as_tuple(), sinv(tx)),
            (x.exp().as_tuple(), sexp(tx)),
            (x.sin().as_tuple(), ssin(tx)),
            (x.cos().as_tuple(), scos(tx)),
        ):
            assert np.allclose(via_class, via_tuple, rtol=0, atol=1e-15), (
                f"class and tuple paths disagree: {via_class} vs {via_tuple}")


def test_the_optimizer_consumes_a_soft_derivative():
    """The optimizer never computes a derivative itself -- it consumes g_delta.
    This asserts that the g_delta driving the ablation above is genuinely
    produced by soft-number propagation, and that the optimizer's step actually
    depends on it: perturbing only the soft-derivative function changes where
    the optimizer lands."""
    theta = np.array([0.4, -0.2, 0.7, 0.1, -0.5])
    grad = np.array([0.3, 0.1, -0.2, 0.05, 0.4])

    a = SoftOpt(N, g_delta, lr=0.05, seed=3).step(theta.copy(), grad)

    # Note a uniform rescaling would NOT do: the Newton ratio -D1/D2 is
    # scale-invariant, correctly. Shift the derivative instead, which changes
    # the ratio, and check the step follows.
    b = SoftOpt(N, lambda t, d: g_delta(t, d) + 1.0, lr=0.05, seed=3).step(
        theta.copy(), grad)

    assert not np.allclose(a, b), (
        "changing the soft derivative left the step unchanged -- the "
        "optimizer would not be using it")

    # And the soft path must be what produced it: g_delta here is generated by
    # soft_compile, so SoftNumber operations must fire on every call.
    import softopt.soft_number as sn
    count = {"n": 0}
    originals = {n: getattr(sn.SoftNumber, n)
                 for n in ("__mul__", "__rmul__", "__add__", "__radd__", "__pow__")}

    def instrument(orig):
        def wrapped(self, *args):
            count["n"] += 1
            return orig(self, *args)
        return wrapped

    try:
        for name, orig in originals.items():
            setattr(sn.SoftNumber, name, instrument(orig))
        SoftOpt(N, g_delta, lr=0.05, seed=3).step(theta.copy(), grad)
    finally:
        for name, orig in originals.items():
            setattr(sn.SoftNumber, name, orig)

    assert count["n"] > 0, (
        "the optimizer completed a step without any soft-number operation "
        "running -- the algebra is not in the path")
