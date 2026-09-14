"""
soft_compile: turn a plain-Python model into the exact directional-
derivative function (`g_delta`) that SoftOpt needs -- without writing any
soft-number arithmetic yourself.

    from softopt import SoftOpt, soft_compile

    def my_model(theta):          # plain Python. numpy works too.
        x, y = theta
        return x**2 + np.exp(y) * x

    g_delta = soft_compile(my_model, n_params=2)
    opt = SoftOpt(2, g_delta, lr=0.02)

Your model is traced once with SoftNumber objects in place of floats, so
every operation carries its exact derivative alongside its value. This is
exact -- not a finite-difference approximation.

Your model must be built from arithmetic and the elementary functions the
soft algebra defines (+ - * / ** sqrt exp sin cos, and numpy's versions of
those). Branching on a parameter's *value* (`if x > 0:`) traces only the
branch taken at that point, which is correct locally but means the
compiled function is valid piecewise -- see `verify=True`.
"""
import numpy as np

from .soft_number import SoftNumber


def soft_compile(model_fn, n_params=None, verify=True, verify_tol=1e-5):
    """model_fn: takes a sequence of scalars, returns a scalar.
    n_params: needed only for verification.
    verify: check the compiled derivative against finite differences at a
        random point before returning it. Catches the common mistakes --
        a model that isn't actually differentiable, uses an unsupported
        operation, or silently drops the traced parameters."""
    def g_delta(theta, delta):
        soft = [SoftNumber(float(d), float(t)) for t, d in zip(theta, delta)]
        result = model_fn(soft)
        if not isinstance(result, SoftNumber):
            raise TypeError(
                "the compiled model returned a plain number, not a SoftNumber "
                "-- its output doesn't depend on the traced parameters, or an "
                "operation in it discarded them (e.g. float(), np.array(...), "
                "or comparison-based rounding)"
            )
        return result.a

    if verify:
        if n_params is None:
            raise ValueError("pass n_params to verify the compiled model, "
                              "or verify=False to skip the check")
        _verify(model_fn, g_delta, n_params, verify_tol)

    return g_delta


def _verify(model_fn, g_delta, n_params, tol):
    """Compare the exact compiled derivative against a finite difference of
    the same model evaluated with plain floats."""
    rng = np.random.default_rng(0)
    for attempt in range(5):
        theta = rng.normal(0.5, 1.0, n_params)
        direction = rng.normal(0, 1, n_params)
        try:
            exact = g_delta(theta, direction)
        except ZeroDivisionError:
            continue  # landed on a singular point; try another
        h = 1e-6
        try:
            f_plus = float(model_fn(list(theta + h * direction)))
            f_minus = float(model_fn(list(theta - h * direction)))
        except Exception as e:
            raise RuntimeError(
                f"the model raised {type(e).__name__} when called with plain "
                f"floats: {e}. soft_compile needs a model that works with both "
                "floats and SoftNumbers."
            ) from e
        approx = (f_plus - f_minus) / (2 * h)

        scale = max(abs(approx), abs(exact), 1.0)
        if abs(exact - approx) / scale < tol:
            return  # verified
        if abs(approx) < 1e-12 and abs(exact) < 1e-12:
            continue  # flat direction here; uninformative, try another

        raise RuntimeError(
            f"compiled derivative ({exact:.8g}) disagrees with a finite "
            f"difference of your model ({approx:.8g}). Your model may use an "
            "operation the soft algebra doesn't define, or may not be "
            "differentiable at this point. Pass verify=False to skip this "
            "check if you're confident it's correct."
        )
    # every attempt landed somewhere uninformative -- don't fail, but don't
    # claim verification either
    import warnings
    warnings.warn(
        "soft_compile could not verify the model: every test point was "
        "either singular or flat. The compiled function is returned unchecked.",
        RuntimeWarning,
    )
