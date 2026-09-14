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

Every call re-runs your model with SoftNumber objects in place of floats, so
each operation carries its exact derivative alongside its value. The result is
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
        if np.any(np.isinf([result.a, result.b])) or np.any(np.isnan([result.a, result.b])):
            raise FloatingPointError(
                f"the model produced a non-finite value ({result.b}) or "
                f"derivative ({result.a}) at this point -- usually overflow in "
                "exp/pow, or a division approaching zero. Returning it would "
                "put nan or inf straight into your parameters."
            )
        return result.a

    if verify:
        if n_params is None:
            raise ValueError("pass n_params to verify the compiled model, "
                              "or verify=False to skip the check")
        _verify(model_fn, g_delta, n_params, verify_tol)

    return g_delta


def _verify(model_fn, g_delta, n_params, tol, n_points=6):
    """Compare the exact compiled derivative against finite differences of the
    same model, at several scattered points rather than one.

    A single passing point proves very little: a model that branches can be
    correct on one side and silently wrong on the other, and an operation that
    discards the traced derivative (float(), np.array on the parameters) may
    only sit inside one branch. Checking a spread of points catches that."""
    rng = np.random.default_rng(0)
    checked = 0
    attempts = 0
    while checked < n_points and attempts < n_points * 6:
        attempts += 1
        # spread the test points over several scales and both signs, so that
        # branch conditions like `if x > 0` are exercised on both sides
        spread = rng.choice([0.3, 1.0, 3.0])
        theta = rng.normal(0.0, spread, n_params)
        direction = rng.normal(0, 1, n_params)
        try:
            exact = g_delta(theta, direction)
        except TypeError:
            raise          # the model dropped the traced parameters -- a real fault
        except (ZeroDivisionError, ValueError):
            continue       # landed outside the model's domain; try elsewhere
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

        # A complex or nan result means this point is outside the model's real
        # domain (a fractional power of a negative, say). That is a bad test
        # point, not a bad model -- skip it and sample elsewhere.
        if isinstance(exact, complex) or isinstance(approx, complex):
            continue
        if np.isnan(exact) or np.isnan(approx):
            continue
        # Infinity is different: the model really did blow up here, and a
        # tolerance check would let it through since every nan/inf comparison
        # is False.
        if np.isinf(exact) or np.isinf(approx):
            raise FloatingPointError(
                f"at theta={np.array2string(theta, precision=4)} the model "
                f"overflowed (compiled {exact}, finite difference {approx}). "
                "Usually exp/pow overflow, or a division approaching zero."
            )

        scale = max(abs(approx), abs(exact), 1.0)
        if abs(approx) < 1e-12 and abs(exact) < 1e-12:
            continue  # flat here; uninformative, doesn't count as a check
        checked += 1
        if abs(exact - approx) / scale >= tol:
            raise RuntimeError(
                f"at theta={np.array2string(theta, precision=4)} the compiled "
                f"derivative ({exact:.8g}) disagrees with a finite difference of "
                f"your model ({approx:.8g}).\n\nIf your model branches, one branch "
                "may use an operation that discards the traced derivative -- "
                "float(), np.array() on the parameter list, or a call into a "
                "library that only accepts plain numbers. Pass verify=False to "
                "skip this check if you are confident the model is correct."
            )

    if checked < n_points:
        import warnings
        warnings.warn(
            f"soft_compile verified the model at only {checked} of {n_points} "
            "test points -- the rest were singular or flat, so they proved "
            "nothing either way. Treat the compiled derivative as checked less "
            "thoroughly than usual.",
            RuntimeWarning,
        )
