"""
SoftOpt — an optimization layer built on Klein–Maimon soft-number calculus that closes the gap between your
model (a simulator, a circuit, a physical law) and the real system you measure.

    from softopt import SoftOpt          # numpy version (slope="model" or "measured"; calibrate=True)
    from softopt import spsa_gradient    # gradient, direction and measured slope from 2 readings
    from softopt import SoftOptTorch     # PyTorch version (torch.optim.Optimizer)
    from softopt import soft_compile     # turn a plain-Python model into g_delta
    from softopt import SoftNumber       # the algebraic primitive itself

calibrate=True (new in 0.7.0): soft_compile(model, n_params, n_bias=k) for model(theta, beta), then
SoftOpt(..., calibrate=True, bias_scale=s) -- the measurements also calibrate the model's uncertain parameters.
See README.md.
"""
from .numpy_optimizer import SoftOpt, spsa_gradient
from .compile import soft_compile
from .soft_number import (
    SoftNumber, sadd, sneg, ssub, smul, sinv, sdiv,
    spow, sroot, ssqrt, ssin, scos, sexp, slog,
)


def __getattr__(name):
    # torch is optional: import the PyTorch optimizer only when asked for.
    if name == "SoftOptTorch":
        from .torch_optimizer import SoftOptTorch
        return SoftOptTorch
    raise AttributeError(f"module 'softopt' has no attribute {name!r}")


__all__ = [
    "SoftOpt", "SoftOptTorch", "SoftNumber", "soft_compile", "spsa_gradient",
    "sadd", "sneg", "ssub", "smul", "sinv", "sdiv",
    "spow", "sroot", "ssqrt", "ssin", "scos", "sexp", "slog",
]
__version__ = "0.7.0"