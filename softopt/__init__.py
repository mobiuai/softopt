"""
SoftOpt — a standalone optimizer built on Klein–Maimon soft-number
calculus, for problems with a known, differentiable computation graph
(quantum circuits, physical models, projection models, ...).

    from softopt import SoftOpt          # numpy version
    from softopt import SoftOptTorch     # PyTorch version (torch.optim.Optimizer)
    from softopt import soft_compile     # turn a plain-Python model into g_delta
    from softopt import SoftNumber       # the algebraic primitive itself

See README.md for the exact requirement your problem must satisfy
(the "known computation graph" criterion) and worked examples.
"""
from .numpy_optimizer import SoftOpt
from .compile import soft_compile
from .soft_number import (
    SoftNumber, sadd, sneg, ssub, smul, sinv, sdiv,
    spow, sroot, ssqrt, ssin, scos, sexp, slog,
)

__all__ = [
    "SoftOpt", "SoftNumber", "soft_compile",
    "sadd", "sneg", "ssub", "smul", "sinv", "sdiv",
    "spow", "sroot", "ssqrt", "ssin", "scos", "sexp", "slog",
]
__version__ = "0.5.0"
