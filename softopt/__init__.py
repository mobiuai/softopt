"""
SoftOpt — a standalone optimizer built on Klein–Maimon soft-number
calculus, for problems with a known, differentiable computation graph
(quantum circuits, physical models, projection models, ...).

    from softopt import SoftOpt          # numpy version
    from softopt import SoftOptTorch     # PyTorch version (torch.optim.Optimizer)
    from softopt import SoftNumber       # the algebraic primitive itself

See README.md for the exact requirement your problem must satisfy
(the "known computation graph" criterion) and worked examples.
"""
from .numpy_optimizer import SoftOpt
from .soft_number import (
    SoftNumber, sadd, sneg, ssub, smul, sinv, sdiv,
    spow, sroot, ssqrt, ssin, scos, sexp,
)

__all__ = [
    "SoftOpt", "SoftNumber",
    "sadd", "sneg", "ssub", "smul", "sinv", "sdiv",
    "spow", "sroot", "ssqrt", "ssin", "scos", "sexp",
]
__version__ = "0.2.0"

try:
    from .torch_optimizer import SoftOpt as SoftOptTorch
    __all__.append("SoftOptTorch")
except ImportError:
    # torch is an optional dependency (pip install softopt[torch])
    pass
