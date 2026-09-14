# SoftOpt

**A standalone optimizer that excels against leading market optimizers like Adam, in problems with a known, differentiable computation graph.**

SoftOpt is free, fully local, and requires no server, license key, or network access — the same way `torch.optim.Adam` requires none. It is not a wrapper around Adam or any other optimizer: it is a complete, drop-in optimizer with its own Adam-equivalent base update, plus an exact Newton-style correction built on Klein–Maimon soft-number calculus (*Foundations of Soft Logic*, Klein & Maimon, Springer 2024).

## Where it helps

If your problem has a **known computation graph** — a quantum circuit, a physical simulator, a projection or measurement model, anything you can write down exactly, even if the *measurements* of it are noisy — SoftOpt computes an exact directional derivative and curvature of that model on every step, and uses them to correct the optimizer's trajectory. Validated, with real hardware-noise-model data, across:

- **Quantum chemistry (VQE)** — H₂, H₄, BeH₂, HeH⁺, and larger multireference molecules
- **Quantum control (GRAPE)** — the single cleanest result across every domain tested
- **Computer vision** — multi-camera bundle adjustment / camera calibration
- **Finance** — portfolio optimization (Markowitz mean-variance)
- **Condensed-matter physics** — quasicrystal and spin-chain models (Ising, XY, Heisenberg, SSH, Kitaev)
- Pharmacokinetics, logistic regression, and more

## Where it does *not* help

Full reinforcement learning (or anything else with a continuously **moving target** — a policy, an adversary, a non-stationary distribution) is outside SoftOpt's validated scope. The mechanism needs a *fixed* objective to compute a meaningful correction against; a moving target breaks that assumption. Use plain Adam/SGD there.

## Installation

```bash
pip install softopt
```

## Quick start

```python
import numpy as np
from softopt import SoftOpt, soft_compile

# Write your model once, in plain Python. No special arithmetic.
def my_model(theta):
    x, y = theta
    return (x - 2.0)**2 + np.exp(y) * x

# soft_compile turns it into the exact directional-derivative function
# SoftOpt needs, and verifies the result against your own model.
g_delta = soft_compile(my_model, n_params=2)

opt = SoftOpt(2, g_delta, lr=0.02)

for step in range(num_steps):
    grad_estimate = my_gradient_estimate(theta)   # e.g. from SPSA on real hardware
    theta = opt.step(theta, grad_estimate)
```

Your model needs to be built from arithmetic and the elementary functions the soft algebra defines: `+ - * / **` (including negative and fractional exponents, and `2.0 ** x`), `abs`, `sqrt`, `exp`, `log`, `log10`, `sin`, `cos`, `tanh`, comparisons, and numpy's versions of those. Loops and `if`/`else` branching on parameter values work too — the compiled derivative is then valid on the branch your parameters are actually in.

The one thing to avoid is converting a traced value back to a plain number mid-model — `float(x)`, or `np.array(params)` on the parameter list, both silently discard the derivative. `soft_compile`'s verification catches this and says so.

If you'd rather write the derivative function yourself — because your model lives in a simulator SoftOpt can't trace, or because you want the speed of a hand-tuned implementation — pass any `g_delta(theta, delta)` that returns the exact directional derivative, and skip `soft_compile` entirely.

## Validated results

All results below use IBM's `FakeFez` noise model via Qiskit + Aer, or realistic finite-sample/measurement noise for the non-quantum domains, with `torch.optim.Adam` as the baseline. Improvement is the reduction in gap to the known optimum (or, for Portfolio, the reduction in loss).

**Quantum chemistry (VQE)**

| Domain | Improvement | Win rate |
|---|---|---|
| H₂ | 66.6% | 5/5 |
| H₄ | 89.8% | 5/5 |
| C₁₃Cl₂ (13-term Hamiltonian, incl. a 4-body term) | 81.3% | 5/5 |
| BeH₂ | 94.7% | 5/5 |
| HeH⁺ | 90.9% | 5/5 |

**Quantum control**

| Domain | Improvement | Win rate |
|---|---|---|
| GRAPE (2-qubit) | 84.0% | 20/20 |

**Condensed-matter & spin models**

| Domain | Improvement | Win rate |
|---|---|---|
| Ferromagnetic Ising (6-qubit chain) | 82.1% | 5/5 |
| Transverse Ising (6-qubit chain) | 71.2% | 5/5 |
| XY model (6-qubit chain) | 62.3% | 5/5 |
| Antiferromagnetic Heisenberg (6-qubit chain) | 61.3% | 5/5 |
| SSH model (topological, 6-qubit chain) | 71.7% | 5/5 |
| Kitaev chain (6-qubit) | 68.8% | 5/5 |
| Fibonacci chain (classical antiferromagnetic XY, N=16) | 95.4% | 10/10 |
| Penrose quasicrystal XY-model (50 sites) | 146.4% | 10/10 |

**Computer vision**

| Domain | Improvement | Win rate |
|---|---|---|
| Camera calibration (bundle adjustment, realistic pixel + outlier noise) | 93.9% | 17/20 |

**Finance**

| Domain | Improvement | Win rate |
|---|---|---|
| Portfolio optimization (realistic backtest noise) | ~58x lower loss | 20/20 |

See `benchmarks/` for the exact, runnable scripts behind every one of these numbers, including the raw per-seed results.

## The math

The full soft-number algebra is available from the package (`from softopt import SoftNumber, sadd, smul, ...`), each operation cited to its exact source in *Foundations of Soft Logic* (Klein & Maimon, Springer 2024):

| Operation | Book source | Formula |
|---|---|---|
| `sadd(a,b)` | §3.3.1, p.19 | `(a+c, b+d)` |
| `smul(a,b)` | §3.3.2, p.19 | `(ad+bc, bd)` |
| `sinv(a,b)` | Lemma 3.1, p.20 | `(−a/b², 1/b)` |
| `spow(x, n)` | Lemma 3.3, p.21 | `(n·a·bⁿ⁻¹, bⁿ)` |
| `sroot(x, n)`, `ssqrt(x)` | Lemma 3.4 (p.22) & 3.5 (p.23) | inverts `spow` |
| `ssin`, `scos`, `sexp` | Lemma 6.1 (p.40) & §6.2 (p.41) | `f(a,b) = (a·f′(b), f(b))` |
| `sdiv(x,y)` | composition | `smul(x, sinv(y))` |

`SoftNumber` wraps these with operator syntax (`x + y`, `x * y`, `x ** 3`, `x.sqrt()`) and is verified, axiom by axiom, against the book's own proofs that bridge numbers form an abelian group under addition and a ring under both operations (`tests/test_soft_number.py`).

## What SoftOpt is *not*

- Not a claim to beat specialized full-Jacobian second-order solvers (Levenberg–Marquardt, L-BFGS) where those are already practical — SoftOpt's validated niche is genuine improvement over first-order optimizers (Adam, SGD) already in use, particularly where switching to a full second-order method isn't practical (embedded in a larger pipeline, high dimensionality, or measurement noise).
- Not a general-purpose black-box optimizer — it requires a known computation graph, as described above.

## License

MIT. See [LICENSE](LICENSE).
