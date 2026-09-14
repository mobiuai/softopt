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
pip install softopt          # numpy version only
pip install softopt[torch]   # + the PyTorch optimizer
```

## Quick start

### NumPy

```python
from softopt import SoftOpt

# g_delta(theta, delta) must return the EXACT directional derivative of
# your true, differentiable objective along `delta`, at `theta` — computed
# from your own known model (a circuit, a projection, a physical law),
# not estimated from noisy measurements.
opt = SoftOpt(n_params, g_delta, lr=0.02)

for step in range(num_steps):
    grad_estimate = my_gradient_estimate(theta)   # e.g. from SPSA on real hardware
    theta = opt.step(theta, grad_estimate)
```

### PyTorch

```python
from softopt import SoftOptTorch

opt = SoftOptTorch(model.parameters(), model, loss_fn, lr=1e-3)

for batch in data:
    loss = opt.step(batch)   # one call: backward + Adam update + soft-number correction
```

`loss_fn(model, batch)` must be an exactly re-evaluatable, differentiable function of the model's own parameters — a full-batch loss, a physics simulator, a known projection model. SoftOpt uses PyTorch's own forward-mode autodiff (`torch.func.jvp`) to get the same exact directional derivative and curvature the NumPy version computes by hand.

## Two correction modes

SoftOpt ships with two ways of turning the computed derivative/curvature into a parameter update:

- **`newton`** (default) — a bounded Newton step `t* = clip(-D1/D2, bounds)`. Best when the curvature (D2) is consistently one-signed along random directions — true for essentially every gradient-based physics/circuit optimization problem we tested (VQE, GRAPE, camera calibration, portfolio, ...).
- **`mobius`** — a bounded, sign-safe step derived from the book's own Möbius map (Ch. 5.3), for problems whose curvature is *not* reliably one-signed. Pass `mode="mobius"` to `SoftOpt`/`SoftOptTorch` if `newton` underperforms plain Adam on your problem — that pattern is itself informative about your landscape's curvature.

**How do I know which one to use?** Right now, empirically: run a short comparison against plain Adam with `mode="newton"` first; if it clearly loses, try `mode="mobius"`. There is also an experimental `mode="auto"` that samples curvature sign near your starting point and picks for you — we tested it honestly and it is **not yet reliable** (on GRAPE, a domain we know needs `newton` with high confidence, it only picked correctly 40% of the time across random starting points). It's included so you can inspect `opt.detected_mode` and help us characterize when it works, but don't depend on it yet.

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
