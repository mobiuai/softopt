# SoftOpt

**A standalone optimizer that excels against leading market optimizers like Adam, in problems with a known, differentiable computation graph.**

Each step is an update followed by an exact Newton correction computed from your own model, through Klein–Maimon soft-number calculus (*Foundations of Soft Logic*, Klein & Maimon, Springer 2024). Free, open source, and runs locally.

## Is SoftOpt the right tool for your problem?

One question decides it, and you can answer it without running anything.

**Can you get a residual vector?** That is: can you compute
`model(theta) - data` and get back an *array*, rather than a single number?

- **Yes** → use `scipy.optimize.least_squares`. It builds the full Jacobian and
  exploits the least-squares structure, and it will beat SoftOpt on both
  accuracy and speed. This holds regardless of how large or how noisy the
  problem is. We measured it: on solar-cell parameter extraction, `least_squares`
  reached 7.9 mA against SoftOpt's 37.3 mA, and did it 21x faster.

- **No** → SoftOpt is built for this case. Each evaluation returns **one number,
  and it is noisy**: a quantum measurement, a laser-tracker sweep, a backtest, a
  simulator that returns a score. There is no residual vector for
  `least_squares` to work with, and finite-difference methods like L-BFGS-B end
  up differentiating the noise rather than the model.

SoftOpt's advantage comes from taking the derivative **from your model** instead
of from your measurements. That only matters when the measurements are the noisy
part. Every validated domain below is of the second kind; that is not a
convenience of selection, it is where the mechanism applies.

## Where it helps

If your problem has a **known computation graph** — a quantum circuit, a physical simulator, a projection or measurement model, anything you can write down exactly, even if the *measurements* of it are noisy — SoftOpt computes an exact directional derivative of that model on every step, plus the curvature along the same direction, and uses them to correct the optimizer's trajectory. Validated, with real hardware-noise-model data, across:

- **Quantum chemistry (VQE)** — H₂, H₄, BeH₂, HeH⁺, and larger multireference molecules
- **Quantum control (GRAPE)** — the single cleanest result across every domain tested
- **Computer vision** — multi-camera bundle adjustment / camera calibration
- **Finance** — portfolio optimization (Markowitz mean-variance)
- **Condensed-matter physics** — quasicrystal and spin-chain models (Ising, XY, Heisenberg, SSH, Kitaev)
- Pharmacokinetics, logistic regression, and more

## Where it does *not* help

Full reinforcement learning (or anything else with a continuously **moving target** — a policy, an adversary, a non-stationary distribution) is outside SoftOpt's validated scope. The mechanism needs a *fixed* objective to compute a meaningful correction against; a moving target breaks that assumption. Use plain Adam/SGD there.

**Barren plateaus.** In a deep hardware-efficient circuit at random
initialisation, the gradient decays exponentially in the qubit count and the
shot-noise measurement of it is larger than the thing it measures. An exact
derivative still helps there — it comes from the circuit, so it stays exact
however flat the landscape gets — but it does not remove the barrier.

We measured this at 4–10 qubits on a transverse-field Ising chain: SoftOpt
covers **1.4–1.6× as much of the distance to the ground state** as Adam, and
that ratio held across three seed streams, five shot budgets (512 to 65536),
five SPSA probe sizes and three different Hamiltonians — 316 wins in 320 runs
across 16 conditions. The advantage *grows* as the measurement degrades, which
is the mechanism.

But at 10 qubits both arms covered only a few percent of the way to the ground
state. 1.5× of very little is still very little. **If the plateau is your
problem, this is not the fix.** The scan and its replication are in
`benchmarks_barren_plateau/`.

## Installation

```bash
pip install softopt            # numpy only
pip install "softopt[torch]"   # adds SoftOptTorch
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

**Parameter scale.** SoftOpt's correction moves each parameter by at most 0.5 per step, a natural size for angles and other parameters of order 1. If your parameters live on a very different scale (pixel offsets of 0.03, prices in the thousands), rescale them to roughly ±1, or set the bound to match: `SoftOpt(n, g_delta, lr=..., newton_lo=-b, newton_hi=b)`.

If you'd rather write the derivative function yourself — because your model lives in a simulator SoftOpt can't trace, or because you want the speed of a hand-tuned implementation — pass any `g_delta(theta, delta)` that returns the exact directional derivative, and skip `soft_compile` entirely.

### Exact curvature (new in 0.6.4)

The Newton correction needs two numbers along the probe direction: the slope D1 and the curvature D2. Since 0.6.4 both come from **one pass of your model on nested soft numbers**, with no step size. Each parameter enters as `θ + δ·ε₁ + δ·ε₂`, where each soft axis obeys the book's `ε² = 0`; the mixed `ε₁ε₂` part of the result is D2 exactly (Lemma 6.1 applied twice). `soft_compile` sets this up for you, and `SoftOpt` uses it automatically.

A hand-written `g_delta` can opt in the same way by carrying an attribute `g_delta.d1_d2 = lambda theta, delta: (D1, D2)`. Without it, D2 is a central finite difference of your exact `g_delta` (`h_fd`, the 0.6.3 behaviour, unchanged). `exact_d2=False` forces the finite difference.

On the H4 benchmark (FakeFez, 10 seeds) the exact-curvature engine (`benchmarks/vqe/soft_su2_exact.py`, `h4_exact_d2.py`) gives the same result as the published one: 400 vs 409 mHa at step 100 (p=0.56), both 10/10 against Adam with 67% of the gap closed. `examples/soft_tensor_demo.py` shows that the same nested construction on tensors reproduces SoftOptTorch's nested `torch.func.jvp` to rounding (4e-16).

## Where the slope comes from: `slope="model"` or `slope="measured"`

Every correction is a Newton step, `-slope / curvature`, taken along one direction. The curvature always comes from your model. The slope has two possible sources:

| | `slope="model"` (default) | `slope="measured"` (new in 0.6.1) |
|---|---|---|
| slope from | your model's exact derivative | your readings, along the direction you probed |
| converges to | the **model's** optimum | the **real system's** optimum |
| choose it when | the model is accurate and the readings are very noisy | the model may be biased, and the readings are clean enough to show the slope |

With `slope="model"`, a model whose optimum is in the wrong place pulls the answer to the wrong place. `slope="measured"` uses the model only for the shape of the landscape, so the readings decide where the optimum is. An SPSA probe already measures that slope, so the measured mode costs no extra readings:

```python
from softopt import SoftOpt, soft_compile, spsa_gradient

opt = SoftOpt(n, soft_compile(my_model, n_params=n), lr=0.05, slope="measured")

for step in range(num_steps):
    grad, direction, slope = spsa_gradient(measure, theta, c=0.1)   # 2 readings
    theta = opt.step(theta, grad, direction=direction, measured_slope=slope)
```

`measure(theta)` returns one reading of the real system. If you already compute your own SPSA estimate, pass its direction `d` and `(f(θ+cd) - f(θ-cd)) / 2c` as `measured_slope`.

If your model is exact, you do not need either mode, or any readings: optimise the model directly.

## PyTorch

If your known model is written in PyTorch, `SoftOptTorch` runs the same step as a `torch.optim.Optimizer`: an Adam update from the gradient, then the bounded Newton correction. The directional derivative and the curvature along the probe direction come from PyTorch's forward-mode AD (`torch.func.jvp`, nested once), which is single-axis soft-number propagation executed by PyTorch's own engine. It runs wherever your model runs: CPU, CUDA or Apple MPS.

```bash
pip install "softopt[torch]"
```

```python
import torch
from softopt import SoftOptTorch

class HeatModel(torch.nn.Module):          # any known, differentiable model
    def __init__(self):
        super().__init__()
        self.k = torch.nn.Parameter(torch.tensor([0.5]))
        self.T0 = torch.nn.Parameter(torch.tensor([20.0]))
    def forward(self, t):
        return self.T0 * torch.exp(-self.k * t)

def loss_fn(model, batch):                 # a fixed objective, re-evaluated exactly
    t, measured = batch
    return ((model(t) - measured) ** 2).mean()

model = HeatModel().to("cuda")             # or "cpu" / "mps"
opt = SoftOptTorch(model.parameters(), model, loss_fn, lr=1e-2)

for step in range(num_steps):
    loss = opt.step((t, measured))         # forward, backward, Adam, correction
```

`loss_fn(model, batch)` must be the same objective on every call, as in every domain above: a fixed dataset, a simulator, a physical model. Each parameter moves by at most `newton_bound` in the correction (default: the learning rate). BatchNorm and MaxPool layers are handled; the correction pass never updates BatchNorm running statistics.

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

**Quantum chemistry on a real molecule, and against COBYLA**

| Test | Baseline | Result | Win rate |
|---|---|---|---|
| H₄ molecule (STO-3G, 8 qubits, 185 Pauli terms), 100 steps | Adam | 67% smaller gap to the ground state (409 vs 1227 mHa) | 10/10 |
| H₂, same budget of 180 device readings | COBYLA | 0.11 vs 5.81 mHa above the ground state; chemical accuracy in 10/10 runs vs 3/10 | 10/10 |

Both run SoftOpt with its default settings on IBM's `FakeFez` noise model: `benchmarks/vqe/h4_real_molecule.py`, `benchmarks/vqe/h2_vs_cobyla.py`.

**Quantum machine learning**

| Test | Baseline | Result | Win rate |
|---|---|---|---|
| Variational quantum classifier (Iris, 4 qubits, 24 parameters), same budget of 300 readings, 20 seeds | Adam | 25% lower training loss; test accuracy on the device 70.0% vs 66.3% | 20/20 (loss), 15/20 + 3 ties (accuracy) |
| Variational quantum classifier (breast cancer, 8 qubits, 48 parameters), same budget, 10 seeds | Adam | 6% lower training loss; test accuracy on the device 90.0% vs 88.3% | 10/10 (loss), 7/10 + 2 ties (accuracy) |
| Same two classifiers | COBYLA | lower training loss (Iris 16/20, cancer 10/10); test accuracy tied | 16/20, 10/10 (loss) |

`benchmarks/qml/vqc.py --dataset iris|cancer`

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
| Camera calibration (bundle adjustment, realistic pixel + outlier noise) | 91.6% | 16/20 |

**Finance**

| Domain | Improvement | Win rate |
|---|---|---|
| Portfolio optimization (realistic backtest noise) | ~58x lower loss | 20/20 |

See `benchmarks/` for the exact, runnable scripts behind every one of these numbers, including the raw per-seed results.

A performance table cannot show *why* a method wins. The ablation runs the optimizer's own code path while replacing the genuine soft-computed curvature with a frozen constant, or with real curvature measured at an unrelated point — everything else held identical.

On GRAPE, 1000 seeds per arm:

| curvature fed to the correction | mean gap | genuine wins | p |
|---|---|---|---|
| genuine | 5.63e-06 | — | — |
| none (Adam alone) | 2.52e-04 | 985/1000 | 3.3e-163 |
| frozen constant | 4.84e-01 | 1000/1000 | 3.3e-165 |
| foreign point, fresh direction | 4.12e-01 | 1000/1000 | 3.3e-165 |
| foreign point, same direction | 4.26e-01 | 1000/1000 | 3.3e-165 |

A correction fed a non-genuine value does not merely help less — it lands about 1600× worse than applying no correction at all. Run it yourself:

```bash
cd benchmarks/grape && python3 causal_ablation.py --seeds 1000
python3 tests/test_causal.py          # the same test, smaller and faster
```

### Two interfaces to the same algebra

`SoftNumber` gives you operator syntax (`x * y`, `x.exp()`) and is what `soft_compile` traces your model with. The tuple functions (`smul`, `sadd`, `ssin`, ...) are the same operations applied to whole numpy arrays at once, which is what the benchmark engines use — a quantum circuit propagates 2ⁿ amplitudes per gate, and one Python object per amplitude would not be workable.

They are the same algebra and produce identical numbers; `tests/test_causal.py` asserts that agreement to 1e-15 across every operation, and also asserts that soft-number operations actually fire during an optimizer step.

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
