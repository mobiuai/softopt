"""
DEEP BENCHMARK 1/3: quantum control -- SoftOpt vs a REAL specialized
optimal-control tool (QuTiP-qtrl's GRAPE with L-BFGS-B), plus the
SoftOpt-vs-plain-Adam comparison that is the actual validated claim.

IMPORTANT, READ THIS FIRST: this script deliberately runs TWO
DIFFERENT parameterizations of "quantum control," because they answer
two different questions:

  (A) QuTiP-qtrl's own GRAPE/L-BFGS-B on standard CNOT-gate synthesis
      via piecewise-constant Hamiltonian pulses -- a SPECIALIZED tool
      built exactly for this problem. Expect this to be extremely fast
      and accurate (in our own testing: ~0.05s, fidelity error ~4e-8).
      This answers: "how good is the best specialized alternative?"

  (B) Plain Adam vs SoftOpt-corrected-Adam on a gate-based ansatz
      (rotation gates + fixed entangler) targeting Bell-state fidelity
      -- the validated comparison from today's development, representing
      how someone using a generic PyTorch/TensorFlow/PennyLane-style
      gradient-descent pipeline (NOT specialized optimal-control
      software) would approach quantum control. This answers: "does
      SoftOpt help the tool most people actually reach for?"

Both are legitimate, real quantum-control benchmarks; they are simply
not the SAME task, so don't read (A) and (B) as a head-to-head. The
honest conclusion from today's development: SoftOpt beats (B)'s plain
Adam decisively and reliably; it does not and is not claimed to beat
(A), a specialized tool, for the class of problem (A) is built for.

USAGE:
    pip install numpy scipy qutip qutip-qtrl
    python3 script1_grape_vs_qutip.py
"""
import numpy as np, sys, time
sys.path.insert(0, '.')
from softopt import SoftOpt
from grape_lemma61 import g_delta, energy_exact, N_PARAMS

# ---------- Part A: QuTiP-qtrl's own GRAPE (specialized tool) ----------
def run_qutip_grape():
    import qutip as qt
    from qutip_qtrl.pulseoptim import optimize_pulse_unitary

    sx, sy, sz, si = qt.sigmax(), qt.sigmay(), qt.sigmaz(), qt.identity(2)
    H_d = qt.tensor(sz, sz)
    H_c = [qt.tensor(sx, si), qt.tensor(si, sx), qt.tensor(sy, si), qt.tensor(si, sy)]
    U_0 = qt.identity(4)
    U_targ = qt.gates.cnot()

    t0 = time.time()
    result = optimize_pulse_unitary(
        H_d, H_c, U_0, U_targ,
        num_tslots=20, evo_time=5.0,
        alg='GRAPE', optim_method='LBFGSB',
        max_iter=200, max_wall_time=60,
        fid_err_targ=1e-6, gen_stats=True,
    )
    dt = time.time() - t0
    print(f"  QuTiP-qtrl GRAPE (L-BFGS-B), CNOT synthesis:")
    print(f"    time={dt:.3f}s  final fidelity error={result.fid_err:.2e}  "
          f"iterations={result.num_iter}  ({result.termination_reason})")

# ---------- Part B: plain Adam vs SoftOpt on the gate-based ansatz ----------
SPSA_C = 0.1
LR = 0.05

class Adam:
    def __init__(self): self.m=self.v=None; self.t=0
    def step(self, x, g, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t+=1
        if self.m is None: self.m=np.zeros_like(g); self.v=np.zeros_like(g)
        self.m=b1*self.m+(1-b1)*g; self.v=b2*self.v+(1-b2)*g*g
        mh=self.m/(1-b1**self.t); vh=self.v/(1-b2**self.t)
        return x - lr*mh/(np.sqrt(vh)+eps)

def spsa_grad(theta, delta, rng):
    Ep = energy_exact(theta + SPSA_C*delta)
    Em = energy_exact(theta - SPSA_C*delta)
    return (Ep-Em)/(2*SPSA_C)*delta

def run_plain_adam(seed, t_steps=100):
    rng = np.random.default_rng(seed*1000)
    theta = np.random.default_rng(seed).uniform(-1, 1, N_PARAMS)
    opt = Adam()
    for _ in range(t_steps):
        delta = rng.choice([-1.,1.], size=N_PARAMS)
        g = spsa_grad(theta, delta, rng)
        theta = opt.step(theta, g, LR)
    return energy_exact(theta)

def run_softopt(seed, t_steps=100):
    rng = np.random.default_rng(seed*1000)
    theta = np.random.default_rng(seed).uniform(-1, 1, N_PARAMS)
    opt = SoftOpt(N_PARAMS, g_delta, lr=LR, seed=seed*1000+555)
    for _ in range(t_steps):
        delta = rng.choice([-1.,1.], size=N_PARAMS)
        g = spsa_grad(theta, delta, rng)
        theta = opt.step(theta, g)
    return energy_exact(theta)

def run_part_b(seeds=100):
    from scipy.stats import wilcoxon
    print(f"\n  Plain Adam vs SoftOpt, gate-based Bell-state ansatz, {seeds} seeds:")
    plain = np.array([run_plain_adam(s) for s in range(seeds)])
    soft = np.array([run_softopt(s) for s in range(seeds)])
    print(f"    plain Adam mean={plain.mean():.6f}   SoftOpt mean={soft.mean():.6f}")
    wins = int(np.sum(soft < plain))
    p = wilcoxon(soft, plain).pvalue
    print(f"    SoftOpt beats plain Adam: {wins}/{seeds}  p={p:.4g}")

if __name__ == "__main__":
    print("=== Part A: QuTiP-qtrl's specialized GRAPE (L-BFGS-B) ===")
    try:
        run_qutip_grape()
    except ImportError as e:
        print(f"  [skipped -- install qutip and qutip-qtrl: {e}]")

    print("\n=== Part B: SoftOpt vs plain Adam (the validated claim) ===")
    run_part_b(seeds=100)
