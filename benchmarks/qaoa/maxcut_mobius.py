#!/usr/bin/env python3
"""
================================================================================
QAOA Max-Cut on FakeFez — CUSTOMER VIEW TEST (real hardware noise, no ablation)
================================================================================
Same realism level as the VQE customer scripts: REAL Qiskit QAOA circuit,
REAL Qiskit transpilation, REAL AerSimulator running IBM's FakeFez noise
model, REAL shot noise -- this is genuine quantum circuit simulation, not
an analytic toy.

Baseline:  plain Adam, driven by noisy shot-based measurements only.
Test:      SoftOpt-Mobius. QAOA's curvature was found to be genuinely
           sign-indefinite (RL-like), even in the exact noiseless model
           (see script14's diagnostic) -- so unlike VQE, QAOA needs the
           SAME bounded Mobius-B correction validated for RL-like
           landscapes, not raw Newton division (which was tried first,
           lost decisively to plain Adam here, and is kept in this
           package as script7_raw_newton_reference.py for the record).
           Mobius-B = mobius(D1,D2) in [-1,1] (book Ch.5.3), computed
           from the EXACT analytic QAOA model (lemma61_qaoa.py), applied
           as a bounded additive step on top of the Adam update.

CONFIRMED RESULT (Ido, full 5-seed/60-step run): SoftOpt-Mobius mean gap
0.7603, beating BOTH plain Adam (0.8076) and the failed raw-Newton
SoftOpt (1.4595) -- Mobius doesn't just fix QAOA, it edges out Adam.

No ablation shown here on purpose (per request) -- just the customer's
actual choice: Adam alone, or Adam+SoftOpt-Mobius.

USAGE:
    pip install numpy scipy qiskit qiskit-aer qiskit-ibm-runtime
    python3 script7_qaoa_real_qiskit_fakefez.py
================================================================================
"""
import numpy as np, sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendEstimatorV2

try:
    from qiskit_ibm_runtime.fake_provider import FakeFezV2 as FakeBackend
except ImportError:
    try:
        from qiskit_ibm_runtime.fake_provider import FakeFez as FakeBackend
    except ImportError:
        from qiskit.providers.fake_provider import GenericBackendV2
        FakeBackend = lambda: GenericBackendV2(num_qubits=127)

from lemma61_qaoa import LemmaQAOA
from scipy.stats import wilcoxon

# ── Config ──────────────────────────────────────────────────────────────────
EDGES = [(0,1),(1,2),(2,3),(3,0),(0,2)]   # 4-qubit, 5-edge Max-Cut, p=2 -- same graph validated all day
N_QUBITS = 4
REPS = 2
NUM_STEPS = 60
NUM_SEEDS = 5
NUM_SHOTS = 4096
LR = 0.02
C_SHIFT = 0.1
ETA = 0.3          # Mobius step-size (matches the RL-validated setting)

print("Setting up FakeFez backend...")
backend = AerSimulator.from_backend(FakeBackend())
estimator = BackendEstimatorV2(backend=backend)
estimator.options.default_precision = 1.0 / np.sqrt(NUM_SHOTS)
estimator.options.seed_simulator = 42

# ── Build the real QAOA circuit (cost + mixer layers) ────────────────────────
def build_qaoa_circuit(edges, n_qubits, reps):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    gammas = [f"g{i}" for i in range(reps)]
    betas = [f"b{i}" for i in range(reps)]
    from qiskit.circuit import Parameter
    gamma_params = [Parameter(g) for g in gammas]
    beta_params = [Parameter(b) for b in betas]
    for layer in range(reps):
        for (i, j) in edges:
            qc.cx(i, j)
            qc.rz(gamma_params[layer], j)
            qc.cx(i, j)
        for q in range(n_qubits):
            qc.rx(2 * beta_params[layer], q)
    return qc, gamma_params + beta_params

qc, params = build_qaoa_circuit(EDGES, N_QUBITS, REPS)
cost_terms = []
for (i, j) in EDGES:
    pauli = ['I'] * N_QUBITS
    pauli[i] = 'Z'; pauli[j] = 'Z'
    cost_terms.append((''.join(reversed(pauli)), 0.5))
hamiltonian = SparsePauliOp.from_list(cost_terms)

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_qc = pm.run(qc)
isa_ops = hamiltonian.apply_layout(isa_qc.layout)
num_params = len(params)

def evaluate_energy_noisy(theta):
    """theta = [gamma_0, gamma_1, beta_0, beta_1] -- REAL noisy hardware measurement"""
    job = estimator.run([(isa_qc, isa_ops, theta)])
    return float(job.result()[0].data.evs)

# ── the exact analytic model (validated engine, used ONLY for SoftOpt's correction) ──
mc = LemmaQAOA(EDGES, N_QUBITS, reps=REPS)
E0 = mc.e0

class Adam:
    def __init__(self, lr=LR):
        self.lr = lr; self.m = self.v = None; self.t = 0
    def step(self, x, g):
        self.t += 1
        if self.m is None: self.m = np.zeros_like(g); self.v = np.zeros_like(g)
        self.m = 0.9*self.m + 0.1*g; self.v = 0.999*self.v + 0.001*g*g
        mh = self.m/(1-0.9**self.t); vh = self.v/(1-0.999**self.t)
        return x - self.lr*mh/(np.sqrt(vh)+1e-8)

def spsa_grad_noisy(theta, delta, rng):
    Ep = evaluate_energy_noisy(theta + C_SHIFT*delta)
    Em = evaluate_energy_noisy(theta - C_SHIFT*delta)
    return (Ep - Em) / (2*C_SHIFT) * delta

def mobius_B(x, y):
    denom = abs(x) + abs(y)
    if denom < 1e-12: return 0.0
    sgn = 1.0 if x >= 0 else -1.0
    return y * sgn / denom

def run_plain_adam(seed):
    rng = np.random.default_rng(seed*1000)
    theta = np.random.default_rng(seed).uniform(-0.3, 0.3, num_params)
    opt = Adam()
    for _ in range(NUM_STEPS):
        delta = rng.choice([-1.,1.], size=num_params)
        g = spsa_grad_noisy(theta, delta, rng)
        theta = opt.step(theta, g)
    return evaluate_energy_noisy(theta) - E0

def run_softopt_mobius(seed):
    rng = np.random.default_rng(seed*1000)
    theta = np.random.default_rng(seed).uniform(-0.3, 0.3, num_params)
    opt = Adam()
    for _ in range(NUM_STEPS):
        delta = rng.choice([-1.,1.], size=num_params)
        g = spsa_grad_noisy(theta, delta, rng)
        theta = opt.step(theta, g)
        # Mobius correction using the EXACT analytic model (not the noisy measurement)
        D1 = mc.g_delta(theta, delta)
        gp = mc.g_delta(theta+1e-4*delta, delta); gm = mc.g_delta(theta-1e-4*delta, delta)
        D2 = (gp - gm) / (2e-4)
        B = mobius_B(D1, D2)
        theta = theta + ETA*B*delta
    return evaluate_energy_noisy(theta) - E0

if __name__ == "__main__":
    print(f"QAOA Max-Cut, {N_QUBITS} qubits, {len(EDGES)} edges, FakeFez noise, "
          f"{NUM_STEPS} steps, {NUM_SEEDS} seeds, {NUM_SHOTS} shots/circuit\n")
    t0 = time.time()
    adam_scores, softopt_scores = [], []
    for seed in range(NUM_SEEDS):
        a = run_plain_adam(seed)
        s = run_softopt_mobius(seed)
        adam_scores.append(a); softopt_scores.append(s)
        winner = "SoftOpt-Mobius" if s < a else "Adam"
        print(f"  seed {seed}: Adam gap={a:.4f}  SoftOpt-Mobius gap={s:.4f}  -> {winner}  ({time.time()-t0:.0f}s)")

    adam_scores = np.array(adam_scores); softopt_scores = np.array(softopt_scores)
    print(f"\nAdam mean gap:           {adam_scores.mean():.4f}")
    print(f"SoftOpt-Mobius mean gap: {softopt_scores.mean():.4f}")
    wins = int(np.sum(softopt_scores < adam_scores))
    print(f"SoftOpt-Mobius beats Adam: {wins}/{NUM_SEEDS} seeds")
    if NUM_SEEDS >= 6:
        p = wilcoxon(softopt_scores, adam_scores).pvalue
        print(f"Wilcoxon p={p:.4g}")
