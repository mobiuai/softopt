#!/usr/bin/env python3
"""
================================================================================
REFERENCE ONLY: QAOA Max-Cut with the ORIGINAL raw-Newton SoftOpt
================================================================================
Kept for the record -- this is the FIRST version of the QAOA customer
test, using SoftOpt's standard raw-Newton correction (t*=clip(-D1/D2,
bounds)). It LOST decisively to plain Adam (5/5, Adam mean=0.8076 vs
this version's mean=1.4595) because QAOA's curvature was found to be
genuinely sign-indefinite (see script14's diagnostic) -- raw division
assumes one-signed D2 (true for VQE, false for QAOA). The corrected,
recommended version is script7_qaoa_real_qiskit_fakefez.py, which uses
the Mobius-bounded correction instead and beats both this AND plain Adam.

USAGE (for comparison/record only):
    python3 script7_raw_newton_reference.py
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
from softopt import SoftOpt
from scipy.stats import wilcoxon

EDGES = [(0,1),(1,2),(2,3),(3,0),(0,2)]
N_QUBITS = 4
REPS = 2
NUM_STEPS = 60
NUM_SEEDS = 5
NUM_SHOTS = 4096
LR = 0.02
C_SHIFT = 0.1

print("Setting up FakeFez backend...")
backend = AerSimulator.from_backend(FakeBackend())
estimator = BackendEstimatorV2(backend=backend)
estimator.options.default_precision = 1.0 / np.sqrt(NUM_SHOTS)
estimator.options.seed_simulator = 42

def build_qaoa_circuit(edges, n_qubits, reps):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    from qiskit.circuit import Parameter
    gamma_params = [Parameter(f"g{i}") for i in range(reps)]
    beta_params = [Parameter(f"b{i}") for i in range(reps)]
    for layer in range(reps):
        for (i, j) in edges:
            qc.cx(i, j); qc.rz(gamma_params[layer], j); qc.cx(i, j)
        for q in range(n_qubits):
            qc.rx(2 * beta_params[layer], q)
    return qc, gamma_params + beta_params

qc, params = build_qaoa_circuit(EDGES, N_QUBITS, REPS)
cost_terms = []
for (i, j) in EDGES:
    pauli = ['I'] * N_QUBITS; pauli[i]='Z'; pauli[j]='Z'
    cost_terms.append((''.join(reversed(pauli)), 0.5))
hamiltonian = SparsePauliOp.from_list(cost_terms)
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_qc = pm.run(qc)
isa_ops = hamiltonian.apply_layout(isa_qc.layout)
num_params = len(params)

def evaluate_energy_noisy(theta):
    job = estimator.run([(isa_qc, isa_ops, theta)])
    return float(job.result()[0].data.evs)

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

def run_plain_adam(seed):
    rng = np.random.default_rng(seed*1000)
    theta = np.random.default_rng(seed).uniform(-0.3, 0.3, num_params)
    opt = Adam()
    for _ in range(NUM_STEPS):
        delta = rng.choice([-1.,1.], size=num_params)
        g = spsa_grad_noisy(theta, delta, rng)
        theta = opt.step(theta, g)
    return evaluate_energy_noisy(theta) - E0

def run_softopt_raw_newton(seed):
    rng = np.random.default_rng(seed*1000)
    theta = np.random.default_rng(seed).uniform(-0.3, 0.3, num_params)
    opt = SoftOpt(num_params, mc.g_delta, lr=LR, seed=seed*1000+555)
    for _ in range(NUM_STEPS):
        delta = rng.choice([-1.,1.], size=num_params)
        g = spsa_grad_noisy(theta, delta, rng)
        theta = opt.step(theta, g)
    return evaluate_energy_noisy(theta) - E0

if __name__ == "__main__":
    print(f"QAOA Max-Cut (RAW NEWTON, reference-only), {N_QUBITS} qubits, {len(EDGES)} edges, "
          f"FakeFez noise, {NUM_STEPS} steps, {NUM_SEEDS} seeds\n")
    t0 = time.time()
    adam_scores, softopt_scores = [], []
    for seed in range(NUM_SEEDS):
        a = run_plain_adam(seed)
        s = run_softopt_raw_newton(seed)
        adam_scores.append(a); softopt_scores.append(s)
        winner = "SoftOpt(raw)" if s < a else "Adam"
        print(f"  seed {seed}: Adam gap={a:.4f}  SoftOpt(raw) gap={s:.4f}  -> {winner}  ({time.time()-t0:.0f}s)")
    adam_scores = np.array(adam_scores); softopt_scores = np.array(softopt_scores)
    print(f"\nAdam mean gap:         {adam_scores.mean():.4f}")
    print(f"SoftOpt(raw) mean gap: {softopt_scores.mean():.4f}")
    print(f"SoftOpt(raw) beats Adam: {int(np.sum(softopt_scores < adam_scores))}/{NUM_SEEDS} seeds")
