#!/usr/bin/env python3
"""
QAOA Maximum Independent Set on FakeFez -- SoftOpt-Mobius vs real Adam
(Mobius replaces raw-Newton, matching the fix that worked for Max-Cut)
"""
import numpy as np, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

try:
    from qiskit_ibm_runtime.fake_provider import FakeFezV2 as FakeBackend
except ImportError:
    try:
        from qiskit_ibm_runtime.fake_provider import FakeFez as FakeBackend
    except ImportError:
        from qiskit.providers.fake_provider import GenericBackendV2
        FakeBackend = lambda: GenericBackendV2(num_qubits=127)

from qaoa_mis_lemma61 import g_delta as g_delta_mis
from itertools import combinations
from scipy.stats import wilcoxon

N_QUBITS = 5
P = 5
NUM_STEPS = 100
NUM_SEEDS = 5
SHOTS = 4096
C_SHIFT = 0.1
LR = 0.02
PENALTY = 2.0
ETA = 0.3

np.random.seed(42)
EDGES = [(i, j) for i in range(N_QUBITS) for j in range(i+1, N_QUBITS)
         if np.random.random() < 0.5]
if not EDGES:
    EDGES = [(0, 1), (1, 2)]

def is_independent(nodes, edges):
    return not any(i in nodes and j in nodes for i, j in edges)

optimal_mis_size = max(
    size for size in range(N_QUBITS + 1)
    for nodes in combinations(range(N_QUBITS), size)
    if is_independent(nodes, EDGES)
)

print("Setting up FakeFez backend...")
backend = AerSimulator.from_backend(FakeBackend())
backend.set_options(seed_simulator=42)
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
n_params = 2 * P

def create_mis_qaoa(params_np):
    gammas = params_np[:P]; betas = params_np[P:]
    qc = QuantumCircuit(N_QUBITS)
    for i in range(N_QUBITS): qc.h(i)
    for layer in range(P):
        gamma = gammas[layer]
        for i in range(N_QUBITS): qc.rz(gamma, i)
        for i, j in EDGES: qc.rzz(PENALTY * gamma / 2, i, j)
        for i in range(N_QUBITS): qc.rx(2 * betas[layer], i)
    return qc

def get_mis_cost(bitstring):
    selected = sum(bitstring)
    violations = sum(1 for i, j in EDGES if bitstring[i] == 1 and bitstring[j] == 1)
    return -selected + PENALTY * violations

def evaluate_qaoa(params_np, shots=SHOTS):
    qc = create_mis_qaoa(params_np)
    qc.measure_all()
    qc_t = pm.run(qc)
    job = backend.run(qc_t, shots=shots)
    counts = job.result().get_counts()
    total_cost = 0
    for bitstring, count in counts.items():
        bits = [int(b) for b in bitstring[::-1]]
        total_cost += get_mis_cost(bits) * count
    return total_cost / shots

def spsa_gradient(params_np, delta):
    e_plus = evaluate_qaoa(params_np + C_SHIFT * delta)
    e_minus = evaluate_qaoa(params_np - C_SHIFT * delta)
    e_center = evaluate_qaoa(params_np)
    grad = (e_plus - e_minus) / (2 * C_SHIFT) * delta
    return e_center, grad

class Adam:
    def __init__(self, lr=LR): self.lr=lr; self.m=self.v=None; self.t=0
    def step(self, x, g):
        self.t+=1
        if self.m is None: self.m=np.zeros_like(g); self.v=np.zeros_like(g)
        self.m=0.9*self.m+0.1*g; self.v=0.999*self.v+0.001*g*g
        mh=self.m/(1-0.9**self.t); vh=self.v/(1-0.999**self.t)
        return x-self.lr*mh/(np.sqrt(vh)+1e-8)

def mobius_B(x, y):
    denom = abs(x)+abs(y)
    if denom < 1e-12: return 0.0
    sgn = 1.0 if x>=0 else -1.0
    return y*sgn/denom

def g_delta(theta, delta):
    return g_delta_mis(theta, delta, N_QUBITS, EDGES, P, PENALTY)

def run_plain_adam(seed):
    np.random.seed(seed)
    init_params = np.random.uniform(-np.pi, np.pi, n_params)
    np.random.seed(seed * 1000)
    spsa_deltas = [np.random.choice([-1, 1], size=n_params) for _ in range(NUM_STEPS)]
    params = init_params.copy()
    opt = Adam()
    best = float('inf')
    for step in range(NUM_STEPS):
        e, g = spsa_gradient(params, spsa_deltas[step])
        best = min(best, e)
        params = opt.step(params, g)
    return best

def run_softopt_mobius(seed):
    np.random.seed(seed)
    init_params = np.random.uniform(-np.pi, np.pi, n_params)
    np.random.seed(seed * 1000)
    spsa_deltas = [np.random.choice([-1, 1], size=n_params) for _ in range(NUM_STEPS)]
    params = init_params.copy()
    opt = Adam()
    best = float('inf')
    for step in range(NUM_STEPS):
        delta = spsa_deltas[step]
        e, g = spsa_gradient(params, delta)
        best = min(best, e)
        params = opt.step(params, g)
        D1 = g_delta(params, delta)
        gp = g_delta(params+1e-4*delta, delta); gm = g_delta(params-1e-4*delta, delta)
        D2 = (gp-gm)/(2e-4)
        B = mobius_B(D1, D2)
        params = params + ETA*B*delta
    return best

if __name__ == "__main__":
    print("=" * 70)
    print("QAOA Maximum Independent Set on FakeFez -- SoftOpt-Mobius vs real Adam")
    print("=" * 70)
    print(f"Graph: {N_QUBITS} nodes, {len(EDGES)} edges | Optimal MIS: {optimal_mis_size} -> target cost: {-optimal_mis_size}")
    print(f"p={P} | steps={NUM_STEPS} | seeds={NUM_SEEDS} | shots={SHOTS}")
    print("=" * 70)

    adam_results, soft_results = [], []
    for seed in range(NUM_SEEDS):
        print(f"\n  Seed {seed+1}/{NUM_SEEDS}")
        a = run_plain_adam(seed)
        s = run_softopt_mobius(seed)
        winner = "SoftOpt-Mobius" if s < a else "Adam"
        print(f"    Adam: {a:.4f} | SoftOpt-Mobius: {s:.4f} -> {winner}")
        adam_results.append(a); soft_results.append(s)

    adam_arr = np.array(adam_results); soft_arr = np.array(soft_results)
    b_mean, m_mean = adam_arr.mean(), soft_arr.mean()
    improvement = (b_mean - m_mean) / abs(b_mean) * 100 if b_mean != 0 else 0
    wins = int(np.sum(soft_arr < adam_arr))

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Optimal target: {-optimal_mis_size}")
    print(f"  Adam (baseline): {b_mean:.4f}")
    print(f"  SoftOpt-Mobius:  {m_mean:.4f}")
    print(f"\n  Improvement: {improvement:+.1f}%")
    print(f"  Win rate: {wins}/{NUM_SEEDS} ({100*wins/NUM_SEEDS:.0f}%)")
    print("=" * 70)
