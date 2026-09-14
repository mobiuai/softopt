#!/usr/bin/env python3
"""
================================================================================
QAOA Maximum Independent Set on FakeFez -- SoftOpt vs real torch.optim.Adam
================================================================================
Matches test_fakefez_qaoa_mis_new.py's exact circuit and cost function:
5 qubits, p=5 QAOA layers, native Qiskit RZZ gates (not CNOT-decomposed --
a different, simpler circuit convention from script7's Max-Cut test),
penalty=2.0 for edge violations. Real FakeFez noise, real shot noise.

Lemma-6.1 engine (qaoa_mis_lemma61.py) verified bit-exact against
Qiskit's own Statevector, and D1 verified vs finite differences
(diff~3.85e-10) after fixing a real bug caught during construction: the
RX mixer angle needs the same factor of 2 as the reference circuit's
`rx(2*beta)` -- initially missed, exactly the kind of "missing factor"
bug caught multiple times today; fixed and re-verified before use here.

USAGE:
    pip install numpy scipy qiskit qiskit-aer qiskit-ibm-runtime torch
    python3 script13_qaoa_mis_real_qiskit_fakefez.py
================================================================================
"""
import numpy as np, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import torch

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
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
from softopt import SoftOpt
from itertools import combinations

N_QUBITS = 5
P = 5
NUM_STEPS = 100
NUM_SEEDS = 5
SHOTS = 4096
C_SHIFT = 0.1
LR = 0.02
PENALTY = 2.0

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


def g_delta(theta, delta):
    return g_delta_mis(theta, delta, N_QUBITS, EDGES, P, PENALTY)


def torch_adam_step(theta_torch, opt_torch, grad_np):
    opt_torch.zero_grad()
    theta_torch.grad = torch.tensor(grad_np, dtype=theta_torch.dtype)
    opt_torch.step()
    return theta_torch.detach().numpy().copy()


def main():
    print("=" * 70)
    print("QAOA Maximum Independent Set on FakeFez -- SoftOpt vs real Adam")
    print("=" * 70)
    print(f"Graph: {N_QUBITS} nodes, {len(EDGES)} edges | Optimal MIS: {optimal_mis_size} "
          f"-> target cost: {-optimal_mis_size}")
    print(f"p={P} | steps={NUM_STEPS} | seeds={NUM_SEEDS} | shots={SHOTS}")
    print("=" * 70)

    adam_results, soft_results = [], []
    for seed in range(NUM_SEEDS):
        print(f"\n  Seed {seed+1}/{NUM_SEEDS}")
        np.random.seed(seed)
        init_params = np.random.uniform(-np.pi, np.pi, n_params)
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=n_params) for _ in range(NUM_STEPS)]

        params = init_params.copy()
        theta_torch = torch.tensor(params.copy(), requires_grad=True)
        opt_torch = torch.optim.Adam([theta_torch], lr=LR)
        adam_best = float('inf')
        for step in range(NUM_STEPS):
            e, g = spsa_gradient(params, spsa_deltas[step])
            adam_best = min(adam_best, e)
            params = torch_adam_step(theta_torch, opt_torch, g)

        params = init_params.copy()
        soft_opt = SoftOpt(n_params, g_delta, lr=LR, seed=seed + 999)
        soft_best = float('inf')
        for step in range(NUM_STEPS):
            e, g = spsa_gradient(params, spsa_deltas[step])
            soft_best = min(soft_best, e)
            params = soft_opt.step(params, g)

        winner = "SoftOpt" if soft_best < adam_best else "Adam"
        print(f"    Adam: {adam_best:.4f} | SoftOpt: {soft_best:.4f} -> {winner}")
        adam_results.append(adam_best); soft_results.append(soft_best)

    adam_arr = np.array(adam_results); soft_arr = np.array(soft_results)
    b_mean, m_mean = adam_arr.mean(), soft_arr.mean()
    improvement = (b_mean - m_mean) / abs(b_mean) * 100 if b_mean != 0 else 0
    wins = int(np.sum(soft_arr < adam_arr))

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Optimal target: {-optimal_mis_size}")
    print(f"  Adam (baseline): {b_mean:.4f}")
    print(f"  SoftOpt:         {m_mean:.4f}")
    print(f"\n  Improvement: {improvement:+.1f}%")
    print(f"  Win rate: {wins}/{NUM_SEEDS} ({100*wins/NUM_SEEDS:.0f}%)")
    print("=" * 70)

if __name__ == "__main__":
    main()
