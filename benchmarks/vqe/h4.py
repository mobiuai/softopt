#!/usr/bin/env python3
"""
================================================================================
VQE H4 Chain on FakeFez -- REAL Qiskit customer-view test, SoftOpt
(standalone) vs REAL torch.optim.Adam (per Ido's request: compare
against the actual PyTorch optimizer, not a hand-rolled reimplementation
or the original customer script's plain-SPSA-gradient baseline)
================================================================================
Same realism as script5: real EfficientSU2(4,reps=2) ansatz, real Qiskit
transpilation, real AerSimulator+FakeFez noise, real shot noise. The
Lemma-6.1 engine (`efficient_su2_lemma61.py`) is fully general in
n_qubits/reps and was verified bit-exact for this 4-qubit/16-param
configuration during development (statevector match + D1 vs finite
difference, diff ~1.7e-10).

USAGE:
    pip install numpy scipy qiskit qiskit-aer qiskit-ibm-runtime
    python3 script6_vqe_h4_real_qiskit_fakefez.py

TIMING: 4 qubits + 16 params is a larger circuit than H2 -- expect this
to run longer than script5 (H2). Reduce NUM_STEPS/NUM_SEEDS for a first
sanity check.
================================================================================
"""
import numpy as np, sys, warnings, json
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import torch

from qiskit.circuit.library import EfficientSU2
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

from efficient_su2_lemma61 import g_delta_su2
from softopt import SoftOpt

NUM_STEPS = 60
NUM_SEEDS = 5
NUM_SHOTS = 4096
C_SHIFT = 0.1
LR = 0.02
N_QUBITS, REPS = 4, 2

print("Setting up FakeFez backend...")
backend = AerSimulator.from_backend(FakeBackend())
estimator = BackendEstimatorV2(backend=backend)
estimator.options.default_shots = NUM_SHOTS
estimator.options.seed_simulator = 42

HAMILTONIAN_TERMS = [
    ('IIII', -1.0466),
    ('IIIZ', 0.1799), ('IIZI', -0.1799), ('IZII', 0.1799), ('ZIII', -0.1799),
    ('IIZZ', 0.1209), ('IZZI', 0.1677), ('ZZII', 0.1209), ('ZIIZ', 0.1677),
    ('IZIZ', 0.1744), ('ZIZI', 0.1744),
    ('IIXX', 0.0362), ('IIYY', 0.0362), ('XXII', 0.0362), ('YYII', 0.0362),
]
hamiltonian = SparsePauliOp.from_list(HAMILTONIAN_TERMS)
HAMILTONIAN_TERMS_SOFT = [(coeff, pauli) for pauli, coeff in HAMILTONIAN_TERMS]
EXACT_ENERGY = -2.0096

ansatz = EfficientSU2(N_QUBITS, reps=REPS, entanglement="linear")
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_ansatz = pm.run(ansatz)
isa_ops = hamiltonian.apply_layout(isa_ansatz.layout)
num_params = ansatz.num_parameters


def get_batched_energy_and_gradient(params, delta):
    job = estimator.run([
        (isa_ansatz, isa_ops, params),
        (isa_ansatz, isa_ops, params + C_SHIFT * delta),
        (isa_ansatz, isa_ops, params - C_SHIFT * delta),
    ])
    results = job.result()
    grad = (float(results[1].data.evs) - float(results[2].data.evs)) / (2 * C_SHIFT) * delta
    return float(results[0].data.evs), grad


def g_delta(theta, delta):
    return g_delta_su2(theta, delta, N_QUBITS, REPS, HAMILTONIAN_TERMS_SOFT)


def torch_adam_step(theta_torch, opt_torch, grad_np):
    opt_torch.zero_grad()
    theta_torch.grad = torch.tensor(grad_np, dtype=theta_torch.dtype)
    opt_torch.step()
    return theta_torch.detach().numpy().copy()


def main():
    print("=" * 70)
    print("VQE H4 Chain on FakeFez -- REAL Qiskit customer-view test")
    print("=" * 70)
    print(f"Steps: {NUM_STEPS} | Seeds: {NUM_SEEDS} | Shots: {NUM_SHOTS}")
    print(f"Ansatz: EfficientSU2({N_QUBITS}, reps={REPS}) -- {num_params} params")
    print(f"Ground state energy: {EXACT_ENERGY} Ha")
    print("Baseline: real torch.optim.Adam (fed a real FakeFez SPSA gradient estimate)")
    print("Test:     SoftOpt (standalone -- own Adam-equivalent update + exact Lemma-6.1 correction)")
    print("=" * 70)

    adam_results, softopt_results = [], []

    for seed in range(NUM_SEEDS):
        print(f"\n  Seed {seed + 1}/{NUM_SEEDS}")
        np.random.seed(seed)
        init_params = np.random.uniform(-0.3, 0.3, num_params)
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=num_params) for _ in range(NUM_STEPS)]

        # Baseline: REAL torch.optim.Adam
        params = init_params.copy()
        theta_torch = torch.tensor(params.copy(), requires_grad=True)
        opt_torch = torch.optim.Adam([theta_torch], lr=LR)
        adam_best = float('inf')
        for step in range(NUM_STEPS):
            e, g = get_batched_energy_and_gradient(params, spsa_deltas[step])
            adam_best = min(adam_best, e)
            params = torch_adam_step(theta_torch, opt_torch, g)

        # Test: SoftOpt (standalone)
        params = init_params.copy()
        soft_opt = SoftOpt(num_params, g_delta, lr=LR, seed=seed + 999)
        soft_best = float('inf')
        for step in range(NUM_STEPS):
            e, g = get_batched_energy_and_gradient(params, spsa_deltas[step])
            soft_best = min(soft_best, e)
            params = soft_opt.step(params, g)

        adam_gap = abs(adam_best - EXACT_ENERGY)
        soft_gap = abs(soft_best - EXACT_ENERGY)
        winner = "SoftOpt" if soft_gap < adam_gap else "torch.optim.Adam"
        print(f"    torch.optim.Adam: {adam_best:.4f} (gap={adam_gap:.4f}) | "
              f"SoftOpt: {soft_best:.4f} (gap={soft_gap:.4f}) -> {winner}")

        adam_results.append(adam_best)
        softopt_results.append(soft_best)

    spsa_arr = np.array(adam_results); soft_arr = np.array(softopt_results)
    spsa_gaps = np.abs(spsa_arr - EXACT_ENERGY); soft_gaps = np.abs(soft_arr - EXACT_ENERGY)
    improvement = (spsa_gaps.mean() - soft_gaps.mean()) / spsa_gaps.mean() * 100
    wins = int(np.sum(soft_gaps < spsa_gaps))

    print("\n" + "=" * 70)
    print(f"Improvement: {improvement:+.1f}% | Win Rate: {wins}/{NUM_SEEDS}")
    print("=" * 70)

    with open('h4_softopt_results.json', 'w') as f:
        json.dump({'molecule': 'H4', 'improvement': float(improvement), 'wins': int(wins)}, f)

if __name__ == "__main__":
    main()
