#!/usr/bin/env python3
"""
================================================================================
VQE C13Cl2 Half-Mobius model on FakeFez -- REAL Qiskit customer-view test
================================================================================
Same EfficientSU2(4,reps=2) ansatz/engine as H4 (script6), same real
FakeFez realism, but the more complex 13-term Hamiltonian (including a
4-body ZZZZ multireference term) from vqe_c13cl2_fakefez_customer.py.
Verified during construction: statevector match=exact, D1 vs finite
difference diff=3.45e-09 (the 4-body term doesn't break anything --
the Lemma-6.1 engine is generic over any Pauli-string Hamiltonian).

Baseline: real torch.optim.Adam (per Ido's standing preference).

USAGE:
    pip install numpy scipy qiskit qiskit-aer qiskit-ibm-runtime
    python3 script11_vqe_c13cl2_real_qiskit_fakefez.py

TIMING: same ansatz size as H4 -- expect similar runtime to script6.
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
C_SHIFT = 0.12
LR = 0.02
N_QUBITS, REPS = 4, 2

print("Setting up FakeFez backend...")
backend = AerSimulator.from_backend(FakeBackend())
estimator = BackendEstimatorV2(backend=backend)
estimator.options.default_shots = NUM_SHOTS
estimator.options.seed_simulator = 42

HAMILTONIAN_TERMS = [
    ('IIII', -8.50), ('ZZZZ', 0.15), ('ZZII', 0.25), ('IIZZ', 0.25),
    ('ZIZI', 0.12), ('IZIZ', 0.12), ('XXII', 0.08), ('IIXX', 0.08),
    ('YYII', 0.08), ('IIYY', 0.08), ('ZIIZ', 0.10), ('ZIII', -0.15), ('IIIZ', -0.15),
]
hamiltonian = SparsePauliOp.from_list(HAMILTONIAN_TERMS)
HAMILTONIAN_TERMS_SOFT = [(coeff, pauli) for pauli, coeff in HAMILTONIAN_TERMS]

H_matrix = hamiltonian.to_matrix()
if hasattr(H_matrix, 'toarray'): H_matrix = H_matrix.toarray()
GROUND_STATE = float(min(np.linalg.eigvalsh(H_matrix)))

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
    print("VQE C13Cl2 Half-Mobius on FakeFez -- REAL Qiskit customer-view test")
    print("=" * 70)
    print(f"Steps: {NUM_STEPS} | Seeds: {NUM_SEEDS} | Shots: {NUM_SHOTS}")
    print(f"Ansatz: EfficientSU2({N_QUBITS}, reps={REPS}) -- {num_params} params")
    print(f"Ground state energy: {GROUND_STATE:.4f}")
    print("Baseline: real torch.optim.Adam")
    print("Test:     SoftOpt (standalone)")
    print("=" * 70)

    adam_results, soft_results = [], []
    for seed in range(NUM_SEEDS):
        print(f"\n  Seed {seed+1}/{NUM_SEEDS}")
        np.random.seed(seed)
        init_params = np.random.uniform(-0.3, 0.3, num_params)
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=num_params) for _ in range(NUM_STEPS)]

        params = init_params.copy()
        theta_torch = torch.tensor(params.copy(), requires_grad=True)
        opt_torch = torch.optim.Adam([theta_torch], lr=LR)
        adam_best = float('inf')
        for step in range(NUM_STEPS):
            e, g = get_batched_energy_and_gradient(params, spsa_deltas[step])
            adam_best = min(adam_best, e)
            params = torch_adam_step(theta_torch, opt_torch, g)

        params = init_params.copy()
        soft_opt = SoftOpt(num_params, g_delta, lr=LR, seed=seed + 999)
        soft_best = float('inf')
        for step in range(NUM_STEPS):
            e, g = get_batched_energy_and_gradient(params, spsa_deltas[step])
            soft_best = min(soft_best, e)
            params = soft_opt.step(params, g)

        adam_gap = abs(adam_best - GROUND_STATE)
        soft_gap = abs(soft_best - GROUND_STATE)
        winner = "SoftOpt" if soft_gap < adam_gap else "torch.optim.Adam"
        print(f"    Adam: {adam_best:.4f} (gap={adam_gap:.4f}) | "
              f"SoftOpt: {soft_best:.4f} (gap={soft_gap:.4f}) -> {winner}")
        adam_results.append(adam_best); soft_results.append(soft_best)

    adam_arr = np.array(adam_results); soft_arr = np.array(soft_results)
    adam_gaps = np.abs(adam_arr - GROUND_STATE); soft_gaps = np.abs(soft_arr - GROUND_STATE)
    adam_mean, soft_mean = adam_gaps.mean(), soft_gaps.mean()
    improvement = (adam_mean - soft_mean) / adam_mean * 100
    wins = int(np.sum(soft_gaps < adam_gaps))

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Ground state:               {GROUND_STATE:.4f}")
    print(f"  torch.optim.Adam mean gap:  {adam_mean*1000:.2f} mHa")
    print(f"  SoftOpt mean gap:           {soft_mean*1000:.2f} mHa")
    print(f"\n  Improvement: {improvement:+.1f}%")
    print(f"  Win rate: {wins}/{NUM_SEEDS} ({100*wins/NUM_SEEDS:.0f}%)")
    print("=" * 70)

if __name__ == "__main__":
    main()
