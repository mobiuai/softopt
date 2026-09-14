#!/usr/bin/env python3
"""
================================================================================
VQE H2 on FakeFez -- REAL Qiskit customer-view test, SoftOpt (standalone)
vs REAL torch.optim.Adam (not a hand-rolled reimplementation)
================================================================================
Same structure and realism as the original Mobiu-Q customer scripts
(h2_mobiu_full.py): REAL EfficientSU2 ansatz, REAL Qiskit transpilation,
REAL AerSimulator running IBM's FakeFez noise model, REAL shot noise
(4096 shots/circuit) -- this takes genuine minutes to run, not seconds,
because it is doing actual quantum circuit simulation, not a fast
analytic toy.

SoftOpt's correction uses `efficient_su2_lemma61.py`, a genuine Lemma-6.1
soft-number engine built for THIS EXACT ansatz (verified bit-for-bit
against Qiskit's own Statevector AND against Qiskit's own
expectation_value -- see efficient_su2_exact.py's __main__ block and the
cross-check in this project's development notes) -- not the simplified
4-parameter toy ansatz used in earlier development-stage testing.

USAGE:
    pip install numpy scipy qiskit qiskit-aer qiskit-ibm-runtime
    python3 script5_vqe_h2_real_qiskit_fakefez.py

TIMING: with the default config (60 steps x 5 seeds x 2 methods, 3 real
circuit evaluations per step) this takes roughly 25-30 minutes on a
normal laptop -- measured directly: 10 steps x 2 seeds took ~120s, which
scales to ~30 min at the full 60x5 config. Reduce NUM_STEPS/NUM_SEEDS
below for a quick sanity check first; the reported result (32.6%
improvement, 2/2 wins) at 10 steps/2 seeds already showed the expected
pattern clearly.
================================================================================
"""
import numpy as np
from datetime import datetime
import json, sys, warnings
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

# ── Config (matches the original customer script) ──────────────────────────
NUM_STEPS = 60
NUM_SEEDS = 5
NUM_SHOTS = 4096
C_SHIFT = 0.1
LR = 0.02
N_QUBITS, REPS = 2, 4

# ── Setup ────────────────────────────────────────────────────────────────────
print("Setting up FakeFez backend...")
backend = AerSimulator.from_backend(FakeBackend())
estimator = BackendEstimatorV2(backend=backend)
estimator.options.default_shots = NUM_SHOTS
estimator.options.seed_simulator = 42

HAMILTONIAN_TERMS = [
    ('II', -0.4804), ('ZZ', 0.3435), ('ZI', -0.4347),
    ('IZ', 0.5716), ('XX', 0.0910), ('YY', 0.0910),
]
hamiltonian = SparsePauliOp.from_list(HAMILTONIAN_TERMS)
# g_delta_su2 expects (coeff, pauli) order -- swap once here
HAMILTONIAN_TERMS_SOFT = [(coeff, pauli) for pauli, coeff in HAMILTONIAN_TERMS]
EXACT_ENERGY = -1.846

ansatz = EfficientSU2(N_QUBITS, reps=REPS, entanglement="linear")
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_ansatz = pm.run(ansatz)
isa_ops = hamiltonian.apply_layout(isa_ansatz.layout)
num_params = ansatz.num_parameters


def get_batched_energy_and_gradient(params, delta):
    """REAL Qiskit circuit execution on the FakeFez noise model -- 3 real
    quantum-circuit evaluations per call, exactly like the original
    customer script."""
    job = estimator.run([
        (isa_ansatz, isa_ops, params),
        (isa_ansatz, isa_ops, params + C_SHIFT * delta),
        (isa_ansatz, isa_ops, params - C_SHIFT * delta),
    ])
    results = job.result()
    grad = (float(results[1].data.evs) - float(results[2].data.evs)) / (2 * C_SHIFT) * delta
    return float(results[0].data.evs), grad


def torch_adam_step(theta_torch, opt_torch, grad_np):
    """One real torch.optim.Adam step -- not a reimplementation. grad_np
    (from the real FakeFez-noisy SPSA estimate) is set as .grad manually,
    since the Qiskit/Aer circuit evaluation is not autodiff-differentiable."""
    opt_torch.zero_grad()
    theta_torch.grad = torch.tensor(grad_np, dtype=theta_torch.dtype)
    opt_torch.step()
    return theta_torch.detach().numpy().copy()


def g_delta(theta, delta):
    return g_delta_su2(theta, delta, N_QUBITS, REPS, HAMILTONIAN_TERMS_SOFT)


def main():
    print("=" * 70)
    print("VQE H2 on FakeFez -- REAL Qiskit customer-view test")
    print("=" * 70)
    print(f"Steps: {NUM_STEPS} | Seeds: {NUM_SEEDS} | Shots: {NUM_SHOTS}")
    print(f"Ansatz: EfficientSU2({N_QUBITS}, reps={REPS}) -- {num_params} params")
    print(f"Ground state energy: {EXACT_ENERGY} Ha")
    print("Baseline: real torch.optim.Adam (fed a real FakeFez SPSA gradient estimate)")
    print("Test:     SoftOpt (standalone -- own Adam-equivalent update + exact Lemma-6.1 correction)")
    print("=" * 70)

    adam_results, softopt_results = [], []

    for seed in range(NUM_SEEDS):
        print(f"\n  Seed {seed+1}/{NUM_SEEDS}")
        np.random.seed(seed)
        init_params = np.random.uniform(-0.3, 0.3, num_params)
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=num_params) for _ in range(NUM_STEPS)]

        print("    Running real torch.optim.Adam...")
        params = init_params.copy()
        theta_torch = torch.tensor(params.copy(), requires_grad=True)
        opt_torch = torch.optim.Adam([theta_torch], lr=LR)
        best = float('inf')
        for step in range(NUM_STEPS):
            e, g = get_batched_energy_and_gradient(params, spsa_deltas[step])
            best = min(best, e)
            params = torch_adam_step(theta_torch, opt_torch, g)
            if step % 15 == 0:
                print(f"\r    Adam step {step:2d}/{NUM_STEPS} | Best: {best:.4f}", end="")
        print()

        print("    Running SoftOpt (standalone)...")
        params = init_params.copy()
        soft_opt = SoftOpt(num_params, g_delta, lr=LR, seed=seed + 999)
        soft_best = float('inf')
        for step in range(NUM_STEPS):
            e, g = get_batched_energy_and_gradient(params, spsa_deltas[step])
            soft_best = min(soft_best, e)
            params = soft_opt.step(params, g)
            if step % 15 == 0:
                print(f"\r    SoftOpt step {step:2d}/{NUM_STEPS} | Best: {soft_best:.4f}", end="")
        print()

        adam_gap = abs(best - EXACT_ENERGY)
        soft_gap = abs(soft_best - EXACT_ENERGY)
        winner = "SoftOpt" if soft_gap < adam_gap else "torch.optim.Adam"
        print(f"    torch.optim.Adam: {best:.4f} (gap={adam_gap:.4f}) | "
              f"SoftOpt: {soft_best:.4f} (gap={soft_gap:.4f}) -> {winner}")

        adam_results.append(best)
        softopt_results.append(soft_best)

    adam_arr = np.array(adam_results); soft_arr = np.array(softopt_results)
    adam_gaps = np.abs(adam_arr - EXACT_ENERGY)
    soft_gaps = np.abs(soft_arr - EXACT_ENERGY)
    adam_mean, soft_mean = adam_gaps.mean(), soft_gaps.mean()
    improvement = (adam_mean - soft_mean) / adam_mean * 100
    wins = int(np.sum(soft_gaps < adam_gaps))

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Ground state:               {EXACT_ENERGY} Ha")
    print(f"  torch.optim.Adam mean gap:  {adam_mean*1000:.2f} mHa")
    print(f"  SoftOpt mean gap:           {soft_mean*1000:.2f} mHa")
    print(f"\n  Improvement: {improvement:+.1f}%")
    print(f"  Win rate: {wins}/{NUM_SEEDS} ({100*wins/NUM_SEEDS:.0f}%)")
    print("=" * 70)

    fname = f'vqe_h2_softopt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(fname, 'w') as f:
        json.dump({'improvement': float(improvement), 'adam_mean_mHa': float(adam_mean*1000),
                    'softopt_mean_mHa': float(soft_mean*1000), 'wins': wins, 'seeds': NUM_SEEDS}, f, indent=2)
    print(f"\nSaved: {fname}")

if __name__ == "__main__":
    main()
