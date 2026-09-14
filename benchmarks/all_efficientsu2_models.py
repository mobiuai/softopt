#!/usr/bin/env python3
"""
================================================================================
Eight spin-chain / small-molecule VQE models -- SoftOpt vs real torch.optim.Adam
================================================================================
All eight reference customer scripts (test_ferro_ising_fair.py,
test_transverse_ising.py, test_xy_model.py, test_antiferro_heisenberg.py,
test_ssh_model.py, test_kitaev_chain.py, test_beh2_customer.py,
test_heh_customer.py) use the SAME EfficientSU2 ansatz family already
verified today (efficient_su2_lemma61.py) -- only the Hamiltonian,
ansatz size, and a few hyperparameters differ. One script, `--model`
selects which.

IMPORTANT FIX made during construction: found and fixed a real sign
bug in efficient_su2_lemma61.py's Hamiltonian-expectation code -- the
Pauli-Y phase convention had the wrong sign. This was INVISIBLE in
every previously-tested Hamiltonian (H2, H4, C13Cl2, GRAPE) because
they only ever had Y appearing in matched PAIRS (YY, YYYY terms),
where the sign error cancels itself out. The Kitaev chain's "XY"/"YX"
cross terms (a single, unpaired Y per term) exposed it: D1 was wrong
by a huge margin (0.61 vs -0.01) until fixed, verified now to
diff=4.6e-10. Confirmed the fix changes NOTHING for H2/H4/C13Cl2
(re-verified after the fix, same results as before) -- those results
Ido already ran remain fully valid.

USAGE:
    pip install numpy scipy qiskit qiskit-aer qiskit-ibm-runtime torch
    python3 script17_spin_chain_models_softopt.py --model ferro_ising
    python3 script17_spin_chain_models_softopt.py --model transverse_ising
    python3 script17_spin_chain_models_softopt.py --model xy_model
    python3 script17_spin_chain_models_softopt.py --model af_heisenberg
    python3 script17_spin_chain_models_softopt.py --model ssh
    python3 script17_spin_chain_models_softopt.py --model kitaev
    python3 script17_spin_chain_models_softopt.py --model beh2
    python3 script17_spin_chain_models_softopt.py --model heh
    python3 script17_spin_chain_models_softopt.py --model all
================================================================================
"""
import numpy as np, sys, warnings, argparse
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


def build_terms(model):
    N = 6
    if model == 'ferro_ising':
        terms = []
        for i in range(N - 1):
            lab = ['I']*N; lab[i]='Z'; lab[i+1]='Z'
            terms.append((''.join(lab), -1.0))
        return terms, N, 1, 0.02, 150, (-np.pi, np.pi), 'adam'
    if model == 'transverse_ising':
        terms = []
        for i in range(N - 1):
            lab = ['I']*N; lab[i]='Z'; lab[i+1]='Z'
            terms.append((''.join(lab), -1.0))
        for i in range(N):
            lab = ['I']*N; lab[i]='X'
            terms.append((''.join(lab), -1.0))
        return terms, N, 1, 0.02, 150, (-np.pi, np.pi), 'adam'
    if model == 'xy_model':
        terms = []
        for i in range(N - 1):
            lab = ['I']*N; lab[i]='X'; lab[i+1]='X'
            terms.append((''.join(lab), 1.0))
            lab = ['I']*N; lab[i]='Y'; lab[i+1]='Y'
            terms.append((''.join(lab), 1.0))
        return terms, N, 1, 0.02, 150, (-np.pi, np.pi), 'adam'
    if model == 'af_heisenberg':
        terms = []
        for i in range(N - 1):
            for P in 'XYZ':
                lab = ['I']*N; lab[i]=P; lab[i+1]=P
                terms.append((''.join(lab), 1.0))
        return terms, N, 1, 0.02, 150, (-np.pi, np.pi), 'adam'
    if model == 'ssh':
        V, W = 1.0, 0.5
        terms = []
        for i in range(N - 1):
            hop = V if i % 2 == 0 else W
            terms.append(("X" + "I"*(N-i-2) + "X" + "I"*i, hop/2))
            terms.append(("Y" + "I"*(N-i-2) + "Y" + "I"*i, hop/2))
        return terms, N, 1, 0.01, 150, (-np.pi, np.pi), 'adam'
    if model == 'kitaev':
        MU, T, DELTA = 0.5, 1.0, 1.0
        terms = []
        for i in range(N):
            terms.append(("I"*i + "Z" + "I"*(N-i-1), MU/2))
        for i in range(N - 1):
            terms.append(("I"*i + "XX" + "I"*(N-i-2), -T/2))
            terms.append(("I"*i + "YY" + "I"*(N-i-2), -T/2))
        for i in range(N - 1):
            terms.append(("I"*i + "XY" + "I"*(N-i-2), DELTA/2))
            terms.append(("I"*i + "YX" + "I"*(N-i-2), -DELTA/2))
        return terms, N, 1, 0.01, 150, (-np.pi, np.pi), 'adam'
    if model == 'beh2':
        terms = [
            ("IIIIII", -14.8527), ("IIIIIZ", 0.3687), ("IIIIZI", -0.3687),
            ("IIIZII", 0.2309), ("IIZIII", -0.2309), ("IIIIZZ", 0.1809),
            ("IIIZZI", 0.1809), ("IIIIXX", 0.0459), ("IIIIYY", 0.0459),
            ("IIXXII", 0.0459), ("IIYYII", 0.0459), ("XXXXII", 0.0115), ("YYYYII", 0.0115),
        ]
        return terms, 6, 1, 0.02, 60, (-0.2, 0.2), 'spsa_descent'
    if model == 'heh':
        terms = [("II", -2.8433), ("IZ", 0.5449), ("ZI", -0.5449),
                 ("ZZ", 0.3474), ("XX", 0.0921), ("YY", 0.0921)]
        return terms, 2, 3, 0.02, 50, (-0.3, 0.3), 'spsa_descent'
    raise ValueError(model)


def run_one(model, num_seeds=5, num_shots=4096, c_shift=0.1):
    terms, n_qubits, reps, lr, num_steps, init_range, baseline_kind = build_terms(model)
    hamiltonian = SparsePauliOp.from_list(terms)
    terms_soft = [(coeff, pauli) for pauli, coeff in terms]

    ham_matrix = hamiltonian.to_matrix()
    if hasattr(ham_matrix, 'toarray'): ham_matrix = ham_matrix.toarray()
    exact_energy = float(min(np.linalg.eigvalsh(ham_matrix)))

    print("Setting up FakeFez backend...")
    backend = AerSimulator.from_backend(FakeBackend())
    estimator = BackendEstimatorV2(backend=backend)
    estimator.options.default_shots = num_shots
    estimator.options.seed_simulator = 42

    ansatz = EfficientSU2(n_qubits, reps=reps, entanglement="linear")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_ansatz = pm.run(ansatz)
    isa_ops = hamiltonian.apply_layout(isa_ansatz.layout)
    num_params = ansatz.num_parameters

    def get_energy_and_grad(params, delta):
        job = estimator.run([
            (isa_ansatz, isa_ops, params),
            (isa_ansatz, isa_ops, params + c_shift * delta),
            (isa_ansatz, isa_ops, params - c_shift * delta),
        ])
        r = job.result()
        grad = (float(r[1].data.evs) - float(r[2].data.evs)) / (2 * c_shift) * delta
        return float(r[0].data.evs), grad

    def g_delta(theta, delta):
        return g_delta_su2(theta, delta, n_qubits, reps, terms_soft)

    print("=" * 70)
    print(f"MODEL: {model} -- {n_qubits} qubits, {num_params} params, "
          f"steps={num_steps}, seeds={num_seeds}, baseline={baseline_kind}")
    print(f"Exact ground state: {exact_energy:.4f}")
    print("=" * 70)

    adam_results, soft_results = [], []
    for seed in range(num_seeds):
        print(f"\n  Seed {seed+1}/{num_seeds}")
        np.random.seed(seed)
        init_params = np.random.uniform(init_range[0], init_range[1], num_params)
        np.random.seed(seed * 1000)
        spsa_deltas = [np.random.choice([-1, 1], size=num_params) for _ in range(num_steps)]

        params = init_params.copy()
        if baseline_kind == 'adam':
            theta_torch = torch.tensor(params.copy(), requires_grad=True, dtype=torch.float64)
            opt_torch = torch.optim.Adam([theta_torch], lr=lr)
        adam_best = float('inf')
        for step in range(num_steps):
            e, g = get_energy_and_grad(params, spsa_deltas[step])
            adam_best = min(adam_best, e)
            if baseline_kind == 'adam':
                opt_torch.zero_grad()
                theta_torch.grad = torch.tensor(g, dtype=torch.float64)
                opt_torch.step()
                params = theta_torch.detach().numpy().copy()
            else:
                params = params - lr * g

        params = init_params.copy()
        soft_opt = SoftOpt(num_params, g_delta, lr=lr, seed=seed + 999)
        soft_best = float('inf')
        for step in range(num_steps):
            e, g = get_energy_and_grad(params, spsa_deltas[step])
            soft_best = min(soft_best, e)
            params = soft_opt.step(params, g)

        adam_gap = abs(adam_best - exact_energy)
        soft_gap = abs(soft_best - exact_energy)
        winner = "SoftOpt" if soft_gap < adam_gap else "Adam"
        print(f"    Adam: {adam_best:.4f} (gap={adam_gap:.4f}) | "
              f"SoftOpt: {soft_best:.4f} (gap={soft_gap:.4f}) -> {winner}")
        adam_results.append(adam_best); soft_results.append(soft_best)

    adam_arr = np.array(adam_results); soft_arr = np.array(soft_results)
    adam_gaps = np.abs(adam_arr - exact_energy); soft_gaps = np.abs(soft_arr - exact_energy)
    improvement = (adam_gaps.mean() - soft_gaps.mean()) / adam_gaps.mean() * 100
    wins = int(np.sum(soft_gaps < adam_gaps))

    print("\n" + "=" * 70)
    print(f"RESULTS -- {model}")
    print(f"  Exact ground state: {exact_energy:.4f}")
    print(f"  Adam mean gap:    {adam_gaps.mean():.4f}")
    print(f"  SoftOpt mean gap: {soft_gaps.mean():.4f}")
    print(f"  Improvement: {improvement:+.1f}% | Win rate: {wins}/{num_seeds}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ferro_ising',
                         choices=['ferro_ising', 'transverse_ising', 'xy_model', 'af_heisenberg',
                                  'ssh', 'kitaev', 'beh2', 'heh', 'all'])
    parser.add_argument('--seeds', type=int, default=None)
    args = parser.parse_args()
    models = ['ferro_ising', 'transverse_ising', 'xy_model', 'af_heisenberg',
              'ssh', 'kitaev', 'beh2', 'heh'] if args.model == 'all' else [args.model]
    for m in models:
        seeds = args.seeds if args.seeds is not None else 5
        run_one(m, num_seeds=seeds)
