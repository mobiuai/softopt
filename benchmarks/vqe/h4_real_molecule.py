#!/usr/bin/env python3
"""
SoftOpt vs Adam on IBM's FakeFez, on a REAL molecule: H4 (STO-3G,
Jordan-Wigner, 8 qubits, 185 Pauli terms) -- the same protocol as h2.py /
h4.py in this folder, which use small model Hamiltonians.

Unchanged from the brief:
  * ansatz: Qiskit's EfficientSU2, linear entanglement, transpiled for FakeFez;
  * device: AerSimulator.from_backend(FakeFez), 4096 shots per reading;
  * SPSA: 3 readings per step (x, x+c*d, x-c*d), c = 0.1, start ~ U(-0.3, 0.3);
  * optimiser: SoftOpt exactly as released (default settings, slope="model");
    Adam = SoftOpt's own code path with the correction off, whose update is
    identical to torch.optim.Adam (checked: 2e-16 after 100 steps).
    Same learning rate for both (0.02), no schedule, no tuning;
  * the model is the noiseless circuit, and SoftOpt's slope comes from
    Lemma 6.1 soft-number propagation (soft_su2.py: the same mathematics as
    benchmarks/efficient_su2_lemma61.py, vectorised; checked identical).

New: real molecular Hamiltonians (STO-3G, PySCF + OpenFermion, Jordan-Wigner):
H4 (8 qubits, 185 Pauli terms; hamiltonians/h4_paulis.json). The error is measured
from the exact ground state (FCI). The exact energy is recorded along the
run, so the result is shown at step 60 (the brief's length) and at the end.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings('ignore')
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.abspath(os.path.join(_here, '..', '..')))      # the repo root (softopt/)
from softopt import SoftOpt                                                  # noqa: E402
from soft_su2 import g_delta_su2_fast, energy                               # noqa: E402

from qiskit.circuit.library import EfficientSU2                             # noqa: E402
from qiskit.quantum_info import SparsePauliOp                               # noqa: E402
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # noqa: E402
from qiskit_aer import AerSimulator                                         # noqa: E402
from qiskit.primitives import BackendEstimatorV2                            # noqa: E402
try:
    from qiskit_ibm_runtime.fake_provider import FakeFezV2 as FakeBackend
except ImportError:
    from qiskit_ibm_runtime.fake_provider import FakeFez as FakeBackend

CHEM_ACC = 1.6  # mHa
LR, C, INIT = 0.02, 0.1, 0.3          # as in the brief's scripts


class Molecule:
    def __init__(self, name, reps, shots, final_shots):
        d = json.load(open(os.path.join(_here, 'hamiltonians', f'{name}_paulis.json')))
        self.name, self.nq, self.reps = name, d['n_qubits'], reps
        self.ground, self.e_hf = d['e_fci'], d['e_hf']
        op = SparsePauliOp.from_list([(p, c) for p, c in d['terms']]).simplify()
        self.Hs = op.to_matrix(sparse=True)
        self.n_terms = len(op)
        backend = AerSimulator.from_backend(FakeBackend(), method='density_matrix')   # same noise model; exact mixed state, then shots
        ansatz = EfficientSU2(self.nq, reps=reps, entanglement='linear')
        self.isa = generate_preset_pass_manager(backend=backend, optimization_level=1).run(ansatz)
        self.ops = op.apply_layout(self.isa.layout)
        self.n = ansatz.num_parameters
        self.est = BackendEstimatorV2(backend=backend)
        self.est.options.default_shots = shots
        self.est.options.seed_simulator = 42
        self.final_est = BackendEstimatorV2(backend=backend)
        self.final_est.options.default_shots = final_shots
        self.final_est.options.seed_simulator = 7

    def exact(self, x):
        return energy(x, self.nq, self.reps, self.Hs)

    def g_delta(self, x, d):
        return g_delta_su2_fast(x, d, self.nq, self.reps, self.Hs)


def run_all(S, arms, seeds, steps, every, say):
    """All runs advance in lock-step; each step's readings go to the device in one batched call."""
    runs = []
    for seed in range(seeds):
        x0 = np.random.default_rng(seed).uniform(-INIT, INIT, S.n)
        dirs = np.random.default_rng(seed * 1000).choice([-1.0, 1.0], size=(steps, S.n))
        for arm in arms:
            opt = SoftOpt(S.n, S.g_delta, lr=LR, seed=seed + 999)
            runs.append(dict(seed=seed, arm=arm, x=x0.copy(), dirs=dirs, opt=opt,
                             hook='plain' if arm == 'adam' else None,
                             traj=[1e3 * (S.exact(x0) - S.ground)]))
    t0 = time.time()
    for k in range(steps):
        pubs = []
        for r in runs:
            d = r['dirs'][k]
            pubs += [(S.isa, S.ops, r['x']), (S.isa, S.ops, r['x'] + C * d), (S.isa, S.ops, r['x'] - C * d)]
        ev = [float(p.data.evs) for p in S.est.run(pubs).result()]
        for i, r in enumerate(runs):
            d = r['dirs'][k]
            slope = (ev[3 * i + 1] - ev[3 * i + 2]) / (2 * C)
            r['x'] = r['opt'].step(r['x'], slope * d, _test_magnitude_source=r['hook'])
            if (k + 1) % every == 0 or k + 1 == steps:
                r['traj'].append(1e3 * (S.exact(r['x']) - S.ground))
        if (k + 1) % every == 0:
            med = {a: np.median([r['traj'][-1] for r in runs if r['arm'] == a]) for a in arms}
            say(f'   step {k + 1:4d}/{steps}  ' + '  '.join(f'{a} {v:8.1f}' for a, v in med.items()) +
                f'   mHa   ({time.time() - t0:.0f}s)')
    dev = [float(p.data.evs) for p in S.final_est.run([(S.isa, S.ops, r['x']) for r in runs]).result()]
    res = {a: {'traj': [None] * seeds, 'device': [None] * seeds} for a in arms}
    for r, e in zip(runs, dev):
        res[r['arm']]['traj'][r['seed']] = r['traj']
        res[r['arm']]['device'][r['seed']] = 1e3 * (e - S.ground)
    return res


def compare(res, idx, label, seeds, say):
    a = np.array([t[idx] for t in res['adam']['traj']])
    s = np.array([t[idx] for t in res['softopt']['traj']])
    d = s - a
    p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    gain = 100 * (np.median(a) - np.median(s)) / np.median(a)
    say(f'  {label:22s} Adam {np.median(a):9.2f}   SoftOpt {np.median(s):9.2f}   '
        f'SoftOpt better in {int((d < 0).sum())}/{seeds}   gap to ground closed {gain:+.0f}%   p={p:.2g}'
        f'   chem.acc. {int((a < CHEM_ACC).sum())} vs {int((s < CHEM_ACC).sum())}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--molecules', nargs='+', default=['h4'], choices=['h4'])
    ap.add_argument('--reps', type=int, default=2, help='EfficientSU2 repetitions (the brief used 2 for H4)')
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--every', type=int, default=10)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--shots', type=int, default=4096)
    ap.add_argument('--final-shots', type=int, default=65536)
    ap.add_argument('--quick', action='store_true', help='smoke test: 2 seeds, 4 steps')
    ap.add_argument('--out', default='results/h4_real_molecule_results.json')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.steps, args.every = 1, 2, 1
    say = lambda *a: print(*a, flush=True)                                    # noqa: E731
    arms = ['adam', 'softopt']
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    out = {'config': vars(args), 'molecules': {}}
    for name in args.molecules:
        t0 = time.time()
        S = Molecule(name, args.reps, args.shots, args.final_shots)
        say(f'\n{name}: {S.nq} qubits, {S.n_terms} Pauli terms, EfficientSU2 reps={S.reps} ({S.n} params); '
            f'FakeFez, {args.shots} shots; lr {LR}; {args.steps} steps; {args.seeds} seeds')
        say(f'  Hartree-Fock is {1e3 * (S.e_hf - S.ground):.1f} mHa above the ground state')
        res = run_all(S, arms, args.seeds, args.steps, args.every, say)
        out['molecules'][name] = dict(ground=S.ground, n_params=S.n, every=args.every, res=res)
        with open(args.out, 'w') as fh:
            json.dump(out, fh)
        say(f'\n  error above the ground state (mHa), median over seeds   ({time.time() - t0:.0f}s)')
        i60 = min(60 // args.every, len(res['adam']['traj'][0]) - 1)
        compare(res, i60, f'exact, step {i60 * args.every}', args.seeds, say)
        compare(res, -1, f'exact, step {args.steps}', args.seeds, say)
        a, s = np.array(res['adam']['device']), np.array(res['softopt']['device'])
        say(f'  {"device (FakeFez), end":22s} Adam {np.median(a):9.2f}   SoftOpt {np.median(s):9.2f}   '
            f'SoftOpt better in {int((s < a).sum())}/{args.seeds}')
    say(f'\nresults written to {args.out}')


if __name__ == '__main__':
    main()
