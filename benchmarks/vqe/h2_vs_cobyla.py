#!/usr/bin/env python3
"""
SoftOpt vs COBYLA on H2, IBM FakeFez noise model.

Same setup as h2.py in this folder: Qiskit's EfficientSU2(2, reps=4), real
transpilation, AerSimulator.from_backend(FakeFez), 4096 shots per reading.

  SoftOpt  as released (default settings, slope="model"), lr 0.02,
           60 steps x 3 readings (SPSA probe) = 180 device readings.
           Slope from Lemma 6.1 soft-number propagation (efficient_su2_lemma61.py).
  COBYLA   scipy's COBYLA with default settings, the same start point and the
           same budget: 180 device readings (one reading per evaluation).

Score: exact energy of the final parameters above the ground state (mHa),
and the energy the device itself returns there (65536 shots).

    pip install numpy scipy qiskit qiskit-aer qiskit-ibm-runtime
    python3 h2_vs_cobyla.py            # 10 seeds
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.abspath(os.path.join(_here, '..', '..')))      # the repo root (softopt/)
from softopt import SoftOpt                                                  # noqa: E402
from efficient_su2_lemma61 import g_delta_su2                               # noqa: E402
from efficient_su2_exact import efficient_su2_statevector                   # noqa: E402

from qiskit.circuit.library import EfficientSU2                             # noqa: E402
from qiskit.quantum_info import SparsePauliOp                               # noqa: E402
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # noqa: E402
from qiskit_aer import AerSimulator                                         # noqa: E402
from qiskit.primitives import BackendEstimatorV2                            # noqa: E402
try:
    from qiskit_ibm_runtime.fake_provider import FakeFezV2 as FakeBackend
except ImportError:
    from qiskit_ibm_runtime.fake_provider import FakeFez as FakeBackend

TERMS = [('II', -0.4804), ('ZZ', 0.3435), ('ZI', -0.4347), ('IZ', 0.5716), ('XX', 0.0910), ('YY', 0.0910)]
NQ, REPS, LR, STEPS, INIT, C = 2, 4, 0.02, 60, 0.3, 0.1
CHEM_ACC = 1.6  # mHa


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--shots', type=int, default=4096)
    ap.add_argument('--final-shots', type=int, default=65536)
    ap.add_argument('--out', default='results/h2_vs_cobyla_results.json')
    args = ap.parse_args()

    H = SparsePauliOp.from_list(TERMS)
    Hm = H.to_matrix()
    ground = float(np.linalg.eigvalsh(Hm).min())
    backend = AerSimulator.from_backend(FakeBackend())
    ansatz = EfficientSU2(NQ, reps=REPS, entanglement='linear')
    isa = generate_preset_pass_manager(backend=backend, optimization_level=1).run(ansatz)
    ops = H.apply_layout(isa.layout)
    n = ansatz.num_parameters
    est = BackendEstimatorV2(backend=backend)
    est.options.default_shots = args.shots
    est.options.seed_simulator = 42
    final = BackendEstimatorV2(backend=backend)
    final.options.default_shots = args.final_shots
    final.options.seed_simulator = 7
    soft_terms = [(c, p) for p, c in TERMS]

    def exact(x):
        psi = efficient_su2_statevector(NQ, REPS, np.asarray(x))
        return float(np.real(np.conj(psi) @ Hm @ psi))

    def device(x):
        return float(final.run([(isa, ops, np.asarray(x))]).result()[0].data.evs)

    t0 = time.time()
    # ---- SoftOpt: all seeds advance together, one batched device call per step
    runs = []
    for seed in range(args.seeds):
        x0 = np.random.default_rng(seed).uniform(-INIT, INIT, n)
        dirs = np.random.default_rng(seed * 1000).choice([-1.0, 1.0], size=(STEPS, n))
        opt = SoftOpt(n, lambda th, d: g_delta_su2(th, d, NQ, REPS, soft_terms), lr=LR, seed=seed + 999)
        runs.append(dict(x=x0, dirs=dirs, opt=opt))
    for k in range(STEPS):
        pubs = []
        for r in runs:
            d = r['dirs'][k]
            pubs += [(isa, ops, r['x']), (isa, ops, r['x'] + C * d), (isa, ops, r['x'] - C * d)]
        ev = [float(p.data.evs) for p in est.run(pubs).result()]
        for i, r in enumerate(runs):
            d = r['dirs'][k]
            slope = (ev[3 * i + 1] - ev[3 * i + 2]) / (2 * C)
            r['x'] = r['opt'].step(r['x'], slope * d)
    so = dict(exact=[exact(r['x']) - ground for r in runs], device=[device(r['x']) - ground for r in runs])

    # ---- COBYLA: same start, same device, same number of readings
    co = dict(exact=[], device=[], readings=[])
    for seed in range(args.seeds):
        x0 = np.random.default_rng(seed).uniform(-INIT, INIT, n)
        count = [0]

        def f(x):
            count[0] += 1
            return float(est.run([(isa, ops, x)]).result()[0].data.evs)

        x = minimize(f, x0, method='COBYLA', options=dict(maxiter=3 * STEPS)).x
        co['exact'].append(exact(x) - ground)
        co['device'].append(device(x) - ground)
        co['readings'].append(count[0])

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(dict(config=vars(args), ground=ground, softopt=so, cobyla=co), fh, indent=1)
    print(f'H2: {NQ} qubits, {n} params; SoftOpt {STEPS} steps x 3 = {3 * STEPS} readings; COBYLA used '
          f'{int(np.median(co["readings"]))} readings (median); {args.seeds} seeds  ({time.time() - t0:.0f}s)')
    print('error above the ground state, median over seeds (mHa)')
    for score in ('exact', 'device'):
        s, c = 1e3 * np.array(so[score]), 1e3 * np.array(co[score])
        d = s - c
        p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        print(f'  [{score:6s}] SoftOpt {np.median(s):8.2f}   COBYLA {np.median(c):8.2f}   SoftOpt better in '
              f'{int((d < 0).sum())}/{args.seeds}   gap closed {100 * (np.median(c) - np.median(s)) / np.median(c):+.0f}%'
              f'   p={p:.2g}   chemical accuracy {int((s < CHEM_ACC).sum())} vs {int((c < CHEM_ACC).sum())}')


if __name__ == '__main__':
    main()
