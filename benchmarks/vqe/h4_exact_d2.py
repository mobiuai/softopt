#!/usr/bin/env python3
"""
H4 on FakeFez (the h4_real_molecule.py protocol, unchanged) with ONE addition: a third arm whose
curvature D2 is exact -- nested soft numbers through the circuit (soft_su2_exact.py), no h_fd --
next to the released arm (D2 = finite difference of the exact soft D1, h_fd = 1e-4) and Adam.

Everything else is identical: start points, SPSA directions, lr 0.02, default SoftOpt settings,
the same probe seeds. The device's shot noise depends on each reading's place in the batch, so the
experiment runs TWICE with a fresh estimator (same simulator seed): [adam, softopt] -- exactly the
published h4_real_molecule.py run -- and [adam, softopt_exact]. The two SoftOpt arms then receive
the same noise realisations, Adam must come out identical in both runs (checked), and any
difference between 'softopt' and 'softopt_exact' is due to the source of D2 alone.

    python3 h4_exact_d2.py --quick        # smoke test
    python3 h4_exact_d2.py                # 10 seeds x 100 steps, as h4_real_molecule.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import h4_real_molecule as B                                                # noqa: E402  (Molecule, constants)
from softopt import SoftOpt, __version__                                    # noqa: E402
from soft_su2_exact import d1_d2_su2_fast                                   # noqa: E402

ARMS = ('adam', 'softopt', 'softopt_exact')


def run_all(S, arms, seeds, steps, every, say):
    def g_fd(x, d):                       # the released path: SoftOpt finite-differences D1 for D2
        return S.g_delta(x, d)

    def g_exact(x, d):
        return S.g_delta(x, d)
    g_exact.d1_d2 = lambda x, d: d1_d2_su2_fast(x, d, S.nq, S.reps, S.Hs)

    runs = []
    for seed in range(seeds):
        x0 = np.random.default_rng(seed).uniform(-B.INIT, B.INIT, S.n)
        dirs = np.random.default_rng(seed * 1000).choice([-1.0, 1.0], size=(steps, S.n))
        for arm in arms:
            opt = SoftOpt(S.n, g_exact if arm == 'softopt_exact' else g_fd, lr=B.LR, seed=seed + 999)
            assert (opt._d1_d2 is not None) == (arm == 'softopt_exact')
            runs.append(dict(seed=seed, arm=arm, x=x0.copy(), dirs=dirs, opt=opt,
                             hook='plain' if arm == 'adam' else None,
                             traj=[1e3 * (S.exact(x0) - S.ground)], t=0.0, newton=0))
    t0 = time.time()
    for k in range(steps):
        pubs = []
        for r in runs:
            d = r['dirs'][k]
            pubs += [(S.isa, S.ops, r['x']), (S.isa, S.ops, r['x'] + B.C * d), (S.isa, S.ops, r['x'] - B.C * d)]
        ev = [float(p.data.evs) for p in S.est.run(pubs).result()]
        for i, r in enumerate(runs):
            d = r['dirs'][k]
            slope = (ev[3 * i + 1] - ev[3 * i + 2]) / (2 * B.C)
            t1 = time.time()
            r['x'] = r['opt'].step(r['x'], slope * d, _test_magnitude_source=r['hook'])
            r['t'] += time.time() - t1
            if (k + 1) % every == 0 or k + 1 == steps:
                r['traj'].append(1e3 * (S.exact(r['x']) - S.ground))
        if (k + 1) % every == 0:
            med = {a: np.median([r['traj'][-1] for r in runs if r['arm'] == a]) for a in arms}
            say(f'   step {k + 1:4d}/{steps}  ' + '  '.join(f'{a} {v:8.1f}' for a, v in med.items()) +
                f'   mHa   ({time.time() - t0:.0f}s)')
    dev = [float(p.data.evs) for p in S.final_est.run([(S.isa, S.ops, r['x']) for r in runs]).result()]
    res = {a: {'traj': [None] * seeds, 'device': [None] * seeds, 'opt_time': 0.0} for a in arms}
    for r, e in zip(runs, dev):
        res[r['arm']]['traj'][r['seed']] = r['traj']
        res[r['arm']]['device'][r['seed']] = 1e3 * (e - S.ground)
        res[r['arm']]['opt_time'] += r['t']
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--every', type=int, default=10)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--shots', type=int, default=4096)
    ap.add_argument('--final-shots', type=int, default=65536)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='results/h4_exact_d2_results.json')
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.steps, a.every = 1, 2, 1
    say = lambda *x: print(*x, flush=True)                                  # noqa: E731
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    t0 = time.time()
    S = B.Molecule('h4', a.reps, a.shots, a.final_shots)
    say(f'softopt {__version__}; H4: {S.nq} qubits, {S.n_terms} Pauli terms, EfficientSU2 reps={S.reps} ({S.n} params); '
        f'FakeFez {a.shots} shots; lr {B.LR}; {a.steps} steps; {a.seeds} seeds; arms {", ".join(ARMS)}')
    th = np.random.default_rng(0).uniform(-B.INIT, B.INIT, S.n); d = np.ones(S.n)
    D1, D2 = d1_d2_su2_fast(th, d, S.nq, S.reps, S.Hs)
    h = 1e-4
    fd = (S.g_delta(th + h * d, d) - S.g_delta(th - h * d, d)) / (2 * h)
    say(f'check: D1 exact {D1:.10f} vs soft_su2 {S.g_delta(th, d):.10f};  D2 exact {D2:.10f} vs finite difference {fd:.10f}')
    say('\nrun 1/2: adam + softopt (released D2 path; the published protocol)')
    r1 = run_all(S, ('adam', 'softopt'), a.seeds, a.steps, a.every, say)
    say('\nrun 2/2: adam + softopt_exact (fresh estimator, same simulator seed)')
    S = B.Molecule('h4', a.reps, a.shots, a.final_shots)
    r2 = run_all(S, ('adam', 'softopt_exact'), a.seeds, a.steps, a.every, say)
    same = r1['adam']['traj'] == r2['adam']['traj']
    say(f'check: Adam identical in both runs (same readings): {same}')
    res = dict(adam=r1['adam'], softopt=r1['softopt'], softopt_exact=r2['softopt_exact'])
    json.dump(dict(config=vars(a), ground=S.ground, res=res), open(a.out, 'w'))
    say(f'\nerror above the ground state (mHa), median over seeds   ({time.time() - t0:.0f}s)')
    i60 = min(60 // a.every, len(res['adam']['traj'][0]) - 1)
    for idx, lab in ((i60, f'exact energy, step {i60 * a.every}'), (-1, f'exact energy, step {a.steps}')):
        v = {arm: np.array([t[idx] for t in res[arm]['traj']]) for arm in ARMS}
        say(f'  {lab:26s} ' + '   '.join(f'{arm} {np.median(v[arm]):8.2f}' for arm in ARMS))
        for x, y in (('softopt', 'adam'), ('softopt_exact', 'adam'), ('softopt_exact', 'softopt')):
            dd = v[x] - v[y]
            p = stats.wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
            say(f'      {x} vs {y}: better in {int((dd < 0).sum())}/{a.seeds}, median diff {np.median(dd):+.3f} mHa, '
                f'max |diff| {np.abs(dd).max():.3g}, p={p:.2g}')
    dv = {arm: np.array(res[arm]['device']) for arm in ARMS}
    say(f'  {"device (FakeFez), end":26s} ' + '   '.join(f'{arm} {np.median(dv[arm]):8.2f}' for arm in ARMS))
    say(f'  optimiser time (all seeds): ' + ', '.join(f'{arm} {res[arm]["opt_time"]:.1f}s' for arm in ARMS))
    say(f'\nresults written to {a.out}')


if __name__ == '__main__':
    main()
