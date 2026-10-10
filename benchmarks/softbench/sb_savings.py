#!/usr/bin/env python3
"""Measurement savings on SoftBench: how many readings does each method need to reach the quality Adam reaches
with the full 200? Same problems, seeds, lrs (from softbench_<p>.json) and code as softbench.py; the current point
is recorded after every step (COBYLA: its best-seen point). Per seed: quality target q = Adam's gap at 200
readings; readings(method) = first reading count at which its gap <= q; saving = 200 / readings. A method that
never reaches q gets 'not reached'. Also readings to a fixed 5% gap.
    python3 sb_savings.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import softbench as B

P_ = ['vqe', 'beam', 'arm', 'reactor', 'pk', 'portfolio', 'thermal', 'battery', 'epidemic', 'shift', 'exact']
ARMS = ['adam', 'cobyla', 'softopt', 'softopt_measured', 'cal_beta']


def first(h, thr):
    for r, g in h:
        if g <= thr:
            return r
    return None


out = {}
print(f'{"domain":10s}' + ''.join(f'{a + " (x/n)":>26s}' for a in ARMS[1:]) + '   readings to 5% gap: adam / cal_beta')
for p in P_:
    d = json.load(open(f'softbench_{p}.json')); lr = d['best_lr']; C = B.PROBLEMS[p]
    H = {a: [] for a in ARMS}
    for s in range(10):
        P0 = C(s); fs = B.optimum(P0); f0 = P0.truth(np.zeros(P0.n))
        for a in ARMS:
            P = C(s); hist = []
            g = None if a in ('adam', 'cobyla') else dict(free=B.soft_compile(lambda th: P.model(th, [0.0] * P.nb), n_params=P.n, verify=False),
                                                          beta=B.soft_compile(P.model, n_params=P.n, n_bias=P.nb, verify=False))
            x = B.run(P, a, s, lr, g, hist)
            gh = [(r, (P.truth(z) - fs) / max(f0 - fs, 1e-12)) for r, z in hist]
            H[a].append(gh)
    row = {}
    line = f'{p:10s}'
    for a in ARMS[1:]:
        sav, nr = [], 0
        for s in range(10):
            q = H['adam'][s][-1][1]
            r = first(H[a][s], q)
            if r is None:
                nr += 1; sav.append(0.0)
            else:
                sav.append(200.0 / max(r, 1))
        row[a] = dict(saving_median=float(np.median(sav)), not_reached=nr, savings=sav)
        line += f'{np.median(sav):17.2f}x ({10 - nr:2d}/10)'
    r5a = [first(h, 0.05) for h in H['adam']]; r5c = [first(h, 0.05) for h in H['cal_beta']]
    f = lambda v: f'{int(np.median([x for x in v if x is not None]))} ({sum(x is not None for x in v)}/10)' if any(x is not None for x in v) else 'never'   # noqa: E731
    line += f'   {f(r5a)} / {f(r5c)}'
    print(line, flush=True)
    out[p] = dict(row=row, r5_adam=r5a, r5_cal=r5c)
json.dump(out, open('sb_savings.json', 'w'))
