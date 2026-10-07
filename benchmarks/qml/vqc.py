#!/usr/bin/env python3
"""
Training a variational quantum classifier on IBM's FakeFez -- SoftOpt as
built vs Adam and COBYLA, same budget of device readings.

Task: Iris, versicolor vs virginica (the two overlapping classes), 4 features
-> 4 qubits. 60 training flowers, 40 held-out test flowers (fixed split).

Circuit (what a customer would write in Qiskit):
    RY(x_q) on every qubit q            -- angle encoding of the 4 features
    EfficientSU2(4, reps=2, linear)     -- 24 trainable parameters
    prediction  f(x) = <Z on qubit 0>,  class = sign(f)
Loss: mean squared error to the labels +-1 over the training set.

One device READING = the loss on the whole training set (60 circuits,
4096 shots each, batched). Every method pays in readings:
    adam     Adam on the SPSA gradient, 3 readings per step  (lr 0.02, probe 0.1)
    softopt  SoftOpt as released (defaults), same lr, probe and readings;
             its slope and curvature come from the circuit by Lemma 6.1
             soft-number propagation (softopt.soft_number via soft_su2.py)
    cobyla   scipy COBYLA, defaults, 1 reading per evaluation
All start from the same parameters; budget = 3 x steps readings each.

Scores on the final parameters:
    test accuracy on the device (FakeFez, 4096 shots per flower)  <- what the customer gets
    test accuracy and training loss of the noiseless circuit

    pip install numpy scipy scikit-learn qiskit qiskit-aer qiskit-ibm-runtime
    python3 vqc.py --quick
    python3 vqc.py                    # 10 seeds x 100 steps
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
sys.path.insert(0, os.path.join(_here, '..', 'vqe'))                        # soft_su2.py
sys.path.insert(0, os.path.abspath(os.path.join(_here, '..', '..')))      # the repo root (softopt/)
from softopt import SoftOpt                                                  # noqa: E402
from softopt.soft_number import sadd, smul                                  # noqa: E402
from soft_su2 import _gate, _ry_soft, _rz_soft, _cnot                       # noqa: E402

from qiskit import QuantumCircuit                                           # noqa: E402
from qiskit.circuit import ParameterVector                                  # noqa: E402
from qiskit.circuit.library import EfficientSU2                             # noqa: E402
from qiskit.quantum_info import SparsePauliOp                               # noqa: E402
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # noqa: E402
from qiskit_aer import AerSimulator                                         # noqa: E402
from qiskit.primitives import BackendEstimatorV2                            # noqa: E402
try:
    from qiskit_ibm_runtime.fake_provider import FakeFezV2 as FakeBackend
except ImportError:
    from qiskit_ibm_runtime.fake_provider import FakeFez as FakeBackend

NQ, REPS, LR, C, INIT = 4, 2, 0.02, 0.1, 0.3
NP = NQ * 2 * (REPS + 1)


def data(name='iris'):
    if name == 'iris':
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)
        m = y > 0                                          # versicolor (1) vs virginica (2)
        X, y = X[m], np.where(y[m] == 1, 1.0, -1.0)
        n_tr, n_te = 60, 40
    else:                                                  # breast cancer: 30 features -> 8 principal components
        from sklearn.datasets import load_breast_cancer
        from sklearn.decomposition import PCA
        X, y = load_breast_cancer(return_X_y=True)
        X = (X - X.mean(0)) / X.std(0)
        X = PCA(n_components=8, random_state=0).fit_transform(X)
        y = np.where(y == 1, 1.0, -1.0)
        n_tr, n_te = 60, 60
    X = (X - X.min(0)) / (X.max(0) - X.min(0)) * np.pi     # features -> angles in [0, pi]
    idx = np.random.default_rng(0).permutation(len(X))
    tr, te = idx[:n_tr], idx[n_tr:n_tr + n_te]
    return X[tr], y[tr], X[te], y[te]


# ------------------------------------------------------------- soft model
def soft_predict(theta, delta, x):
    """<Z_0> of the circuit for one input, as a soft number (D1 along delta, value)."""
    dim = 2 ** NQ
    a = np.zeros(dim, complex); b = np.zeros(dim, complex); b[0] = 1.0
    pair = (a, b)
    for q in range(NQ):                                    # encoding: data, no trainable part (potential 0)
        pair = _gate(pair, _ry_soft((0.0, x[q])), q, NQ)
    k = 0
    for layer in range(REPS + 1):
        for q in range(NQ):
            pair = _gate(pair, _ry_soft((delta[k], theta[k])), q, NQ); k += 1
        for q in range(NQ):
            pair = _gate(pair, _rz_soft((delta[k], theta[k])), q, NQ); k += 1
        if layer < REPS:
            for q in range(NQ - 1):
                pair = _cnot(pair, q, q + 1, NQ)
    a, b = pair
    z0 = np.where(np.arange(dim) & 1, -1.0, 1.0)           # Z on qubit 0 (least significant bit)
    pot, val = smul((np.conj(a), np.conj(b)), (z0 * a, z0 * b))
    return float(np.real(pot.sum())), float(np.real(val.sum()))


def soft_loss(theta, delta, X, Y):
    """Mean squared error as a soft number: sum of smul(f - y, f - y) over the set."""
    tot = (0.0, 0.0)
    for x, y in zip(X, Y):
        d, v = soft_predict(theta, delta, x)
        r = (d, v - y)
        tot = sadd(tot, smul(r, r))
    return tot[0] / len(X), tot[1] / len(X)


def exact_predict(theta, X):
    return np.array([soft_predict(theta, np.zeros(NP), x)[1] for x in X])


# ------------------------------------------------------------- device
class Device:
    def __init__(self, shots):
        backend = AerSimulator.from_backend(FakeBackend(), method='density_matrix')
        xs = ParameterVector('x', NQ)
        qc = QuantumCircuit(NQ)
        for q in range(NQ):
            qc.ry(xs[q], q)
        ans = EfficientSU2(NQ, reps=REPS, entanglement='linear')
        qc.compose(ans, inplace=True)
        self.order = list(qc.parameters)                   # qiskit sorts parameters by name
        self.th_names = [p.name for p in ans.parameters]
        self.isa = generate_preset_pass_manager(backend=backend, optimization_level=1).run(qc)
        self.obs = SparsePauliOp('I' * (NQ - 1) + 'Z').apply_layout(self.isa.layout)
        self.est = BackendEstimatorV2(backend=backend)
        self.est.options.default_shots = shots
        self.est.options.seed_simulator = 42
        self.reads = 0

    def _values(self, theta, X):
        th = dict(zip(self.th_names, theta))
        rows = []
        for x in X:
            v = []
            for p in self.order:
                v.append(x[int(p.name.split('[')[1][:-1])] if p.name.startswith('x[') else th[p.name])
            rows.append(v)
        return np.array(rows)

    def predict_many(self, thetas, X):
        """One batched call: the predictions for every theta in `thetas` on every x in X."""
        pubs = [(self.isa, self.obs, self._values(t, X)) for t in thetas]
        return [np.asarray(r.data.evs) for r in self.est.run(pubs).result()]


def run_all(dev, Xtr, Ytr, seeds, steps, say):
    loss = lambda f: float(np.mean((f - Ytr) ** 2))                         # noqa: E731
    runs = []
    for seed in range(seeds):
        x0 = np.random.default_rng(seed).uniform(-INIT, INIT, NP)
        dirs = np.random.default_rng(seed * 1000).choice([-1.0, 1.0], size=(steps, NP))
        for arm in ('adam', 'softopt'):
            opt = SoftOpt(NP, lambda th, d: soft_loss(th, d, Xtr, Ytr)[0], lr=LR, seed=seed + 999)
            runs.append(dict(seed=seed, arm=arm, x=x0.copy(), dirs=dirs, opt=opt,
                             hook='plain' if arm == 'adam' else None))
    t0 = time.time()
    for k in range(steps):
        thetas = []
        for r in runs:
            d = r['dirs'][k]
            thetas += [r['x'], r['x'] + C * d, r['x'] - C * d]
        f = dev.predict_many(thetas, Xtr)
        for i, r in enumerate(runs):
            d = r['dirs'][k]
            slope = (loss(f[3 * i + 1]) - loss(f[3 * i + 2])) / (2 * C)
            r['x'] = r['opt'].step(r['x'], slope * d, _test_magnitude_source=r['hook'])
        if (k + 1) % 10 == 0:
            say(f'   step {k + 1}/{steps}  ({time.time() - t0:.0f}s)')
    out = {(r['seed'], r['arm']): r['x'] for r in runs}
    for seed in range(seeds):                                               # COBYLA, same budget
        x0 = np.random.default_rng(seed).uniform(-INIT, INIT, NP)
        fn = lambda th: loss(dev.predict_many([th], Xtr)[0])                # noqa: E731
        out[(seed, 'cobyla')] = minimize(fn, x0, method='COBYLA', options=dict(maxiter=3 * steps)).x
        say(f'   cobyla seed {seed + 1}/{seeds}  ({time.time() - t0:.0f}s)')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--shots', type=int, default=4096)
    ap.add_argument('--dataset', default='iris', choices=['iris', 'cancer'],
                    help='iris: 4 qubits, 24 parameters; cancer: breast cancer, 8 PCA features, 8 qubits, 48 parameters')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='vqc_results.json')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.steps = 2, 4
    global NQ, NP
    NQ = 4 if args.dataset == 'iris' else 8
    NP = NQ * 2 * (REPS + 1)
    say = lambda *a: print(*a, flush=True)                                  # noqa: E731
    Xtr, Ytr, Xte, Yte = data(args.dataset)
    # checks: soft slope vs finite difference; our circuit vs Qiskit's own statevector
    th = np.random.default_rng(1).uniform(-1, 1, NP); dl = np.random.default_rng(2).choice([-1.0, 1.0], NP)
    D1 = soft_loss(th, dl, Xtr, Ytr)[0]
    h = 1e-6
    fd = (soft_loss(th + h * dl, 0 * dl, Xtr, Ytr)[1] - soft_loss(th - h * dl, 0 * dl, Xtr, Ytr)[1]) / (2 * h)
    from qiskit.quantum_info import Statevector
    dev = Device(args.shots)
    vals = dev._values(th, Xtr[:1])[0]
    bound = QuantumCircuit(NQ)
    xs = ParameterVector('x', NQ)
    for q in range(NQ):
        bound.ry(xs[q], q)
    bound.compose(EfficientSU2(NQ, reps=REPS, entanglement='linear'), inplace=True)
    sv = Statevector(bound.assign_parameters(dict(zip(list(bound.parameters), vals))))
    z_qiskit = float(np.real(sv.expectation_value(SparsePauliOp('I' * (NQ - 1) + 'Z'))))
    say(f'checks: soft slope {D1:.8f} vs finite difference {fd:.8f};  '
        f'<Z0> ours {exact_predict(th, Xtr[:1])[0]:.8f} vs Qiskit {z_qiskit:.8f}')
    say(f'{"Iris versicolor vs virginica" if args.dataset == "iris" else "Breast cancer (8 PCA features)"}: {len(Xtr)} train / {len(Xte)} test; {NQ} qubits, {NP} parameters; '
        f'FakeFez, {args.shots} shots; {args.steps} steps x 3 = {3 * args.steps} readings per method; {args.seeds} seeds')
    t0 = time.time()
    final = run_all(dev, Xtr, Ytr, args.seeds, args.steps, say)
    arms = ('adam', 'softopt', 'cobyla')
    keys = [(s, a) for s in range(args.seeds) for a in arms]
    dev_te = dev.predict_many([final[k] for k in keys], Xte)
    res = {a: dict(device_test_acc=[], exact_test_acc=[], exact_train_loss=[]) for a in arms}
    for (s, a), f in zip(keys, dev_te):
        res[a]['device_test_acc'].append(float(np.mean(np.sign(f) == Yte)))
        res[a]['exact_test_acc'].append(float(np.mean(np.sign(exact_predict(final[(s, a)], Xte)) == Yte)))
        res[a]['exact_train_loss'].append(float(np.mean((exact_predict(final[(s, a)], Xtr) - Ytr) ** 2)))
    with open(args.out, 'w') as fh:
        json.dump(dict(config=vars(args), res=res), fh, indent=1)
    say(f'\nresults ({time.time() - t0:.0f}s), median over {args.seeds} seeds')
    say(f'  {"":8s} {"test accuracy (device)":>24s} {"test accuracy (exact)":>23s} {"train loss (exact)":>20s}')
    for a in arms:
        r = res[a]
        say(f'  {a:8s} {np.median(r["device_test_acc"]):24.1%} {np.median(r["exact_test_acc"]):23.1%} '
            f'{np.median(r["exact_train_loss"]):20.4f}')
    for comp in ('adam', 'cobyla'):
        for m, better in (('exact_train_loss', -1), ('device_test_acc', 1)):
            s, c = np.array(res['softopt'][m]), np.array(res[comp][m])
            d = better * (s - c)
            p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            say(f'  softopt vs {comp:6s} {m:17s}: better in {int((d > 0).sum())}, tie {int((d == 0).sum())}, '
                f'worse {int((d < 0).sum())}   p={p:.2g}')
    say(f'\nresults written to {args.out}')


if __name__ == '__main__':
    main()
