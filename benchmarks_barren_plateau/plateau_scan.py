"""
Does the advantage grow with the depth of the plateau?

    python3 plateau_scan.py            # the full scan
    python3 plateau_scan.py --quick    # a short version, ~10 minutes

The single-configuration result showed SoftOpt ahead 12/12 at 8 qubits. That
says the derivative helps; it does not say why. If the mechanism is what we
claim -- an exact derivative surviving where a measured one is lost in shot
noise -- then the advantage should track how badly the measurement is losing.

So this scans the qubit count. As n grows the partial derivatives shrink like
2^-n while shot noise stays where it is, so the measurement degrades and the
exact derivative does not. A rising advantage supports the mechanism. A flat
one would mean something else is doing the work.

The measurement to watch is the last column: the ratio of shot noise to the
exact partial derivative. It is the depth of the plateau, and everything else
should be read against it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# The two circuit modules live beside this file, or one directory up in the
# benchmarks tree. Look in both rather than assuming a layout.
_here = Path(__file__).resolve().parent
for _p in (_here, _here / "vqe", _here.parent / "vqe",
           _here.parent / "benchmarks" / "vqe"):
    if (_p / "efficient_su2_exact.py").exists():
        sys.path.insert(0, str(_p))
        break
else:
    sys.exit(
        "Could not find efficient_su2_exact.py and efficient_su2_lemma61.py.\n"
        "Put both beside this script -- they are in the SoftOpt repository\n"
        "under benchmarks/vqe/.")

from efficient_su2_exact import efficient_su2_statevector
from efficient_su2_lemma61 import g_delta_su2

try:
    from softopt import SoftOpt
except ImportError:
    sys.exit("pip install softopt")

SHOTS = 4096
PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
         "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1]).astype(complex)}


def terms_for(n):
    """Transverse-field Ising chain on n sites."""
    t = []
    for i in range(n - 1):
        s = ["I"] * n; s[i] = s[i + 1] = "Z"
        t.append((1.0, "".join(reversed(s))))
    for i in range(n):
        s = ["I"] * n; s[i] = "X"
        t.append((0.5, "".join(reversed(s))))
    return t


class System:
    """One (qubits, layers) configuration, with its exact ground state."""

    def __init__(self, n_qubits, reps):
        self.n, self.reps = n_qubits, reps
        self.n_params = n_qubits * 2 * (reps + 1)
        self.terms = terms_for(n_qubits)
        H = np.zeros((2 ** n_qubits, 2 ** n_qubits), dtype=complex)
        for coeff, p in self.terms:
            m = PAULI[p[-1]]
            for ch in reversed(p[:-1]):
                m = np.kron(PAULI[ch], m)
            H = H + coeff * m
        self.H = H
        self.ground = float(np.min(np.linalg.eigvalsh(H)))
        self.noise = sum(abs(c) for c, _ in self.terms) / np.sqrt(SHOTS)

    def energy(self, theta):
        psi = efficient_su2_statevector(self.n, self.reps, theta)
        return float(np.real(np.conj(psi) @ self.H @ psi))

    def measured(self, theta, rng):
        return self.energy(theta) + rng.normal(0.0, self.noise)

    def g_delta(self, theta, delta):
        return g_delta_su2(theta, delta, self.n, self.reps, self.terms)

    def plateau_depth(self, samples=25, seed=0):
        """Shot noise divided by the typical exact partial derivative.
        Above 1 the measurement cannot resolve the gradient at all."""
        rng = np.random.default_rng(seed)
        mags = []
        for _ in range(samples):
            th = rng.uniform(-np.pi, np.pi, self.n_params)
            d = np.zeros(self.n_params)
            d[rng.integers(self.n_params)] = 1.0
            mags.append(abs(self.g_delta(th, d)))
        typical = float(np.median(mags))
        # a central difference at step c sees noise of sqrt(2)*sigma/(2c)
        c = 0.1
        return (np.sqrt(2) * self.noise / (2 * c)) / max(typical, 1e-12), typical


class Adam:
    def __init__(self, lr): self.lr=lr; self.m=self.v=None; self.t=0
    def step(self, x, g):
        if self.m is None: self.m=np.zeros_like(x); self.v=np.zeros_like(x)
        self.t += 1
        self.m = 0.9*self.m + 0.1*g
        self.v = 0.999*self.v + 0.001*g*g
        mh=self.m/(1-0.9**self.t); vh=self.v/(1-0.999**self.t)
        return x - self.lr*mh/(np.sqrt(vh)+1e-8)


def run(sys_, arm, seed, steps, lr=0.05, c=0.1):
    """Both arms see the identical sequence of noisy measurements. Only the
    step differs."""
    rng = np.random.default_rng(seed * 1000)
    th = np.random.default_rng(seed).uniform(-np.pi, np.pi, sys_.n_params)
    opt = (Adam(lr) if arm == "adam"
           else SoftOpt(sys_.n_params, sys_.g_delta, lr=lr, seed=seed + 999))
    best = np.inf
    for _ in range(steps):
        d = rng.choice([-1.0, 1.0], size=sys_.n_params)
        g = ((sys_.measured(th + c*d, rng) - sys_.measured(th - c*d, rng))
             / (2*c)) * d
        th = opt.step(th, g)
        best = min(best, sys_.energy(th))
    return best - sys_.ground


def main(configs, seeds, steps, out):
    rows = []
    for n_qubits, reps in configs:
        s = System(n_qubits, reps)
        depth, typical = s.plateau_depth()
        t0 = time.perf_counter()
        a = np.array([run(s, "adam", k, steps) for k in range(seeds)])
        b = np.array([run(s, "softopt", k, steps) for k in range(seeds)])
        # how much of the distance to the ground state each arm covered
        span = abs(s.ground)
        rows.append(dict(qubits=n_qubits, reps=reps, params=s.n_params,
                         ground=s.ground, plateau_depth=depth,
                         typical_partial=typical,
                         adam_gap=float(np.median(a)),
                         soft_gap=float(np.median(b)),
                         wins=int(np.sum(b < a)), seeds=seeds,
                         advantage=float(np.median(a) - np.median(b)),
                         advantage_frac=float((np.median(a)-np.median(b))/span),
                         seconds=time.perf_counter() - t0,
                         adam_all=a.tolist(), soft_all=b.tolist()))
        r = rows[-1]
        print(f"  {n_qubits}q/{reps}L  params={s.n_params:<4} "
              f"depth={depth:>5.1f}  Adam={r['adam_gap']:.3f}  "
              f"SoftOpt={r['soft_gap']:.3f}  "
              f"advantage={r['advantage_frac']*100:>5.1f}%  "
              f"wins={r['wins']}/{seeds}  ({r['seconds']:.0f}s)", flush=True)
        json.dump(rows, open(out, "w"), indent=2)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="plateau_scan.json")
    args = ap.parse_args()

    if args.quick:
        configs, seeds, steps = [(4, 6), (6, 8), (8, 10)], 6, 120
    else:
        configs, seeds, steps = [(4, 6), (5, 7), (6, 8), (7, 9), (8, 10),
                                 (9, 11), (10, 12)], 15, 300

    print("Barren-plateau scan. 'depth' is shot noise over the typical exact")
    print("partial derivative: above 1, the measurement cannot see the gradient.")
    print(f"{seeds} seeds, {steps} steps per run.\n")
    rows = main(configs, seeds, steps, args.out)

    print(f"\nWrote {args.out}")
    d = np.array([r["plateau_depth"] for r in rows])
    v = np.array([r["advantage_frac"] for r in rows])
    if len(d) > 2:
        from scipy.stats import spearmanr
        print(f"advantage vs plateau depth: rho = {spearmanr(d, v).correlation:+.3f}")
        print("A positive correlation supports the mechanism: the exact")
        print("derivative helps most where the measured one is least usable.")
