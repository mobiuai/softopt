"""
Replicating the barren-plateau result before it goes anywhere public.

    python3 plateau_replicate.py --quick     # ~15 min, the core checks
    python3 plateau_replicate.py             # ~2 hours, all of them

The first scan found SoftOpt covering about 1.5x as much of the distance to
the ground state as Adam, and found that ratio holding across a 15x range of
plateau depth. That is a clean result and it came from one run of one setup,
so before it is claimed anywhere it has to survive the things that could have
produced it by accident.

Four checks, each one able to overturn the finding:

  seeds       Different noise streams and different starting points. If the
              ratio moves, fifteen seeds was not enough.

  shots       The plateau was defined against 4096-shot noise. At 1024 the
              measurement is worse and at 16384 it is better; the ratio should
              move with it in the right direction, or the mechanism is not
              what we think.

  step        SPSA's probe size c was fixed at 0.1 throughout. A result that
              only appears at one probe size is a tuning artefact.

  hamiltonian The target was a transverse-field Ising chain every time. A
              different observable should not change the ratio.

Each check reports the progress ratio -- how far SoftOpt got toward the ground
state, over how far Adam got. That is the quantity the claim rests on, and it
is the one that stayed flat in the first scan.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
for _p in (_here, _here / "vqe", _here.parent / "vqe",
           _here.parent / "benchmarks" / "vqe"):
    if (_p / "efficient_su2_exact.py").exists():
        sys.path.insert(0, str(_p))
        break
else:
    sys.exit("Put efficient_su2_exact.py and efficient_su2_lemma61.py beside "
             "this script (SoftOpt repo, benchmarks/vqe/).")

from efficient_su2_exact import efficient_su2_statevector
from efficient_su2_lemma61 import g_delta_su2

try:
    from softopt import SoftOpt
except ImportError:
    sys.exit("pip install softopt")

PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
         "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1]).astype(complex)}


def ising(n):
    """Transverse-field Ising chain -- the target used in the first scan."""
    t = []
    for i in range(n - 1):
        s = ["I"] * n; s[i] = s[i + 1] = "Z"
        t.append((1.0, "".join(reversed(s))))
    for i in range(n):
        s = ["I"] * n; s[i] = "X"
        t.append((0.5, "".join(reversed(s))))
    return t


def heisenberg(n):
    """A different observable entirely: XX + YY + ZZ couplings."""
    t = []
    for i in range(n - 1):
        for ax in "XYZ":
            s = ["I"] * n; s[i] = s[i + 1] = ax
            t.append((0.4, "".join(reversed(s))))
    return t


def random_pauli_sum(n, seed=0, n_terms=None):
    """Weight-2 Pauli terms drawn at random -- no physical structure at all."""
    rng = np.random.default_rng(seed)
    n_terms = n_terms or 2 * n
    t = []
    for _ in range(n_terms):
        s = ["I"] * n
        i, j = rng.choice(n, 2, replace=False)
        s[i] = rng.choice(list("XYZ")); s[j] = rng.choice(list("XYZ"))
        t.append((float(rng.uniform(0.3, 1.0)), "".join(reversed(s))))
    return t


HAMILTONIANS = {"ising": ising, "heisenberg": heisenberg,
                "random": random_pauli_sum}


class System:
    def __init__(self, n_qubits, reps, terms_fn=ising, shots=4096):
        self.n, self.reps, self.shots = n_qubits, reps, shots
        self.n_params = n_qubits * 2 * (reps + 1)
        self.terms = terms_fn(n_qubits)
        H = np.zeros((2 ** n_qubits, 2 ** n_qubits), dtype=complex)
        for coeff, p in self.terms:
            m = PAULI[p[-1]]
            for ch in reversed(p[:-1]):
                m = np.kron(PAULI[ch], m)
            H = H + coeff * m
        self.H = H
        self.ground = float(np.min(np.linalg.eigvalsh(H)))
        self.top = float(np.max(np.linalg.eigvalsh(H)))
        self.noise = sum(abs(c) for c, _ in self.terms) / np.sqrt(shots)

    def energy(self, theta):
        psi = efficient_su2_statevector(self.n, self.reps, theta)
        return float(np.real(np.conj(psi) @ self.H @ psi))

    def measured(self, theta, rng):
        return self.energy(theta) + rng.normal(0.0, self.noise)

    def g_delta(self, theta, delta):
        return g_delta_su2(theta, delta, self.n, self.reps, self.terms)

    def depth(self, samples=25, seed=0, c=0.1):
        rng = np.random.default_rng(seed)
        mags = []
        for _ in range(samples):
            th = rng.uniform(-np.pi, np.pi, self.n_params)
            d = np.zeros(self.n_params)
            d[rng.integers(self.n_params)] = 1.0
            mags.append(abs(self.g_delta(th, d)))
        return (np.sqrt(2) * self.noise / (2*c)) / max(float(np.median(mags)), 1e-12)


class Adam:
    def __init__(self, lr): self.lr=lr; self.m=self.v=None; self.t=0
    def step(self, x, g):
        if self.m is None: self.m=np.zeros_like(x); self.v=np.zeros_like(x)
        self.t += 1
        self.m = 0.9*self.m + 0.1*g
        self.v = 0.999*self.v + 0.001*g*g
        mh=self.m/(1-0.9**self.t); vh=self.v/(1-0.999**self.t)
        return x - self.lr*mh/(np.sqrt(vh)+1e-8)


def run(sys_, arm, seed, steps, lr=0.05, c=0.1, seed_offset=0):
    rng = np.random.default_rng(seed * 1000 + seed_offset)
    th = np.random.default_rng(seed + seed_offset).uniform(
        -np.pi, np.pi, sys_.n_params)
    opt = (Adam(lr) if arm == "adam"
           else SoftOpt(sys_.n_params, sys_.g_delta, lr=lr, seed=seed + 999))
    best = np.inf
    for _ in range(steps):
        d = rng.choice([-1.0, 1.0], size=sys_.n_params)
        g = ((sys_.measured(th + c*d, rng) - sys_.measured(th - c*d, rng))
             / (2*c)) * d
        th = opt.step(th, g)
        best = min(best, sys_.energy(th))
    return best


def progress_ratio(sys_, seeds, steps, c=0.1, seed_offset=0):
    """How far each arm got from a random start toward the ground state.

    Measured against the energy at a random initialisation, not against zero:
    a random deep circuit starts near the middle of the spectrum, and scoring
    against zero would flatter whichever arm happens to start lower.
    """
    start_energy = float(np.mean([
        sys_.energy(np.random.default_rng(k + seed_offset).uniform(
            -np.pi, np.pi, sys_.n_params)) for k in range(seeds)]))
    span = start_energy - sys_.ground
    a = np.array([run(sys_, "adam", k, steps, c=c, seed_offset=seed_offset)
                  for k in range(seeds)])
    b = np.array([run(sys_, "softopt", k, steps, c=c, seed_offset=seed_offset)
                  for k in range(seeds)])
    pa = (start_energy - a) / span
    pb = (start_energy - b) / span
    from scipy.stats import wilcoxon
    return dict(adam_progress=float(np.median(pa)),
                soft_progress=float(np.median(pb)),
                ratio=float(np.median(pb) / max(np.median(pa), 1e-12)),
                wins=int(np.sum(b < a)), seeds=seeds,
                p=float(wilcoxon(b, a).pvalue) if seeds > 5 else float("nan"))


def report(label, res, extra=""):
    print(f"  {label:<26}{res['adam_progress']*100:>7.1f}%"
          f"{res['soft_progress']*100:>8.1f}%{res['ratio']:>8.2f}"
          f"{res['wins']:>6}/{res['seeds']}{res['p']:>10.1e}  {extra}",
          flush=True)


def main(quick, out):
    rows = []
    seeds = 10 if quick else 20
    steps = 150 if quick else 300
    print("Progress toward the ground state, from a random start.")
    print("'ratio' is SoftOpt's progress over Adam's -- the quantity claimed.\n")

    print(f"  {'check':<26}{'Adam':>7}{'SoftOpt':>8}{'ratio':>8}{'wins':>8}{'p':>10}")
    print("  " + "-" * 70)

    # 1. fresh seeds: the same setup, a different draw of noise and starts
    for off in ([0, 5000] if quick else [0, 5000, 10000]):
        s = System(6, 8)
        r = progress_ratio(s, seeds, steps, seed_offset=off)
        report(f"seeds, offset {off}", r)
        rows.append(dict(check="seeds", offset=off, qubits=6, **r))

    # 2. shot budget: the plateau is defined against it
    for shots in ([1024, 4096, 16384] if quick else [512, 1024, 4096, 16384, 65536]):
        s = System(6, 8, shots=shots)
        r = progress_ratio(s, seeds, steps)
        report(f"shots = {shots}", r, f"depth {s.depth():.1f}")
        rows.append(dict(check="shots", shots=shots, depth=s.depth(), **r))

    # 3. SPSA probe size: a result that needs one value of c is tuning
    for c in ([0.05, 0.1, 0.2] if quick else [0.02, 0.05, 0.1, 0.2, 0.4]):
        s = System(6, 8)
        r = progress_ratio(s, seeds, steps, c=c)
        report(f"probe c = {c}", r)
        rows.append(dict(check="probe", c=c, **r))

    # 4. a different observable
    for name in (["ising", "heisenberg"] if quick
                 else ["ising", "heisenberg", "random"]):
        s = System(6, 8, terms_fn=HAMILTONIANS[name])
        r = progress_ratio(s, seeds, steps)
        report(f"hamiltonian: {name}", r, f"depth {s.depth():.1f}")
        rows.append(dict(check="hamiltonian", name=name, **r))

    json.dump(rows, open(out, "w"), indent=2)
    ratios = np.array([r["ratio"] for r in rows])
    print(f"\n  ratio across all {len(rows)} conditions: "
          f"median {np.median(ratios):.2f}, "
          f"range [{ratios.min():.2f}, {ratios.max():.2f}], "
          f"sd {ratios.std():.3f}")
    print(f"  wrote {out}")
    print("\n  The claim survives if the ratio stays put. A ratio that moves")
    print("  with shots is still informative -- it would say the advantage")
    print("  tracks how bad the measurement is, which is the mechanism.")
    print("  A ratio that moves with the probe size or the Hamiltonian would")
    print("  mean the first scan measured a tuning artefact.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="plateau_replicate.json")
    a = ap.parse_args()
    t0 = time.perf_counter()
    main(a.quick, a.out)
    print(f"  {(time.perf_counter()-t0)/60:.0f} minutes")
