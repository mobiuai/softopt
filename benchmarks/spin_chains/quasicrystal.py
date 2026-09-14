#!/usr/bin/env python3
"""
================================================================================
Penrose Quasicrystal XY Model -- SoftOpt vs real torch.optim.Adam
================================================================================
Matches quasicrystal.py's exact protocol: de Bruijn pentagrid Penrose P3
tiling, 50 vertices, 150 steps, 10 seeds, SPSA gradient estimation,
identical init/deltas per seed for fairness. Uses the already-validated
quasicrystal_lemma61.py engine (one of the nine original domains) --
only change: real torch.optim.Adam (not the old scalar Mobiu controller)
as the baseline, SoftOpt (standalone) as the test arm.

USAGE:
    pip install numpy scipy torch
    python3 script16_quasicrystal_vs_real_adam.py
================================================================================
"""
import numpy as np, sys, torch
from scipy.stats import wilcoxon
sys.path.insert(0, '.')
from quasicrystal_lemma61 import energy_exact, g_delta
from softopt import SoftOpt

NUM_SEEDS = 10
NUM_STEPS = 150
LR = 0.02
SPSA_C = 0.1
MAX_VERTICES = 50


def generate_penrose_graph(max_n=3, max_verts=MAX_VERTICES):
    angles = np.array([2 * np.pi * k / 5 for k in range(5)])
    nx, ny = np.cos(angles), np.sin(angles)
    gamma = np.zeros(5)
    raw = []
    for k1 in range(5):
        for k2 in range(k1 + 1, 5):
            A = np.array([[nx[k1], ny[k1]], [nx[k2], ny[k2]]])
            det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
            if abs(det) < 1e-8: continue
            Ainv = np.array([[A[1,1], -A[0,1]], [-A[1,0], A[0,0]]]) / det
            for j1 in range(-max_n, max_n + 1):
                for j2 in range(-max_n, max_n + 1):
                    rhs = np.array([j1 + gamma[k1], j2 + gamma[k2]])
                    raw.append(Ainv @ rhs)
    unique = {}
    for v in raw:
        key = (round(v[0], 4), round(v[1], 4))
        unique[key] = np.array([float(key[0]), float(key[1])])
    verts = np.array(list(unique.values()))
    dists = np.linalg.norm(verts, axis=1)
    idx = np.argsort(dists)[:max_verts]
    verts = verts[idx]
    sample_d = []
    for i in range(min(30, len(verts))):
        for j in range(i + 1, min(30, len(verts))):
            d = np.linalg.norm(verts[i] - verts[j])
            if d > 0.05: sample_d.append(d)
    sample_d.sort()
    d_short = sample_d[0]
    phi = (1 + np.sqrt(5)) / 2
    d_long = d_short * phi
    threshold = d_long * 1.05
    edges = [(i, j) for i in range(len(verts)) for j in range(i + 1, len(verts))
             if np.linalg.norm(verts[i] - verts[j]) <= threshold]
    return verts, edges


def spsa_grad_and_energy(params, edges, delta, c=SPSA_C):
    e_plus = energy_exact(params + c * delta, edges)
    e_minus = energy_exact(params - c * delta, edges)
    grad = (e_plus - e_minus) / (2 * c) * delta
    energy = energy_exact(params, edges)
    return grad, energy


def run_adam(init_params, edges, spsa_deltas):
    params = init_params.copy()
    theta_torch = torch.tensor(params.copy(), requires_grad=True)
    opt_torch = torch.optim.Adam([theta_torch], lr=LR)
    best = np.inf
    for step in range(NUM_STEPS):
        grad, energy = spsa_grad_and_energy(params, edges, spsa_deltas[step])
        best = min(best, energy)
        opt_torch.zero_grad()
        theta_torch.grad = torch.tensor(grad, dtype=theta_torch.dtype)
        opt_torch.step()
        params = theta_torch.detach().numpy().copy()
    return best


def run_softopt(init_params, edges, spsa_deltas, seed):
    params = init_params.copy()
    N = len(init_params)
    opt = SoftOpt(N, lambda th, d: g_delta(th, d, edges), lr=LR, seed=seed + 999)
    best = np.inf
    for step in range(NUM_STEPS):
        grad, energy = spsa_grad_and_energy(params, edges, spsa_deltas[step])
        best = min(best, energy)
        params = opt.step(params, grad)
    return best


def main():
    print("Generating Penrose P3 tiling (de Bruijn pentagrid)...")
    verts, edges = generate_penrose_graph(max_n=3, max_verts=MAX_VERTICES)
    N = len(verts)
    print(f"Vertices: {N}  |  Edges: {len(edges)}")

    print(f"\n{'='*72}")
    print(f"PENROSE QUASICRYSTAL XY MODEL -- SoftOpt vs real torch.optim.Adam")
    print(f"Sites: {N}  |  Bonds: {len(edges)}  |  Steps: {NUM_STEPS}  |  Seeds: {NUM_SEEDS}")
    print(f"{'='*72}\n")

    adam_best_list, soft_best_list = [], []
    for seed in range(NUM_SEEDS):
        np.random.seed(seed * 1000)
        init_params = np.random.uniform(-np.pi, np.pi, N)
        spsa_deltas = [np.random.choice([-1.0, 1.0], size=N) for _ in range(NUM_STEPS)]

        print(f"Seed {seed+1:2d}/{NUM_SEEDS} | ", end="", flush=True)
        adam_best = run_adam(init_params, edges, spsa_deltas)
        print(f"Adam: {adam_best:+.4f} | ", end="", flush=True)
        soft_best = run_softopt(init_params, edges, spsa_deltas, seed)
        print(f"SoftOpt: {soft_best:+.4f}", end="")

        adam_best_list.append(adam_best); soft_best_list.append(soft_best)
        winner = "SoftOpt" if soft_best < adam_best else "Adam"
        print(f"  -> {winner}")

    adam_arr = np.array(adam_best_list); soft_arr = np.array(soft_best_list)
    wins = int(np.sum(soft_arr < adam_arr))
    improvement = (adam_arr.mean() - soft_arr.mean()) / (abs(adam_arr.mean()) + 1e-10) * 100
    _, p_val = wilcoxon(adam_arr, soft_arr)

    print(f"\n{'='*72}")
    print("FINAL RESULTS")
    print(f"Adam    best energy: {adam_arr.mean():+.4f} +/- {adam_arr.std():.4f}")
    print(f"SoftOpt best energy: {soft_arr.mean():+.4f} +/- {soft_arr.std():.4f}")
    print(f"Improvement:         {improvement:+.1f}%")
    print(f"Win rate:            {wins}/{NUM_SEEDS}")
    print(f"p-value (Wilcoxon):  {p_val:.6f}")
    print(f"{'='*72}")

if __name__ == "__main__":
    main()
