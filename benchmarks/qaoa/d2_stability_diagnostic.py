#!/usr/bin/env python3
"""
================================================================================
DIAGNOSTIC: QAOA D2 (curvature) sign-stability under REAL FakeFez noise
================================================================================
The same diagnostic used for RL earlier today (check whether D2's sign
is consistent across random probe directions -- VQE-like, one-signed
and easy for raw Newton division -- or indefinite -- RL-like, needing
bounded/Mobius treatment instead) -- now applied to QAOA's exact
analytic model (lemma61_qaoa.py, the same engine SoftOpt's correction
uses in script7), to see whether QAOA's curvature is inherently
RL-like even in the NOISELESS exact model, independent of FakeFez.

This is cheap (no real Qiskit calls) and should run in seconds --
if D2 is already sign-indefinite in the exact model, that alone would
explain script7's failure without needing to probe real hardware
noise at all. If D2 is one-signed here (like the original all-day
finding), the failure must come from something specific to how real
shot noise interacts with the correction, which would need a second,
more expensive diagnostic using real FakeFez measurements directly
(sketched at the bottom, not run here) to pin down.

USAGE:
    pip install numpy
    python3 script14_qaoa_d2_stability_diagnostic.py
================================================================================
"""
import numpy as np, sys
sys.path.insert(0, '.')
from lemma61_qaoa import LemmaQAOA

EDGES = [(0,1),(1,2),(2,3),(3,0),(0,2)]
mc = LemmaQAOA(EDGES, 4, reps=2)
N_PARAMS = mc.n_params

def D1_D2(theta, delta, h=1e-4):
    d1 = mc.g_delta(theta, delta)
    gp = mc.g_delta(theta + h*delta, delta)
    gm = mc.g_delta(theta - h*delta, delta)
    d2 = (gp - gm) / (2*h)
    return d1, d2

def main():
    print("=" * 70)
    print("QAOA D2 sign-stability, exact analytic model (no real Qiskit calls)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    n_test_points = 8
    n_directions = 10

    flip_counts = []
    for pt in range(n_test_points):
        theta0 = rng.uniform(-0.3, 0.3, N_PARAMS)
        signs = []
        for _ in range(n_directions):
            delta = rng.choice([-1., 1.], size=N_PARAMS)
            _, d2 = D1_D2(theta0, delta)
            signs.append(np.sign(d2))
        n_pos = sum(1 for s in signs if s > 0)
        n_neg = sum(1 for s in signs if s < 0)
        flip_counts.append(min(n_pos, n_neg))
        print(f"  point {pt}: {n_pos} positive / {n_neg} negative D2 (out of {n_directions} directions)")

    print()
    total_flips = sum(flip_counts)
    print(f"Total minority-sign count across all points/directions: {total_flips}/{n_test_points*n_directions}")
    if total_flips == 0:
        print("-> D2 is FULLY one-signed everywhere tested -- VQE-like in the exact")
        print("   model. If script7 still fails on real hardware, the cause is")
        print("   specific to how real FakeFez shot noise interacts with the")
        print("   correction (e.g. SPSA gradient noise swamping a small, correct")
        print("   D1/D2 signal at this landscape's scale) -- NOT an inherent")
        print("   sign-indefinite-curvature problem like RL's.")
    else:
        print("-> D2 shows genuine sign flips -- RL-like indefinite curvature,")
        print("   even in the noiseless exact model. This alone could explain")
        print("   why raw Newton division underperforms here, independent of")
        print("   any real hardware noise.")

if __name__ == "__main__":
    main()
