"""
Penrose quasicrystal XY-model energy, curvature via genuine Klein-Maimon
Lemma 6.1 soft-number propagation (same methodology as lemma61_vqe.py),
now for a classical (non-quantum-circuit) domain with up to ~50 angle
parameters -- directly testing whether the dimensionality/convexity
hypothesis (VQE/QAOA succeed because they're few-parameter and smooth;
RL fails because it's high-dimensional and indefinite) extends to a
genuinely different physical system in the "middle" parameter-count range.

E(theta) = -(1/|edges|) * sum_{(i,j) in edges} cos(theta_i - theta_j)
"""
import numpy as np

def smul(x, y):
    a1,b1 = x; a2,b2 = y
    return (a1*b2 + a2*b1, b1*b2)
def sadd(x, y):
    return (x[0]+y[0], x[1]+y[1])
def ssub(x, y):
    return (x[0]-y[0], x[1]-y[1])
def scos(x):
    a,b = x
    return (-a*np.sin(b), np.cos(b))

def energy_exact(theta, edges):
    if not edges:
        return 0.0
    return -sum(np.cos(theta[i]-theta[j]) for i,j in edges) / len(edges)

def g_delta(theta0, delta, edges):
    """Exact directional derivative of the XY energy along delta at theta0,
    via ONE application of Klein-Maimon's exact single-axis soft-number
    calculus (Lemma 6.1) -- no jet, no extension, no finite differencing
    for this part."""
    n = len(edges)
    if n == 0:
        return 0.0
    E = (0.0, 0.0)
    for i, j in edges:
        ti = (delta[i], theta0[i])
        tj = (delta[j], theta0[j])
        diff = ssub(ti, tj)
        c = scos(diff)
        E = sadd(E, c)
    E = (-E[0]/n, -E[1]/n)
    return E[0]  # potential component = g_delta(theta0)

def D1_D2(theta0, delta, edges, h=1e-4):
    """D1: exact (Lemma 6.1). D2: ordinary finite difference of the exact,
    noiseless g_delta function -- honestly not itself claimed as soft."""
    d1 = g_delta(theta0, delta, edges)
    gp = g_delta(theta0 + h*delta, delta, edges)
    gm = g_delta(theta0 - h*delta, delta, edges)
    d2 = (gp - gm) / (2*h)
    return d1, d2

if __name__ == "__main__":
    # quick self-check vs finite differences on the exact energy
    rng = np.random.default_rng(0)
    N = 12
    edges = [(i, (i+1) % N) for i in range(N)] + [(i, (i+3) % N) for i in range(N)]
    theta0 = rng.uniform(-np.pi, np.pi, N)
    delta = rng.choice([-1.0, 1.0], size=N)

    d1, d2 = D1_D2(theta0, delta, edges)
    h = 1e-6
    ep = energy_exact(theta0 + h*delta, edges)
    em = energy_exact(theta0 - h*delta, edges)
    d1_fd = (ep - em) / (2*h)
    print(f"D1 (soft, exact)      = {d1:.8f}")
    print(f"D1 (finite-diff check)= {d1_fd:.8f}")
    print(f"D1 diff = {abs(d1-d1_fd):.2e}")
    print(f"D2 (FD on exact g_delta) = {d2:.6f}")
