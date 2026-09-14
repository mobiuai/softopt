"""
Fibonacci-chain antiferromagnetic XY model, genuine Lemma-6.1 soft-number
propagation -- same methodology as quasicrystal_lemma61.py, weighted
bonds (J_S=1.0 short, J_L=1/phi long) from the Fibonacci substitution
L->LS, S->L. AF sign convention (minimize +sum J*cos, matching the
reference fibonacci_benchmark.py's energy_af): frustration matters.
"""
import numpy as np

PHI = (1 + np.sqrt(5)) / 2

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

def fibonacci_bonds(n):
    word = "L"
    while len(word) < n - 1:
        word = "".join("LS" if c == "L" else "L" for c in word)
    word = word[:n - 1]
    bonds = []
    for i, c in enumerate(word):
        J = 1.0 if c == "S" else 1.0 / PHI
        bonds.append((i, i + 1, J))
    return bonds

def energy_exact(theta, bonds):
    return sum(J * np.cos(theta[i] - theta[j]) for i, j, J in bonds)

def g_delta(theta0, delta, bonds):
    """Exact directional derivative of the AF energy along delta, via ONE
    application of Lemma 6.1 (no jet, no finite-differencing here)."""
    E = (0.0, 0.0)
    for i, j, J in bonds:
        ti = (delta[i], theta0[i])
        tj = (delta[j], theta0[j])
        diff = ssub(ti, tj)
        c = scos(diff)
        E = sadd(E, (J*c[0], J*c[1]))
    return E[0]

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 16
    bonds = fibonacci_bonds(N)
    theta0 = rng.uniform(-np.pi, np.pi, N)
    delta = rng.choice([-1.0, 1.0], size=N)

    d1 = g_delta(theta0, delta, bonds)
    h = 1e-6
    ep = energy_exact(theta0 + h*delta, bonds)
    em = energy_exact(theta0 - h*delta, bonds)
    d1_fd = (ep - em) / (2*h)
    print(f"D1 (soft, exact)      = {d1:.8f}")
    print(f"D1 (finite-diff check)= {d1_fd:.8f}")
    print(f"D1 diff = {abs(d1-d1_fd):.2e}")
