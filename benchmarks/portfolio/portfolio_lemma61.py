"""
Portfolio optimization (Markowitz mean-variance) -- a THIRD genuinely
different domain (finance, not physics/chemistry/vision). Known model:
fixed expected returns mu and covariance Sigma. Weights w=softmax(theta)
(unconstrained params -> valid portfolio automatically). Objective:
minimize -(w.mu - lambda/2 * w'Sigma w) [maximize mean-variance utility].

Genuine Lemma-6.1 soft-number propagation, now including exp (their own
elementary-function rule: exp((a,b)) = (a*exp(b), exp(b))) plus the
already-used soft division for softmax normalization.
"""
import numpy as np
from softopt import smul, sadd, sexp, sinv, sdiv

N_ASSETS = 6
LAMBDA_RISK = 3.0

rng0 = np.random.default_rng(11)
MU = rng0.uniform(0.02, 0.15, N_ASSETS)
A = rng0.normal(0, 0.2, size=(N_ASSETS, N_ASSETS))
SIGMA = A @ A.T + 0.05*np.eye(N_ASSETS)  # guaranteed PSD covariance

def softmax_exact(theta):
    e = np.exp(theta - theta.max())
    return e / e.sum()

def energy_exact(theta):
    w = softmax_exact(theta)
    utility = w @ MU - 0.5*LAMBDA_RISK * (w @ SIGMA @ w)
    return -utility  # minimize negative utility

E0 = None  # unknown true optimum in closed form for softmax-constrained Markowitz; not needed for gap-based comparison here (we compare RAW objective values)

def g_delta(theta0, delta):
    n = N_ASSETS
    # Softmax is shift-invariant, so subtract the max before exponentiating.
    # Without this, a large common offset in theta overflows exp() and the
    # whole soft derivative comes back NaN even though the model is unchanged.
    shift = float(np.max(theta0))
    th = [(delta[i], theta0[i] - shift) for i in range(n)]
    exps = [sexp(t) for t in th]
    Z = (0.0, 0.0)
    for e in exps:
        Z = sadd(Z, e)
    w = [sdiv(e, Z) for e in exps]

    # w . mu
    wmu = (0.0, 0.0)
    for i in range(n):
        wmu = sadd(wmu, smul(w[i], (0.0, MU[i])))

    # w' Sigma w
    wSw = (0.0, 0.0)
    for i in range(n):
        Swi = (0.0, 0.0)
        for j in range(n):
            Swi = sadd(Swi, smul(w[j], (0.0, SIGMA[i,j])))
        wSw = sadd(wSw, smul(w[i], Swi))

    utility_pot = wmu[0] - 0.5*LAMBDA_RISK*wSw[0]
    return -utility_pot  # D1 of the negative utility

def D1_D2(theta0, delta, h=1e-4):
    d1 = g_delta(theta0, delta)
    gp = g_delta(theta0 + h*delta, delta)
    gm = g_delta(theta0 - h*delta, delta)
    d2 = (gp - gm) / (2*h)
    return d1, d2

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    theta0 = rng.normal(0, 0.5, N_ASSETS)
    delta = rng.choice([-1.,1.], size=N_ASSETS)
    d1 = g_delta(theta0, delta)
    h = 1e-6
    d1_fd = (energy_exact(theta0+h*delta) - energy_exact(theta0-h*delta)) / (2*h)
    print(f"D1 (soft, exact)      = {d1:.8f}")
    print(f"D1 (finite-diff check)= {d1_fd:.8f}")
    print(f"diff = {abs(d1-d1_fd):.2e}")
    print(f"energy at theta0 = {energy_exact(theta0):.6f}")
