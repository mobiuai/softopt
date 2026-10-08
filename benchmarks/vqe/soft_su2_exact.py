"""
Exact second order for the EfficientSU2 soft statevector (soft_su2.py): the same Lemma 6.1
propagation, but every amplitude is a NESTED soft number -- a soft number whose two parts are
themselves soft numbers:

    x = ((e, a), (c, v))   =   v + c*eps2 + a*eps1 + e*eps1*eps2,     eps1^2 = eps2^2 = 0

Each parameter enters as theta + delta*eps1 + delta*eps2, i.e. ((0, delta), (delta, theta)).
Products are the book's soft product (Section 3.3.2) applied at both levels; functions follow
Lemma 6.1 applied twice. The energy <psi|H|psi> then reads
    v = E,  a = c = D1 (the directional derivative),  e = D2 (the exact directional curvature)
from ONE pass, with no step size. soft_su2.g_delta_su2_fast gives the same D1 (checked in
test_exact()); D2 replaces the finite difference (g(x+h d) - g(x-h d)) / 2h that SoftOpt
otherwise takes.
"""
import numpy as np


# ---- the soft product, applied recursively: leaves are numbers / numpy arrays
def sadd(x, y):
    if isinstance(x, tuple):
        return (sadd(x[0], y[0]), sadd(x[1], y[1]))
    return x + y


def smul(x, y):
    """(a1, b1) * (a2, b2) = (a1*b2 + a2*b1, b1*b2) -- Section 3.3.2; recursive for nesting."""
    if isinstance(x, tuple):
        return (sadd(smul(x[0], y[1]), smul(y[0], x[1])), smul(x[1], y[1]))
    return x * y


def tmap(f, x):
    return (tmap(f, x[0]), tmap(f, x[1])) if isinstance(x, tuple) else f(x)


def lift2(z):
    """A constant as a nested soft number: ((0, 0), (0, z))."""
    return ((0.0 * z, 0.0 * z), (0.0 * z, z))


def f2(f, df, ddf, theta, delta):
    """Lemma 6.1 twice: f(theta + d eps1 + d eps2) = ((d^2 f'', d f'), (d f', f))."""
    return ((delta * delta * ddf(theta), delta * df(theta)), (delta * df(theta), f(theta)))


def _gate2(state, U, q, n):
    ax = n - 1 - q
    S = tmap(lambda v: np.moveaxis(v.reshape([2] * n), ax, 0), state)
    S0, S1 = tmap(lambda v: v[0], S), tmap(lambda v: v[1], S)
    u00, u01, u10, u11 = U
    n0 = sadd(smul(u00, S0), smul(u01, S1))
    n1 = sadd(smul(u10, S0), smul(u11, S1))
    return tmap(lambda pair: np.moveaxis(np.stack(pair), 0, ax).reshape(-1), _zip(n0, n1))


def _zip(x, y):
    return (_zip(x[0], y[0]), _zip(x[1], y[1])) if isinstance(x, tuple) else [x, y]   # list = leaf pair


def _ry2(theta, delta):
    c = f2(lambda t: np.cos(t / 2), lambda t: -0.5 * np.sin(t / 2), lambda t: -0.25 * np.cos(t / 2), theta, delta)
    s = f2(lambda t: np.sin(t / 2), lambda t: 0.5 * np.cos(t / 2), lambda t: -0.25 * np.sin(t / 2), theta, delta)
    return (c, tmap(lambda v: -v, s), s, c)


def _rz2(theta, delta):
    v0 = f2(lambda t: np.exp(-0.5j * t), lambda t: -0.5j * np.exp(-0.5j * t), lambda t: -0.25 * np.exp(-0.5j * t), theta, delta)
    v1 = f2(lambda t: np.exp(0.5j * t), lambda t: 0.5j * np.exp(0.5j * t), lambda t: -0.25 * np.exp(0.5j * t), theta, delta)
    z = lift2(0.0 + 0j)
    return (v0, z, z, v1)


def _cnot2(state, c, t, n):
    def sw(v):
        V = np.moveaxis(v.reshape([2] * n), [n - 1 - c, n - 1 - t], [0, 1]).copy()
        V[1, 0], V[1, 1] = V[1, 1].copy(), V[1, 0].copy()
        return np.moveaxis(V, [0, 1], [n - 1 - c, n - 1 - t]).reshape(-1)
    return tmap(sw, state)


def soft_state2(theta, delta, n, reps):
    dim = 2 ** n
    v = np.zeros(dim, complex); v[0] = 1.0
    state, k = lift2(v), 0
    for layer in range(reps + 1):
        for q in range(n):
            state = _gate2(state, _ry2(theta[k], delta[k]), q, n); k += 1
        for q in range(n):
            state = _gate2(state, _rz2(theta[k], delta[k]), q, n); k += 1
        if layer < reps:
            for q in range(n - 1):
                state = _cnot2(state, q, q + 1, n)
    return state


def d1_d2_su2_fast(theta, delta, n, reps, Hs):
    """(D1, D2) of <psi|H|psi> along delta, exactly, from one nested-soft pass."""
    psi = soft_state2(theta, delta, n, reps)
    Hpsi = tmap(lambda v: Hs @ v, psi)
    E = smul(tmap(np.conj, psi), Hpsi)
    (e, a), (c, v) = tmap(lambda x: float(np.real(np.sum(x))), E)
    return a, e


def test_exact(n=4, reps=2, trials=5):
    from soft_su2 import g_delta_su2_fast
    rng = np.random.default_rng(0)
    H = rng.normal(size=(2 ** n, 2 ** n)) + 1j * rng.normal(size=(2 ** n, 2 ** n)); H = H + H.conj().T
    worst = (0.0, 0.0)
    for _ in range(trials):
        npar = 2 * n * (reps + 1)
        th, d = rng.uniform(-1, 1, npar), rng.choice([-1.0, 1.0], npar)
        D1, D2 = d1_d2_su2_fast(th, d, n, reps, H)
        g = lambda x: g_delta_su2_fast(x, d, n, reps, H)                    # noqa: E731
        h = 1e-4
        D2fd = (g(th + h * d) - g(th - h * d)) / (2 * h)
        worst = (max(worst[0], abs(D1 - g(th)) / max(1, abs(D1))), max(worst[1], abs(D2 - D2fd) / max(1, abs(D2))))
    return worst


if __name__ == '__main__':
    e1, e2 = test_exact()
    print(f'D1 vs soft_su2 (exact first order): max rel diff {e1:.1e}')
    print(f'D2 vs finite difference of soft_su2 D1 (h=1e-4): max rel diff {e2:.1e}')
