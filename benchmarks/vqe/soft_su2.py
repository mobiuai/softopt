"""
Lemma 6.1 soft-number propagation through Qiskit's EfficientSU2 (linear),
vectorised: the same mathematics as benchmarks/efficient_su2_lemma61.py
(every amplitude is a complex soft number (a, b) = (potential, real); gates
act by the soft product smul((a1,b1),(a2,b2)) = (a1*b2 + a2*b1, b1*b2)),
but the state is held as two numpy arrays instead of a Python list of
pairs, and <psi|H|psi> uses the Hamiltonian as a sparse matrix instead of a
loop over Pauli strings. Checked against the original function in
test_soft_su2() -- identical to rounding.
"""
import numpy as np


def smul(x, y):
    a1, b1 = x
    a2, b2 = y
    return (a1 * b2 + a2 * b1, b1 * b2)


def _gate(pair, U_soft, q, n):
    """Apply a soft 2x2 gate U_soft = ((a00,a01,a10,a11),(b00,b01,b10,b11)) to qubit q."""
    a, b = pair
    ax = n - 1 - q
    A = np.moveaxis(a.reshape([2] * n), ax, 0)
    B = np.moveaxis(b.reshape([2] * n), ax, 0)
    (u00a, u01a, u10a, u11a), (u00b, u01b, u10b, u11b) = U_soft
    n0 = smul((u00a, u00b), (A[0], B[0])); m0 = smul((u01a, u01b), (A[1], B[1]))
    n1 = smul((u10a, u10b), (A[0], B[0])); m1 = smul((u11a, u11b), (A[1], B[1]))
    A2 = np.stack([n0[0] + m0[0], n1[0] + m1[0]]); B2 = np.stack([n0[1] + m0[1], n1[1] + m1[1]])
    return (np.moveaxis(A2, 0, ax).reshape(-1), np.moveaxis(B2, 0, ax).reshape(-1))


def _ry_soft(th):
    a, b = th
    c = (-0.5 * a * np.sin(b / 2), np.cos(b / 2))
    s = (0.5 * a * np.cos(b / 2), np.sin(b / 2))
    return ((c[0], -s[0], s[0], c[0]), (c[1], -s[1], s[1], c[1]))


def _rz_soft(th):
    a, b = th
    v0, v1 = np.exp(-0.5j * b), np.exp(0.5j * b)
    return ((a * (-0.5j) * v0, 0.0, 0.0, a * (0.5j) * v1), (v0, 0.0, 0.0, v1))


def _cnot(pair, c, t, n):
    out = []
    for v in pair:
        V = np.moveaxis(v.reshape([2] * n), [n - 1 - c, n - 1 - t], [0, 1]).copy()
        V[1, 0], V[1, 1] = V[1, 1].copy(), V[1, 0].copy()
        out.append(np.moveaxis(V, [0, 1], [n - 1 - c, n - 1 - t]).reshape(-1))
    return tuple(out)


def soft_state(theta, delta, n, reps):
    dim = 2 ** n
    a = np.zeros(dim, complex); b = np.zeros(dim, complex); b[0] = 1.0
    pair, k = (a, b), 0
    for layer in range(reps + 1):
        for q in range(n):
            pair = _gate(pair, _ry_soft((delta[k], theta[k])), q, n); k += 1
        for q in range(n):
            pair = _gate(pair, _rz_soft((delta[k], theta[k])), q, n); k += 1
        if layer < reps:
            for q in range(n - 1):
                pair = _cnot(pair, q, q + 1, n)
    return pair


def g_delta_su2_fast(theta, delta, n, reps, Hs):
    """Soft <psi|H|psi>; returns its potential part = exact directional derivative."""
    a, b = soft_state(theta, delta, n, reps)
    Ha, Hb = Hs @ a, Hs @ b
    pot, _ = smul((np.conj(a), np.conj(b)), (Ha, Hb))
    return float(np.real(np.sum(pot)))


def energy(theta, n, reps, Hs):
    _, b = soft_state(theta, np.zeros_like(theta), n, reps)
    return float(np.real(np.vdot(b, Hs @ b)))
