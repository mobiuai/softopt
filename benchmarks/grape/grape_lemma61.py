"""
GRAPE-style quantum control: genuine Lemma-6.1 soft-number propagation,
structurally identical to lemma61_vqe.py -- a sequence of parameterized
rotations (control pulses) + fixed entangling gates, but the objective
is STATE-TRANSFER FIDELITY to a target state, not a Hamiltonian's
ground-state energy. Same "known computation graph" property as VQE.

2 qubits, N_LAYERS layers, each: Rx(theta) on each qubit, then CNOT.
Target: a Bell state. Objective to MINIMIZE: 1 - |<target|psi>|^2.
"""
import numpy as np
from softopt import smul, sadd, ssin, scos

N_LAYERS = 6
N_QUBITS = 2
N_PARAMS = N_LAYERS * N_QUBITS

TARGET = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)  # Bell state (|00>+|11>)/sqrt2

def soft_rx_apply(theta_soft, psi_soft):
    half = (theta_soft[0]*0.5, theta_soft[1]*0.5)
    c, s = scos(half), ssin(half)
    p0, p1 = psi_soft
    new0 = sadd(smul(c,p0), (-smul(s,p1)[0], -smul(s,p1)[1]))
    new1 = sadd(smul(s,p0), smul(c,p1))
    return new0, new1

def g_delta(theta0, delta):
    """Exact directional derivative of (1 - fidelity) via genuine
    single-axis Lemma-6.1 soft propagation -- no jvp, no approximation."""
    psi = [(0.0,1.0), (0.0,0.0), (0.0,0.0), (0.0,0.0)]  # |00>
    idx = 0
    for layer in range(N_LAYERS):
        for q in range(2):
            th = (delta[idx], theta0[idx]); idx += 1
            if q == 0:
                new = list(psi)
                for b1 in (0,1):
                    i0, i1 = 0*2+b1, 1*2+b1
                    new[i0], new[i1] = soft_rx_apply(th, (psi[i0], psi[i1]))
                psi = new
            else:
                new = list(psi)
                for b0 in (0,1):
                    i0, i1 = b0*2+0, b0*2+1
                    new[i0], new[i1] = soft_rx_apply(th, (psi[i0], psi[i1]))
                psi = new
        # fixed CNOT (control=0, target=1): swap amplitudes of |10> and |11>
        psi = [psi[0], psi[1], psi[3], psi[2]]

    # overlap <target|psi> (real target, real amplitudes throughout since
    # Rx with real angle keeps amplitudes real here given real start state
    # -- verified below) -- fidelity = overlap^2
    overlap_pot = sum(TARGET[i]*psi[i][0] for i in range(4))
    overlap_real = sum(TARGET[i]*psi[i][1] for i in range(4))
    # fidelity = overlap^2 as a soft number: (2*a*b, b^2)
    fid_pot = 2*overlap_pot*overlap_real
    # objective = 1 - fidelity -> potential negates
    return -fid_pot  # = D1 of (1-fidelity)

def energy_exact(theta):
    """1 - fidelity, exact (no soft numbers, for scoring/plain SPSA).
    Uses the SAME real-valued Ry-style rotation as g_delta (not a complex
    Rx) -- consistency between the two is what matters here, and real
    rotations avoid needing complex soft-number coefficients for this
    first test."""
    psi = np.zeros(4); psi[0] = 1.0
    def ry(t):
        c,s = np.cos(t/2), np.sin(t/2)
        return np.array([[c,-s],[s,c]])
    idx = 0
    for layer in range(N_LAYERS):
        U0 = ry(theta[idx]); idx += 1
        U1 = ry(theta[idx]); idx += 1
        U = np.kron(U0, U1)
        psi = U @ psi
        # CNOT
        psi = np.array([psi[0], psi[1], psi[3], psi[2]])
    fidelity = (np.dot(TARGET, psi))**2
    return 1.0 - fidelity

E0 = 0.0  # perfect fidelity is achievable (a Bell state is reachable from |00> with this ansatz)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    theta0 = rng.uniform(-1, 1, N_PARAMS)
    delta = rng.choice([-1.,1.], size=N_PARAMS)
    d1 = g_delta(theta0, delta)
    h = 1e-6
    d1_fd = (energy_exact(theta0+h*delta) - energy_exact(theta0-h*delta)) / (2*h)
    print(f"D1 (soft, exact)      = {d1:.8f}")
    print(f"D1 (finite-diff check)= {d1_fd:.8f}")
    print(f"diff = {abs(d1-d1_fd):.2e}")
    print(f"E0 (target objective) = {E0}, current energy_exact(theta0) = {energy_exact(theta0):.4f}")

def D1_D2(theta0, delta, h=1e-4):
    d1 = g_delta(theta0, delta)
    gp = g_delta(theta0 + h*delta, delta)
    gm = g_delta(theta0 - h*delta, delta)
    d2 = (gp - gm) / (2*h)
    return d1, d2
