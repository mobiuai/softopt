"""
Genuine Lemma-6.1 soft-number propagation through the EXACT QAOA-MIS
circuit family from test_fakefez_qaoa_mis_new.py: per layer,
RZ(gamma) on each qubit, RZZ(penalty*gamma/2) on each edge (Qiskit's
NATIVE ZZ-rotation gate, verified phase convention exp(-i*theta/2) for
matching bits / exp(+i*theta/2) for differing bits), RX(2*beta) on
each qubit. Complex soft numbers (potential, real), matching the
approach used for efficient_su2_lemma61.py and lemma61_qaoa.py.
"""
import numpy as np

def smul(x, y):
    a1,b1=x; a2,b2=y
    return (a1*b2 + a2*b1, b1*b2)
def sadd(x, y):
    return (x[0]+y[0], x[1]+y[1])

def s_phase(theta_soft, sign):
    """exp(sign*i*theta/2) as a soft number (Lemma 6.1's general rule)."""
    a, b = theta_soft
    val = np.exp(sign * 1j * b / 2)
    deriv = sign * 1j / 2 * val
    return (a*deriv, val)

def s_rx(theta_soft):
    """Soft 2x2 RX matrix entries: [[c,-i*s],[-i*s,c]]."""
    a, b = theta_soft
    half_a, half_b = a*0.5, b*0.5
    c = (-half_a*np.sin(half_b), np.cos(half_b))
    s = (half_a*np.cos(half_b), np.sin(half_b))
    nis = (-1j*s[0], -1j*s[1])  # -i*s
    return c, nis

def apply_rz_soft(psi_pot, psi_real, q, n, theta_soft):
    ph0 = s_phase(theta_soft, -1.0)
    ph1 = s_phase(theta_soft, +1.0)
    psi_pot = psi_pot.reshape([2]*n); psi_real = psi_real.reshape([2]*n)
    axis = n - 1 - q
    psi_pot = np.moveaxis(psi_pot, axis, 0); psi_real = np.moveaxis(psi_real, axis, 0)
    shape = psi_pot.shape
    p0 = (psi_pot[0].reshape(-1), psi_real[0].reshape(-1))
    p1 = (psi_pot[1].reshape(-1), psi_real[1].reshape(-1))
    new0 = smul(ph0, p0); new1 = smul(ph1, p1)
    out_pot = np.stack([new0[0].reshape(shape[1:]), new1[0].reshape(shape[1:])], axis=0)
    out_real = np.stack([new0[1].reshape(shape[1:]), new1[1].reshape(shape[1:])], axis=0)
    out_pot = np.moveaxis(out_pot, 0, axis).reshape(-1); out_real = np.moveaxis(out_real, 0, axis).reshape(-1)
    return out_pot, out_real

def apply_rzz_soft(psi_pot, psi_real, qi, qj, n, theta_soft):
    dim = 2**n
    idxs = np.arange(dim)
    bi = (idxs >> qi) & 1; bj = (idxs >> qj) & 1
    same = (bi == bj)
    ph_same = s_phase(theta_soft, -1.0); ph_diff = s_phase(theta_soft, +1.0)
    new_pot = np.where(same, ph_same[0]*psi_real + ph_same[1]*psi_pot, ph_diff[0]*psi_real + ph_diff[1]*psi_pot)
    new_real = np.where(same, ph_same[1]*psi_real, ph_diff[1]*psi_real)
    return new_pot, new_real

def apply_rx_soft(psi_pot, psi_real, q, n, theta_soft):
    c, nis = s_rx(theta_soft)
    psi_pot = psi_pot.reshape([2]*n); psi_real = psi_real.reshape([2]*n)
    axis = n - 1 - q
    psi_pot = np.moveaxis(psi_pot, axis, 0); psi_real = np.moveaxis(psi_real, axis, 0)
    shape = psi_pot.shape
    p0 = (psi_pot[0].reshape(-1), psi_real[0].reshape(-1))
    p1 = (psi_pot[1].reshape(-1), psi_real[1].reshape(-1))
    c_p0 = smul(c, p0); nis_p1 = smul(nis, p1)
    new0 = (c_p0[0]+nis_p1[0], c_p0[1]+nis_p1[1])
    nis_p0 = smul(nis, p0); c_p1 = smul(c, p1)
    new1 = (nis_p0[0]+c_p1[0], nis_p0[1]+c_p1[1])
    out_pot = np.stack([new0[0].reshape(shape[1:]), new1[0].reshape(shape[1:])], axis=0)
    out_real = np.stack([new0[1].reshape(shape[1:]), new1[1].reshape(shape[1:])], axis=0)
    out_pot = np.moveaxis(out_pot, 0, axis).reshape(-1); out_real = np.moveaxis(out_real, 0, axis).reshape(-1)
    return out_pot, out_real

def statevector_exact(n_qubits, edges, p_layers, penalty, theta):
    """Plain-complex (non-soft) exact statevector, for verification."""
    dim = 2**n_qubits
    psi = np.zeros(dim, dtype=complex); psi[:] = 1.0/np.sqrt(dim)  # H on all
    gammas, betas = theta[:p_layers], theta[p_layers:]
    for layer in range(p_layers):
        g = gammas[layer]
        idxs = np.arange(dim)
        for q in range(n_qubits):
            bit = (idxs >> q) & 1
            psi = psi * np.where(bit==0, np.exp(-1j*g/2), np.exp(1j*g/2))
        for (i,j) in edges:
            bi = (idxs>>i)&1; bj=(idxs>>j)&1
            theta_zz = penalty*g/2
            psi = psi * np.where(bi==bj, np.exp(-1j*theta_zz/2), np.exp(1j*theta_zz/2))
        b = betas[layer]
        for q in range(n_qubits):
            psi_r = psi.reshape([2]*n_qubits)
            axis = n_qubits-1-q
            psi_r = np.moveaxis(psi_r, axis, 0)
            shape = psi_r.shape
            p0 = psi_r[0].reshape(-1); p1 = psi_r[1].reshape(-1)
            c, s = np.cos(b), np.sin(b)
            new0 = c*p0 - 1j*s*p1; new1 = -1j*s*p0 + c*p1
            out = np.stack([new0.reshape(shape[1:]), new1.reshape(shape[1:])], axis=0)
            out = np.moveaxis(out, 0, axis)
            psi = out.reshape(-1)
    return psi

def g_delta(theta0, delta, n_qubits, edges, p_layers, penalty):
    dim = 2**n_qubits
    psi_pot = np.zeros(dim, dtype=complex)
    psi_real = np.full(dim, 1.0/np.sqrt(dim), dtype=complex)
    idx = 0
    for layer in range(p_layers):
        g_th = (delta[idx], theta0[idx]); idx += 1  # placeholder, real idx below
    idx = 0
    for layer in range(p_layers):
        gamma_th = (delta[layer], theta0[layer])
        for q in range(n_qubits):
            psi_pot, psi_real = apply_rz_soft(psi_pot, psi_real, q, n_qubits, gamma_th)
        # RZZ uses theta = penalty*gamma/2 -- scale the soft number linearly
        zz_th = (penalty*gamma_th[0]/2, penalty*gamma_th[1]/2)
        for (i, j) in edges:
            psi_pot, psi_real = apply_rzz_soft(psi_pot, psi_real, i, j, n_qubits, zz_th)
        beta_th = (2*delta[p_layers+layer], 2*theta0[p_layers+layer])
        for q in range(n_qubits):
            psi_pot, psi_real = apply_rx_soft(psi_pot, psi_real, q, n_qubits, beta_th)

    # cost = sum over computational basis states of P(bitstring)*mis_cost(bitstring)
    # expressed as an operator expectation: diag(cost) is REAL and diagonal in Z basis
    idxs = np.arange(dim)
    total_pot = 0.0; total_real = 0.0
    for state in range(dim):
        bits = [(state>>q)&1 for q in range(n_qubits)]
        selected = sum(bits)
        violations = sum(1 for (i,j) in edges if bits[i]==1 and bits[j]==1)
        cost = -selected + penalty*violations
        amp_pot = psi_pot[state]; amp_real = psi_real[state]
        prob_term = smul((amp_pot, amp_real), (np.conj(amp_pot), np.conj(amp_real)))
        total_pot += cost * np.real(prob_term[0])
        total_real += cost * np.real(prob_term[1])
    return float(total_pot)

def energy_exact(theta, n_qubits, edges, p_layers, penalty):
    psi = statevector_exact(n_qubits, edges, p_layers, penalty, theta)
    idxs = np.arange(2**n_qubits)
    total = 0.0
    for state in range(2**n_qubits):
        bits = [(state>>q)&1 for q in range(n_qubits)]
        selected = sum(bits)
        violations = sum(1 for (i,j) in edges if bits[i]==1 and bits[j]==1)
        cost = -selected + penalty*violations
        total += cost * abs(psi[state])**2
    return total

if __name__ == "__main__":
    import warnings; warnings.filterwarnings('ignore')
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    n_qubits, p_layers, penalty = 5, 5, 2.0
    rng = np.random.default_rng(0)
    edges = [(i,j) for i in range(n_qubits) for j in range(i+1,n_qubits) if rng.uniform()<0.5]
    if not edges: edges=[(0,1),(1,2)]
    theta = rng.uniform(-np.pi, np.pi, 2*p_layers)

    # cross-check against real qiskit
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits): qc.h(i)
    gammas, betas = theta[:p_layers], theta[p_layers:]
    from qiskit.circuit import Parameter
    for layer in range(p_layers):
        g = gammas[layer]
        for i in range(n_qubits): qc.rz(g, i)
        for (i,j) in edges: qc.rzz(penalty*g/2, i, j)
        for i in range(n_qubits): qc.rx(2*betas[layer], i)
    sv_qiskit = Statevector(qc).data
    sv_mine = statevector_exact(n_qubits, edges, p_layers, penalty, theta)
    print("statevector match:", np.allclose(sv_qiskit, sv_mine))

    d1 = g_delta(theta, rng.choice([-1.,1.],size=2*p_layers), n_qubits, edges, p_layers, penalty)
    delta = rng.choice([-1.,1.],size=2*p_layers)
    d1 = g_delta(theta, delta, n_qubits, edges, p_layers, penalty)
    h=1e-6
    e_p = energy_exact(theta+h*delta, n_qubits, edges, p_layers, penalty)
    e_m = energy_exact(theta-h*delta, n_qubits, edges, p_layers, penalty)
    d1_fd = (e_p-e_m)/(2*h)
    print(f"D1 soft={d1:.6f}  D1 fd={d1_fd:.6f}  diff={abs(d1-d1_fd):.2e}")
    print(f"energy at theta = {energy_exact(theta, n_qubits, edges, p_layers, penalty):.4f}")
