"""
Genuine Lemma-6.1 soft-number propagation through Qiskit's ACTUAL
EfficientSU2 ansatz (verified bit-exact in efficient_su2_exact.py) --
not a hand-picked toy ansatz. Complex soft numbers (a,b), a=potential,
b=real, both complex, exactly as used for QAOA earlier today.
"""
import numpy as np

def smul(x, y):
    a1,b1=x; a2,b2=y
    return (a1*b2 + a2*b1, b1*b2)
def sadd(x, y):
    return (x[0]+y[0], x[1]+y[1])

def s_ry(theta_soft):
    """Returns the soft 2x2 Ry matrix as ((a00,a01,a10,a11),(b00,...))."""
    a, b = theta_soft
    half_a, half_b = a*0.5, b*0.5
    c = (-half_a*np.sin(half_b), np.cos(half_b))
    s = (half_a*np.cos(half_b), np.sin(half_b))
    return c, s  # RY = [[c,-s],[s,c]]

def s_rz_phase(theta_soft, sign):
    """exp(sign * i * theta/2) as a soft number, via Lemma 6.1's general
    rule f(a,b)=(a*f'(b), f(b)) with f(t)=exp(sign*i*t/2)."""
    a, b = theta_soft
    val = np.exp(sign * 1j * b / 2)
    deriv = sign * 1j / 2 * val
    return (a*deriv, val)

def apply_ry_soft(psi, q, n, theta_soft):
    c, s = s_ry(theta_soft)
    psi = [list(v) for v in psi]  # each entry is (potential, real) complex pair
    psi_arr_pot = np.array([v[0] for v in psi]).reshape([2]*n)
    psi_arr_real = np.array([v[1] for v in psi]).reshape([2]*n)
    axis = n - 1 - q
    psi_arr_pot = np.moveaxis(psi_arr_pot, axis, 0)
    psi_arr_real = np.moveaxis(psi_arr_real, axis, 0)
    shape = psi_arr_pot.shape
    p0_pot, p1_pot = psi_arr_pot[0].reshape(-1), psi_arr_pot[1].reshape(-1)
    p0_real, p1_real = psi_arr_real[0].reshape(-1), psi_arr_real[1].reshape(-1)
    # new0 = c*p0 - s*p1 ; new1 = s*p0 + c*p1
    c_p0 = smul(c, (p0_pot, p0_real)); s_p1 = smul(s, (p1_pot, p1_real))
    new0 = (c_p0[0]-s_p1[0], c_p0[1]-s_p1[1])
    s_p0 = smul(s, (p0_pot, p0_real)); c_p1 = smul(c, (p1_pot, p1_real))
    new1 = (s_p0[0]+c_p1[0], s_p0[1]+c_p1[1])
    out_pot = np.stack([new0[0].reshape(shape[1:]), new1[0].reshape(shape[1:])], axis=0)
    out_real = np.stack([new0[1].reshape(shape[1:]), new1[1].reshape(shape[1:])], axis=0)
    out_pot = np.moveaxis(out_pot, 0, axis).reshape(-1)
    out_real = np.moveaxis(out_real, 0, axis).reshape(-1)
    return list(zip(out_pot, out_real))

def apply_rz_soft(psi, q, n, theta_soft):
    ph0 = s_rz_phase(theta_soft, -1.0)  # bit=0 -> exp(-i theta/2)
    ph1 = s_rz_phase(theta_soft, +1.0)  # bit=1 -> exp(+i theta/2)
    psi_arr_pot = np.array([v[0] for v in psi]).reshape([2]*n)
    psi_arr_real = np.array([v[1] for v in psi]).reshape([2]*n)
    axis = n - 1 - q
    psi_arr_pot = np.moveaxis(psi_arr_pot, axis, 0)
    psi_arr_real = np.moveaxis(psi_arr_real, axis, 0)
    shape = psi_arr_pot.shape
    p0 = (psi_arr_pot[0].reshape(-1), psi_arr_real[0].reshape(-1))
    p1 = (psi_arr_pot[1].reshape(-1), psi_arr_real[1].reshape(-1))
    new0 = smul(ph0, p0); new1 = smul(ph1, p1)
    out_pot = np.stack([new0[0].reshape(shape[1:]), new1[0].reshape(shape[1:])], axis=0)
    out_real = np.stack([new0[1].reshape(shape[1:]), new1[1].reshape(shape[1:])], axis=0)
    out_pot = np.moveaxis(out_pot, 0, axis).reshape(-1)
    out_real = np.moveaxis(out_real, 0, axis).reshape(-1)
    return list(zip(out_pot, out_real))

def apply_cnot_soft(psi, control, target, n):
    psi_arr_pot = np.array([v[0] for v in psi]).reshape([2]*n)
    psi_arr_real = np.array([v[1] for v in psi]).reshape([2]*n)
    c_axis = n - 1 - control; t_axis = n - 1 - target
    psi_arr_pot = np.moveaxis(psi_arr_pot, [c_axis, t_axis], [0, 1])
    psi_arr_real = np.moveaxis(psi_arr_real, [c_axis, t_axis], [0, 1])
    new_pot = psi_arr_pot.copy(); new_real = psi_arr_real.copy()
    new_pot[1, 0] = psi_arr_pot[1, 1]; new_pot[1, 1] = psi_arr_pot[1, 0]
    new_real[1, 0] = psi_arr_real[1, 1]; new_real[1, 1] = psi_arr_real[1, 0]
    new_pot = np.moveaxis(new_pot, [0, 1], [c_axis, t_axis]).reshape(-1)
    new_real = np.moveaxis(new_real, [0, 1], [c_axis, t_axis]).reshape(-1)
    return list(zip(new_pot, new_real))

def g_delta_su2(theta0, delta, n_qubits, reps, hamiltonian_terms):
    """hamiltonian_terms: list of (coeff, pauli_string) e.g. [(-0.48,'II'),...]
    with Qiskit's own qubit-0-is-rightmost string convention."""
    dim = 2**n_qubits
    psi = [(0.0+0j, 0.0+0j)] * dim
    psi[0] = (0.0+0j, 1.0+0j)
    idx = 0
    for layer in range(reps+1):
        for q in range(n_qubits):
            th = (delta[idx], theta0[idx]); idx += 1
            psi = apply_ry_soft(psi, q, n_qubits, th)
        for q in range(n_qubits):
            th = (delta[idx], theta0[idx]); idx += 1
            psi = apply_rz_soft(psi, q, n_qubits, th)
        if layer < reps:
            for q in range(n_qubits-1):
                psi = apply_cnot_soft(psi, q, q+1, n_qubits)

    # <psi|H|psi> as a soft number, H diagonal-or-Pauli decomposed.
    # For general Pauli terms we apply each Pauli op to <psi| (bra side is
    # just conj, real part of the expectation), summing coeff * <psi|P|psi>.
    pot_total = 0.0+0j; real_total = 0.0+0j
    idxs = np.arange(dim)
    for coeff, pauli in hamiltonian_terms:
        # pauli string: qiskit convention, rightmost char = qubit 0
        flips = 0; phase = np.ones(dim, dtype=complex)
        for qi, ch in enumerate(reversed(pauli)):
            bit = (idxs >> qi) & 1
            if ch == 'X':
                flips ^= (1 << qi)
            elif ch == 'Y':
                flips ^= (1 << qi)
                phase = phase * (-1j * (-1.0)**bit)
            elif ch == 'Z':
                phase = phase * ((-1.0)**bit)
        flipped_idx = idxs ^ flips
        # <psi|P|psi> = sum_i conj(psi_i) * phase_i * psi_{flipped_i}
        pot_i = np.array([psi[i][0] for i in range(dim)])
        real_i = np.array([psi[i][1] for i in range(dim)])
        conj_pot = np.conj(pot_i); conj_real = np.conj(real_i)
        flip_pot = pot_i[flipped_idx]; flip_real = real_i[flipped_idx]
        # (conj_psi) * phase * (flip_psi) as soft number product of two
        # soft numbers: (conj_real,conj_pot as "other" soft number)... use smul twice
        term1 = smul((conj_pot, conj_real), (phase*flip_pot, phase*flip_real))
        pot_total += coeff * np.sum(term1[0])
        real_total += coeff * np.sum(term1[1])
    return float(np.real(pot_total))

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_qubits, reps = 2, 4
    n_params = n_qubits*2*(reps+1)
    theta0 = rng.uniform(-1, 1, n_params)
    delta = rng.choice([-1.,1.], size=n_params)

    hterms = [(-0.4804,'II'), (0.3435,'ZZ'), (-0.4347,'ZI'), (0.5716,'IZ'), (0.0910,'XX'), (0.0910,'YY')]

    d1 = g_delta_su2(theta0, delta, n_qubits, reps, hterms)

    from efficient_su2_exact import efficient_su2_statevector
    def energy_exact(theta):
        psi = efficient_su2_statevector(n_qubits, reps, theta)
        H = np.zeros((4,4), dtype=complex)
        I2=np.eye(2); X=np.array([[0,1],[1,0]]); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1,-1])
        pmap = {'I':I2,'X':X,'Y':Y,'Z':Z}
        for coeff, pauli in hterms:
            mats = [pmap[c] for c in reversed(pauli)]
            term = mats[0]
            for m in mats[1:]: term = np.kron(m, term)
            H += coeff*term
        return float(np.real(np.conj(psi) @ H @ psi))

    h = 1e-6
    d1_fd = (energy_exact(theta0+h*delta) - energy_exact(theta0-h*delta)) / (2*h)
    print(f"D1 (soft, exact)      = {d1:.8f}")
    print(f"D1 (finite-diff check)= {d1_fd:.8f}")
    print(f"diff = {abs(d1-d1_fd):.2e}")
    print(f"energy at theta0 = {energy_exact(theta0):.6f}")
