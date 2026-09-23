"""
Exact numpy replica of Qiskit's EfficientSU2(n_qubits, reps, entanglement='linear')
ansatz -- verified bit-for-bit against Qiskit's own Statevector simulation.
Qiskit statevector convention: qubit 0 is the LEAST significant bit
(index = ... + b1*2 + b0).
"""
import numpy as np

def ry(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -s], [s, c]])

def rz(theta):
    return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]])

def apply_1q(psi, U, q, n):
    """Apply single-qubit gate U to qubit q (0=LSB) of an n-qubit statevector."""
    psi = psi.reshape([2]*n)
    # numpy axis ordering: axis 0 = MOST significant qubit = qubit (n-1)
    axis = n - 1 - q
    psi = np.moveaxis(psi, axis, 0)
    shape = psi.shape
    psi = psi.reshape(2, -1)
    psi = U @ psi
    psi = psi.reshape(shape)
    psi = np.moveaxis(psi, 0, axis)
    return psi.reshape(-1)

def apply_cnot(psi, control, target, n):
    psi = psi.reshape([2]*n)
    c_axis = n - 1 - control
    t_axis = n - 1 - target
    psi = np.moveaxis(psi, [c_axis, t_axis], [0, 1])
    shape = psi.shape
    new = psi.copy()
    new[1, 0] = psi[1, 1]
    new[1, 1] = psi[1, 0]
    new = np.moveaxis(new, [0, 1], [c_axis, t_axis])
    return new.reshape(-1)

def efficient_su2_statevector(n_qubits, reps, theta):
    """theta: flat array, length n_qubits*2*(reps+1), matching Qiskit's own
    parameter ordering (Ry,Ry,...,Rz,Rz,... per layer, linear CNOT chain
    between layers, no CNOT after the final layer)."""
    dim = 2**n_qubits
    psi = np.zeros(dim, dtype=complex); psi[0] = 1.0
    idx = 0
    for layer in range(reps+1):
        for q in range(n_qubits):
            psi = apply_1q(psi, ry(theta[idx]), q, n_qubits); idx += 1
        for q in range(n_qubits):
            psi = apply_1q(psi, rz(theta[idx]), q, n_qubits); idx += 1
        if layer < reps:
            for q in range(n_qubits-1):
                psi = apply_cnot(psi, q, q+1, n_qubits)
    return psi

if __name__ == "__main__":
    from qiskit.circuit.library import EfficientSU2
    from qiskit.quantum_info import Statevector
    import warnings; warnings.filterwarnings('ignore')

    rng = np.random.default_rng(0)
    params = rng.uniform(-1, 1, 20)
    ansatz = EfficientSU2(2, reps=4, entanglement='linear')
    bound = ansatz.assign_parameters(params)
    sv_qiskit = Statevector(bound).data

    sv_mine = efficient_su2_statevector(2, 4, params)
    print("Qiskit: ", sv_qiskit)
    print("Mine:   ", sv_mine)
    print("max diff:", np.max(np.abs(sv_qiskit - sv_mine)))
