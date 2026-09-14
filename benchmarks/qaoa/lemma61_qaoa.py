"""
Same approach as lemma61_vqe.py, for QAOA Max-Cut. g_delta computed via
ONE application of the single-axis soft number system (complex-valued);
D2 via finite difference on g_delta. Rebuilt using the same reshape/take
mixer-application pattern already verified in twin_zero_qaoa.py.
"""
import numpy as np

def smul(x, y):
    a1,b1 = x; a2,b2 = y
    return (a1*b2 + a2*b1, b1*b2)

class LemmaQAOA:
    def __init__(self, edges, n_qubits, reps=2):
        self.edges = edges; self.n = n_qubits; self.dim = 2**n_qubits; self.reps = reps
        self.n_params = 2*reps
        idx = np.arange(self.dim)
        cost = np.zeros(self.dim)
        for (i,j) in edges:
            bi = (idx >> (self.n-1-i)) & 1; bj = (idx >> (self.n-1-j)) & 1
            cost += 0.5*(1.0 - 2.0*(bi^bj))
        self.cost_diag = cost
        self.e0 = float(np.min(cost))

    def g_delta(self, theta0, delta):
        dim = self.dim
        A = np.zeros(dim, dtype=complex)
        B = np.full(dim, 1.0/np.sqrt(dim), dtype=complex)
        for layer in range(self.reps):
            gamma0, dgamma = theta0[layer], delta[layer]
            c_ = self.cost_diag
            ph_b = np.exp(-1j*c_*gamma0)
            ph_a = ph_b * (-1j*c_*dgamma)
            newA = A*ph_b + B*ph_a
            newB = B*ph_b
            A, B = newA, newB

            beta0, dbeta = theta0[self.reps+layer], delta[self.reps+layer]
            cv, sv = np.cos(beta0), np.sin(beta0)
            c_soft = (-sv*dbeta, cv); ms_soft = (-1j*cv*dbeta, -1j*sv)
            A = A.reshape([2]*self.n); B = B.reshape([2]*self.n)
            for q in range(self.n):
                A0,B0 = np.take(A,0,axis=q), np.take(B,0,axis=q)
                A1,B1 = np.take(A,1,axis=q), np.take(B,1,axis=q)
                cA0,cB0 = smul(c_soft,(A0,B0))
                cA1,cB1 = smul(c_soft,(A1,B1))
                mA0,mB0 = smul(ms_soft,(A0,B0))
                mA1,mB1 = smul(ms_soft,(A1,B1))
                new0 = (cA0+mA1, cB0+mB1)
                new1 = (mA0+cA1, mB0+cB1)
                A = np.stack([new0[0], new1[0]], axis=q)
                B = np.stack([new0[1], new1[1]], axis=q)
            A, B = A.reshape(-1), B.reshape(-1)

        cA, cB = np.conj(A), np.conj(B)
        pA = A*cB + B*cA
        Ea = np.sum(self.cost_diag*pA).real
        return float(Ea)

    def energy_true(self, theta0):
        dim = self.dim
        B = np.full(dim, 1.0/np.sqrt(dim), dtype=complex)
        for layer in range(self.reps):
            gamma0 = theta0[layer]
            B = B * np.exp(-1j*self.cost_diag*gamma0)
            beta0 = theta0[self.reps+layer]
            cv, sv = np.cos(beta0), np.sin(beta0)
            B = B.reshape([2]*self.n)
            for q in range(self.n):
                B0, B1 = np.take(B,0,axis=q), np.take(B,1,axis=q)
                new0 = cv*B0 - 1j*sv*B1
                new1 = -1j*sv*B0 + cv*B1
                B = np.stack([new0,new1], axis=q)
            B = B.reshape(-1)
        return float(np.sum(self.cost_diag * (B*np.conj(B)).real))

    def D1_D2(self, theta0, delta, h=1e-5):
        d1 = self.g_delta(theta0, delta)
        gp = self.g_delta(theta0 + h*delta, delta)
        gm = self.g_delta(theta0 - h*delta, delta)
        d2 = (gp - gm) / (2*h)
        return d1, d2

if __name__ == "__main__":
    import soft_jet_qaoa as SJQ
    edges = [(0,1),(1,2),(2,3),(3,0),(0,2)]
    mc_new = LemmaQAOA(edges, 4, reps=2)
    mc_ref = SJQ.MaxCutQAOA(edges, 4, reps=2)
    rng = np.random.default_rng(0)
    max_d1, max_d2, max_e = 0.0, 0.0, 0.0
    for _ in range(15):
        theta0 = rng.uniform(-1, 1, mc_new.n_params)
        delta = rng.choice([-1.0,1.0], size=mc_new.n_params)
        d1_mine, d2_mine = mc_new.D1_D2(theta0, delta)
        E_ref, d1_ref, halfd2_ref = mc_ref.energy_jet_along(theta0, delta)
        e_mine = mc_new.energy_true(theta0)
        max_e = max(max_e, abs(e_mine-E_ref))
        max_d1 = max(max_d1, abs(d1_mine-d1_ref))
        max_d2 = max(max_d2, abs(d2_mine-2*halfd2_ref))
    print(f"max|E diff|={max_e:.2e}  max|D1 diff|={max_d1:.2e}  max|D2 diff| (FD, h=1e-5)={max_d2:.2e}")
