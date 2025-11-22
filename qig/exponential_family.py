"""
Quantum exponential family and BKM metric interface for the quantum inaccessible game.

This module contains:
- local operator bases (Pauli, Gell-Mann, generalised Gell-Mann);
- construction of the full operator basis {F_a};
- the `QuantumExponentialFamily` class providing ρ(θ), log-partition ψ(θ),
  Fisher/BKM metric G(θ), and the marginal-entropy constraint.
"""

from typing import Tuple, List

import numpy as np
from scipy.linalg import expm, eigh

from qig.core import marginal_entropies


# ============================================================================
# Operator Basis: Local Lie Algebra Generators
# ============================================================================

def pauli_basis(site: int, n_sites: int) -> List[np.ndarray]:
    """
    Create Pauli operator basis for site 'site' in an n_sites qubit system.

    Returns [sigma_x, sigma_y, sigma_z] tensored with identity on other sites.
    """
    # Define Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    paulis = [X, Y, Z]
    operators: List[np.ndarray] = []

    for P in paulis:
        # Build tensor product: I ⊗ ... ⊗ P ⊗ ... ⊗ I
        op = None
        for i in range(n_sites):
            current = P if i == site else I
            op = current if op is None else np.kron(op, current)
        operators.append(op)

    return operators


def gell_mann_matrices() -> List[np.ndarray]:
    """
    Return the 8 Gell-Mann matrices (generators of SU(3)).
    """
    gm: List[np.ndarray] = []

    # λ1 and λ2 (off-diagonal in 01 block)
    gm.append(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex))
    gm.append(np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex))

    # λ3 (diagonal in 01 block)
    gm.append(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex))

    # λ4 and λ5 (off-diagonal in 02 block)
    gm.append(np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex))
    gm.append(np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex))

    # λ6 and λ7 (off-diagonal in 12 block)
    gm.append(np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex))
    gm.append(np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex))

    # λ8 (diagonal)
    gm.append(np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex) / np.sqrt(3))

    return gm


def qutrit_basis(site: int, n_sites: int) -> List[np.ndarray]:
    """
    Create Gell-Mann operator basis for site 'site' in n_sites qutrit system.
    """
    I = np.eye(3, dtype=complex)
    gm = gell_mann_matrices()
    operators: List[np.ndarray] = []

    for G in gm:
        op = None
        for i in range(n_sites):
            current = G if i == site else I
            op = current if op is None else np.kron(op, current)
        operators.append(op)

    return operators


def create_operator_basis(n_sites: int, d: int) -> Tuple[list, list]:
    """
    Create full operator basis {F_a} for quantum exponential family.

    Parameters
    -----------
    n_sites : int
        Number of sites
    d : int
        Local dimension (2 for qubits, 3 for qutrits, d>=2 in general)
    """
    operators: List[np.ndarray] = []
    labels: List[str] = []

    if d == 2:
        # Qubits: Pauli basis
        pauli_names = ["X", "Y", "Z"]
        for site in range(n_sites):
            ops = pauli_basis(site, n_sites)
            operators.extend(ops)
            labels.extend([f"{name}_{site+1}" for name in pauli_names])

    elif d == 3:
        # Qutrits: Gell-Mann basis
        for site in range(n_sites):
            ops = qutrit_basis(site, n_sites)
            operators.extend(ops)
            labels.extend([f"λ{k+1}_{site+1}" for k in range(8)])

    elif d >= 4:
        # Higher dimensions: use generalised Gell-Mann matrices
        for site in range(n_sites):
            # Create d^2-1 Hermitian traceless operators
            basis_ops: List[np.ndarray] = []
            # Off-diagonal symmetric and antisymmetric
            for i in range(d):
                for j in range(i + 1, d):
                    # Symmetric: |i><j| + |j><i|
                    op = np.zeros((d, d), dtype=complex)
                    op[i, j] = 1
                    op[j, i] = 1
                    basis_ops.append(op)

                    # Antisymmetric: -i|i><j| + i|j><i|
                    op = np.zeros((d, d), dtype=complex)
                    op[i, j] = -1j
                    op[j, i] = 1j
                    basis_ops.append(op)

            # Diagonal traceless
            for k in range(d - 1):
                op = np.zeros((d, d), dtype=complex)
                for i in range(k + 1):
                    op[i, i] = 1
                op[k + 1, k + 1] = -(k + 1)
                op = op / np.sqrt(k + 1 + (k + 1) ** 2)
                basis_ops.append(op)

            # Tensor with identity on other sites
            for op_local in basis_ops:
                op_full = None
                for s in range(n_sites):
                    current = op_local if s == site else np.eye(d, dtype=complex)
                    op_full = current if op_full is None else np.kron(op_full, current)
                operators.append(op_full)
                labels.append(f"H{len(operators)}_{site+1}")

    else:
        raise ValueError(f"Dimension d={d} must be >= 2.")

    return operators, labels


# ============================================================================
# Quantum Exponential Family
# ============================================================================


class QuantumExponentialFamily:
    """
    Quantum exponential family: ρ(θ) = exp(∑ θ_a F_a - ψ(θ))
    """

    def __init__(self, n_sites: int, d: int):
        """
        Initialise quantum exponential family.

        Parameters
        -----------
        n_sites : int
            Number of subsystems
        d : int
            Local dimension (2 for qubits, 3 for qutrits)
        """
        self.n_sites = n_sites
        self.d = d
        self.dims = [d] * n_sites
        self.D = d**n_sites

        # Create operator basis
        self.operators, self.labels = create_operator_basis(n_sites, d)
        self.n_params = len(self.operators)

        print(f"Initialised {n_sites}-site system with d={d}")
        print(f"Hilbert space dimension: {self.D}")
        print(f"Number of parameters: {self.n_params}")

    def rho_from_theta(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute ρ(θ) = exp(K(θ) - ψ(θ)) where K(θ) = ∑ θ_a F_a.
        """
        K = np.zeros((self.D, self.D), dtype=complex)
        for theta_a, F_a in zip(theta, self.operators):
            K += theta_a * F_a
        rho_unnorm = expm(K)
        Z = np.trace(rho_unnorm)
        rho = rho_unnorm / Z
        return rho

    def log_partition(self, theta: np.ndarray) -> float:
        """
        Compute log partition function ψ(θ) = log Tr(exp(∑ θ_a F_a)).
        """
        K = sum(theta_a * F_a for theta_a, F_a in zip(theta, self.operators))
        return np.log(np.trace(expm(K))).real

    def fisher_information(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute Fisher information (BKM metric) G(θ) = ∇∇ψ(θ) using the
        Kubo-Mori / BKM inner product.

        For a quantum exponential family
            ρ(θ) = exp(K(θ)) / Z(θ),  K(θ) = ∑_a θ_a F_a,
        the Bogoliubov-Kubo-Mori metric can be written as
        \[
            G_{ab}(θ)
            = \int_0^1 \mathrm{Tr}\!\left(
                ρ(θ)^s  \tilde F_a  ρ(θ)^{1-s} \tilde F_b
              \right)\,\mathrm{d}s,
        \]
        where \(\tilde F_a = F_a - \mathrm{Tr}[ρ(θ) F_a]\,\mathbb I\) are
        centred sufficient statistics.  In the eigenbasis of ρ(θ), this
        reduces to the standard spectral representation with the
        Morozova-Chentsov function
            c(λ, μ) = (log λ - log μ)/(λ - μ)
        (with the diagonal limit c(λ, λ) = 1/λ).  When all F_a commute with
        ρ(θ) (the classical/diagonal case), this expression reduces to the
        usual covariance Fisher information matrix.

        This implementation:
        - diagonalises ρ(θ) = U diag(p) U† (respecting the Hilbert space
          structure);
        - centres each F_a in that basis;
        - applies the BKM kernel c(p_i, p_j) to all matrix elements, taking
          care with ordering (A_a[i,j] A_b[j,i]) and Hermitian conjugation;
        - and finally symmetrises G to guard against small numerical
          asymmetries.
        """
        # Step 1: spectral decomposition of ρ(θ)
        rho = self.rho_from_theta(theta)
        eigvals, U = eigh(rho)  # ρ is Hermitian

        # Regularise eigenvalues to avoid log(0); keep normalisation approximate
        eps_p = 1e-14
        p = np.clip(np.real(eigvals), eps_p, None)

        # Step 2: build centred sufficient statistics in eigenbasis of ρ
        D = self.D
        n = self.n_params
        A_tilde = np.zeros((n, D, D), dtype=complex)

        I = np.eye(D, dtype=complex)
        for a, F_a in enumerate(self.operators):
            # Centre F_a: F_a - ⟨F_a⟩_ρ I
            mean_Fa = np.trace(rho @ F_a).real
            A_a = F_a - mean_Fa * I
            # Transform to eigenbasis of ρ
            A_tilde[a] = U.conj().T @ A_a @ U

        # Step 3: construct BKM kernel k(p_i, p_j)
        p_i = p[:, None]
        p_j = p[None, :]
        diff = p_i - p_j
        log_diff = np.log(p_i) - np.log(p_j)

        # k(λ, μ) = (λ - μ)/(log λ - log μ), with diagonal limit k(λ, λ) = λ
        # This is the standard Kubo–Mori / BKM kernel. In the diagonal
        # (commuting) case it reduces to the classical covariance weight p_i.
        k = np.zeros_like(diff)
        off_diag = np.abs(diff) > 1e-14
        k[off_diag] = diff[off_diag] / log_diff[off_diag]
        # Diagonal terms: limit (λ-μ)/(log λ - log μ) → λ as μ→λ
        diag_mask = np.eye(len(p), dtype=bool)
        k[diag_mask] = p

        # Step 4: assemble G_ab = Σ_{i,j} c(p_i, p_j) A_a[i,j] A_b[j,i]
        G = np.zeros((n, n))
        for a in range(n):
            A_a = A_tilde[a]
            for b in range(a, n):
                A_b = A_tilde[b]
                # Note: A_b[j,i] = (A_b.T.conj())[i,j]; we use that to respect
                # operator ordering and Hermitian structure.
                prod = A_a * A_b.T.conj()
                Gab = np.sum(k * prod)
                # BKM inner product is real for Hermitian observables; take Re.
                Gab_real = float(np.real(Gab))
                G[a, b] = Gab_real
                G[b, a] = Gab_real

        # Symmetrise to enforce exact symmetry numerically
        G = 0.5 * (G + G.T)
        return G

    def marginal_entropy_constraint(
        self, theta: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute constraint value C(θ) = ∑_i h_i and gradient ∇C.
        """
        rho = self.rho_from_theta(theta)
        h = marginal_entropies(rho, self.dims)
        C = float(np.sum(h))

        # TK: In the classical game this was analytic
        
        # Compute gradient via finite differences
        eps = 1e-5
        grad_C = np.zeros(self.n_params)
        for i in range(self.n_params):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            rho_plus = self.rho_from_theta(theta_plus)
            h_plus = marginal_entropies(rho_plus, self.dims)
            C_plus = float(np.sum(h_plus))
            grad_C[i] = (C_plus - C) / eps

        return C, grad_C


__all__ = [
    "pauli_basis",
    "gell_mann_matrices",
    "qutrit_basis",
    "create_operator_basis",
    "QuantumExponentialFamily",
]



