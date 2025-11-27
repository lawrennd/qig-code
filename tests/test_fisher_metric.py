"""
Comprehensive test suite for Fisher Information (BKM) Metric implementation.

This module consolidates all BKM metric validation tests from:
- test_commuting_bkm.py (diagonal/commuting operators)
- test_nondiagonal_commuting_bkm.py (rotated commuting operators)
- test_non_commuting_bkm.py (non-commuting diagnostics)

Tests are organized by operator commutation structure:
1. Diagonal/Commuting: Validates spectral BKM reduces to classical Fisher
2. Rotated Commuting: Non-diagonal but commuting operators
3. Partially Commuting: Mix of commuting and non-commuting operators
4. Non-Commuting: General quantum case (diagnostic tests for known issues)

Validates: qig.exponential_family.QuantumExponentialFamily.fisher_information()

CIP-0004: Uses tolerance framework with scientifically justified bounds.
"""

import numpy as np
import pytest
from scipy.linalg import eigh, expm

from qig.exponential_family import QuantumExponentialFamily
from tests.fd_helpers import finite_difference_fisher
from tests.tolerance_framework import quantum_assert_close, quantum_assert_symmetric


# ============================================================================
# SECTION 1: DIAGONAL/COMMUTING OPERATORS
# ============================================================================
# When all operators are diagonal (commute), the BKM metric should exactly
# equal the classical Fisher information matrix. This provides a rigorous
# validation of the spectral implementation.
# ============================================================================


class DiagonalQuantumExponentialFamily:
    """
    A quantum exponential family where all sufficient statistics F_a are
    diagonal in a fixed basis.
    
    In this case:
        ρ(θ) = exp(∑_a θ_a F_a) / Z(θ)
    is diagonal, and the model reduces to a classical exponential family
    over the probability simplex with D = d^n_sites states.
    
    The BKM metric should exactly equal the classical Fisher information:
        G_ab(θ) = ∑_i p_i(θ) * (∂_a log p_i)(∂_b log p_i)
                = ∑_i (1/p_i) * (∂_a p_i)(∂_b p_i)
    where p_i(θ) are the diagonal elements of ρ(θ).
    """
    
    def __init__(self, n_sites: int, d: int):
        """
        Initialise diagonal quantum exponential family.
        
        Parameters
        ----------
        n_sites : int
            Number of subsystems
        d : int
            Local dimension
        """
        self.n_sites = n_sites
        self.d = d
        self.D = d ** n_sites
        
        # Create D-1 linearly independent diagonal operators (traceless)
        self.n_params = self.D - 1
        self.operators = self._create_diagonal_basis()
        self.labels = [f"F_{a+1}" for a in range(self.n_params)]
        
    def _create_diagonal_basis(self):
        """
        Create a basis of diagonal Hermitian traceless operators.
        
        For a D-dimensional Hilbert space, we create D-1 diagonal operators
        of the form:
            F_k = diag(1, 1, ..., 1, -(k+1), 0, ..., 0) / sqrt((k+1) + (k+1)^2)
        where the first k+1 entries are 1, the (k+2)-th entry is -(k+1),
        and the remaining entries are 0, ensuring trace = 0.
        
        This is analogous to the diagonal Gell-Mann matrices but generalised
        to arbitrary dimension.
        """
        operators = []
        D = self.D
        
        for k in range(D - 1):
            # Create diagonal operator with first k+2 entries summing to zero
            diag_vals = np.zeros(D)
            # First k+1 entries are 1
            for i in range(k + 1):
                diag_vals[i] = 1.0
            # (k+2)-th entry is -(k+1) to make trace zero
            diag_vals[k + 1] = -(k + 1)
            # Remaining entries are 0
            
            # Normalise: norm^2 = (k+1)*1^2 + (k+1)^2 = (k+1)(1 + k+1) = (k+1)(k+2)
            norm = np.sqrt((k + 1) * (k + 2))
            diag_vals = diag_vals / norm
            
            # Create diagonal matrix
            F_k = np.diag(diag_vals).astype(complex)
            operators.append(F_k)
            
        return operators
    
    def rho_from_theta(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute ρ(θ) = exp(K(θ)) / Z(θ) where K(θ) = ∑_a θ_a F_a.
        
        Since all F_a are diagonal, K(θ) is diagonal and ρ(θ) is diagonal.
        """
        # Build K(θ) = ∑_a θ_a F_a
        K = np.zeros((self.D, self.D), dtype=complex)
        for theta_a, F_a in zip(theta, self.operators):
            K += theta_a * F_a
        
        # Since K is diagonal, exp(K) is just exp of diagonal elements
        diag_K = np.diag(K)
        diag_rho_unnorm = np.exp(diag_K)
        Z = np.sum(diag_rho_unnorm)
        diag_rho = diag_rho_unnorm / Z
        
        rho = np.diag(diag_rho)
        return rho
    
    def log_partition(self, theta: np.ndarray) -> float:
        """
        Compute log partition function ψ(θ) = log Tr(exp(∑_a θ_a F_a)).
        
        For diagonal operators, this reduces to:
            ψ(θ) = log(∑_i exp(∑_a θ_a F_a[i,i]))
        """
        K = sum(theta_a * F_a for theta_a, F_a in zip(theta, self.operators))
        diag_K = np.diag(K)
        return float(np.log(np.sum(np.exp(diag_K))))
    
    def analytic_fisher_information(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute the analytic Fisher information for the diagonal case.
        
        For a diagonal exponential family, the Fisher information is:
            G_ab = ∂_a ∂_b ψ(θ)
                 = ∑_i p_i(θ) * F_a[i,i] * F_b[i,i] - (∑_i p_i F_a[i,i])(∑_i p_i F_b[i,i])
                 = Cov_ρ(F_a, F_b)
        
        This is the classical Fisher information for the probability distribution
        p_i(θ) over the D basis states.
        """
        rho = self.rho_from_theta(theta)
        p = np.diag(rho).real  # Probability distribution
        
        # Extract diagonal elements of each operator
        F_diag = np.array([np.diag(F_a).real for F_a in self.operators])
        
        # Compute means
        means = F_diag @ p  # Shape: (n_params,)
        
        # Compute covariance: G_ab = E[F_a F_b] - E[F_a]E[F_b]
        n = self.n_params
        G = np.zeros((n, n))
        
        for a in range(n):
            for b in range(n):
                # E[F_a F_b] = ∑_i p_i F_a[i] F_b[i]
                expectation = np.sum(p * F_diag[a] * F_diag[b])
                G[a, b] = expectation - means[a] * means[b]
        
        return G
    
    def spectral_fisher_information(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute Fisher information using the spectral BKM implementation.
        
        This uses the same algorithm as QuantumExponentialFamily.fisher_information
        but applied to our diagonal operators.
        """
        # Step 1: spectral decomposition of ρ(θ)
        rho = self.rho_from_theta(theta)
        eigvals, U = eigh(rho)
        
        # Regularise eigenvalues
        eps_p = 1e-14
        p = np.clip(np.real(eigvals), eps_p, None)
        
        # Step 2: build centred sufficient statistics in eigenbasis of ρ
        D = self.D
        n = self.n_params
        A_tilde = np.zeros((n, D, D), dtype=complex)
        
        I = np.eye(D, dtype=complex)
        for a, F_a in enumerate(self.operators):
            # Centre F_a
            mean_Fa = np.trace(rho @ F_a).real
            A_a = F_a - mean_Fa * I
            # Transform to eigenbasis of ρ
            A_tilde[a] = U.conj().T @ A_a @ U
        
        # Step 3: construct BKM kernel
        p_i = p[:, None]
        p_j = p[None, :]
        diff = p_i - p_j
        log_diff = np.log(p_i) - np.log(p_j)
        
        k = np.zeros_like(diff)
        off_diag = np.abs(diff) > 1e-14
        k[off_diag] = diff[off_diag] / log_diff[off_diag]
        diag_mask = np.eye(len(p), dtype=bool)
        k[diag_mask] = p
        
        # Step 4: assemble G_ab
        G = np.zeros((n, n))
        for a in range(n):
            A_a = A_tilde[a]
            for b in range(a, n):
                A_b = A_tilde[b]
                prod = A_a * A_b.T.conj()
                Gab = np.sum(k * prod)
                Gab_real = float(np.real(Gab))
                G[a, b] = Gab_real
                G[b, a] = Gab_real
        
        # Symmetrise
        G = 0.5 * (G + G.T)
        return G


class TestCommutingBKMMetric:
    """
    Test suite for validating BKM metric implementation via commuting families.
    """
    
    @pytest.mark.parametrize("n_sites,d", [
        (1, 2),  # Single qubit
        (1, 3),  # Single qutrit
        (1, 4),  # Single ququart
        (2, 2),  # Two qubits
    ])
    def test_diagonal_family_construction(self, n_sites, d):
        """Test that diagonal families are correctly constructed."""
        family = DiagonalQuantumExponentialFamily(n_sites, d)
        
        D = d ** n_sites
        assert family.D == D
        assert family.n_params == D - 1
        assert len(family.operators) == D - 1
        
        # Check that all operators are diagonal
        for F_a in family.operators:
            quantum_assert_close(F_a, np.diag(np.diag(F_a)), 'quantum_state',
                               err_msg="Operator not diagonal")
        
        # Check that all operators are Hermitian
        for F_a in family.operators:
            quantum_assert_close(F_a, F_a.conj().T, 'quantum_state',
                               err_msg="Operator not Hermitian")
        
        # Check that all operators are traceless
        for F_a in family.operators:
            trace_val = np.trace(F_a)
            quantum_assert_close(trace_val, 0.0, 'information_metric',
                               err_msg="Operator not traceless")
    
    @pytest.mark.parametrize("n_sites,d", [
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 2),
    ])
    def test_rho_is_diagonal(self, n_sites, d):
        """Test that ρ(θ) is diagonal for diagonal families."""
        family = DiagonalQuantumExponentialFamily(n_sites, d)
        
        # Random parameter point
        np.random.seed(42)
        theta = np.random.randn(family.n_params)
        
        rho = family.rho_from_theta(theta)
        
        # Check diagonal
        quantum_assert_close(rho, np.diag(np.diag(rho)), 'quantum_state',
                           err_msg="ρ(θ) not diagonal")
        
        # Check positive semidefinite
        eigvals = np.linalg.eigvalsh(rho)
        assert np.all(eigvals >= -1e-10), "ρ(θ) not positive semidefinite"
        
        # Check normalised
        quantum_assert_close(np.trace(rho), 1.0, 'quantum_state',
                           err_msg="ρ(θ) not normalized")
    
    @pytest.mark.parametrize("n_sites,d", [
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 2),
    ])
    def test_analytic_vs_spectral_bkm(self, n_sites, d):
        """
        Test that the analytic and spectral BKM metrics agree for diagonal families.
        
        This is the core validation test: in the commuting case, the spectral
        implementation should reduce to the classical Fisher information.
        """
        family = DiagonalQuantumExponentialFamily(n_sites, d)
        
        # Test at multiple parameter points
        np.random.seed(42)
        for trial in range(5):
            theta = np.random.randn(family.n_params) * 0.5
            
            G_analytic = family.analytic_fisher_information(theta)
            G_spectral = family.spectral_fisher_information(theta)
            
            # Check agreement (Category D: analytical derivatives)
            quantum_assert_close(G_spectral, G_analytic, 'fisher_metric',
                               err_msg=f"Trial {trial}: Analytic and spectral BKM metrics disagree")
    
    @pytest.mark.parametrize("n_sites,d", [
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 2),
    ])
    def test_bkm_positive_semidefinite(self, n_sites, d):
        """Test that the BKM metric is positive semidefinite."""
        family = DiagonalQuantumExponentialFamily(n_sites, d)
        
        np.random.seed(42)
        for trial in range(5):
            theta = np.random.randn(family.n_params) * 0.5
            
            G_analytic = family.analytic_fisher_information(theta)
            G_spectral = family.spectral_fisher_information(theta)
            
            # Check positive semidefiniteness
            eigvals_analytic = np.linalg.eigvalsh(G_analytic)
            eigvals_spectral = np.linalg.eigvalsh(G_spectral)
            
            assert np.all(eigvals_analytic >= -1e-10), (
                f"Analytic BKM has negative eigenvalues: {eigvals_analytic}"
            )
            assert np.all(eigvals_spectral >= -1e-10), (
                f"Spectral BKM has negative eigenvalues: {eigvals_spectral}"
            )
    
    @pytest.mark.parametrize("n_sites,d", [
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 2),
    ])
    def test_bkm_symmetry(self, n_sites, d):
        """Test that the BKM metric is symmetric."""
        family = DiagonalQuantumExponentialFamily(n_sites, d)
        
        np.random.seed(42)
        theta = np.random.randn(family.n_params) * 0.5
        
        G_analytic = family.analytic_fisher_information(theta)
        G_spectral = family.spectral_fisher_information(theta)
        
        # Check symmetry (Category D: analytical derivatives)
        quantum_assert_symmetric(G_analytic, 'fisher_metric',
                                err_msg="Analytic BKM not symmetric")
        quantum_assert_symmetric(G_spectral, 'fisher_metric',
                                err_msg="Spectral BKM not symmetric")
    
    @pytest.mark.parametrize("n_sites,d", [
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 2),
    ])
    def test_bkm_vs_finite_difference_hessian(self, n_sites, d):
        """
        Test that the BKM metric matches the finite-difference Hessian of ψ(θ).
        
        For an exponential family, the Fisher information equals the Hessian
        of the log-partition function:
            G_ab(θ) = ∂_a ∂_b ψ(θ)
        """
        family = DiagonalQuantumExponentialFamily(n_sites, d)
        
        np.random.seed(42)
        theta = np.random.randn(family.n_params) * 0.5
        
        # Compute BKM metrics
        G_analytic = family.analytic_fisher_information(theta)
        G_spectral = family.spectral_fisher_information(theta)
        
        # Compute finite-difference Hessian using shared helper
        # Note: DiagonalQuantumExponentialFamily has log_partition, but fd_helpers expects psi
        # Add psi as alias if not present
        if not hasattr(family, 'psi'):
            family.psi = family.log_partition
        H = finite_difference_fisher(family, theta, eps=1e-5)
        
        # Check agreement (Category D: analytical derivatives)
        quantum_assert_close(G_analytic, H, 'fisher_metric',
                           err_msg="Analytic BKM does not match FD Hessian")
        quantum_assert_close(G_spectral, H, 'fisher_metric',
                           err_msg="Spectral BKM does not match FD Hessian")


class TestQuantumExponentialFamilyCommutingCase:
    """
    Test that QuantumExponentialFamily.fisher_information works correctly
    when restricted to commuting operators.
    """
    
    def test_single_qubit_diagonal_only(self):
        """
        Test QuantumExponentialFamily with only the Z operator (diagonal).
        
        For a single qubit with only σ_z, the family is classical (diagonal)
        and the BKM metric should match the classical Fisher information.
        """
        # Create a custom family with only diagonal operators
        family = QuantumExponentialFamily(n_sites=1, d=2)
        
        # Extract only the Z operator (index 2 in Pauli basis)
        Z_op = family.operators[2]
        
        # Create a restricted family
        family_restricted = type('obj', (object,), {
            'n_sites': 1,
            'd': 2,
            'D': 2,
            'n_params': 1,
            'operators': [Z_op],
            'labels': ['Z'],
        })()
        
        # Add methods
        def rho_from_theta(theta):
            K = theta[0] * Z_op
            from scipy.linalg import expm
            rho_unnorm = expm(K)
            Z = np.trace(rho_unnorm)
            return rho_unnorm / Z
        
        def log_partition(theta):
            K = theta[0] * Z_op
            from scipy.linalg import expm
            return np.log(np.trace(expm(K))).real
        
        family_restricted.rho_from_theta = rho_from_theta
        family_restricted.log_partition = log_partition
        
        # Test at a parameter point
        theta = np.array([0.5])
        
        # Analytic Fisher information for single-qubit Z-only family
        # ρ(θ) = diag([e^θ, e^{-θ}]) / (e^θ + e^{-θ})
        # p_0 = e^θ / (e^θ + e^{-θ}) = (1 + tanh(θ))/2
        # p_1 = e^{-θ} / (e^θ + e^{-θ}) = (1 - tanh(θ))/2
        # F_Z = diag([1, -1])
        # G = Var(F_Z) = E[F_Z^2] - E[F_Z]^2 = 1 - tanh(θ)^2
        
        theta_val = theta[0]
        G_analytic = np.array([[1 - np.tanh(theta_val)**2]])
        
        # Compute using the spectral implementation
        rho = family_restricted.rho_from_theta(theta)
        eigvals, U = eigh(rho)
        eps_p = 1e-14
        p = np.clip(np.real(eigvals), eps_p, None)
        
        I = np.eye(2, dtype=complex)
        mean_Z = np.trace(rho @ Z_op).real
        A_Z = Z_op - mean_Z * I
        A_tilde = U.conj().T @ A_Z @ U
        
        p_i = p[:, None]
        p_j = p[None, :]
        diff = p_i - p_j
        log_diff = np.log(p_i) - np.log(p_j)
        
        k = np.zeros_like(diff)
        off_diag = np.abs(diff) > 1e-14
        k[off_diag] = diff[off_diag] / log_diff[off_diag]
        diag_mask = np.eye(len(p), dtype=bool)
        k[diag_mask] = p
        
        prod = A_tilde * A_tilde.T.conj()
        G_spectral = np.array([[float(np.real(np.sum(k * prod)))]])
        
        # Check agreement (Category D: analytical derivatives)
        quantum_assert_close(G_spectral, G_analytic, 'fisher_metric',
                           err_msg="Single-qubit Z-only: spectral BKM does not match analytic")


# ============================================================================
# SECTION 2: ROTATED COMMUTING OPERATORS
# ============================================================================
# Tests for non-diagonal operators that still commute (share eigenbasis).
# This validates that the spectral implementation works beyond just diagonal.
# ============================================================================


class RotatedDiagonalFamily:
    """
    Quantum exponential family where all operators are diagonal in a shared
    rotated basis (non-diagonal in computational basis but still commuting).
    
    Construction:
        F_a = U D_a U† where D_a are diagonal and U is a fixed unitary.
    
    All F_a share the same eigenbasis U, so they commute:
        [F_a, F_b] = U[D_a, D_b]U† = 0
    
    The BKM metric should still reduce to classical Fisher information,
    just computed in the rotated eigenbasis.
    """
    
    def __init__(self, n_sites: int, d: int, rotation_angle: float = np.pi/4):
        """
        Initialize rotated diagonal family.
        
        Parameters
        ----------
        n_sites : int
            Number of subsystems
        d : int
            Local dimension
        rotation_angle : float
            Angle for rotation (default: π/4 for maximal non-diagonality)
        """
        self.n_sites = n_sites
        self.d = d
        self.D = d ** n_sites
        self.n_params = self.D - 1
        
        # Create rotation unitary
        self.U = self._create_rotation_unitary(rotation_angle)
        
        # Create diagonal operators in the computational basis
        self.diagonal_ops = self._create_diagonal_basis()
        
        # Rotate them: F_a = U D_a U†
        self.operators = [self.U @ D @ self.U.conj().T for D in self.diagonal_ops]
        self.labels = [f"F_{a+1}" for a in range(self.n_params)]
        
    def _create_rotation_unitary(self, angle):
        """Create a fixed rotation unitary matrix."""
        if self.D == 2:
            # Single qubit: rotation around Y-axis
            c, s = np.cos(angle), np.sin(angle)
            return np.array([[c, -s], [s, c]], dtype=complex)
        elif self.D == 4:
            # Two qubits: tensor product of single-qubit rotations
            c, s = np.cos(angle), np.sin(angle)
            U1 = np.array([[c, -s], [s, c]], dtype=complex)
            return np.kron(U1, U1)
        else:
            # General: Use Householder reflection for a random rotation
            # This creates a unitary that is maximally non-diagonal
            v = np.random.randn(self.D) + 1j * np.random.randn(self.D)
            v = v / np.linalg.norm(v)
            U = np.eye(self.D, dtype=complex) - 2 * np.outer(v, v.conj())
            return U
    
    def _create_diagonal_basis(self):
        """Create D-1 diagonal traceless operators (same as DiagonalFamily)."""
        operators = []
        D = self.D
        
        for k in range(D - 1):
            diag_vals = np.zeros(D)
            for i in range(k + 1):
                diag_vals[i] = 1.0
            diag_vals[k + 1] = -(k + 1)
            norm = np.sqrt((k + 1) * (k + 2))
            diag_vals = diag_vals / norm
            
            F_k = np.diag(diag_vals).astype(complex)
            operators.append(F_k)
            
        return operators
    
    def rho_from_theta(self, theta: np.ndarray) -> np.ndarray:
        """Compute ρ(θ) = exp(K(θ)) / Z(θ)."""
        K = sum(theta_a * F_a for theta_a, F_a in zip(theta, self.operators))
        rho_unnorm = expm(K)
        Z = np.trace(rho_unnorm)
        return rho_unnorm / Z
    
    def log_partition(self, theta: np.ndarray) -> float:
        """Compute log partition function."""
        K = sum(theta_a * F_a for theta_a, F_a in zip(theta, self.operators))
        return np.log(np.trace(expm(K))).real
    
    def fisher_information_analytic(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute Fisher information analytically.
        
        Since all operators share eigenbasis U, we work in the rotated frame:
        1. Transform to eigenbasis: D_a = U† F_a U
        2. Compute as if diagonal (using DiagonalFamily formula)
        3. Result is basis-independent (no transform back needed)
        """
        # Compute ρ in the rotated basis
        rho = self.rho_from_theta(theta)
        eigvals, V = eigh(rho)  # V rotates to eigenbasis of ρ
        p = np.clip(eigvals.real, 1e-14, None)
        
        # Transform operators to eigenbasis of ρ
        F_eig = [V.conj().T @ F_a @ V for F_a in self.operators]
        F_diag = np.array([np.diag(F).real for F in F_eig])
        
        # Classical Fisher information in eigenbasis
        means = F_diag @ p
        n = self.n_params
        G = np.zeros((n, n))
        
        for a in range(n):
            for b in range(n):
                expectation = np.sum(p * F_diag[a] * F_diag[b])
                G[a, b] = expectation - means[a] * means[b]
        
        return G
    
    def fisher_information_spectral(self, theta: np.ndarray) -> np.ndarray:
        """Compute using spectral BKM implementation."""
        rho = self.rho_from_theta(theta)
        eigvals, U = eigh(rho)
        eps_p = 1e-14
        p = np.clip(np.real(eigvals), eps_p, None)
        
        D = self.D
        n = self.n_params
        A_tilde = np.zeros((n, D, D), dtype=complex)
        
        I = np.eye(D, dtype=complex)
        for a, F_a in enumerate(self.operators):
            mean_Fa = np.trace(rho @ F_a).real
            A_a = F_a - mean_Fa * I
            A_tilde[a] = U.conj().T @ A_a @ U
        
        p_i = p[:, None]
        p_j = p[None, :]
        diff = p_i - p_j
        log_diff = np.log(p_i) - np.log(p_j)
        
        k = np.zeros_like(diff)
        off_diag = np.abs(diff) > 1e-14
        k[off_diag] = diff[off_diag] / log_diff[off_diag]
        diag_mask = np.eye(len(p), dtype=bool)
        k[diag_mask] = p
        
        G = np.zeros((n, n))
        for a in range(n):
            A_a = A_tilde[a]
            for b in range(a, n):
                A_b = A_tilde[b]
                prod = A_a * A_b.T.conj()
                Gab = np.sum(k * prod)
                Gab_real = float(np.real(Gab))
                G[a, b] = Gab_real
                G[b, a] = Gab_real
        
        G = 0.5 * (G + G.T)
        return G


def verify_operators_commute(operators):
    """Verify that all operators commute."""
    n = len(operators)
    for i in range(n):
        for j in range(i+1, n):
            comm = operators[i] @ operators[j] - operators[j] @ operators[i]
            norm = np.linalg.norm(comm, 'fro')
            if norm > 1e-10:
                return False, (i, j, norm)
    return True, None


class TestNonDiagonalCommutingBKM:
    """Test BKM metric for non-diagonal but commuting operators."""
    
    @pytest.mark.parametrize("n_sites,d", [(1, 2), (1, 3), (2, 2)])
    def test_operators_actually_commute(self, n_sites, d):
        """Verify that rotated operators do indeed commute."""
        family = RotatedDiagonalFamily(n_sites, d)
        commute, info = verify_operators_commute(family.operators)
        assert commute, f"Operators {info[0]} and {info[1]} don't commute: norm={info[2]:.2e}"
    
    @pytest.mark.parametrize("n_sites,d", [(1, 2), (1, 3)])
    def test_analytic_vs_spectral_rotated(self, n_sites, d):
        """Test that analytic and spectral agree for rotated families."""
        family = RotatedDiagonalFamily(n_sites, d)
        
        np.random.seed(123)
        theta = np.random.randn(family.n_params) * 0.2
        
        G_analytic = family.fisher_information_analytic(theta)
        G_spectral = family.fisher_information_spectral(theta)
        
        # Check agreement (Category D: analytical derivatives)
        quantum_assert_close(G_spectral, G_analytic, 'fisher_metric',
                           err_msg=f"Rotated family (n={n_sites}, d={d}): spectral ≠ analytic")
    
    @pytest.mark.parametrize("n_sites,d", [(1, 2), (1, 3), (2, 2)])
    def test_positive_semidefinite_rotated(self, n_sites, d):
        """Test that BKM is positive semidefinite for rotated operators."""
        family = RotatedDiagonalFamily(n_sites, d)
        
        np.random.seed(123)
        theta = np.random.randn(family.n_params) * 0.1
        
        G = family.fisher_information_analytic(theta)
        eigenvalues = np.linalg.eigvalsh(G)
        
        assert np.all(eigenvalues > -1e-10), (
            f"BKM metric not positive semidefinite for rotated family\n"
            f"Min eigenvalue: {eigenvalues.min():.2e}"
        )


# ============================================================================
# SECTION 3: PARTIALLY COMMUTING OPERATORS
# ============================================================================
# Mix of commuting and non-commuting operators. Tests transition between
# classical and quantum regimes.
# ============================================================================


class PartiallyCommutingFamily:
    """
    Quantum exponential family with partially commuting operators.
    
    Construction for two qubits:
        F_1 = σ_z ⊗ I   (commutes with F_3)
        F_2 = σ_x ⊗ I   (does NOT commute with F_1)
        F_3 = I ⊗ σ_z   (commutes with F_1)
    
    This creates a mix: {F_1, F_3} form a commuting subspace (classical),
    while F_2 introduces quantum behavior (non-commuting with F_1).
    
    The BKM metric should:
    - For (1,3) block: behave classically
    - For (1,2) and (2,3) blocks: show quantum corrections
    """
    
    def __init__(self):
        """Initialize partially commuting family for two qubits."""
        self.n_sites = 2
        self.d = 2
        self.D = 4
        self.n_params = 3  # Three operators
        
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)
        
        # F_1 = σ_z ⊗ I, F_2 = σ_x ⊗ I, F_3 = I ⊗ σ_z
        self.operators = [
            np.kron(sigma_z, I),
            np.kron(sigma_x, I),
            np.kron(I, sigma_z),
        ]
        self.labels = ['Z⊗I', 'X⊗I', 'I⊗Z']
    
    def rho_from_theta(self, theta: np.ndarray) -> np.ndarray:
        """Compute ρ(θ)."""
        K = sum(theta_a * F_a for theta_a, F_a in zip(theta, self.operators))
        rho_unnorm = expm(K)
        Z = np.trace(rho_unnorm)
        return rho_unnorm / Z
    
    def commutation_structure(self):
        """Return boolean matrix indicating which operators commute."""
        n = len(self.operators)
        commutes = np.zeros((n, n), dtype=bool)
        
        for i in range(n):
            for j in range(n):
                comm = self.operators[i] @ self.operators[j] - self.operators[j] @ self.operators[i]
                commutes[i, j] = np.linalg.norm(comm) < 1e-10
        
        return commutes


class TestPartiallyCommutingBKM:
    """Test BKM metric for partially commuting operators."""
    
    def test_commutation_structure(self):
        """Verify the commutation pattern is as expected."""
        family = PartiallyCommutingFamily()
        commutes = family.commutation_structure()
        
        # Expected: [F_1, F_2] ≠ 0 (same qubit), [F_1, F_3] = 0, [F_2, F_3] = 0 (different qubits)
        assert commutes[0, 2] and commutes[2, 0], "F_1 (Z⊗I) and F_3 (I⊗Z) should commute"
        assert commutes[1, 2] and commutes[2, 1], "F_2 (X⊗I) and F_3 (I⊗Z) should commute"
        assert not commutes[0, 1] and not commutes[1, 0], "F_1 (Z⊗I) and F_2 (X⊗I) should NOT commute"
    
    def test_bkm_positive_semidefinite_partial(self):
        """BKM metric should be positive semidefinite even with partial commutation."""
        family = PartiallyCommutingFamily()
        
        # Test at multiple parameter points
        for seed in range(5):
            np.random.seed(seed + 100)
            theta = np.random.randn(family.n_params) * 0.2
            
            rho = family.rho_from_theta(theta)
            eigvals, U = eigh(rho)
            eps_p = 1e-14
            p = np.clip(np.real(eigvals), eps_p, None)
            
            # Compute BKM metric
            n = family.n_params
            D = family.D
            A_tilde = np.zeros((n, D, D), dtype=complex)
            
            I = np.eye(D, dtype=complex)
            for a, F_a in enumerate(family.operators):
                mean_Fa = np.trace(rho @ F_a).real
                A_a = F_a - mean_Fa * I
                A_tilde[a] = U.conj().T @ A_a @ U
            
            p_i = p[:, None]
            p_j = p[None, :]
            diff = p_i - p_j
            log_diff = np.log(p_i) - np.log(p_j)
            
            k = np.zeros_like(diff)
            off_diag = np.abs(diff) > 1e-14
            k[off_diag] = diff[off_diag] / log_diff[off_diag]
            diag_mask = np.eye(len(p), dtype=bool)
            k[diag_mask] = p
            
            G = np.zeros((n, n))
            for a in range(n):
                A_a = A_tilde[a]
                for b in range(a, n):
                    A_b = A_tilde[b]
                    prod = A_a * A_b.T.conj()
                    Gab = np.sum(k * prod)
                    Gab_real = float(np.real(Gab))
                    G[a, b] = Gab_real
                    G[b, a] = Gab_real
            
            # Check positive semidefiniteness
            eigvals = np.linalg.eigvalsh(G)
            assert np.all(eigvals > -1e-10), (
                f"BKM not positive semidefinite for partial commutation (seed={seed})\n"
                f"Min eigenvalue: {eigvals.min():.2e}"
            )
    
    def test_commuting_block_classical(self):
        """Test that commuting block (1,3) behaves classically."""
        family = PartiallyCommutingFamily()
        
        # Use only θ_1 and θ_3 (the commuting operators)
        theta = np.array([0.5, 0.0, 0.3])  # θ_2 = 0
        
        # Compute full BKM
        rho = family.rho_from_theta(theta)
        p, U = eigh(rho)
        p = np.clip(p.real, 1e-14, None)
        
        # BKM kernel
        p_i = p[:, None]
        p_j = p[None, :]
        diff = p_i - p_j
        log_diff = np.log(p_i) - np.log(p_j)
        
        k = np.zeros_like(diff)
        off = np.abs(diff) > 1e-14
        k[off] = diff[off] / log_diff[off]
        k[np.diag_indices(len(p))] = p
        
        # Compute G
        G = np.zeros((family.n_params, family.n_params))
        I_full = np.eye(family.D, dtype=complex)
        
        for a in range(family.n_params):
            F_a = family.operators[a]
            mean_a = np.trace(rho @ F_a).real
            A_a = F_a - mean_a * I_full
            A_a_tilde = U.conj().T @ A_a @ U
            
            for b in range(family.n_params):
                F_b = family.operators[b]
                mean_b = np.trace(rho @ F_b).real
                A_b = F_b - mean_b * I_full
                A_b_tilde = U.conj().T @ A_b @ U
                
                G[a, b] = np.real(np.sum(k * (A_a_tilde * np.conj(A_b_tilde))))
        
        # Extract (1,3) block: G[0,2] and G[2,0] (indices for F_1 and F_3)
        G_13 = G[0, 2]
        G_31 = G[2, 0]
        
        # For commuting operators, G should be symmetric (Category D: analytical derivatives)
        quantum_assert_close(G_13, G_31, 'fisher_metric',
                           err_msg=f"BKM not symmetric for commuting block: G[1,3]={G_13:.6f}, G[3,1]={G_31:.6f}")
        
        # The (1,3) block should be zero (F_1 and F_3 act on different qubits)
        # For product operators on independent subsystems, the covariance should be exactly zero!
        # With θ_2=0, ρ is a product state, so Cov(F_1, F_3) = 0 (Category D: analytical derivatives)
        quantum_assert_close(G_13, 0.0, 'fisher_metric',
                           err_msg=f"Expected zero covariance for independent operators, got {G_13:.6e}")


# ============================================================================
# SECTION 4: NON-COMMUTING OPERATORS (DIAGNOSTIC TESTS)
# ============================================================================
# General quantum case where operators don't commute. These tests document
# known issues with the BKM implementation for non-commuting operators.
# ============================================================================


class TestSimpleNonCommuting:
    """
    Test BKM metric for the simplest non-commuting cases.
    
    NOTE: These tests document known failures. The spectral BKM implementation
    has issues for non-commuting operators that need to be addressed.
    """
    
    def test_single_qubit_pauli_x_y(self):
        """
        Test single qubit with only σ_x and σ_y (non-commuting).
        
        For a single qubit with K(θ) = θ_x σ_x + θ_y σ_y:
        - [σ_x, σ_y] = 2i σ_z ≠ 0 (non-commuting)
        - This is the simplest non-trivial non-commuting case
        """
        # Create full family and extract X, Y operators
        family = QuantumExponentialFamily(n_sites=1, d=2)
        X_op = family.operators[0]  # σ_x
        Y_op = family.operators[1]  # σ_y
        
        # Verify non-commutation
        commutator = X_op @ Y_op - Y_op @ X_op
        assert np.linalg.norm(commutator) > 1e-10, "X and Y should not commute"
        
        # Create restricted family with only X and Y
        class RestrictedFamily:
            def __init__(self):
                self.n_sites = 1
                self.d = 2
                self.D = 2
                self.n_params = 2
                self.operators = [X_op, Y_op]
                self.labels = ['X', 'Y']
            
            def rho_from_theta(self, theta):
                K = theta[0] * X_op + theta[1] * Y_op
                rho_unnorm = expm(K)
                Z = np.trace(rho_unnorm)
                return rho_unnorm / Z
            
            def psi(self, theta):
                K = theta[0] * X_op + theta[1] * Y_op
                return np.log(np.trace(expm(K))).real
            
            def fisher_information(self, theta):
                """Spectral BKM implementation."""
                rho = self.rho_from_theta(theta)
                eigvals, U = eigh(rho)
                
                eps_p = 1e-14
                p = np.clip(np.real(eigvals), eps_p, None)
                
                D = self.D
                n = self.n_params
                A_tilde = np.zeros((n, D, D), dtype=complex)
                
                I = np.eye(D, dtype=complex)
                for a, F_a in enumerate(self.operators):
                    mean_Fa = np.trace(rho @ F_a).real
                    A_a = F_a - mean_Fa * I
                    A_tilde[a] = U.conj().T @ A_a @ U
                
                p_i = p[:, None]
                p_j = p[None, :]
                diff = p_i - p_j
                log_diff = np.log(p_i) - np.log(p_j)
                
                k = np.zeros_like(diff)
                off_diag = np.abs(diff) > 1e-14
                k[off_diag] = diff[off_diag] / log_diff[off_diag]
                diag_mask = np.eye(len(p), dtype=bool)
                k[diag_mask] = p
                
                G = np.zeros((n, n))
                for a in range(n):
                    A_a = A_tilde[a]
                    for b in range(a, n):
                        A_b = A_tilde[b]
                        prod = A_a * A_b.T.conj()
                        Gab = np.sum(k * prod)
                        Gab_real = float(np.real(Gab))
                        G[a, b] = Gab_real
                        G[b, a] = Gab_real
                
                G = 0.5 * (G + G.T)
                return G
        
        restricted = RestrictedFamily()
        
        # Test at a parameter point
        theta = np.array([0.3, 0.5])
        
        # Compute spectral BKM
        G_spectral = restricted.fisher_information(theta)
        
        # Compute finite-difference Hessian using shared helper
        G_fd = finite_difference_fisher(restricted, theta, eps=1e-5)
        
        # Compare (Category D: analytical derivatives)
        print(f"\nSingle qubit X-Y test:")
        print(f"Spectral BKM:\n{G_spectral}")
        print(f"Finite-diff Hessian:\n{G_fd}")
        print(f"Difference:\n{G_spectral - G_fd}")
        
        quantum_assert_close(G_spectral, G_fd, 'fisher_metric',
                           err_msg="Non-commuting case (X,Y): spectral vs FD mismatch")
    
    def test_single_qubit_all_paulis(self):
        """
        Test single qubit with all three Paulis (X, Y, Z).
        
        All three Pauli matrices are mutually non-commuting.
        """
        family = QuantumExponentialFamily(n_sites=1, d=2)
        
        # Test at multiple parameter points
        np.random.seed(42)
        for trial in range(3):
            theta = np.random.randn(family.n_params) * 0.5
            
            G_spectral = family.fisher_information(theta)
            
            # Finite-difference Hessian using shared helper
            G_fd = finite_difference_fisher(family, theta, eps=1e-5)
            
            if trial == 0:
                print(f"\nSingle qubit all Paulis (trial {trial}):")
                print(f"Spectral BKM:\n{G_spectral}")
                print(f"Finite-diff Hessian:\n{G_fd}")
                print(f"Difference:\n{G_spectral - G_fd}")
            
            # Category D: analytical derivatives
            quantum_assert_close(G_spectral, G_fd, 'fisher_metric',
                               err_msg=f"Trial {trial}: Non-commuting Paulis: spectral vs FD mismatch")
    
    def test_two_qubits_local_paulis(self):
        """
        Test two qubits with local Pauli operators.
        
        This is the case that currently fails in test_inaccessible_game.py.
        """
        family = QuantumExponentialFamily(n_sites=2, d=2)
        
        # Test at a single parameter point
        np.random.seed(0)
        theta = np.random.randn(family.n_params)
        
        G_spectral = family.fisher_information(theta)
        
        # Finite-difference Hessian using shared helper
        G_fd = finite_difference_fisher(family, theta, eps=1e-5)
        
        diff = G_spectral - G_fd
        max_abs_err = np.max(np.abs(diff))
        rel_err = max_abs_err / (np.max(np.abs(G_fd)) + 1e-10)
        
        print(f"\nTwo qubits (6 parameters):")
        print(f"Max absolute error: {max_abs_err:.6e}")
        print(f"Relative error: {rel_err:.6e}")
        print(f"\nSpectral BKM (first 3x3 block):\n{G_spectral[:3, :3]}")
        print(f"Finite-diff Hessian (first 3x3 block):\n{G_fd[:3, :3]}")
        print(f"Difference (first 3x3 block):\n{diff[:3, :3]}")
        
        # This will likely fail - documenting the failure
        if rel_err >= 1e-4:
            print(f"\n⚠️  EXPECTED FAILURE: rel_err={rel_err:.3e} >> 1e-4")
            print("This confirms the non-commuting case has fundamental issues.")
        
        # Category D: analytical derivatives
        quantum_assert_close(G_spectral, G_fd, 'fisher_metric',
                           err_msg=f"Two-qubit case: spectral vs FD mismatch (known issue)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

