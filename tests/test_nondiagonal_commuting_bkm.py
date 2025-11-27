"""
Extension of BKM validation: Non-diagonal commuting and partially commuting families.

This module completes the validation plan from backlog task
2025-11-22_commuting-bkm-validation-plan.md by testing:

1. Non-diagonal commuting operators: Operators that share an eigenbasis but are
   not diagonal in the computational basis (rotated diagonal operators).
   
2. Partially commuting families: Families where some operators commute with
   each other but not all, to test the transition between classical and quantum.

This extends test_commuting_bkm.py which validated the diagonal case.
"""

import numpy as np
import pytest
from scipy.linalg import eigh, expm

from qig.exponential_family import QuantumExponentialFamily
from tests.tolerance_framework import quantum_assert_close


# ============================================================================
# Non-Diagonal Commuting Operators
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
    
    def rho_from_theta(self, theta):
        """Compute density matrix ρ(θ) = exp(∑ θ_a F_a) / Z."""
        K = sum(theta_a * F_a for theta_a, F_a in zip(theta, self.operators))
        rho_unnorm = expm(K)
        Z = np.trace(rho_unnorm)
        return rho_unnorm / Z
    
    def fisher_information_analytic(self, theta):
        """
        Compute analytic Fisher information for rotated diagonal family.
        
        Since all F_a share eigenbasis U, we can work in that basis:
            D_a = U† F_a U  (diagonal)
            rho_diag = U† ρ U  (diagonal)
        
        Then G_ab = Cov(D_a, D_b) computed with rho_diag.
        """
        rho = self.rho_from_theta(theta)
        
        # Transform to shared eigenbasis
        rho_diag = self.U.conj().T @ rho @ self.U
        p = np.diag(rho_diag).real  # Eigenvalues
        
        # Get operators in eigenbasis
        D_ops = [self.U.conj().T @ F @ self.U for F in self.operators]
        
        # Classical Fisher information in eigenbasis
        G = np.zeros((self.n_params, self.n_params))
        for a in range(self.n_params):
            for b in range(self.n_params):
                D_a_diag = np.diag(D_ops[a]).real
                D_b_diag = np.diag(D_ops[b]).real
                
                mean_a = np.sum(p * D_a_diag)
                mean_b = np.sum(p * D_b_diag)
                cov = np.sum(p * D_a_diag * D_b_diag) - mean_a * mean_b
                G[a, b] = cov
        
        return G


def verify_operators_commute(operators, tol=1e-12):
    """Check that all operators pairwise commute."""
    n = len(operators)
    for i in range(n):
        for j in range(i+1, n):
            commutator = operators[i] @ operators[j] - operators[j] @ operators[i]
            comm_norm = np.linalg.norm(commutator)
            if comm_norm > tol:
                return False, (i, j, comm_norm)
    return True, None


# ============================================================================
# Partially Commuting Families
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
            np.kron(sigma_z, I),  # F_1
            np.kron(sigma_x, I),  # F_2  
            np.kron(I, sigma_z),  # F_3
        ]
        self.labels = ["Z⊗I", "X⊗I", "I⊗Z"]
        
        # Commutation structure:
        # [F_1, F_2] ≠ 0  (quantum - both on first qubit)
        # [F_1, F_3] = 0   (classical - different qubits)
        # [F_2, F_3] = 0  (classical - different qubits)
        
    def rho_from_theta(self, theta):
        """Compute density matrix ρ(θ)."""
        K = sum(theta_a * F_a for theta_a, F_a in zip(theta, self.operators))
        rho_unnorm = expm(K)
        Z = np.trace(rho_unnorm)
        return rho_unnorm / Z
    
    def commutation_structure(self):
        """Return matrix indicating which operators commute."""
        n = self.n_params
        commutes = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                comm = self.operators[i] @ self.operators[j] - self.operators[j] @ self.operators[i]
                commutes[i, j] = np.linalg.norm(comm) < 1e-12
        return commutes


# ============================================================================
# Tests for Non-Diagonal Commuting Families
# ============================================================================

class TestNonDiagonalCommutingBKM:
    """Test BKM metric for non-diagonal but commuting operators."""
    
    @pytest.mark.parametrize("n_sites,d", [(1, 2), (1, 3), (2, 2)])
    def test_operators_actually_commute(self, n_sites, d):
        """Verify that rotated operators do commute."""
        family = RotatedDiagonalFamily(n_sites, d)
        commute, info = verify_operators_commute(family.operators)
        assert commute, f"Operators {info[0]} and {info[1]} don't commute: norm={info[2]:.2e}"
    
    @pytest.mark.parametrize("n_sites,d", [(1, 2), (1, 3)])
    def test_analytic_vs_spectral_rotated(self, n_sites, d):
        """Compare analytic and spectral BKM for rotated operators."""
        family = RotatedDiagonalFamily(n_sites, d)
        
        # Random parameter point
        np.random.seed(42)
        theta = np.random.randn(family.n_params) * 0.1
        
        # Analytic Fisher information (in rotated basis)
        G_analytic = family.fisher_information_analytic(theta)
        
        # Spectral implementation via QuantumExponentialFamily
        # We need to manually compute it since we're using custom operators
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
        
        # Compute G_spectral
        G_spectral = np.zeros((family.n_params, family.n_params))
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
                
                G_spectral[a, b] = np.real(np.sum(k * (A_a_tilde * np.conj(A_b_tilde))))
        
        # Compare
        # Check agreement (Category D: analytical derivatives)
        quantum_assert_close(G_spectral, G_analytic, 'fisher_metric',
                           err_msg=f"Rotated family (n={n_sites}, d={d}): spectral ≠ analytic")
    
    @pytest.mark.parametrize("n_sites,d", [(1, 2), (1, 3), (2, 2)])
    def test_positive_semidefinite_rotated(self, n_sites, d):
        """Verify BKM metric is positive semidefinite for rotated operators."""
        family = RotatedDiagonalFamily(n_sites, d)
        
        np.random.seed(123)
        theta = np.random.randn(family.n_params) * 0.1
        
        G = family.fisher_information_analytic(theta)
        eigvals = np.linalg.eigvalsh(G)
        
        assert np.all(eigvals > -1e-10), (
            f"BKM metric not positive semidefinite for rotated family\n"
            f"Min eigenvalue: {eigvals.min():.2e}"
        )


# ============================================================================
# Tests for Partially Commuting Families
# ============================================================================

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
            np.random.seed(seed)
            theta = np.random.randn(family.n_params) * 0.2
            
            # Compute BKM using spectral implementation
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
            
            # Check positive semidefiniteness
            eigvals = np.linalg.eigvalsh(G)
            assert np.all(eigvals > -1e-10), (
                f"BKM not positive semidefinite for partial commutation (seed={seed})\n"
                f"Min eigenvalue: {eigvals.min():.2e}"
            )
    
    def test_commuting_block_classical(self):
        """
        For the (1,3) block (Z⊗I and I⊗Z which commute), the BKM metric
        should behave classically (match the diagonal case).
        """
        family = PartiallyCommutingFamily()
        
        np.random.seed(24)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

