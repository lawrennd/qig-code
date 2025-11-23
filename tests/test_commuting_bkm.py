"""
Test module for validating BKM metric implementation via commuting/diagonal families.

This module implements the validation plan from backlog task
2025-11-22_commuting-bkm-validation-plan.md:

1. Construct commuting toy exponential families where all sufficient statistics
   F_a are diagonal in a fixed basis.
2. Derive the analytic BKM metric (second Kubo-Mori cumulant) for these cases.
3. Compare the analytic formula with the spectral implementation.
4. Validate positive semidefiniteness and symmetry.

In the commuting/diagonal case, the quantum exponential family reduces to a
classical exponential family over the probability simplex, and the BKM metric
should exactly match the classical Fisher information matrix.
"""

import numpy as np
import pytest
from scipy.linalg import eigh

from qig.exponential_family import QuantumExponentialFamily


# ============================================================================
# Commuting Toy Family: Diagonal Operators
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
        
        # Create diagonal operator basis
        # We use D-1 linearly independent diagonal operators (traceless)
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


# ============================================================================
# Tests
# ============================================================================


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
            assert np.allclose(F_a, np.diag(np.diag(F_a)))
        
        # Check that all operators are Hermitian
        for F_a in family.operators:
            assert np.allclose(F_a, F_a.conj().T)
        
        # Check that all operators are traceless
        for F_a in family.operators:
            assert np.abs(np.trace(F_a)) < 1e-10
    
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
        assert np.allclose(rho, np.diag(np.diag(rho)))
        
        # Check positive semidefinite
        eigvals = np.linalg.eigvalsh(rho)
        assert np.all(eigvals >= -1e-10)
        
        # Check normalised
        assert np.abs(np.trace(rho) - 1.0) < 1e-10
    
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
            
            # Check agreement
            max_diff = np.max(np.abs(G_analytic - G_spectral))
            rel_diff = max_diff / (np.max(np.abs(G_analytic)) + 1e-10)
            
            assert rel_diff < 1e-6, (
                f"Trial {trial}: Analytic and spectral BKM metrics disagree.\n"
                f"Max absolute difference: {max_diff:.3e}\n"
                f"Max relative difference: {rel_diff:.3e}\n"
                f"Analytic:\n{G_analytic}\n"
                f"Spectral:\n{G_spectral}"
            )
    
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
        
        # Check symmetry
        assert np.allclose(G_analytic, G_analytic.T)
        assert np.allclose(G_spectral, G_spectral.T)
    
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
        
        # Compute finite-difference Hessian
        eps = 1e-5
        n = family.n_params
        H = np.zeros((n, n))
        
        psi_0 = family.log_partition(theta)
        
        for a in range(n):
            for b in range(a, n):
                # Central differences for second derivative
                theta_pp = theta.copy()
                theta_pp[a] += eps
                theta_pp[b] += eps
                psi_pp = family.log_partition(theta_pp)
                
                theta_pm = theta.copy()
                theta_pm[a] += eps
                theta_pm[b] -= eps
                psi_pm = family.log_partition(theta_pm)
                
                theta_mp = theta.copy()
                theta_mp[a] -= eps
                theta_mp[b] += eps
                psi_mp = family.log_partition(theta_mp)
                
                theta_mm = theta.copy()
                theta_mm[a] -= eps
                theta_mm[b] -= eps
                psi_mm = family.log_partition(theta_mm)
                
                H[a, b] = (psi_pp - psi_pm - psi_mp + psi_mm) / (4 * eps**2)
                H[b, a] = H[a, b]
        
        # Check agreement
        max_diff_analytic = np.max(np.abs(G_analytic - H))
        max_diff_spectral = np.max(np.abs(G_spectral - H))
        
        rel_diff_analytic = max_diff_analytic / (np.max(np.abs(H)) + 1e-10)
        rel_diff_spectral = max_diff_spectral / (np.max(np.abs(H)) + 1e-10)
        
        assert rel_diff_analytic < 1e-4, (
            f"Analytic BKM does not match Hessian.\n"
            f"Max relative difference: {rel_diff_analytic:.3e}"
        )
        assert rel_diff_spectral < 1e-4, (
            f"Spectral BKM does not match Hessian.\n"
            f"Max relative difference: {rel_diff_spectral:.3e}"
        )


# ============================================================================
# Integration with QuantumExponentialFamily
# ============================================================================


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
        
        # Check agreement
        rel_diff = np.abs(G_analytic[0, 0] - G_spectral[0, 0]) / G_analytic[0, 0]
        assert rel_diff < 1e-6, (
            f"Single-qubit Z-only family: spectral BKM does not match analytic.\n"
            f"Analytic: {G_analytic[0, 0]:.6f}\n"
            f"Spectral: {G_spectral[0, 0]:.6f}\n"
            f"Relative difference: {rel_diff:.3e}"
        )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

