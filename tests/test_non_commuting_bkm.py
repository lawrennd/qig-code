"""
Test module for diagnosing BKM metric issues in non-commuting cases.

The diagonal validation (test_commuting_bkm.py) showed the spectral BKM
implementation is correct when all operators commute. However, tests with
non-commuting operators (Pauli/Gell-Mann bases) show O(1) errors between
the spectral BKM and finite-difference Hessian.

This module systematically tests the transition from commuting to non-commuting
to identify where the implementation breaks down.

Key quantum derivative issues to check:
1. Operator commutation: [F_a, F_b] ≠ 0 in general
2. Operator ordering: ABC ≠ CBA for non-commuting operators
3. Quantum vs classical: ∂_a ρ involves commutators, not just gradients
4. Hilbert space structure: derivatives must respect tensor product structure
5. Each derivative step: parameter space → operator space → density matrix space
"""

import numpy as np
import pytest
from scipy.linalg import expm, eigh

from qig.exponential_family import QuantumExponentialFamily
from tests.fd_helpers import finite_difference_fisher
from tests.tolerance_framework import quantum_assert_close


# ============================================================================
# Diagnostic Tests: Simple Non-Commuting Cases
# ============================================================================


class TestSimpleNonCommuting:
    """
    Test BKM metric for the simplest non-commuting cases.
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

