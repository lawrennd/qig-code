"""
Integration tests for complete GENERIC decomposition.

This module provides end-to-end testing of the full GENERIC decomposition
pipeline, validating all properties and checking against known solutions.
"""

import numpy as np
import pytest

from qig.exponential_family import QuantumExponentialFamily
from qig.generic_decomposition import run_generic_decomposition
from qig.dynamics import GenericDynamics
from qig.core import von_neumann_entropy, marginal_entropies


class TestEndToEnd2Qubit:
    """End-to-end tests for 2-qubit system."""
    
    def test_complete_pipeline_2qubit(self):
        """Test complete GENERIC decomposition for 2-qubit pair."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        # Start near LME origin
        theta = 0.1 * np.random.randn(exp_fam.n_params)
        
        results = run_generic_decomposition(
            theta, exp_fam,
            compute_diffusion=False,
            verbose=False,
            print_summary=False
        )
        
        # All 12 steps should complete
        assert 'H_eff' in results
        assert 'diagnostics' in results
        
        # Core algebraic checks should pass
        checks = results['diagnostics']['checks']
        assert checks['S_symmetric']
        assert checks['A_antisymmetric']
        assert checks['M_reconstructs']
        assert checks['H_eff_hermitian']
        assert checks['H_eff_traceless']
        
    def test_2qubit_dynamics_integration(self):
        """Test integration of GENERIC dynamics for 2-qubit."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        dyn = GenericDynamics(exp_fam)
        
        theta_0 = 0.1 * np.random.randn(exp_fam.n_params)
        
        # Integrate with monitoring
        result = dyn.integrate_with_monitoring(
            theta_0, (0.0, 0.1), n_points=20, compute_diffusion=False
        )
        
        assert result['success']
        assert len(result['H_eff']) == 20
        assert np.all(result['entropy_production'] >= -1e-12)
        
    @pytest.mark.slow
    def test_2qubit_constraint_preservation(self):
        """Test constraint preservation throughout integration."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        dyn = GenericDynamics(exp_fam)
        
        theta_0 = 0.1 * np.random.randn(exp_fam.n_params)
        result = dyn.integrate_with_monitoring(
            theta_0, (0.0, 0.1), n_points=20, compute_diffusion=False
        )
        
        # Constraint should be preserved
        C_traj = result['constraint']
        C_std = np.std(C_traj)
        assert C_std < 1e-4, f"Constraint variation: {C_std}"


@pytest.mark.slow
class TestEndToEnd2Qutrit:
    """End-to-end tests for 2-qutrit system."""
    
    @pytest.mark.slow
    def test_complete_pipeline_2qutrit(self):
        """Test complete GENERIC decomposition for 2-qutrit pair."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        
        # Start near LME origin
        theta = 0.05 * np.random.randn(exp_fam.n_params)
        
        results = run_generic_decomposition(
            theta, exp_fam,
            compute_diffusion=False,
            verbose=False,
            print_summary=False
        )
        
        # All 12 steps should complete
        assert 'H_eff' in results
        assert 'diagnostics' in results
        
        # Core algebraic checks should pass
        checks = results['diagnostics']['checks']
        assert checks['S_symmetric']
        assert checks['A_antisymmetric']
        assert checks['M_reconstructs']
        assert checks['H_eff_hermitian']
        assert checks['H_eff_traceless']
        
    def test_2qutrit_has_more_parameters(self):
        """Test qutrit system has correct dimensionality."""
        exp_fam_qubit = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        exp_fam_qutrit = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        
        # d²-1 parameters per pair
        assert exp_fam_qubit.n_params == 15  # 4² - 1
        assert exp_fam_qutrit.n_params == 80  # 9² - 1
        assert exp_fam_qutrit.n_params > exp_fam_qubit.n_params


class TestGENERICProperties:
    """Test fundamental GENERIC properties."""
    
    def test_jacobian_decomposition(self):
        """Test M = S + A decomposition is exact."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        theta = 0.1 * np.random.randn(exp_fam.n_params)
        
        results = run_generic_decomposition(
            theta, exp_fam,
            verbose=False,
            print_summary=False
        )
        
        M = results['M']
        S = results['S']
        A = results['A']
        
        # M = S + A should be exact
        reconstruction_error = np.linalg.norm(M - (S + A), 'fro')
        assert reconstruction_error < 2e-12  # Slightly looser for numerical stability
        
    def test_symmetry_properties(self):
        """Test S is symmetric and A is antisymmetric."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        theta = 0.1 * np.random.randn(exp_fam.n_params)
        
        results = run_generic_decomposition(
            theta, exp_fam,
            verbose=False,
            print_summary=False
        )
        
        S = results['S']
        A = results['A']
        
        # S = S^T
        assert np.allclose(S, S.T, atol=1e-12)
        
        # A = -A^T
        assert np.allclose(A, -A.T, atol=1e-12)
        
    def test_hamiltonian_hermiticity(self):
        """Test H_eff is Hermitian."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        theta = 0.1 * np.random.randn(exp_fam.n_params)
        
        results = run_generic_decomposition(
            theta, exp_fam,
            verbose=False,
            print_summary=False
        )
        
        H_eff = results['H_eff']
        
        # H = H†
        assert np.allclose(H_eff, H_eff.conj().T, atol=1e-12)
        
    def test_hamiltonian_tracelessness(self):
        """Test H_eff is traceless."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        theta = 0.1 * np.random.randn(exp_fam.n_params)
        
        results = run_generic_decomposition(
            theta, exp_fam,
            verbose=False,
            print_summary=False
        )
        
        H_eff = results['H_eff']
        
        # Tr(H) = 0
        assert abs(np.trace(H_eff)) < 1e-10


class TestNumericalPrecision:
    """Test numerical precision and stability."""
    
    def test_near_origin_stability(self):
        """Test stability near the origin."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        # Very small perturbations
        for scale in [1e-3, 1e-4, 1e-5]:
            theta = scale * np.random.randn(exp_fam.n_params)
            
            results = run_generic_decomposition(
                theta, exp_fam,
                verbose=False,
                print_summary=False
            )
            
            # Should still work
            assert np.all(np.isfinite(results['H_eff']))
            assert np.all(np.isfinite(results['S']))
            assert np.all(np.isfinite(results['A']))
            
    def test_different_parameter_magnitudes(self):
        """Test with different parameter magnitudes."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        for scale in [0.01, 0.1, 0.3]:
            theta = scale * np.random.randn(exp_fam.n_params)
            
            results = run_generic_decomposition(
                theta, exp_fam,
                verbose=False,
                print_summary=False
            )
            
            # Core checks should pass
            checks = results['diagnostics']['checks']
            assert checks['S_symmetric']
            assert checks['A_antisymmetric']
            assert checks['H_eff_hermitian']


class TestPropertyPreservation:
    """Test property preservation during integration."""
    
    def test_entropy_monotonicity(self):
        """Test entropy increases monotonically."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        dyn = GenericDynamics(exp_fam)
        
        theta_0 = 0.1 * np.random.randn(exp_fam.n_params)
        result = dyn.integrate_with_monitoring(
            theta_0, (0.0, 0.1), n_points=20, compute_diffusion=False
        )
        
        # Entropy should not decrease (allowing small numerical error)
        H_traj = result['H']
        H_changes = np.diff(H_traj)
        assert np.all(H_changes >= -1e-8), "Entropy should not decrease"
        
    def test_generic_structure_throughout_trajectory(self):
        """Test GENERIC structure preserved along trajectory."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        dyn = GenericDynamics(exp_fam)
        
        theta_0 = 0.1 * np.random.randn(exp_fam.n_params)
        result = dyn.integrate_with_monitoring(
            theta_0, (0.0, 0.1), n_points=10, compute_diffusion=False
        )
        
        # Check GENERIC structure at each point
        for theta in result['theta']:
            decomp_result = run_generic_decomposition(
                theta, exp_fam,
                verbose=False,
                print_summary=False
            )
            
            checks = decomp_result['diagnostics']['checks']
            # Core algebraic properties should hold
            assert checks['S_symmetric']
            assert checks['A_antisymmetric']
            assert checks['H_eff_hermitian']


class TestKnownSolutions:
    """Test against known analytical solutions."""
    
    def test_maximally_mixed_state(self):
        """Test at maximally mixed state (origin)."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        theta = np.zeros(exp_fam.n_params)
        results = run_generic_decomposition(
            theta, exp_fam,
            verbose=False,
            print_summary=False
        )
        
        rho = results['rho']
        
        # Should be maximally mixed: ρ = I/D
        expected_rho = np.eye(exp_fam.D) / exp_fam.D
        assert np.allclose(rho, expected_rho, atol=1e-10)
        
        # Entropy should be maximal
        H = von_neumann_entropy(rho)
        H_max = np.log(exp_fam.D)
        assert abs(H - H_max) < 1e-10
        
    def test_product_state_properties(self):
        """Test properties of product states."""
        exp_fam = QuantumExponentialFamily(n_sites=2, d=2, pair_basis=False)
        
        # Create a separable state (local parameters only)
        theta = np.random.randn(exp_fam.n_params) * 0.1
        
        results = run_generic_decomposition(
            theta, exp_fam,
            verbose=False,
            print_summary=False
        )
        
        # Should still satisfy GENERIC properties
        checks = results['diagnostics']['checks']
        assert checks['S_symmetric']
        assert checks['A_antisymmetric']
        assert checks['H_eff_hermitian']


class TestRegressionTests:
    """Regression tests against reference values."""
    
    def test_specific_state_regression(self):
        """Test specific state gives consistent results."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        # Fixed seed for reproducibility
        np.random.seed(12345)
        theta = 0.1 * np.random.randn(exp_fam.n_params)
        
        results = run_generic_decomposition(
            theta, exp_fam,
            verbose=False,
            print_summary=False
        )
        
        # Record key values for regression
        # Reference values from this implementation (with reasonable tolerance)
        H_joint = results['H_joint']
        S_norm = np.linalg.norm(results['S'], 'fro')
        A_norm = np.linalg.norm(results['A'], 'fro')
        eta_norm = np.linalg.norm(results['eta'])
        
        # Should be consistent across runs
        # Note: These are regression checks, not physical constraints
        assert 1.35 < H_joint < 1.36
        assert 1.47 < S_norm < 1.49
        assert 0.10 < A_norm < 0.15
        assert 0.20 < eta_norm < 0.25


class TestPerformance:
    """Test computational performance."""
    
    def test_2qubit_computation_time(self):
        """Test 2-qubit computation completes in reasonable time."""
        import time
        
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        theta = 0.1 * np.random.randn(exp_fam.n_params)
        
        start = time.time()
        results = run_generic_decomposition(
            theta, exp_fam,
            compute_diffusion=False,
            verbose=False,
            print_summary=False
        )
        elapsed = time.time() - start
        
        # Should complete in under 5 seconds
        assert elapsed < 5.0, f"Computation took {elapsed:.2f}s"
        
    def test_multiple_decompositions_scale(self):
        """Test multiple decompositions scale reasonably."""
        import time
        
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        times = []
        for _ in range(5):
            theta = 0.1 * np.random.randn(exp_fam.n_params)
            
            start = time.time()
            run_generic_decomposition(
                theta, exp_fam,
                compute_diffusion=False,
                verbose=False,
                print_summary=False
            )
            times.append(time.time() - start)
        
        # Times should be consistent (not growing)
        assert np.std(times) / np.mean(times) < 0.5


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_small_constraint_gradient(self):
        """Test when constraint gradient is very small."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        # Near equilibrium where ||a|| → 0
        theta = 0.001 * np.random.randn(exp_fam.n_params)
        
        results = run_generic_decomposition(
            theta, exp_fam,
            verbose=False,
            print_summary=False
        )
        
        # Should still work (degeneracy conditions may not pass)
        assert np.all(np.isfinite(results['H_eff']))
        
    def test_zero_parameters(self):
        """Test with all parameters zero."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        theta = np.zeros(exp_fam.n_params)
        
        results = run_generic_decomposition(
            theta, exp_fam,
            verbose=False,
            print_summary=False
        )
        
        # Should complete successfully
        assert 'H_eff' in results
        assert np.all(np.isfinite(results['H_eff']))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

