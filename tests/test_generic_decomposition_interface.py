"""
Tests for high-level GENERIC decomposition interface.

This module tests the GenericDecomposition class and run_generic_decomposition
convenience function.
"""

import numpy as np
import pytest

from qig.exponential_family import QuantumExponentialFamily
from qig.generic_decomposition import GenericDecomposition, run_generic_decomposition


class TestGenericDecompositionClass:
    """Test GenericDecomposition class."""
    
    def test_initialization(self):
        """Test GenericDecomposition initialization."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        decomp = GenericDecomposition(exp_fam)
        
        assert decomp.exp_fam is exp_fam
        assert decomp.method == 'duhamel'
        assert decomp.compute_diffusion == False
        assert decomp.f_abc is not None
        
    def test_compute_all_basic(self):
        """Test complete computation runs successfully."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        decomp = GenericDecomposition(exp_fam, compute_diffusion=False)
        
        theta = 0.1 * np.random.randn(exp_fam.n_params)
        results = decomp.compute_all(theta, verbose=False)
        
        # Check all expected keys present
        expected_keys = [
            'theta', 'rho', 'psi', 'mu', 'G',
            'H_joint', 'h', 'C', 'a', 'a_norm',
            'nu', 'grad_nu', 'M', 'S', 'A',
            'eta', 'eta_info', 'H_eff', 'D_rho', 'diagnostics'
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
            
    def test_compute_all_with_diffusion(self):
        """Test computation with diffusion operator."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        decomp = GenericDecomposition(exp_fam, compute_diffusion=True)
        
        theta = 0.05 * np.random.randn(exp_fam.n_params)
        results = decomp.compute_all(theta, verbose=False)
        
        assert results['D_rho'] is not None
        assert results['D_rho'].shape == (exp_fam.D, exp_fam.D)
        
    def test_results_have_correct_shapes(self):
        """Test all results have expected shapes."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        decomp = GenericDecomposition(exp_fam)
        
        theta = 0.1 * np.random.randn(exp_fam.n_params)
        results = decomp.compute_all(theta, verbose=False)
        
        n = exp_fam.n_params
        D = exp_fam.D
        
        assert results['theta'].shape == (n,)
        assert results['rho'].shape == (D, D)
        assert isinstance(results['psi'], (int, float, np.number))
        assert results['mu'].shape == (n,)
        assert results['G'].shape == (n, n)
        assert results['a'].shape == (n,)
        assert results['grad_nu'].shape == (n,)
        assert results['M'].shape == (n, n)
        assert results['S'].shape == (n, n)
        assert results['A'].shape == (n, n)
        assert results['eta'].shape == (n,)
        assert results['H_eff'].shape == (D, D)


class TestDiagnostics:
    """Test diagnostic computation."""
    
    def test_diagnostics_computed(self):
        """Test diagnostics are computed."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        decomp = GenericDecomposition(exp_fam)
        
        theta = 0.1 * np.random.randn(exp_fam.n_params)
        results = decomp.compute_all(theta, verbose=False)
        
        assert 'diagnostics' in results
        diag = results['diagnostics']
        
        # Check expected diagnostic keys
        expected_keys = [
            'S_symmetry_error', 'A_antisymmetry_error', 'M_reconstruction_error',
            'H_eff_hermiticity_error', 'H_eff_trace',
            'degeneracy_S_condition', 'degeneracy_A_condition',
            'constraint_tangency', 'checks', 'all_checks_pass'
        ]
        for key in expected_keys:
            assert key in diag, f"Missing diagnostic: {key}"
            
    def test_algebraic_checks_pass(self):
        """Test core algebraic properties pass."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        decomp = GenericDecomposition(exp_fam)
        
        theta = 0.1 * np.random.randn(exp_fam.n_params)
        results = decomp.compute_all(theta, verbose=False)
        
        checks = results['diagnostics']['checks']
        
        # These should always pass (algebraic properties)
        assert checks['S_symmetric'], "S should be symmetric"
        assert checks['A_antisymmetric'], "A should be antisymmetric"
        assert checks['M_reconstructs'], "M = S + A should hold"
        assert checks['H_eff_hermitian'], "H_eff should be Hermitian"
        assert checks['H_eff_traceless'], "H_eff should be traceless"
        
    def test_diagnostics_with_diffusion(self):
        """Test diagnostics include diffusion operator checks."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        decomp = GenericDecomposition(exp_fam, compute_diffusion=True)
        
        theta = 0.05 * np.random.randn(exp_fam.n_params)
        results = decomp.compute_all(theta, verbose=False)
        
        checks = results['diagnostics']['checks']
        
        assert 'D_hermitian' in checks
        assert 'D_traceless' in checks
        assert 'entropy_production_positive' in checks
        
        # Check these pass
        assert checks['D_hermitian'], "D[ρ] should be Hermitian"
        assert checks['D_traceless'], "D[ρ] should be traceless"


class TestPrintSummary:
    """Test summary printing."""
    
    def test_print_summary_without_results(self, capsys):
        """Test print_summary before computation."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        decomp = GenericDecomposition(exp_fam)
        
        decomp.print_summary()
        captured = capsys.readouterr()
        assert "No results yet" in captured.out
        
    def test_print_summary_basic(self, capsys):
        """Test basic summary printing."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        decomp = GenericDecomposition(exp_fam)
        
        theta = 0.1 * np.random.randn(exp_fam.n_params)
        decomp.compute_all(theta, verbose=False)
        
        decomp.print_summary(detailed=False)
        captured = capsys.readouterr()
        
        assert "GENERIC Decomposition Summary" in captured.out
        assert "State:" in captured.out
        assert "GENERIC Decomposition:" in captured.out
        assert "Diagnostics:" in captured.out
        
    def test_print_summary_detailed(self, capsys):
        """Test detailed summary printing."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        decomp = GenericDecomposition(exp_fam)
        
        theta = 0.1 * np.random.randn(exp_fam.n_params)
        decomp.compute_all(theta, verbose=False)
        
        decomp.print_summary(detailed=True)
        captured = capsys.readouterr()
        
        assert "Detailed Diagnostics:" in captured.out


class TestConvenienceFunction:
    """Test run_generic_decomposition convenience function."""
    
    def test_convenience_function_basic(self):
        """Test convenience function runs."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        theta = 0.1 * np.random.randn(exp_fam.n_params)
        
        results = run_generic_decomposition(
            theta, exp_fam,
            verbose=False,
            print_summary=False
        )
        
        assert isinstance(results, dict)
        assert 'H_eff' in results
        assert 'diagnostics' in results
        
    def test_convenience_function_with_options(self):
        """Test convenience function with various options."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        theta = 0.05 * np.random.randn(exp_fam.n_params)
        
        results = run_generic_decomposition(
            theta, exp_fam,
            method='sld',
            compute_diffusion=False,
            verbose=False,
            print_summary=False
        )
        
        assert results['D_rho'] is None
        
    def test_convenience_function_prints_summary(self, capsys):
        """Test convenience function prints summary when requested."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        theta = 0.1 * np.random.randn(exp_fam.n_params)
        
        run_generic_decomposition(
            theta, exp_fam,
            verbose=False,
            print_summary=True
        )
        
        captured = capsys.readouterr()
        assert "GENERIC Decomposition Summary" in captured.out


class TestOriginState:
    """Test decomposition at LME origin."""
    
    def test_origin_state(self):
        """Test decomposition at θ=0 (LME origin)."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        decomp = GenericDecomposition(exp_fam)
        
        theta = np.zeros(exp_fam.n_params)
        results = decomp.compute_all(theta, verbose=False)
        
        # At origin, certain properties should hold
        # ψ(0) = log(Tr[exp(0)]) = log(D) for normalized maximally mixed state
        expected_psi = np.log(exp_fam.D)
        assert abs(results['psi'] - expected_psi) < 1e-10
        
        # Fisher info should be well-defined
        assert np.all(np.isfinite(results['G']))
        
        # Check algebraic properties
        checks = results['diagnostics']['checks']
        assert checks['S_symmetric']
        assert checks['A_antisymmetric']
        assert checks['H_eff_hermitian']


class TestMultipleStates:
    """Test decomposition for multiple states."""
    
    def test_multiple_random_states(self):
        """Test decomposition works for multiple random states."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        decomp = GenericDecomposition(exp_fam)
        
        # Test 5 random states
        for _ in range(5):
            theta = 0.2 * np.random.randn(exp_fam.n_params)
            results = decomp.compute_all(theta, verbose=False)
            
            # All algebraic checks should pass
            checks = results['diagnostics']['checks']
            assert checks['S_symmetric']
            assert checks['A_antisymmetric']
            assert checks['M_reconstructs']
            assert checks['H_eff_hermitian']
            assert checks['H_eff_traceless']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

