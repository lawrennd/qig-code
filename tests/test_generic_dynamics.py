"""
Tests for GENERIC-aware dynamics integration.

This module tests the GenericDynamics class which tracks:
- Effective Hamiltonian H_eff along trajectories
- Diffusion operator D[ρ] (optional, expensive)
- Entropy production rate
- GENERIC structure preservation
"""

import numpy as np
import pytest

from qig.exponential_family import QuantumExponentialFamily
from qig.dynamics import GenericDynamics
from qig.pair_operators import bell_state_density_matrix
from qig.core import von_neumann_entropy, marginal_entropies
from qig.validation import ValidationReport


class TestGenericDynamicsInit:
    """Test GenericDynamics initialization and setup."""
    
    def test_initialization_with_structure_constants(self):
        """Test initialization with pre-computed structure constants."""
        from qig.structure_constants import compute_structure_constants
        
        d = 2
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
        
        f_abc = compute_structure_constants(exp_fam.operators)
        dyn = GenericDynamics(exp_fam, structure_constants=f_abc)
        
        assert dyn.f_abc is f_abc
        assert dyn.exp_family is exp_fam
        
    def test_initialization_computes_structure_constants(self):
        """Test initialization auto-computes structure constants if not provided."""
        d = 2
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
        
        dyn = GenericDynamics(exp_fam)
        
        assert dyn.f_abc is not None
        assert dyn.f_abc.shape == (exp_fam.n_params, exp_fam.n_params, exp_fam.n_params)


class TestGenericDecomposition:
    """Test computation of GENERIC decomposition."""
    
    def test_compute_decomposition_at_origin(self):
        """Test decomposition at LME origin."""
        d = 2
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
        dyn = GenericDynamics(exp_fam)
        
        theta_0 = np.zeros(exp_fam.n_params)
        decomp = dyn.compute_generic_decomposition(theta_0)
        
        # Check all expected keys present
        assert 'M' in decomp
        assert 'S' in decomp
        assert 'A' in decomp
        assert 'H_eff' in decomp
        assert 'eta' in decomp
        assert 'entropy_production' in decomp
        assert 'S_norm' in decomp
        assert 'A_norm' in decomp
        
    def test_decomposition_has_correct_shapes(self):
        """Test all components have correct shapes."""
        d = 2
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
        dyn = GenericDynamics(exp_fam)
        
        theta_0 = np.zeros(exp_fam.n_params)
        decomp = dyn.compute_generic_decomposition(theta_0)
        
        n_params = exp_fam.n_params
        D = exp_fam.D
        
        assert decomp['M'].shape == (n_params, n_params)
        assert decomp['S'].shape == (n_params, n_params)
        assert decomp['A'].shape == (n_params, n_params)
        assert decomp['H_eff'].shape == (D, D)
        assert decomp['eta'].shape == (n_params,)
        assert isinstance(decomp['entropy_production'], (int, float, np.number))
        
    def test_hamiltonian_is_hermitian(self):
        """Test H_eff is Hermitian."""
        d = 2
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
        dyn = GenericDynamics(exp_fam)
        
        theta_0 = 0.1 * np.random.randn(exp_fam.n_params)
        decomp = dyn.compute_generic_decomposition(theta_0)
        
        H_eff = decomp['H_eff']
        assert np.allclose(H_eff, H_eff.conj().T, atol=1e-10)


class TestReversibleDynamics:
    """Test reversible (Hamiltonian) part of dynamics."""
    
    def test_reversible_integration_runs(self):
        """Test reversible integration completes successfully."""
        d = 2
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
        dyn = GenericDynamics(exp_fam)
        
        theta_0 = 0.1 * np.random.randn(exp_fam.n_params)
        result = dyn.integrate_reversible(theta_0, (0.0, 0.1), n_points=20)
        
        assert result['success']
        assert len(result['time']) == 20
        assert result['theta'].shape == (20, exp_fam.n_params)
        
    def test_reversible_conserves_constraint(self):
        """Test reversible part preserves constraint (approximately)."""
        d = 2
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
        dyn = GenericDynamics(exp_fam)
        
        # Start near origin
        theta_0 = 0.05 * np.random.randn(exp_fam.n_params)
        result = dyn.integrate_reversible(theta_0, (0.0, 0.05), n_points=10)
        
        # Check constraint preservation
        rho_0 = exp_fam.rho_from_theta(theta_0)
        h_0 = marginal_entropies(rho_0, exp_fam.dims)
        C_0 = np.sum(h_0)
        
        theta_f = result['theta'][-1]
        rho_f = exp_fam.rho_from_theta(theta_f)
        h_f = marginal_entropies(rho_f, exp_fam.dims)
        C_f = np.sum(h_f)
        
        # Reversible part should approximately conserve (within integration error)
        assert abs(C_f - C_0) < 0.1  # Loose tolerance for short integration


class TestIrreversibleDynamics:
    """Test irreversible (dissipative) part of dynamics."""
    
    def test_irreversible_integration_runs(self):
        """Test irreversible integration completes successfully."""
        d = 2
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
        dyn = GenericDynamics(exp_fam)
        
        theta_0 = 0.1 * np.random.randn(exp_fam.n_params)
        result = dyn.integrate_irreversible(theta_0, (0.0, 0.1), n_points=20)
        
        assert result['success']
        assert len(result['time']) == 20
        assert result['theta'].shape == (20, exp_fam.n_params)
        
    def test_irreversible_increases_entropy(self):
        """Test irreversible part increases entropy."""
        d = 2
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
        dyn = GenericDynamics(exp_fam)
        
        # Start away from equilibrium
        theta_0 = 0.2 * np.random.randn(exp_fam.n_params)
        result = dyn.integrate_irreversible(theta_0, (0.0, 0.1), n_points=20)
        
        # Compute entropies along trajectory
        H_traj = []
        for theta in result['theta']:
            rho = exp_fam.rho_from_theta(theta)
            H_traj.append(von_neumann_entropy(rho))
        
        H_traj = np.array(H_traj)
        
        # Entropy should increase (or stay constant at maximum)
        # Allow small numerical violations
        entropy_changes = np.diff(H_traj)
        assert np.all(entropy_changes >= -1e-6), "Entropy should not decrease"


@pytest.mark.slow
class TestFullDynamicsWithMonitoring:
    """Test full GENERIC dynamics with structure monitoring."""
    
    def test_integration_with_monitoring_runs(self):
        """Test monitoring integration completes."""
        d = 2
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
        dyn = GenericDynamics(exp_fam)
        
        theta_0 = 0.1 * np.random.randn(exp_fam.n_params)
        result = dyn.integrate_with_monitoring(
            theta_0, (0.0, 0.1), n_points=10, compute_diffusion=False
        )
        
        assert result['success']
        assert 'H_eff' in result
        assert 'entropy_production' in result
        assert 'S_norm' in result
        assert 'A_norm' in result
        assert 'cumulative_entropy' in result
        
    def test_monitoring_tracks_generic_structure(self):
        """Test monitoring tracks GENERIC decomposition."""
        d = 2
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
        dyn = GenericDynamics(exp_fam)
        
        theta_0 = 0.1 * np.random.randn(exp_fam.n_params)
        result = dyn.integrate_with_monitoring(
            theta_0, (0.0, 0.1), n_points=10, compute_diffusion=False
        )
        
        # Check all Hamiltonians are Hermitian
        for H_eff in result['H_eff']:
            assert np.allclose(H_eff, H_eff.conj().T, atol=1e-10)
            
        # Check entropy production is non-negative
        assert np.all(result['entropy_production'] >= -1e-12)
        
        # Check cumulative entropy increases
        assert np.all(np.diff(result['cumulative_entropy']) >= -1e-12)
        
    def test_monitoring_with_diffusion_operator(self):
        """Test monitoring can compute diffusion operator (expensive)."""
        d = 2
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
        dyn = GenericDynamics(exp_fam)
        
        theta_0 = 0.05 * np.random.randn(exp_fam.n_params)
        result = dyn.integrate_with_monitoring(
            theta_0, (0.0, 0.05), n_points=5, compute_diffusion=True
        )
        
        assert 'D_rho' in result
        assert len(result['D_rho']) == 5
        
        # Check each D[ρ] is Hermitian and traceless
        for D_rho in result['D_rho']:
            assert np.allclose(D_rho, D_rho.conj().T, atol=1e-10)
            assert abs(np.trace(D_rho)) < 1e-10


@pytest.mark.slow
class TestEntropyProduction:
    """Test entropy production tracking."""
    
    def test_entropy_production_nonnegative(self):
        """Test entropy production is non-negative."""
        d = 2
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
        dyn = GenericDynamics(exp_fam)
        
        # Sample multiple random states
        for _ in range(10):
            theta = 0.2 * np.random.randn(exp_fam.n_params)
            decomp = dyn.compute_generic_decomposition(theta)
            
            # Entropy production should be non-negative
            assert decomp['entropy_production'] >= -1e-12
            
    def test_cumulative_entropy_increases(self):
        """Test cumulative entropy production increases monotonically."""
        d = 2
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
        dyn = GenericDynamics(exp_fam)
        
        theta_0 = 0.1 * np.random.randn(exp_fam.n_params)
        result = dyn.integrate_with_monitoring(
            theta_0, (0.0, 0.1), n_points=20, compute_diffusion=False
        )
        
        cumulative = result['cumulative_entropy']
        
        # Should be monotonically increasing
        assert np.all(np.diff(cumulative) >= -1e-12)
        
        # Should start near zero (cumulative sum of first element times dt)
        assert abs(cumulative[0]) < 1e-3  # Looser tolerance for numerical integration


@pytest.mark.slow
class TestConstraintPreservation:
    """Test constraint preservation in full dynamics."""
    
    def test_full_dynamics_preserves_constraint(self):
        """Test full GENERIC dynamics preserves constraint."""
        d = 2
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
        dyn = GenericDynamics(exp_fam)
        
        theta_0 = 0.1 * np.random.randn(exp_fam.n_params)
        result = dyn.integrate_with_monitoring(
            theta_0, (0.0, 0.1), n_points=20, compute_diffusion=False
        )
        
        # Should have constraint tracking from parent class
        constraint_traj = result['constraint']
        
        # Constraint should be approximately constant
        C_variation = np.std(constraint_traj)
        assert C_variation < 1e-4, f"Constraint variation: {C_variation}"


@pytest.mark.slow
class TestGenericStructurePreservation:
    """Test preservation of GENERIC structure properties."""
    
    def test_symmetric_antisymmetric_norms(self):
        """Test S and A norms are tracked correctly."""
        d = 2
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
        dyn = GenericDynamics(exp_fam)
        
        theta_0 = 0.1 * np.random.randn(exp_fam.n_params)
        result = dyn.integrate_with_monitoring(
            theta_0, (0.0, 0.1), n_points=10, compute_diffusion=False
        )
        
        # Both should be non-negative
        assert np.all(result['S_norm'] >= 0)
        assert np.all(result['A_norm'] >= 0)
        
        # Both should be finite
        assert np.all(np.isfinite(result['S_norm']))
        assert np.all(np.isfinite(result['A_norm']))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

