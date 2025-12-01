"""
Validation tests for analytic su(9) pair GENERIC decomposition.

This module tests symbolic/analytic forms for a qutrit PAIR using the full
su(9) Lie algebra (80 generators). Unlike the local basis, this can represent
entangled states and has non-zero antisymmetric part A.

Related to CIP-0007: Analytic Forms for S and A via Lie Algebra Structure
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from qig.exponential_family import QuantumExponentialFamily
from qig.structure_constants import compute_structure_constants


@pytest.fixture
def qutrit_pair_family():
    """Qutrit pair with full su(9) basis (80 parameters)."""
    return QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)


@pytest.fixture
def random_states(qutrit_pair_family, n_states=100):
    """Generate random test states in interior of state space."""
    np.random.seed(42)
    n_params = qutrit_pair_family.n_params
    
    # Generate states spanning parameter space
    theta_list = []
    for _ in range(n_states):
        theta = np.random.randn(n_params) * 0.5  # Scale to stay reasonable
        theta_list.append(theta)
    
    return theta_list


class TestSU9SymbolicInfrastructure:
    """Test symbolic su(9) infrastructure (Phase 1)."""
    
    def test_symbolic_su9_generators(self):
        """Test symbolic su(9) generators match numerical."""
        from qig.symbolic import symbolic_su9_generators
        from qig.exponential_family import QuantumExponentialFamily
        
        # Get symbolic and numerical generators
        symbolic_gen = symbolic_su9_generators()
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        numerical_gen = exp_fam.operators
        
        assert len(symbolic_gen) == 80, "Should have 80 generators"
        assert len(numerical_gen) == 80, "Should have 80 generators"
        
        # Check each generator
        for i, (sym, num) in enumerate(zip(symbolic_gen, numerical_gen)):
            # Convert symbolic to numpy
            sym_eval = np.array(sym.tolist(), dtype=complex)
            assert_allclose(sym_eval, num, atol=1e-14,
                           err_msg=f"Generator {i+1} mismatch")
    
    def test_symbolic_su9_structure_constants(self):
        """Test symbolic su(9) structure constants."""
        from qig.symbolic import symbolic_su9_structure_constants
        from qig.exponential_family import QuantumExponentialFamily
        
        # Get structure constants
        f_sym = symbolic_su9_structure_constants()
        
        # Verify properties
        assert f_sym.shape == (80, 80, 80), "Shape should be (80, 80, 80)"
        
        # Antisymmetry: f_abc = -f_bac
        for a in range(10):  # Sample a few
            for b in range(10):
                for c in range(10):
                    assert abs(f_sym[a, b, c] + f_sym[b, a, c]) < 1e-12, \
                        f"f_{{{a},{b},{c}}} should be antisymmetric in first two indices"
        
        # Jacobi identity (sample)
        generators = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True).operators
        for a, b, c in [(0, 1, 2), (1, 2, 3), (2, 3, 4)]:
            comm_ab_c = np.trace((generators[a] @ generators[b] - 
                                  generators[b] @ generators[a]) @ generators[c])
            expected = 2j * sum(f_sym[a, b, d] * np.trace(generators[d] @ generators[c])
                               for d in range(80))
            assert abs(comm_ab_c - expected) < 1e-10, "Jacobi identity violated"
    
    def test_su9_generator_properties(self):
        """Test su(9) generators are Hermitian, traceless, etc."""
        from qig.symbolic import verify_su9_generators
        
        results = verify_su9_generators()
        
        # Check all are Hermitian
        assert all(results['hermitian']), \
            "All generators should be Hermitian"
        
        # Check all are traceless
        assert all(results['traceless']), \
            "All generators should be traceless"
        
        # Check orthogonality
        assert results['orthogonal'], \
            "Generators should be orthogonal"
    
    def test_commutation_relations(self):
        """Verify [F_a, F_b] = 2i Σ_c f_abc F_c for su(9)."""
        from qig.symbolic import symbolic_su9_generators, symbolic_su9_structure_constants
        import sympy as sp
        
        generators = symbolic_su9_generators()
        f_abc = symbolic_su9_structure_constants()
        
        # Test a few commutators symbolically (expensive!)
        test_pairs = [(0, 1), (1, 2), (2, 3)]
        
        for a, b in test_pairs:
            F_a = generators[a]
            F_b = generators[b]
            
            # Compute commutator
            comm = F_a * F_b - F_b * F_a
            
            # Compute RHS: 2i Σ_c f_abc F_c
            rhs = sp.zeros(9, 9)
            for c in range(80):
                if abs(f_abc[a, b, c]) > 1e-14:
                    rhs += 2 * sp.I * f_abc[a, b, c] * generators[c]
            
            # Check equality (convert to numerical for efficiency)
            comm_num = np.array(comm.tolist(), dtype=complex)
            rhs_num = np.array(rhs.tolist(), dtype=complex)
            
            error = np.linalg.norm(comm_num - rhs_num)
            print(f"  [F_{a}, F_{b}]: error = {error:.2e}")
            assert error < 1e-10, f"Commutation relation violated for [{a}, {b}]"
    
    def test_sparsity_structure(self):
        """Check sparsity of su(9) structure constants."""
        from qig.symbolic import symbolic_su9_structure_constants
        
        f_abc = symbolic_su9_structure_constants()
        
        # Count non-zero elements
        n_nonzero = np.count_nonzero(np.abs(f_abc) > 1e-14)
        total = f_abc.size
        sparsity = 100 * (1 - n_nonzero / total)
        
        print(f"  Non-zero structure constants: {n_nonzero} / {total}")
        print(f"  Sparsity: {sparsity:.1f}%")
        
        # Should be very sparse (~99%)
        assert sparsity > 95, "Structure constants should be sparse"
        assert n_nonzero < 5000, "Should have < 5000 non-zero elements"


class TestSU9PairDensityMatrix:
    """Test symbolic density matrix for su(9) pair (Phase 2)."""
    
    @pytest.mark.skip(reason="Phase 2: Not yet implemented")
    def test_density_matrix(self):
        """Test symbolic density matrix for su(9) pair."""
        pass


class TestSU9ConstraintGeometry:
    """Test constraint geometry for su(9) pair (Phase 3)."""
    
    @pytest.mark.skip(reason="Phase 3: Not yet implemented")
    def test_constraint_gradient(self):
        """Test that Gθ ≠ -a for entangling operators."""
        pass


class TestSU9AntisymmetricPart:
    """Test antisymmetric part for su(9) pair (Phase 4)."""
    
    @pytest.mark.skip(reason="Phase 4: Not yet implemented")
    def test_antisymmetric_part_nonzero(self):
        """Test that A ≠ 0 for su(9) pair basis."""
        pass

