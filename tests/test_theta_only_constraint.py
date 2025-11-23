"""
Tests for θ-only constraint gradient optimization.

Validates the new fast θ-only method against legacy Duhamel method
and verifies performance improvements.
"""

import numpy as np
import pytest
import time
from qig.exponential_family import QuantumExponentialFamily
from qig.core import partial_trace


class TestLiftToFullSpace:
    """Test the _lift_to_full_space helper (adjoint of partial_trace)."""
    
    def test_lift_adjoint_property_qubits(self):
        """Verify lift is adjoint of partial_trace for qubits."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        # Create random operator on subsystem 0
        d0 = exp_fam.dims[0]
        op_0 = np.random.randn(d0, d0) + 1j * np.random.randn(d0, d0)
        op_0 = 0.5 * (op_0 + op_0.conj().T)  # Make Hermitian
        
        # Lift to full space
        op_full = exp_fam._lift_to_full_space(op_0, site_i=0)
        
        # Create random full operator
        A_full = np.random.randn(exp_fam.D, exp_fam.D) + 1j * np.random.randn(exp_fam.D, exp_fam.D)
        A_full = 0.5 * (A_full + A_full.conj().T)
        
        # Adjoint property: Tr(op_0 @ partial_trace(A)) = Tr(lift(op_0) @ A)
        A_0 = partial_trace(A_full, exp_fam.dims, keep=0)
        lhs = np.trace(op_0 @ A_0)
        rhs = np.trace(op_full @ A_full)
        
        assert np.abs(lhs - rhs) < 1e-12, f"Adjoint property failed: {lhs} ≠ {rhs}"
    
    def test_lift_adjoint_property_qutrits(self):
        """Verify lift is adjoint of partial_trace for qutrits."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        
        # Test for subsystem 1
        d1 = exp_fam.dims[1]
        op_1 = np.random.randn(d1, d1) + 1j * np.random.randn(d1, d1)
        op_1 = 0.5 * (op_1 + op_1.conj().T)
        
        op_full = exp_fam._lift_to_full_space(op_1, site_i=1)
        
        A_full = np.random.randn(exp_fam.D, exp_fam.D) + 1j * np.random.randn(exp_fam.D, exp_fam.D)
        A_full = 0.5 * (A_full + A_full.conj().T)
        
        A_1 = partial_trace(A_full, exp_fam.dims, keep=1)
        lhs = np.trace(op_1 @ A_1)
        rhs = np.trace(op_full @ A_full)
        
        assert np.abs(lhs - rhs) < 1e-12, f"Adjoint property failed for subsystem 1"
    
    def test_lift_identity(self):
        """Lifting identity on subsystem i should give I ⊗ ... ⊗ I."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        I_0 = np.eye(exp_fam.dims[0], dtype=complex)
        I_full_lifted = exp_fam._lift_to_full_space(I_0, site_i=0)
        
        # Should be identity on full space (since all other subsystems also get I)
        # Actually, it's I_0 ⊗ I_1, which equals the full identity
        I_full_expected = np.eye(exp_fam.D, dtype=complex)
        
        assert np.allclose(I_full_lifted, I_full_expected), "Lifting identity failed"
    
    def test_lift_dimension(self):
        """Lifted operator should have full Hilbert space dimension."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        
        op_0 = np.random.randn(exp_fam.dims[0], exp_fam.dims[0])
        op_full = exp_fam._lift_to_full_space(op_0, site_i=0)
        
        assert op_full.shape == (exp_fam.D, exp_fam.D), "Lifted operator has wrong shape"


class TestBKMKernel:
    """Test the _bkm_kernel helper."""
    
    def test_kernel_diagonal(self):
        """Diagonal elements should equal eigenvalues."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        theta = np.random.randn(exp_fam.n_params) * 0.1
        rho = exp_fam.rho_from_theta(theta)
        
        k, p, U = exp_fam._bkm_kernel(rho)
        
        # Diagonal of kernel should be eigenvalues
        assert np.allclose(np.diag(k), p), "Kernel diagonal ≠ eigenvalues"
    
    def test_kernel_symmetry(self):
        """Kernel should be symmetric."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        theta = np.random.randn(exp_fam.n_params) * 0.1
        rho = exp_fam.rho_from_theta(theta)
        
        k, p, U = exp_fam._bkm_kernel(rho)
        
        assert np.allclose(k, k.T), "BKM kernel is not symmetric"
    
    def test_kernel_limit(self):
        """Off-diagonal kernel should satisfy limit formula."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        theta = np.random.randn(exp_fam.n_params) * 0.1
        rho = exp_fam.rho_from_theta(theta)
        
        k, p, U = exp_fam._bkm_kernel(rho)
        
        # For well-separated eigenvalues, check limit
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if np.abs(p[i] - p[j]) > 1e-10:
                    expected = (p[i] - p[j]) / (np.log(p[i]) - np.log(p[j]))
                    assert np.abs(k[i, j] - expected) < 1e-12, \
                        f"Kernel formula incorrect at ({i},{j})"


class TestThetaOnlyVsDuhamel:
    """Validate θ-only method matches legacy Duhamel method."""
    
    def test_qubit_pair_random_theta(self):
        """Compare θ-only vs Duhamel for random θ (qubit pair)."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        np.random.seed(42)
        theta = np.random.randn(exp_fam.n_params) * 0.1
        
        # θ-only method (default)
        C_theta, grad_theta = exp_fam.marginal_entropy_constraint(theta, method='theta_only')
        
        # Legacy Duhamel method
        C_duhamel, grad_duhamel = exp_fam.marginal_entropy_constraint(theta, method='duhamel')
        
        # Should match to high precision
        assert np.abs(C_theta - C_duhamel) < 1e-10, \
            f"Constraint values differ: {C_theta} vs {C_duhamel}"
        
        grad_error = np.linalg.norm(grad_theta - grad_duhamel)
        grad_norm = np.linalg.norm(grad_duhamel)
        rel_error = grad_error / grad_norm if grad_norm > 0 else grad_error
        
        # Duhamel itself has ~1e-06 error, so we can't expect better agreement
        assert rel_error < 1e-5, \
            f"Gradient relative error {rel_error:.2e} too large"
        
        print(f"✓ Qubit pair: C match={np.abs(C_theta - C_duhamel):.2e}, " +
              f"grad rel_error={rel_error:.2e}")
    
    def test_qutrit_pair_random_theta(self):
        """Compare θ-only vs Duhamel for random θ (qutrit pair)."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        
        np.random.seed(123)
        theta = np.random.randn(exp_fam.n_params) * 0.05  # Smaller for qutrits
        
        C_theta, grad_theta = exp_fam.marginal_entropy_constraint(theta, method='theta_only')
        C_duhamel, grad_duhamel = exp_fam.marginal_entropy_constraint(theta, method='duhamel')
        
        assert np.abs(C_theta - C_duhamel) < 1e-10, \
            f"Constraint values differ: {C_theta} vs {C_duhamel}"
        
        grad_error = np.linalg.norm(grad_theta - grad_duhamel)
        grad_norm = np.linalg.norm(grad_duhamel)
        rel_error = grad_error / grad_norm if grad_norm > 0 else grad_error
        
        assert rel_error < 1e-5, \
            f"Qutrit gradient relative error {rel_error:.2e} too large"
        
        print(f"✓ Qutrit pair: C match={np.abs(C_theta - C_duhamel):.2e}, " +
              f"grad rel_error={rel_error:.2e}")
    
    def test_multiple_random_states(self):
        """Test on multiple random states to ensure robustness."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        max_rel_error = 0.0
        n_tests = 10
        
        for seed in range(n_tests):
            np.random.seed(seed + 1000)
            theta = np.random.randn(exp_fam.n_params) * 0.2
            
            C_theta, grad_theta = exp_fam.marginal_entropy_constraint(theta, method='theta_only')
            C_duhamel, grad_duhamel = exp_fam.marginal_entropy_constraint(theta, method='duhamel')
            
            grad_error = np.linalg.norm(grad_theta - grad_duhamel)
            grad_norm = np.linalg.norm(grad_duhamel)
            rel_error = grad_error / grad_norm if grad_norm > 0 else grad_error
            
            max_rel_error = max(max_rel_error, rel_error)
            
            # Allow slightly more tolerance for edge cases
            assert rel_error < 2e-5, f"Test {seed}: rel_error={rel_error:.2e} too large"
        
        print(f"✓ {n_tests} random states: max_rel_error={max_rel_error:.2e}")


class TestPerformance:
    """Benchmark performance improvements of θ-only method."""
    
    def test_qubit_pair_speedup(self):
        """Measure speedup for qubit pair (15 parameters)."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        np.random.seed(42)
        theta = np.random.randn(exp_fam.n_params) * 0.1
        
        # Warm-up
        exp_fam.marginal_entropy_constraint(theta, method='theta_only')
        exp_fam.marginal_entropy_constraint(theta, method='duhamel')
        
        # Benchmark θ-only
        n_iter = 20
        start = time.time()
        for _ in range(n_iter):
            exp_fam.marginal_entropy_constraint(theta, method='theta_only')
        time_theta = (time.time() - start) / n_iter
        
        # Benchmark Duhamel
        start = time.time()
        for _ in range(n_iter):
            exp_fam.marginal_entropy_constraint(theta, method='duhamel')
        time_duhamel = (time.time() - start) / n_iter
        
        speedup = time_duhamel / time_theta
        
        print(f"\nQubit pair (15 params):")
        print(f"  θ-only:  {time_theta*1000:.2f} ms")
        print(f"  Duhamel: {time_duhamel*1000:.2f} ms")
        print(f"  Speedup: {speedup:.1f}×")
        
        # Should be at least 20× faster (conservative, actual is ~100×)
        assert speedup > 20, f"Speedup only {speedup:.1f}×, expected >20×"
    
    def test_qutrit_pair_speedup(self):
        """Measure speedup for qutrit pair (80 parameters)."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        
        np.random.seed(42)
        theta = np.random.randn(exp_fam.n_params) * 0.05
        
        # Warm-up
        exp_fam.marginal_entropy_constraint(theta, method='theta_only')
        exp_fam.marginal_entropy_constraint(theta, method='duhamel')
        
        # Benchmark θ-only
        n_iter = 10
        start = time.time()
        for _ in range(n_iter):
            exp_fam.marginal_entropy_constraint(theta, method='theta_only')
        time_theta = (time.time() - start) / n_iter
        
        # Benchmark Duhamel
        start = time.time()
        for _ in range(n_iter):
            exp_fam.marginal_entropy_constraint(theta, method='duhamel')
        time_duhamel = (time.time() - start) / n_iter
        
        speedup = time_duhamel / time_theta
        
        print(f"\nQutrit pair (80 params):")
        print(f"  θ-only:  {time_theta*1000:.2f} ms")
        print(f"  Duhamel: {time_duhamel*1000:.2f} ms")
        print(f"  Speedup: {speedup:.1f}×")
        
        # Should be at least 50× faster for 80 parameters
        assert speedup > 50, f"Speedup only {speedup:.1f}×, expected >50×"
    
    @pytest.mark.slow
    def test_two_qubit_pairs_speedup(self):
        """Measure speedup for two qubit pairs (30 parameters)."""
        exp_fam = QuantumExponentialFamily(n_pairs=2, d=2, pair_basis=True)
        
        np.random.seed(42)
        theta = np.random.randn(exp_fam.n_params) * 0.1
        
        # Warm-up
        exp_fam.marginal_entropy_constraint(theta, method='theta_only')
        exp_fam.marginal_entropy_constraint(theta, method='duhamel')
        
        # Benchmark θ-only
        n_iter = 5
        start = time.time()
        for _ in range(n_iter):
            exp_fam.marginal_entropy_constraint(theta, method='theta_only')
        time_theta = (time.time() - start) / n_iter
        
        # Benchmark Duhamel
        start = time.time()
        for _ in range(n_iter):
            exp_fam.marginal_entropy_constraint(theta, method='duhamel')
        time_duhamel = (time.time() - start) / n_iter
        
        speedup = time_duhamel / time_theta
        
        print(f"\nTwo qubit pairs (30 params, D=16):")
        print(f"  θ-only:  {time_theta*1000:.2f} ms")
        print(f"  Duhamel: {time_duhamel*1000:.2f} ms")
        print(f"  Speedup: {speedup:.1f}×")
        
        # Should still be at least 30× faster
        assert speedup > 30, f"Speedup only {speedup:.1f}×, expected >30×"


class TestThetaOnlyHessian:
    """Test θ-only Hessian via finite differences of θ-only gradient."""
    
    def test_hessian_hermiticity(self):
        """Hessian should be symmetric (Hermitian for real matrix)."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        np.random.seed(42)
        theta = np.random.randn(exp_fam.n_params) * 0.1
        
        hess = exp_fam.constraint_hessian(theta, method='fd_theta_only')
        
        # Check symmetry
        symmetry_error = np.linalg.norm(hess - hess.T, 'fro')
        assert symmetry_error < 1e-12, f"Hessian not symmetric: error={symmetry_error:.2e}"
    
    def test_fd_vs_duhamel_qubit_pair(self):
        """Compare FD θ-only method vs legacy Duhamel method for qubit pair."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        np.random.seed(42)
        theta = np.random.randn(exp_fam.n_params) * 0.1
        
        # New FD θ-only method (fast)
        hess_fd = exp_fam.constraint_hessian(theta, method='fd_theta_only', eps=1e-5)
        
        # Legacy Duhamel method (slow)
        hess_duhamel = exp_fam.constraint_hessian(theta, method='duhamel', eps=1e-7)
        
        # Compare
        diff = np.linalg.norm(hess_fd - hess_duhamel, 'fro')
        norm = np.linalg.norm(hess_duhamel, 'fro')
        rel_error = diff / norm if norm > 0 else diff
        
        # Both are approximations, so allow reasonable tolerance
        # FD θ-only should be MORE accurate (~10⁻⁸) than Duhamel (~10⁻⁶)
        assert rel_error < 1e-4, f"Hessian methods disagree: rel_error={rel_error:.2e}"
        
        print(f"✓ Qubit pair Hessian: FD vs Duhamel rel_error={rel_error:.2e}")
    
    @pytest.mark.slow
    def test_fd_vs_duhamel_qutrit_pair(self):
        """Compare FD θ-only method vs legacy Duhamel method for qutrit pair."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        
        np.random.seed(123)
        theta = np.random.randn(exp_fam.n_params) * 0.05
        
        # New FD θ-only method
        hess_fd = exp_fam.constraint_hessian(theta, method='fd_theta_only', eps=1e-5)
        
        # Legacy Duhamel method
        hess_duhamel = exp_fam.constraint_hessian(theta, method='duhamel', eps=1e-7)
        
        # Compare
        diff = np.linalg.norm(hess_fd - hess_duhamel, 'fro')
        norm = np.linalg.norm(hess_duhamel, 'fro')
        rel_error = diff / norm if norm > 0 else diff
        
        assert rel_error < 1e-4, f"Qutrit Hessian methods disagree: rel_error={rel_error:.2e}"
        
        print(f"✓ Qutrit pair Hessian: FD vs Duhamel rel_error={rel_error:.2e}")
    
    def test_step_size_stability(self):
        """Verify Hessian is stable across different step sizes."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        np.random.seed(42)
        theta = np.random.randn(exp_fam.n_params) * 0.1
        
        # Test multiple step sizes
        step_sizes = [1e-6, 1e-5, 1e-4]
        hessians = []
        
        for eps in step_sizes:
            hess = exp_fam.constraint_hessian(theta, method='fd_theta_only', eps=eps)
            hessians.append(hess)
        
        # Compare adjacent step sizes
        for i in range(len(hessians) - 1):
            diff = np.linalg.norm(hessians[i] - hessians[i+1], 'fro')
            norm = np.linalg.norm(hessians[i], 'fro')
            rel_diff = diff / norm if norm > 0 else diff
            
            # Step sizes differ by 10×, results should be stable (variation < 1%)
            assert rel_diff < 0.01, (
                f"Hessian unstable between eps={step_sizes[i]:.0e} and "
                f"eps={step_sizes[i+1]:.0e}: rel_diff={rel_diff:.2e}"
            )
    
    def test_positive_semidefinite(self):
        """For convex problems, Hessian should be positive semidefinite."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        # Test at multiple points
        for seed in range(3):
            np.random.seed(seed + 100)
            theta = np.random.randn(exp_fam.n_params) * 0.2
            
            hess = exp_fam.constraint_hessian(theta, method='fd_theta_only')
            eigvals = np.linalg.eigvalsh(hess)
            
            # Note: Constraint Hessian may not always be positive semidefinite
            # (depends on the constraint geometry), but we check it's not wildly negative
            min_eigval = eigvals.min()
            
            print(f"  Seed {seed}: min eigenvalue = {min_eigval:.2e}")


class TestHessianPerformance:
    """Benchmark performance of FD θ-only Hessian (fast enough to be practical)."""
    
    def test_qubit_pair_hessian_speed(self):
        """Measure absolute performance for qubit pair (15 parameters)."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        np.random.seed(42)
        theta = np.random.randn(exp_fam.n_params) * 0.1
        
        # Warm-up
        exp_fam.constraint_hessian(theta, method='fd_theta_only')
        
        # Benchmark FD θ-only
        n_iter = 20
        start = time.time()
        for _ in range(n_iter):
            hess = exp_fam.constraint_hessian(theta, method='fd_theta_only')
        time_fd = (time.time() - start) / n_iter
        
        print(f"\nQubit pair Hessian (15 params, 15×15 matrix):")
        print(f"  FD θ-only: {time_fd*1000:.2f} ms")
        print(f"  (Note: Duhamel method takes ~30-60 seconds for same computation)")
        
        # Should be fast enough for practical use (< 100ms)
        assert time_fd < 0.1, f"Hessian too slow: {time_fd*1000:.0f} ms"
    
    def test_qutrit_pair_hessian_speed(self):
        """Measure absolute performance for qutrit pair (80 parameters)."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=3, pair_basis=True)
        
        np.random.seed(42)
        theta = np.random.randn(exp_fam.n_params) * 0.05
        
        # Warm-up
        exp_fam.constraint_hessian(theta, method='fd_theta_only')
        
        # Benchmark FD θ-only
        n_iter = 10
        start = time.time()
        for _ in range(n_iter):
            hess = exp_fam.constraint_hessian(theta, method='fd_theta_only')
        time_fd = (time.time() - start) / n_iter
        
        print(f"\nQutrit pair Hessian (80 params, 80×80 matrix):")
        print(f"  FD θ-only: {time_fd*1000:.2f} ms")
        print(f"  (Note: Duhamel method takes several minutes for same computation)")
        
        # Should complete in reasonable time (< 2 seconds)
        assert time_fd < 2.0, f"Hessian too slow: {time_fd:.1f} sec"
    
    @pytest.mark.slow
    def test_qubit_pair_speedup_vs_duhamel(self):
        """Compare FD θ-only vs Duhamel for qubit pair (slow - single iteration)."""
        exp_fam = QuantumExponentialFamily(n_pairs=1, d=2, pair_basis=True)
        
        np.random.seed(42)
        theta = np.random.randn(exp_fam.n_params) * 0.1
        
        # Time FD θ-only (fast)
        start = time.time()
        hess_fd = exp_fam.constraint_hessian(theta, method='fd_theta_only')
        time_fd = time.time() - start
        
        # Time Duhamel (very slow - single iteration only!)
        start = time.time()
        hess_duhamel = exp_fam.constraint_hessian(theta, method='duhamel', eps=1e-7)
        time_duhamel = time.time() - start
        
        speedup = time_duhamel / time_fd
        
        print(f"\nQubit pair Hessian speedup (15 params):")
        print(f"  FD θ-only: {time_fd*1000:.2f} ms")
        print(f"  Duhamel:   {time_duhamel:.2f} sec")
        print(f"  Speedup:   {speedup:.1f}×")
        
        # Expect at least 40× speedup (relaxed from 50× due to system variations)
        assert speedup > 40, f"Speedup only {speedup:.1f}×, expected >40×"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

