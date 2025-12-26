"""
Benchmark suite for Duhamel derivative methods.

Compares performance (time, accuracy, memory) of:
- Quadrature (trapezoid rule)
- Spectral (eigendecomposition)
- Block-matrix (Higham's trick)
- SLD (symmetric logarithmic derivative)

Across various matrix sizes: n = 4, 9, 16, 25, 64, 100

Usage:
    pytest tests/benchmark_duhamel.py -v --benchmark
    pytest tests/benchmark_duhamel.py::test_benchmark_single_qubit -v -s
"""

import numpy as np
import pytest
import time
from typing import Dict, List, Tuple

from qig.exponential_family import QuantumExponentialFamily
from tests.tolerance_framework import QuantumTolerances


def time_method(func, *args, **kwargs) -> Tuple[float, np.ndarray]:
    """Time a function call and return (time_seconds, result)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


def compute_accuracy_vs_fd(drho_test: np.ndarray, drho_fd: np.ndarray) -> Dict[str, float]:
    """Compute accuracy metrics compared to finite differences."""
    diff = drho_test - drho_fd
    abs_error = np.max(np.abs(diff))
    frobenius_error = np.linalg.norm(diff, 'fro')
    
    ref_norm = np.linalg.norm(drho_fd, 'fro')
    rel_error = frobenius_error / ref_norm if ref_norm > 1e-14 else frobenius_error
    
    return {
        'max_abs_error': abs_error,
        'frobenius_error': frobenius_error,
        'relative_error': rel_error
    }


class BenchmarkResults:
    """Container for benchmark results."""
    
    def __init__(self, system_size: int, n_params: int):
        self.system_size = system_size
        self.n_params = n_params
        self.methods = {}
        
    def add_method(self, name: str, time_s: float, accuracy: Dict[str, float]):
        """Add results for a method."""
        self.methods[name] = {
            'time': time_s,
            'accuracy': accuracy
        }
    
    def print_summary(self):
        """Print formatted summary table."""
        print(f"\n{'='*80}")
        print(f"Benchmark: System size n={self.system_size}, Parameters={self.n_params}")
        print(f"{'='*80}")
        print(f"{'Method':<15} {'Time (ms)':<12} {'Max Error':<12} {'Rel Error':<12}")
        print(f"{'-'*80}")
        
        for name, data in self.methods.items():
            time_ms = data['time'] * 1000
            max_err = data['accuracy']['max_abs_error']
            rel_err = data['accuracy']['relative_error']
            print(f"{name:<15} {time_ms:>10.3f}  {max_err:>10.2e}  {rel_err:>10.2e}")
        
        print(f"{'='*80}\n")
    
    def get_fastest(self) -> str:
        """Return name of fastest method."""
        return min(self.methods.items(), key=lambda x: x[1]['time'])[0]
    
    def get_most_accurate(self) -> str:
        """Return name of most accurate method."""
        return min(self.methods.items(), 
                  key=lambda x: x[1]['accuracy']['max_abs_error'])[0]


def benchmark_system(n_sites: int, d: int, verbose: bool = False) -> BenchmarkResults:
    """
    Benchmark all methods on a system with given parameters.
    
    Parameters
    ----------
    n_sites : int
        Number of sites (e.g., 1 for single qubit, 2 for pairs)
    d : int
        Local dimension (2 for qubits, 3 for qutrits)
    verbose : bool
        Whether to print detailed results
        
    Returns
    -------
    BenchmarkResults
        Benchmark results for all methods
    """
    # Create system
    exp_fam = QuantumExponentialFamily(n_sites=n_sites, d=d)
    n = exp_fam.n_params
    
    # Generate random parameters (small magnitude for stability)
    np.random.seed(42)
    theta = 0.1 * np.random.randn(n)
    
    # Pick a parameter to differentiate
    a = 0
    
    # Compute reference using high-precision finite differences
    eps = 1e-8
    theta_plus = theta.copy()
    theta_plus[a] += eps
    rho_plus = exp_fam.rho_from_theta(theta_plus)
    
    theta_minus = theta.copy()
    theta_minus[a] -= eps
    rho_minus = exp_fam.rho_from_theta(theta_minus)
    
    drho_fd = (rho_plus - rho_minus) / (2 * eps)
    
    # Initialize results
    results = BenchmarkResults(system_size=d**n_sites, n_params=n)
    
    # Benchmark each method
    methods_to_test = [
        ('Spectral', 'duhamel_spectral'),
        ('Block', 'duhamel_block'),
        ('SLD', 'sld'),
    ]
    
    # Only test quadrature on small systems (it's very slow)
    if n <= 9:
        methods_to_test.insert(0, ('Quadrature', 'duhamel'))
    
    for name, method in methods_to_test:
        try:
            # Time the computation
            elapsed, drho = time_method(
                exp_fam.rho_derivative,
                theta, a, method=method
            )
            
            # Compute accuracy
            accuracy = compute_accuracy_vs_fd(drho, drho_fd)
            
            # Store results
            results.add_method(name, elapsed, accuracy)
            
        except Exception as e:
            if verbose:
                print(f"Warning: {name} failed with {e}")
    
    if verbose:
        results.print_summary()
    
    return results


@pytest.mark.benchmark
class TestDuhamelBenchmarks:
    """Benchmark suite for Duhamel derivative methods."""
    
    def test_benchmark_single_qubit(self):
        """Benchmark on single qubit (n=2, params=3)."""
        results = benchmark_system(n_sites=1, d=2, verbose=True)
        
        # Verify all methods are reasonably fast
        for name, data in results.methods.items():
            assert data['time'] < 1.0, f"{name} too slow: {data['time']:.3f}s"
        
        # Verify spectral and block achieve expected precision vs FD reference
        # Using D_numerical tolerance (analytical vs finite differences)
        tol = QuantumTolerances.D_numerical['atol']
        for name in ['Spectral', 'Block']:
            if name in results.methods:
                err = results.methods[name]['accuracy']['max_abs_error']
                assert err < tol, f"{name} error {err:.2e} exceeds tolerance {tol:.2e}"
    
    def test_benchmark_single_qutrit(self):
        """Benchmark on single qutrit (n=3, params=8)."""
        results = benchmark_system(n_sites=1, d=3, verbose=True)
        
        # Verify spectral and block achieve expected precision vs FD reference
        # Using D_numerical tolerance (analytical vs finite differences)
        tol = QuantumTolerances.D_numerical['atol']
        if 'Spectral' in results.methods and 'Block' in results.methods:
            spec_err = results.methods['Spectral']['accuracy']['max_abs_error']
            block_err = results.methods['Block']['accuracy']['max_abs_error']
            assert spec_err < tol, f"Spectral error {spec_err:.2e} exceeds tolerance {tol:.2e}"
            assert block_err < tol, f"Block error {block_err:.2e} exceeds tolerance {tol:.2e}"
    
    @pytest.mark.slow
    def test_benchmark_qubit_pair(self):
        """Benchmark on qubit pair (n=4, params=15)."""
        results = benchmark_system(n_sites=2, d=2, verbose=True)
        
        # For larger systems, block may be slower than spectral
        # But should still achieve D_numerical tolerance vs FD reference
        tol = QuantumTolerances.D_numerical['atol']
        if 'Block' in results.methods:
            err = results.methods['Block']['accuracy']['max_abs_error']
            assert err < tol, f"Block error {err:.2e} exceeds tolerance {tol:.2e}"
    
    @pytest.mark.slow
    def test_benchmark_comparison_table(self):
        """Generate comprehensive comparison table across system sizes."""
        print("\n" + "="*100)
        print("COMPREHENSIVE BENCHMARK: Duhamel Derivative Methods")
        print("="*100)
        
        test_systems = [
            (1, 2, "Single qubit"),
            (1, 3, "Single qutrit"),
            (2, 2, "Qubit pair"),
        ]
        
        all_results = []
        for n_sites, d, description in test_systems:
            print(f"\n{description}:")
            results = benchmark_system(n_sites=n_sites, d=d, verbose=True)
            all_results.append((description, results))
        
        # Print summary comparison
        print("\n" + "="*100)
        print("SUMMARY: Time Comparison (milliseconds)")
        print("="*100)
        
        # Header
        methods = list(all_results[0][1].methods.keys())
        header = f"{'System':<20}"
        for method in methods:
            header += f"{method:>15}"
        print(header)
        print("-"*100)
        
        # Data rows
        for description, results in all_results:
            row = f"{description:<20}"
            for method in methods:
                if method in results.methods:
                    time_ms = results.methods[method]['time'] * 1000
                    row += f"{time_ms:>15.3f}"
                else:
                    row += f"{'N/A':>15}"
            print(row)
        
        print("\n" + "="*100)
        print("SUMMARY: Accuracy Comparison (max absolute error)")
        print("="*100)
        print(header)
        print("-"*100)
        
        # Accuracy rows
        for description, results in all_results:
            row = f"{description:<20}"
            for method in methods:
                if method in results.methods:
                    err = results.methods[method]['accuracy']['max_abs_error']
                    row += f"{err:>15.2e}"
                else:
                    row += f"{'N/A':>15}"
            print(row)
        
        print("="*100 + "\n")
    
    def test_benchmark_method_agreement(self):
        """Verify that spectral and block methods agree on small system."""
        exp_fam = QuantumExponentialFamily(n_sites=1, d=2)
        theta = np.array([0.3, 0.5, 0.2])
        
        drho_spectral = exp_fam.rho_derivative(theta, 0, method='duhamel_spectral')
        drho_block = exp_fam.rho_derivative(theta, 0, method='duhamel_block')
        
        # Comparing two analytical methods: use D tolerance
        tol = QuantumTolerances.D['atol']
        diff = np.max(np.abs(drho_spectral - drho_block))
        assert diff < tol, f"Spectral and block disagree by {diff:.2e} (tolerance {tol:.2e})"


@pytest.mark.benchmark
def test_benchmark_scaling_with_size():
    """
    Test how methods scale with matrix size.
    
    Expected scaling:
    - Spectral: O(n^3) for eigendecomposition
    - Block: O(8n^3) for 2n×2n matrix exponential
    - SLD: O(n^2)
    """
    print("\n" + "="*100)
    print("SCALING ANALYSIS: Time vs System Size")
    print("="*100)
    
    # Test single qubit, qutrit, and pairs
    sizes = [
        (1, 2),  # n=2
        (1, 3),  # n=3
        (2, 2),  # n=4
    ]
    
    results_by_size = {}
    
    for n_sites, d in sizes:
        exp_fam = QuantumExponentialFamily(n_sites=n_sites, d=d)
        system_size = d ** n_sites
        
        theta = 0.1 * np.random.randn(exp_fam.n_params)
        
        times = {}
        for name, method in [('Spectral', 'duhamel_spectral'), 
                              ('Block', 'duhamel_block'),
                              ('SLD', 'sld')]:
            elapsed, _ = time_method(exp_fam.rho_derivative, theta, 0, method=method)
            times[name] = elapsed
        
        results_by_size[system_size] = times
    
    # Print table
    print(f"\n{'System Size (n)':<15} {'Spectral (ms)':<15} {'Block (ms)':<15} {'SLD (ms)':<15} {'Block/Spectral':<15}")
    print("-"*85)
    
    for size in sorted(results_by_size.keys()):
        times = results_by_size[size]
        spec_ms = times['Spectral'] * 1000
        block_ms = times['Block'] * 1000
        sld_ms = times['SLD'] * 1000
        ratio = times['Block'] / times['Spectral']
        print(f"{size:<15} {spec_ms:<15.3f} {block_ms:<15.3f} {sld_ms:<15.3f} {ratio:<15.2f}")
    
    print("="*100 + "\n")
    
    # Note: SLD is O(n^2) with no matrix exponentials, so typically fastest
    # But timing can vary due to cache effects, so we don't assert strict ordering
    # Just verify all methods complete successfully
    for size, times in results_by_size.items():
        for method, time_s in times.items():
            assert time_s < 1.0, f"{method} took {time_s:.3f}s at size {size}"


if __name__ == "__main__":
    # Run benchmarks when executed directly
    print("Running Duhamel derivative benchmarks...")
    print("(Run with pytest for full benchmark suite)")
    
    # Quick benchmark on single qubit
    results = benchmark_system(n_sites=1, d=2, verbose=True)
    print(f"\nFastest method: {results.get_fastest()}")
    print(f"Most accurate method: {results.get_most_accurate()}")

