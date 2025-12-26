"""
Tests for Higham-style block-matrix Fréchet derivatives and cumulants.

This validates:
- `duhamel_derivative_block` matches the existing spectral Duhamel derivative.
- Hessian of ψ computed via 2nd Fréchet derivative matches `fisher_information()`.
- 3rd cumulant contraction computed via block method matches finite-difference method
  on small systems.
"""

import numpy as np
import pytest

from qig.exponential_family import QuantumExponentialFamily
from tests.tolerance_framework import quantum_assert_close, quantum_assert_symmetric


class TestBlockFrechet:
    def test_duhamel_block_matches_spectral_single_qubit(self):
        exp_fam = QuantumExponentialFamily(n_sites=1, d=2)
        np.random.seed(0)
        theta = 0.2 * np.random.randn(exp_fam.n_params)

        # Compare ∂ρ/∂θ_a across all parameters.
        for a in range(exp_fam.n_params):
            drho_block = exp_fam.rho_derivative(theta, a, method="duhamel_block")
            drho_spec = exp_fam.rho_derivative(theta, a, method="duhamel_spectral")

            quantum_assert_close(
                drho_block,
                drho_spec,
                "jacobian",
                err_msg=f"duhamel_block != duhamel_spectral for a={a}",
            )

    def test_hessian_block_matches_fisher_single_qubit(self):
        exp_fam = QuantumExponentialFamily(n_sites=1, d=2)
        theta = np.array([0.3, 0.5, 0.2])

        G = exp_fam.fisher_information(theta)
        H_block = exp_fam.psi_hessian_block(theta)

        quantum_assert_symmetric(H_block, "fisher_metric", err_msg="Hessian not symmetric")
        quantum_assert_close(
            H_block,
            G,
            "jacobian",
            err_msg="Block Hessian of ψ does not match fisher_information()",
        )

    def test_third_cumulant_contraction_block_matches_fd_single_qubit(self):
        exp_fam = QuantumExponentialFamily(n_sites=1, d=2)
        np.random.seed(1)
        theta = 0.2 * np.random.randn(exp_fam.n_params)

        T_fd = exp_fam.third_cumulant_contraction(theta, method="fd")
        T_block = exp_fam.third_cumulant_contraction(theta, method="block")

        quantum_assert_symmetric(T_block, "jacobian", err_msg="Third cumulant not symmetric")
        quantum_assert_close(
            T_block,
            T_fd,
            "jacobian",
            err_msg="Block 3rd cumulant contraction does not match FD",
        )


@pytest.mark.slow
class TestBlockFrechetQutrit:
    def test_duhamel_block_matches_spectral_single_qutrit(self):
        exp_fam = QuantumExponentialFamily(n_sites=1, d=3)
        np.random.seed(2)
        theta = 0.05 * np.random.randn(exp_fam.n_params)

        a = 0
        drho_block = exp_fam.rho_derivative(theta, a, method="duhamel_block")
        drho_spec = exp_fam.rho_derivative(theta, a, method="duhamel_spectral")

        quantum_assert_close(
            drho_block,
            drho_spec,
            "jacobian",
            err_msg="duhamel_block != duhamel_spectral for single qutrit (a=0)",
        )


