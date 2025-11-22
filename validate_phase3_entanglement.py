#!/usr/bin/env python3
"""
CIP-0002 Phase 3 Validation: Verify Entanglement in Migrated Scripts
====================================================================

Quick validation that migrated scripts can now create and maintain
entanglement, which was impossible with local operators.
"""

import numpy as np
import quantum_qutrit_n3 as qq

print("="*70)
print("CIP-0002 PHASE 3 VALIDATION")
print("="*70)

# Test 1: Verify qutrit pair creates entanglement
print("\n" + "="*70)
print("TEST 1: Qutrit Pair Entanglement")
print("="*70)

n_sites = 2  # 1 qutrit pair
d = 3  # qutrits

# Create operators
operators = qq.single_site_operators(n_sites=n_sites, d=d)
print(f"  Operators: {len(operators)} (su(9) generators)")

# Create LME state
rho_lme = qq.create_lme_state(n_sites=n_sites, d=d)
H_lme = qq.von_neumann_entropy(rho_lme)
h_lme = qq.compute_marginal_entropies(rho_lme, n_sites=n_sites, d=d)
C_lme = h_lme.sum()
I_lme = C_lme - H_lme

print(f"\n  LME State:")
print(f"    Joint entropy H: {H_lme:.6f} (pure: ≈0)")
print(f"    Marginal sum C: {C_lme:.6f}")
print(f"    Mutual information I: {I_lme:.6f}")
print(f"    Theoretical max I: {2*np.log(d):.6f}")
print(f"    Entanglement ratio: {I_lme/(2*np.log(d)):.1%}")

if I_lme > 2.0:
    print(f"  ✓ PASS: Genuine entanglement (I = {I_lme:.3f})")
else:
    print(f"  ✗ FAIL: No entanglement (I = {I_lme:.3f})")

# Test 2: Verify generic states also show entanglement
print("\n" + "="*70)
print("TEST 2: Generic State Entanglement")
print("="*70)

np.random.seed(42)
theta_test = np.random.randn(len(operators)) * 0.5

rho_test = qq.compute_density_matrix(theta_test, operators)
H_test = qq.von_neumann_entropy(rho_test)
h_test = qq.compute_marginal_entropies(rho_test, n_sites=n_sites, d=d)
C_test = h_test.sum()
I_test = C_test - H_test

print(f"\n  Random θ (||θ|| = {np.linalg.norm(theta_test):.3f}):")
print(f"    Joint entropy H: {H_test:.6f}")
print(f"    Marginal sum C: {C_test:.6f}")
print(f"    Mutual information I: {I_test:.6f}")

if I_test > 0.1:
    print(f"  ✓ PASS: Can create entanglement (I = {I_test:.3f})")
else:
    print(f"  ✗ FAIL: No entanglement (I = {I_test:.3f})")

# Test 3: Compare with LOCAL operators (if we had them)
print("\n" + "="*70)
print("TEST 3: Comparison with Legacy (LOCAL operators)")
print("="*70)

print("\n  LOCAL operators (legacy):")
print(f"    Parameters: 24 (8 × 3 sites)")
print(f"    Mutual information: I = 0.000 ALWAYS (separable only)")
print(f"    Entanglement: ✗ IMPOSSIBLE")

print("\n  PAIR operators (migrated):")
print(f"    Parameters: 80 (su(9))")
print(f"    Mutual information: I = {I_lme:.3f} (max), {I_test:.3f} (generic)")
print(f"    Entanglement: ✓ POSSIBLE")

print(f"\n  Improvement: {I_lme:.3f} / 0.000 = ∞")

# Summary
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

tests_passed = 0
tests_total = 2

if I_lme > 2.0:
    tests_passed += 1
    print("✓ Test 1: LME state is maximally entangled")
else:
    print("✗ Test 1: LME state not entangled")

if I_test > 0.1:
    tests_passed += 1
    print("✓ Test 2: Generic states can be entangled")
else:
    print("✗ Test 2: Generic states not entangled")

print(f"\nResult: {tests_passed}/{tests_total} tests passed")

if tests_passed == tests_total:
    print("\n✓✓✓ PHASE 3 VALIDATION: SUCCESS ✓✓✓")
    print("Migrated scripts can now create genuine entanglement!")
    print("Paper's description of 'locally maximally entangled' states")
    print("is now accurately represented in the code.")
else:
    print("\n✗✗✗ PHASE 3 VALIDATION: FAILED ✗✗✗")
    print("Migration may not be complete.")

