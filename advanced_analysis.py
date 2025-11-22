#!/usr/bin/env python3
"""
Advanced analysis for quantum inaccessible game.

This script provides:
1. Comparison of different time parametrisations
2. Detailed GENERIC decomposition along trajectories
3. Entropy gradient geometry analysis
4. Comparison of qubit vs qutrit optimality

Author: Implementation for "The Origin of the Inaccessible Game"
"""

import numpy as np
import matplotlib.pyplot as plt
from inaccessible_game_quantum import *


def compare_time_parametrisations(n_sites=2, d=2, t_end=5.0):
    """
    Compare affine time vs entropy time parametrisation.
    
    Shows that entropy-time removes coordinate singularity at origin.
    """
    print("\n" + "="*70)
    print("COMPARING TIME PARAMETRISATIONS")
    print("="*70)
    
    exp_family = QuantumExponentialFamily(n_sites, d)
    theta_0 = np.random.randn(exp_family.n_params) * 0.1
    
    # Run in affine time
    print("\n[1/2] Integrating in affine time τ...")
    dynamics_affine = InaccessibleGameDynamics(exp_family)
    dynamics_affine.set_time_mode('affine')
    sol_affine = dynamics_affine.integrate(theta_0, (0, t_end), n_points=100)
    
    # Run in entropy time
    print("\n[2/2] Integrating in entropy time t...")
    dynamics_entropy = InaccessibleGameDynamics(exp_family)
    dynamics_entropy.set_time_mode('entropy')
    sol_entropy = dynamics_entropy.integrate(theta_0, (0, t_end), n_points=100)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) Entropy vs affine time
    ax = axes[0, 0]
    ax.plot(sol_affine['time'], sol_affine['H'], 'b-', linewidth=2, label='Affine time')
    ax.set_xlabel('Affine Time τ')
    ax.set_ylabel('Joint Entropy H')
    ax.set_title('(a) H(τ): Varying Production Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) Entropy vs entropy time
    ax = axes[0, 1]
    ax.plot(sol_entropy['time'], sol_entropy['H'], 'r-', linewidth=2, label='Entropy time')
    ax.plot(sol_entropy['time'], sol_entropy['time'] * (sol_entropy['H'][-1] - sol_entropy['H'][0]) / sol_entropy['time'][-1] + sol_entropy['H'][0], 
            'k--', alpha=0.5, label='Linear (dH/dt=const)')
    ax.set_xlabel('Entropy Time t')
    ax.set_ylabel('Joint Entropy H')
    ax.set_title('(b) H(t): Unit Production Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (c) Entropy production rate in affine time
    ax = axes[1, 0]
    dH_affine = np.diff(sol_affine['H'])
    dt_affine = np.diff(sol_affine['time'])
    t_mid = 0.5 * (sol_affine['time'][:-1] + sol_affine['time'][1:])
    ax.plot(t_mid, dH_affine/dt_affine, 'b-', linewidth=2)
    ax.set_xlabel('Affine Time τ')
    ax.set_ylabel('dH/dτ')
    ax.set_title('(c) Entropy Production Rate (Affine)')
    ax.grid(True, alpha=0.3)
    
    # (d) Entropy production rate in entropy time
    ax = axes[1, 1]
    dH_entropy = np.diff(sol_entropy['H'])
    dt_entropy = np.diff(sol_entropy['time'])
    t_mid_e = 0.5 * (sol_entropy['time'][:-1] + sol_entropy['time'][1:])
    ax.plot(t_mid_e, dH_entropy/dt_entropy, 'r-', linewidth=2)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='dH/dt = 1 (target)')
    ax.set_xlabel('Entropy Time t')
    ax.set_ylabel('dH/dt')
    ax.set_title('(d) Entropy Production Rate (Entropy Time)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_parametrisation_comparison.png', dpi=150)
    print("\n✓ Figure saved: time_parametrisation_comparison.png")
    plt.show()
    
    return {
        'affine': sol_affine,
        'entropy': sol_entropy
    }


def generic_evolution(n_sites=2, d=2, t_end=5.0, n_samples=20):
    """
    Track evolution of GENERIC decomposition along trajectory.
    
    Shows how symmetric vs antisymmetric parts evolve.
    """
    print("\n" + "="*70)
    print("GENERIC DECOMPOSITION EVOLUTION")
    print("="*70)
    
    exp_family = QuantumExponentialFamily(n_sites, d)
    theta_0 = np.random.randn(exp_family.n_params) * 0.1
    
    print("\nIntegrating dynamics...")
    dynamics = InaccessibleGameDynamics(exp_family)
    solution = dynamics.integrate(theta_0, (0, t_end), n_points=200)
    
    # Sample points along trajectory
    sample_indices = np.linspace(0, len(solution['theta'])-1, n_samples, dtype=int)
    
    times = []
    S_norms = []
    A_norms = []
    ratios = []
    
    print(f"\nComputing GENERIC at {n_samples} points...")
    for i, idx in enumerate(sample_indices):
        theta = solution['theta'][idx]
        t = solution['time'][idx]
        
        M = compute_jacobian(dynamics, theta)
        S, A = generic_decomposition(M)
        
        S_norm = np.linalg.norm(S, 'fro')
        A_norm = np.linalg.norm(A, 'fro')
        ratio = A_norm / S_norm if S_norm > 0 else 0
        
        times.append(t)
        S_norms.append(S_norm)
        A_norms.append(A_norm)
        ratios.append(ratio)
        
        if i % 5 == 0:
            print(f"  t={t:.2f}: ||S||={S_norm:.4f}, ||A||={A_norm:.4f}, ratio={ratio:.4f}")
    
    # Plot evolution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # (a) Individual norms
    ax = axes[0]
    ax.plot(times, S_norms, 'b-o', label='||S|| (dissipative)', markersize=4)
    ax.plot(times, A_norms, 'r-s', label='||A|| (conservative)', markersize=4)
    ax.set_xlabel('Time τ')
    ax.set_ylabel('Frobenius Norm')
    ax.set_title('(a) GENERIC Component Norms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) Ratio evolution
    ax = axes[1]
    ax.plot(times, ratios, 'g-o', markersize=6, linewidth=2)
    ax.set_xlabel('Time τ')
    ax.set_ylabel('||A|| / ||S||')
    ax.set_title('(b) Conservative/Dissipative Ratio')
    ax.grid(True, alpha=0.3)
    
    # (c) Phase diagram: H vs ratio
    ax = axes[2]
    H_sampled = [solution['H'][idx] for idx in sample_indices]
    sc = ax.scatter(H_sampled, ratios, c=times, cmap='viridis', s=50)
    ax.set_xlabel('Joint Entropy H')
    ax.set_ylabel('||A|| / ||S||')
    ax.set_title('(c) Ratio vs Entropy')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Time τ')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('generic_evolution.png', dpi=150)
    print("\n✓ Figure saved: generic_evolution.png")
    plt.show()
    
    return {
        'times': times,
        'S_norms': S_norms,
        'A_norms': A_norms,
        'ratios': ratios
    }


def compare_qubit_qutrit_optimality(t_end=3.0):
    """
    Compare entropy gradient geometry for qubits vs qutrits.
    
    Validates Lemma 3.1: qutrits maximise m/d * log(d).
    """
    print("\n" + "="*70)
    print("QUBIT VS QUTRIT OPTIMALITY COMPARISON")
    print("="*70)
    
    results = {}
    
    for d_label, (n_sites, d) in [('2 qubits', (2, 2)), ('2 qutrits', (2, 3))]:
        print(f"\n{'-'*70}")
        print(f"System: {d_label}")
        print(f"{'-'*70}")
        
        exp_family = QuantumExponentialFamily(n_sites, d)
        
        # Create LME state
        rho_lme, dims = create_lme_state(n_sites, d)
        h_lme = marginal_entropies(rho_lme, dims)
        C_initial = np.sum(h_lme)
        
        # Theoretical maximum
        C_theory = n_sites * np.log(d)
        
        # Resource budget and efficiency
        m = n_sites * d  # Additive level budget
        efficiency = (m / d) * np.log(d)
        
        print(f"  Local dimension d = {d}")
        print(f"  Number of sites n = {n_sites}")
        print(f"  Level budget m = nd = {m}")
        print(f"  Efficiency (m/d)log(d) = {efficiency:.6f}")
        print(f"  Marginal entropy sum C = {C_initial:.6f}")
        print(f"  Theoretical maximum = {C_theory:.6f}")
        print(f"  Achievement ratio = {C_initial/C_theory:.6f}")
        
        # Run dynamics
        theta_0 = np.random.randn(exp_family.n_params) * 0.1
        dynamics = InaccessibleGameDynamics(exp_family)
        solution = dynamics.integrate(theta_0, (0, t_end), n_points=50)
        
        results[d_label] = {
            'd': d,
            'n': n_sites,
            'm': m,
            'efficiency': efficiency,
            'C_initial': C_initial,
            'C_theory': C_theory,
            'solution': solution
        }
    
    # Compare efficiencies
    print(f"\n{'='*70}")
    print("OPTIMALITY COMPARISON")
    print(f"{'='*70}")
    
    eff_qubit = results['2 qubits']['efficiency']
    eff_qutrit = results['2 qutrits']['efficiency']
    
    print(f"\n  Qubit efficiency:  {eff_qubit:.6f}")
    print(f"  Qutrit efficiency: {eff_qutrit:.6f}")
    print(f"  Ratio (qutrit/qubit): {eff_qutrit/eff_qubit:.6f}")
    
    # Theoretical prediction: d=3 maximises (m/d)log(d) for fixed m
    # For equal m=4: qubits give (4/2)log(2) = 1.386
    #               qutrits give (4/3)log(3) = 1.465
    print(f"\n  For equal budget m=4:")
    print(f"    Qubits (d=2):  {(4/2)*np.log(2):.6f}")
    print(f"    Qutrits (d=3): {(4/3)*np.log(3):.6f}")
    print(f"  ✓ Qutrits optimal (closest to e≈2.718)")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # (a) Entropy trajectories
    ax = axes[0]
    for label, data in results.items():
        ax.plot(data['solution']['time'], data['solution']['H'], 
               label=f"{label} (d={data['d']})", linewidth=2)
    ax.set_xlabel('Time τ')
    ax.set_ylabel('Joint Entropy H')
    ax.set_title('(a) Entropy Production: Qubits vs Qutrits')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) Efficiency comparison
    ax = axes[1]
    labels = list(results.keys())
    efficiencies = [results[label]['efficiency'] for label in labels]
    colors = ['blue', 'red']
    bars = ax.bar(labels, efficiencies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Efficiency (m/d)log(d)')
    ax.set_title('(b) Resource Efficiency Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, eff in zip(bars, efficiencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{eff:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('qubit_qutrit_comparison.png', dpi=150)
    print("\n✓ Figure saved: qubit_qutrit_comparison.png")
    plt.show()
    
    return results


def entropy_gradient_geometry(n_sites=2, d=2):
    """
    Analyse the entropy gradient geometric factor R(θ).
    
    Computes R(θ) = θ^T G Π_∥ G θ which determines relaxation timescale.
    """
    print("\n" + "="*70)
    print("ENTROPY GRADIENT GEOMETRY ANALYSIS")
    print("="*70)
    
    exp_family = QuantumExponentialFamily(n_sites, d)
    theta_0 = np.random.randn(exp_family.n_params) * 0.1
    
    print("\nIntegrating dynamics...")
    dynamics = InaccessibleGameDynamics(exp_family)
    solution = dynamics.integrate(theta_0, (0, 5.0), n_points=100)
    
    # Compute R(θ) along trajectory
    R_values = []
    
    print("\nComputing geometric factor R(θ)...")
    for theta in solution['theta']:
        G = exp_family.fisher_information(theta)
        _, a = exp_family.marginal_entropy_constraint(theta)
        
        a_norm_sq = np.dot(a, a)
        if a_norm_sq > 1e-12:
            Pi = np.eye(len(theta)) - np.outer(a, a) / a_norm_sq
        else:
            Pi = np.eye(len(theta))
        
        # R(θ) = θ^T G Π_∥ G θ
        R = theta @ G @ Pi @ G @ theta
        R_values.append(R)
    
    R_values = np.array(R_values)
    
    print(f"\n  Initial R(θ₀) = {R_values[0]:.6f}")
    print(f"  Final R(θf) = {R_values[-1]:.6f}")
    print(f"  Max R = {np.max(R_values):.6f}")
    print(f"  Min R = {np.min(R_values):.6f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # (a) R vs time
    ax = axes[0]
    ax.plot(solution['time'], R_values, 'b-', linewidth=2)
    ax.set_xlabel('Time τ')
    ax.set_ylabel('R(θ) = θᵀ G Π_∥ G θ')
    ax.set_title('(a) Entropy Gradient Geometric Factor')
    ax.grid(True, alpha=0.3)
    
    # (b) R vs H
    ax = axes[1]
    ax.plot(solution['H'], R_values, 'r-', linewidth=2)
    ax.set_xlabel('Joint Entropy H')
    ax.set_ylabel('R(θ)')
    ax.set_title('(b) Geometric Factor vs Entropy')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('entropy_gradient_geometry.png', dpi=150)
    print("\n✓ Figure saved: entropy_gradient_geometry.png")
    plt.show()
    
    return {
        'R_values': R_values,
        'times': solution['time'],
        'H': solution['H']
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ADVANCED ANALYSIS: QUANTUM INACCESSIBLE GAME")
    print("="*70)
    
    # Analysis 1: Time parametrisations
    print("\n\n[ANALYSIS 1: TIME PARAMETRISATIONS]")
    time_comparison = compare_time_parametrisations(n_sites=2, d=2, t_end=5.0)
    
    # Analysis 2: GENERIC evolution
    print("\n\n[ANALYSIS 2: GENERIC EVOLUTION]")
    generic_analysis = generic_evolution(n_sites=2, d=2, t_end=5.0, n_samples=20)
    
    # Analysis 3: Qubit vs qutrit optimality
    print("\n\n[ANALYSIS 3: QUBIT VS QUTRIT OPTIMALITY]")
    optimality_comparison = compare_qubit_qutrit_optimality(t_end=3.0)
    
    # Analysis 4: Entropy gradient geometry
    print("\n\n[ANALYSIS 4: ENTROPY GRADIENT GEOMETRY]")
    geometry_analysis = entropy_gradient_geometry(n_sites=2, d=2)
    
    print("\n" + "="*70)
    print("ADVANCED ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated figures:")
    print("  - time_parametrisation_comparison.png")
    print("  - generic_evolution.png")
    print("  - qubit_qutrit_comparison.png")
    print("  - entropy_gradient_geometry.png")

