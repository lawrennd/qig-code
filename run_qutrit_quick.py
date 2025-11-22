"""
Quick quantum qutrit experiment (skips expensive GENERIC computation).
"""

import numpy as np
import matplotlib.pyplot as plt
import quantum_qutrit_n3 as qq

print("="*70)
print("QUICK QUANTUM QUTRIT EXPERIMENT")
print("="*70)
print()

# Setup
print("Setting up...")
n_sites = 2  # Changed from 3 to 2 for pair operators (CIP-0002)
operators = qq.single_site_operators(n_sites)
print(f"  ✓ Hilbert space dimension: {3**n_sites}")
print()

# Initialize
np.random.seed(42)
theta_init = 0.3 * np.random.randn(len(operators))

rho_init = qq.compute_density_matrix(theta_init, operators)
h_init = qq.compute_marginal_entropies(rho_init, n_sites)
print(f"Initial state:")
print(f"  Marginals: {h_init}")
print(f"  Sum: Σh_i = {np.sum(h_init):.4f}")
print()

# Run dynamics (shorter)
print("Integrating dynamics (500 steps)...")
sol = qq.solve_constrained_quantum_maxent(
    theta_init, operators, n_sites=n_sites, n_steps=500, dt=0.01, verbose=False
)
print(f"  ✓ Steps: {len(sol['trajectory'])}")
print(f"  ✓ Final ||F||: {sol['flow_norms'][-1]:.2e}")
print()

# Final state
rho_final = qq.compute_density_matrix(sol['trajectory'][-1], operators)
h_final = qq.compute_marginal_entropies(rho_final, n_sites)
print(f"Final state:")
print(f"  Marginals: {h_final}")
print(f"  Constraint drift: {abs(np.sum(h_final) - sol['C_init']):.2e}")
print()

# Generate figures
print("Generating figures...")

# 1. Constraint preservation
fig, ax = plt.subplots(figsize=(8, 6))
deviation = np.abs(np.array(sol['constraint_values']) - sol['C_init'])
ax.semilogy(deviation, 'b-', linewidth=2)
ax.set_xlabel(r'Time step')
ax.set_ylabel(r'$|\sum_i h_i - C|$')
ax.set_title('Quantum Qutrit: Constraint Preservation')
ax.grid(True, alpha=0.3, which='both')
plt.savefig('fig_qutrit_constraint.pdf', dpi=300, bbox_inches='tight')
print("  ✓ fig_qutrit_constraint.pdf")

# 2. Convergence
fig, ax = plt.subplots(figsize=(8, 6))
ax.semilogy(sol['flow_norms'], 'b-', linewidth=2)
ax.set_xlabel(r'Time step')
ax.set_ylabel(r'$\|F(\theta)\|$')
ax.set_title('Quantum Qutrit: Convergence')
ax.grid(True, alpha=0.3, which='both')
plt.savefig('fig_qutrit_convergence.pdf', dpi=300, bbox_inches='tight')
print("  ✓ fig_qutrit_convergence.pdf")

# 3. Trajectory
fig, ax = plt.subplots(figsize=(8, 6))
traj = sol['trajectory']
ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.7, linewidth=2, label='Trajectory')
ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Initial')
ax.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=10, label='Final')
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_2$')
ax.set_title('Quantum Qutrit: Parameter Space Trajectory')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('fig_qutrit_trajectory.pdf', dpi=300, bbox_inches='tight')
print("  ✓ fig_qutrit_trajectory.pdf")

# 4. Quantum measures
print("Computing quantum coherence and purity...")
coherences = []
purities = []
for idx in range(0, len(traj), 5):
    rho = qq.compute_density_matrix(traj[idx], operators)
    rho_diag = np.diag(np.diag(rho))
    coherence = np.linalg.norm(rho - rho_diag, 'fro')
    coherences.append(coherence)
    purity = np.real(np.trace(rho @ rho))
    purities.append(purity)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sample_steps = np.arange(0, len(traj), 5)
ax1.plot(sample_steps, coherences, 'purple', linewidth=2)
ax1.set_xlabel(r'Time step')
ax1.set_ylabel(r'Coherence')
ax1.set_title('Off-Diagonal Elements')
ax1.grid(True, alpha=0.3)

ax2.plot(sample_steps, purities, 'orange', linewidth=2)
ax2.set_xlabel(r'Time step')
ax2.set_ylabel(r'Purity $\mathrm{Tr}(\rho^2)$')
ax2.set_title('State Purity')
ax2.grid(True, alpha=0.3)
ax2.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Pure')
ax2.legend()

plt.tight_layout()
plt.savefig('fig_qutrit_quantum_properties.pdf', dpi=300, bbox_inches='tight')
print("  ✓ fig_qutrit_quantum_properties.pdf")

# 5. GENERIC at initial point only
print("\nGENERIC analysis at initial point...")
result_init = qq.analyse_quantum_generic_structure(theta_init, operators, n_sites=n_sites)
print(f"  ||S|| (dissipative): {result_init['norm_S']:.4f}")
print(f"  ||A|| (conservative): {result_init['norm_A']:.4f}")
print(f"  Ratio ||A||/||S||: {result_init['ratio']:.4f}")

print()
print("="*70)
print("✓ QUICK EXPERIMENT COMPLETE")
print("="*70)
print()
print("Generated 4 figures demonstrating quantum qutrit dynamics!")
print()
print("For full GENERIC analysis along trajectory, use run_qutrit_experiment.py")
print("(but it takes ~30 minutes due to expensive numerical differentiation)")

