"""
Run quantum qutrit dynamics experiment and generate figures.

This is the quantum analog of the classical n=3 binary experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import quantum_qutrit_n3 as qq

print("="*70)
print("QUANTUM QUTRIT DYNAMICS EXPERIMENT")
print("="*70)
print()

# Setup
print("Setting up...")
n_sites = 2  # Changed from 3 to 2 for pair operators (CIP-0002)
operators = qq.single_site_operators(n_sites)
print(f"  ✓ Hilbert space dimension: {3**n_sites}")
print(f"  ✓ Parameters: {len(operators)}")
print()

# Initialize at a non-trivial state
# Start with small random parameters (near maximally mixed but not exactly)
print("Initializing at perturbed state...")
np.random.seed(42)
theta_init = 0.3 * np.random.randn(len(operators))

# Verify it has reasonable marginal entropy sum
rho_init = qq.compute_density_matrix(theta_init, operators)
h_init = qq.compute_marginal_entropies(rho_init, n_sites)
print(f"  Initial marginals: {h_init}")
print(f"  Initial sum: Σh_i = {np.sum(h_init):.4f}")
print(f"  Maximum possible: {3*np.log(3):.4f}")
print()

# Run constrained dynamics
print("Integrating constrained dynamics...")
print("  (This may take a few minutes...)")
sol_constrained = qq.solve_constrained_quantum_maxent(
    theta_init,
    operators,
    n_sites=n_sites,
    n_steps=2000,
    dt=0.005,
    convergence_tol=1e-5,
    verbose=False
)
print(f"  ✓ Converged: {sol_constrained['converged']}")
print(f"  ✓ Steps: {sol_constrained['n_steps']}")
print(f"  ✓ Final ||F||: {sol_constrained['flow_norms'][-1]:.2e}")
print()

# Analyze final state
print("Final state:")
rho_final = qq.compute_density_matrix(sol_constrained['trajectory'][-1], operators)
h_final = qq.compute_marginal_entropies(rho_final, n_sites)
print(f"  Final marginals: {h_final}")
print(f"  Final sum: {np.sum(h_final):.6f}")
print(f"  Constraint drift: {abs(np.sum(h_final) - sol_constrained['C_init']):.2e}")
print()

# Create figures
print("Generating figures...")

# Figure 1: Constraint preservation
fig, ax = plt.subplots(figsize=(8, 6))
steps = np.arange(len(sol_constrained['constraint_values']))
deviation = np.abs(np.array(sol_constrained['constraint_values']) - sol_constrained['C_init'])
ax.semilogy(steps, deviation, 'b-', linewidth=2)
ax.set_xlabel(r'Time step $\tau$')
ax.set_ylabel(r'$|\sum_i h_i(\tau) - C|$')
ax.set_title('Constraint Preservation')
ax.grid(True, alpha=0.3, which='both')
ax.axhline(1e-8, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Target')
ax.legend()
plt.tight_layout()
plt.savefig('fig_qutrit_constraint_preservation.pdf', dpi=300, bbox_inches='tight')
print("  ✓ Saved: fig_qutrit_constraint_preservation.pdf")

# Figure 2: Flow norm convergence
fig, ax = plt.subplots(figsize=(8, 6))
ax.semilogy(sol_constrained['flow_norms'], 'b-', linewidth=2)
ax.set_xlabel(r'Time step $\tau$')
ax.set_ylabel(r'$\|F(\theta)\|$')
ax.set_title('Flow Convergence')
ax.grid(True, alpha=0.3, which='both')
ax.axhline(1e-6, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Convergence threshold')
ax.legend()
plt.tight_layout()
plt.savefig('fig_qutrit_convergence.pdf', dpi=300, bbox_inches='tight')
print("  ✓ Saved: fig_qutrit_convergence.pdf")

# Figure 3: Parameter trajectory (first 2 parameters)
fig, ax = plt.subplots(figsize=(8, 6))
traj = sol_constrained['trajectory']
ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.7, linewidth=2)
ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Initial', zorder=5)
ax.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=10, label='Final', zorder=5)
ax.set_xlabel(r'$\theta_1$ (site 1, Gell-Mann 1)')
ax.set_ylabel(r'$\theta_2$ (site 1, Gell-Mann 2)')
ax.set_title('Trajectory in Parameter Space')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_qutrit_trajectory.pdf', dpi=300, bbox_inches='tight')
print("  ✓ Saved: fig_qutrit_trajectory.pdf")

# Figure 4: GENERIC decomposition along trajectory
print("\nComputing GENERIC decomposition along trajectory...")
print("  (This is expensive - sampling every 50 steps)")
sample_indices = np.arange(0, len(traj), 50)
ratios = []
norms_S = []
norms_A = []

for idx in sample_indices:
    if idx % 200 == 0:
        print(f"    Step {idx}/{len(traj)-1}")
    try:
        result = qq.analyse_quantum_generic_structure(
            traj[idx], operators, n_sites=n_sites, eps_diff=1e-5
        )
        ratios.append(result['ratio'])
        norms_S.append(result['norm_S'])
        norms_A.append(result['norm_A'])
    except:
        ratios.append(np.nan)
        norms_S.append(np.nan)
        norms_A.append(np.nan)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sample_indices, ratios, 'g-', linewidth=2)
ax.set_xlabel(r'Time step $\tau$')
ax.set_ylabel(r'$\|A\|/\|S\|$')
ax.set_title('GENERIC Regime Variation')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_qutrit_generic_ratio.pdf', dpi=300, bbox_inches='tight')
print("  ✓ Saved: fig_qutrit_generic_ratio.pdf")

# Figure 5: Component norms
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sample_indices, norms_S, 'r-', linewidth=2, label=r'$\|S\|$ (dissipative)')
ax.plot(sample_indices, norms_A, 'b-', linewidth=2, label=r'$\|A\|$ (conservative)')
ax.set_xlabel(r'Time step $\tau$')
ax.set_ylabel('Frobenius Norm')
ax.set_title('GENERIC Component Norms')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_qutrit_component_norms.pdf', dpi=300, bbox_inches='tight')
print("  ✓ Saved: fig_qutrit_component_norms.pdf")

# Figure 6: Coherence evolution
print("\nComputing coherence evolution...")
coherences = []
purities = []
for idx in range(0, len(traj), 10):
    rho = qq.compute_density_matrix(traj[idx], operators)
    # Coherence: ||ρ - diag(ρ)||_F
    rho_diag = np.diag(np.diag(rho))
    coherence = np.linalg.norm(rho - rho_diag, 'fro')
    coherences.append(coherence)
    # Purity: Tr(ρ²)
    purity = np.real(np.trace(rho @ rho))
    purities.append(purity)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

sample_steps = np.arange(0, len(traj), 10)
ax1.plot(sample_steps, coherences, 'purple', linewidth=2)
ax1.set_xlabel(r'Time step $\tau$')
ax1.set_ylabel(r'Coherence $\|\rho - \mathrm{diag}(\rho)\|_F$')
ax1.set_title('Quantum Coherence Evolution')
ax1.grid(True, alpha=0.3)

ax2.plot(sample_steps, purities, 'orange', linewidth=2)
ax2.set_xlabel(r'Time step $\tau$')
ax2.set_ylabel(r'Purity $\mathrm{Tr}(\rho^2)$')
ax2.set_title('Purity Evolution')
ax2.grid(True, alpha=0.3)
ax2.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Pure state')
ax2.legend()

plt.tight_layout()
plt.savefig('fig_qutrit_quantum_measures.pdf', dpi=300, bbox_inches='tight')
print("  ✓ Saved: fig_qutrit_quantum_measures.pdf")

print()
print("="*70)
print("✓ EXPERIMENT COMPLETE")
print("="*70)
print()
print("Generated figures:")
print("  1. fig_qutrit_constraint_preservation.pdf - Σh_i conservation")
print("  2. fig_qutrit_convergence.pdf - Flow magnitude")
print("  3. fig_qutrit_trajectory.pdf - Parameter space trajectory")
print("  4. fig_qutrit_generic_ratio.pdf - ||A||/||S|| variation")
print("  5. fig_qutrit_component_norms.pdf - ||S|| and ||A|| separately")
print("  6. fig_qutrit_quantum_measures.pdf - Coherence and purity")
print()
print("These are the quantum analogs of your classical n=3 binary experiments!")

