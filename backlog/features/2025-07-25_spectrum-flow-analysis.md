---
title: "Spectrum Flow Analysis (Time × log λ_k Heatmaps)"
id: "2025-07-25_spectrum-flow-analysis"
status: "proposed"
priority: "medium"
created: "2025-07-25"
updated: "2025-07-25"
owner: "Neil"
dependencies: ["CIP-0005", "2025-07-25_charge-classification"]
category: "features"
---

# Task: Spectrum Flow Analysis (Time × log λ_k Heatmaps)

## Description

Implement spectrum flow visualization as time × log λ_k heatmaps to show successive quasi-symmetry plateaus as band gaps open and close. This provides visual insight into how "charges" disappear when λ_k finally grows and reveals the hierarchy of timescales in thermalization.

## Acceptance Criteria

### 1. Spectrum Flow Heatmaps
- [ ] Create time × log λ_k heatmap visualization
- [ ] Track Fisher eigenvalue evolution throughout simulation
- [ ] Use color coding to show eigenvalue magnitude changes
- [ ] Overlay plateau detection events on heatmap

### 2. Band Gap Analysis
- [ ] Identify band gaps: regions where λ_k values cluster into groups
- [ ] Track band gap opening/closing events during evolution
- [ ] Correlate band gaps with charge classification (hard/soft)
- [ ] Show connection between gaps and entropy plateaus

### 3. Quasi-Symmetry Identification
- [ ] Detect quasi-symmetry phases where multiple λ_k ≈ 0
- [ ] Track symmetry breaking: when λ_k values grow and separate
- [ ] Identify emergent conservation laws from persistent small eigenvalues
- [ ] Show hierarchy of symmetry breaking timescales

### 4. Interactive Analysis
- [ ] Enable selection of specific eigenvalue tracks
- [ ] Show corresponding eigenvector evolution for selected λ_k
- [ ] Correlate spectrum changes with entropy and charge dynamics
- [ ] Provide eigenvalue trajectory analysis tools

## Implementation Notes

### Core Spectrum Flow Visualization
```python
def create_spectrum_flow_heatmap(self):
    """Create time × log λ_k heatmap showing eigenvalue evolution."""
    import matplotlib.pyplot as plt
    
    # Prepare data
    n_steps = len(self.fisher_eigenval_history)
    n_evals = len(self.fisher_eigenval_history[0])
    
    # Create log eigenvalue matrix (time × eigenvalue index)
    log_eigenvals = np.zeros((n_steps, n_evals))
    
    for i, evals in enumerate(self.fisher_eigenval_history):
        # Add small regularization to avoid log(0)
        regularized_evals = np.maximum(evals, 1e-15)
        log_eigenvals[i, :] = np.log10(regularized_evals)
    
    # Sort eigenvalues at each time step for consistent tracking
    log_eigenvals_sorted = np.sort(log_eigenvals, axis=1)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(log_eigenvals_sorted.T, aspect='auto', cmap='viridis',
                   origin='lower', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(λ_k)', fontsize=12)
    
    # Labels and title
    ax.set_xlabel('Simulation Step', fontsize=12)
    ax.set_ylabel('Eigenvalue Index (sorted)', fontsize=12)
    ax.set_title('Fisher Spectrum Flow: Time × log λ_k', fontsize=14)
    
    return fig, ax, log_eigenvals_sorted

def overlay_plateau_events(self, ax):
    """Overlay plateau detection events on spectrum flow heatmap."""
    # Add vertical lines for detected plateaus
    for plateau in self.plateau_events:
        ax.axvline(x=plateau['start_step'], color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=plateau['end_step'], color='red', linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(['Plateau Start/End'], loc='upper right')
```

### Band Gap Analysis
```python
def analyze_band_gaps(self, gap_threshold=2.0):
    """Identify and track band gaps in Fisher spectrum."""
    band_gap_history = []
    
    for step, evals in enumerate(self.fisher_eigenval_history):
        log_evals = np.log10(np.maximum(evals, 1e-15))
        sorted_log_evals = np.sort(log_evals)
        
        # Find gaps larger than threshold
        gaps = np.diff(sorted_log_evals)
        gap_indices = np.where(gaps > gap_threshold)[0]
        
        gap_info = {
            'step': step,
            'n_gaps': len(gap_indices),
            'gap_positions': gap_indices.tolist(),
            'gap_sizes': gaps[gap_indices].tolist(),
            'below_gap_count': gap_indices + 1 if len(gap_indices) > 0 else [],
            'total_eigenvals': len(evals)
        }
        
        band_gap_history.append(gap_info)
    
    return band_gap_history

def visualize_band_gap_evolution(self, band_gap_history):
    """Visualize how band gaps evolve over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    steps = [gap['step'] for gap in band_gap_history]
    n_gaps = [gap['n_gaps'] for gap in band_gap_history]
    
    # Plot number of gaps vs time
    ax1.plot(steps, n_gaps, 'b-', linewidth=2)
    ax1.set_ylabel('Number of Band Gaps')
    ax1.set_title('Band Gap Count Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Plot gap positions as scatter
    for gap_info in band_gap_history:
        if gap_info['gap_positions']:
            ax2.scatter([gap_info['step']] * len(gap_info['gap_positions']),
                       gap_info['gap_positions'], 
                       s=50, alpha=0.6, c='red')
    
    ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel('Gap Position (Eigenvalue Index)')
    ax2.set_title('Band Gap Positions Over Time')
    ax2.grid(True, alpha=0.3)
    
    return fig
```

### Quasi-Symmetry Detection
```python
def detect_quasi_symmetries(self, symmetry_threshold=1e-6):
    """Identify quasi-symmetry phases with multiple small eigenvalues."""
    quasi_symmetry_phases = []
    
    for step, evals in enumerate(self.fisher_eigenval_history):
        # Count eigenvalues below threshold
        small_evals = np.sum(evals < symmetry_threshold)
        
        if small_evals > 1:  # Multiple quasi-conserved quantities
            phase_info = {
                'step': step,
                'n_quasi_conserved': small_evals,
                'smallest_eigenval': np.min(evals),
                'quasi_eigenvals': evals[evals < symmetry_threshold].tolist(),
                'symmetry_breaking_scale': np.max(evals[evals < symmetry_threshold])
            }
            quasi_symmetry_phases.append(phase_info)
    
    return quasi_symmetry_phases

def track_symmetry_breaking_cascade(self, quasi_symmetry_phases):
    """Analyze the cascade of symmetry breaking events."""
    breaking_events = []
    
    for i in range(1, len(quasi_symmetry_phases)):
        prev_phase = quasi_symmetry_phases[i-1]
        curr_phase = quasi_symmetry_phases[i]
        
        # Check if symmetries were broken
        if curr_phase['n_quasi_conserved'] < prev_phase['n_quasi_conserved']:
            n_broken = prev_phase['n_quasi_conserved'] - curr_phase['n_quasi_conserved']
            
            breaking_event = {
                'step': curr_phase['step'],
                'symmetries_broken': n_broken,
                'remaining_symmetries': curr_phase['n_quasi_conserved'],
                'breaking_scale': curr_phase['symmetry_breaking_scale']
            }
            breaking_events.append(breaking_event)
    
    return breaking_events
```

### Interactive Eigenvalue Tracking
```python
def create_interactive_spectrum_explorer(self):
    """Create interactive tool for exploring eigenvalue trajectories."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    
    fig, (ax_heatmap, ax_trajectory) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Create spectrum flow heatmap
    _, _, log_eigenvals = self.create_spectrum_flow_heatmap()
    im = ax_heatmap.imshow(log_eigenvals.T, aspect='auto', cmap='viridis',
                          origin='lower', interpolation='nearest')
    
    # Initial trajectory plot
    eigenval_idx = 0
    line, = ax_trajectory.plot(log_eigenvals[:, eigenval_idx], 'r-', linewidth=2)
    ax_trajectory.set_ylabel('log₁₀(λ_k)')
    ax_trajectory.set_xlabel('Simulation Step')
    ax_trajectory.set_title(f'Eigenvalue {eigenval_idx} Trajectory')
    ax_trajectory.grid(True, alpha=0.3)
    
    # Slider for eigenvalue selection
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, 'Eigenvalue Index', 0, log_eigenvals.shape[1]-1, 
                   valinit=0, valfmt='%d')
    
    def update_trajectory(val):
        idx = int(slider.val)
        line.set_ydata(log_eigenvals[:, idx])
        ax_trajectory.set_title(f'Eigenvalue {idx} Trajectory')
        ax_trajectory.relim()
        ax_trajectory.autoscale_view()
        fig.canvas.draw()
    
    slider.on_changed(update_trajectory)
    
    return fig, slider
```

## Physics Insights

### What We Learn
1. *Timescale Hierarchy*: Visual separation of fast/slow modes in spectrum
2. *Symmetry Breaking Cascade*: Sequential breaking of approximate symmetries
3. *Plateau Mechanism*: Band gaps create quasi-equilibrium states
4. *Charge Evolution*: When conserved quantities become non-conserved

### Expected Results
- Clear band structure with gaps separating fast/slow modes
- Gap closing events correlate with entropy plateau endings
- Hierarchical symmetry breaking: higher λ_k values grow first
- Final approach to thermal equilibrium shows gap closure

## Testing Strategy

### Unit Tests
- Verify heatmap data consistency with stored eigenvalue history
- Test band gap detection algorithm with synthetic data
- Check quasi-symmetry identification accuracy

### Physics Validation
- Compare with theoretical predictions for known systems
- Verify symmetry breaking hierarchy matches expectations
- Cross-check with charge classification results

### Visualization Tests
- Test interactive tools for responsiveness and accuracy
- Verify color scaling and data representation
- Check overlay alignment with other analysis tools

## Advanced Features

### Eigenvalue Flow Streamlines
```python
def create_eigenvalue_streamlines(self):
    """Create streamline visualization of eigenvalue flow in 2D projections."""
    # Project high-dimensional eigenvalue space to 2D using PCA
    from sklearn.decomposition import PCA
    
    eigenval_matrix = np.array(self.fisher_eigenval_history)
    pca = PCA(n_components=2)
    eigenval_2d = pca.fit_transform(eigenval_matrix)
    
    # Create vector field showing flow direction
    dx = np.diff(eigenval_2d[:, 0])
    dy = np.diff(eigenval_2d[:, 1])
    
    # Plot streamlines
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.quiver(eigenval_2d[:-1, 0], eigenval_2d[:-1, 1], dx, dy, 
             angles='xy', scale_units='xy', scale=1, alpha=0.7)
    ax.plot(eigenval_2d[:, 0], eigenval_2d[:, 1], 'ro-', alpha=0.8)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title('Eigenvalue Flow in Principal Component Space')
    
    return fig, pca
```

## Related

- CIP: 0005 (MEPP Thermalization)
- Theory: Fisher information geometry, symmetry breaking, critical phenomena
- Files: `mepp.py`, `mepp_demo.ipynb`
- Dependencies: Charge classification, plateau detection

## Progress Updates

### 2025-07-25
Task created with Proposed status. Spectrum flow analysis framework designed with comprehensive visualization and symmetry breaking analysis. 