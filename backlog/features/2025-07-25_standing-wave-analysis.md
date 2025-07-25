---
title: "Standing Wave Analysis and Normal Mode Extraction"
id: "2025-07-25_standing-wave-analysis"
status: "proposed"
priority: "medium"
created: "2025-07-25"
updated: "2025-07-25"
owner: "Neil"
dependencies: ["CIP-0005", "2025-07-25_multi-information-tracking"]
category: "features"
---

# Task: Standing Wave Analysis and Normal Mode Extraction

## Description

Implement standing-wave extractor and normal-mode analysis to visualize particles as information waves. This includes perturbation analysis, oscillatory decay extraction, and particle-as-standing-wave demonstration.

## Acceptance Criteria

### 1. Normal Mode Analysis
- [ ] Implement perturbation-based mode extraction
- [ ] Perturb ρ by +δe_k in correlation directions
- [ ] Observe oscillatory decay e^(-λ_k τ) patterns
- [ ] Extract eigenmode frequencies and decay rates

### 2. Standing Wave Visualization
- [ ] Map correlation modes to spatial patterns
- [ ] Visualize wave amplitude |ψ_k(x)| on lattice
- [ ] Show wave packet evolution and dispersion
- [ ] Demonstrate particle interpretation

### 3. Dispersion Analysis
- [ ] Compute dispersion relation ω(k) for normal modes
- [ ] Test Lorentz-like dispersion: ω_k² = c²k² + m²
- [ ] Use Fourier transforms on ring topology
- [ ] Verify emergent relativistic symmetry

### 4. Topological Features
- [ ] Implement soliton pulse tests
- [ ] Create kink-antikink configurations
- [ ] Track topological charge conservation
- [ ] Demonstrate particle stability

## Implementation Notes

### Core Perturbation Analysis
```python
def extract_normal_modes(self, rho, correlation_direction, delta=1e-6):
    """Extract normal modes via perturbation analysis."""
    # Perturb density matrix in specific correlation direction
    rho_pert = rho + delta * correlation_direction
    rho_pert = rho_pert / np.trace(rho_pert)  # Renormalize
    
    # Evolve perturbation and track decay
    decay_history = []
    current_rho = rho_pert
    
    for step in range(50):  # Short-time evolution
        current_rho = self._evolve_one_step(current_rho)
        perturbation = current_rho - rho  # Extract deviation
        amplitude = np.linalg.norm(perturbation)
        decay_history.append(amplitude)
    
    return decay_history

def analyze_oscillatory_decay(self, decay_history):
    """Extract frequency and decay rate from oscillatory signal."""
    # Fit to A * exp(-λt) * cos(ωt + φ)
    times = np.arange(len(decay_history))
    
    # Use FFT to identify dominant frequency
    fft_vals = np.fft.fft(decay_history)
    freqs = np.fft.fftfreq(len(decay_history))
    
    # Find peak frequency
    dominant_freq_idx = np.argmax(np.abs(fft_vals[1:len(fft_vals)//2])) + 1
    omega = 2 * np.pi * freqs[dominant_freq_idx]
    
    # Fit exponential envelope
    envelope = np.abs(decay_history)
    lambda_decay = -np.polyfit(times, np.log(envelope + 1e-12), 1)[0]
    
    return {'omega': omega, 'lambda': lambda_decay, 'amplitude': np.max(decay_history)}
```

### Spatial Wave Mapping
```python
def map_to_spatial_lattice(self, n_qubits, topology='ring'):
    """Map qubit indices to spatial coordinates."""
    if topology == 'ring':
        positions = [(np.cos(2*np.pi*i/n_qubits), np.sin(2*np.pi*i/n_qubits)) 
                     for i in range(n_qubits)]
    elif topology == '1d_chain':
        positions = [(i, 0) for i in range(n_qubits)]
    elif topology == '2d_square':
        side = int(np.sqrt(n_qubits))
        positions = [(i%side, i//side) for i in range(n_qubits)]
    
    return positions

def visualize_standing_wave(self, mode_pattern, positions):
    """Visualize spatial wave pattern."""
    import matplotlib.pyplot as plt
    
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_coords, y_coords, c=np.real(mode_pattern), 
                         s=100*np.abs(mode_pattern), cmap='RdBu', alpha=0.7)
    plt.colorbar(scatter, label='Wave Amplitude')
    plt.title('Standing Wave Pattern')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
```

### Lorentz Dispersion Test
```python
def test_lorentz_dispersion(self, rho, positions):
    """Test for emergent Lorentz-like dispersion relation."""
    n_sites = len(positions)
    
    # Create θ(x) field from density matrix phases
    theta_field = np.angle(np.diag(rho))
    
    # Fourier transform to momentum space
    theta_k = np.fft.fft(theta_field)
    k_values = 2 * np.pi * np.fft.fftfreq(n_sites)
    
    # Track decay of each mode
    dispersion_data = []
    for k_idx, k in enumerate(k_values[:n_sites//2]):
        # Extract decay rate for this k-mode
        mode_decay = self._track_mode_decay(theta_k[k_idx])
        omega = mode_decay['omega']
        
        dispersion_data.append({'k': k, 'omega': omega})
    
    return dispersion_data

def fit_lorentz_dispersion(self, dispersion_data):
    """Fit ω² = c²k² + m² to dispersion data."""
    k_vals = [d['k'] for d in dispersion_data]
    omega_vals = [d['omega'] for d in dispersion_data]
    
    # Fit ω² vs k²
    k_squared = np.array(k_vals)**2
    omega_squared = np.array(omega_vals)**2
    
    # Linear fit: ω² = c²k² + m²
    coeffs = np.polyfit(k_squared, omega_squared, 1)
    c_squared = coeffs[0]
    m_squared = coeffs[1]
    
    return {'c': np.sqrt(c_squared), 'm': np.sqrt(m_squared), 'fit_coeffs': coeffs}
```

## Physics Insights

### What We Learn
1. *Wave-Particle Duality*: Information modes as particle excitations
2. *Dispersion Relations*: Emergent relativistic-like physics
3. *Topological Stability*: Soliton solutions and conservation laws
4. *Mode Structure*: Fast/slow mode separation in wave picture

### Expected Results
- Clear normal mode spectrum with exponential decay
- Standing wave patterns with spatial structure
- Lorentz-like dispersion for long-wavelength modes
- Stable soliton configurations

## Testing Strategy

### Unit Tests
- Verify perturbation linearity in small-δ limit
- Check mode orthogonality and completeness
- Test FFT accuracy for dispersion analysis

### Physics Validation
- Compare with known analytical solutions
- Verify conservation laws (energy, particle number)
- Check dispersion relation consistency

### Integration Tests
- Validate with different lattice topologies
- Test scaling with system size
- Cross-check with Fisher and multi-info analyses

## Advanced Features

### Soliton Creation
```python
def create_soliton_pulse(self, positions, width=2, amplitude=1):
    """Create topological soliton configuration."""
    n_sites = len(positions)
    theta_profile = np.zeros(n_sites)
    
    # Create kink profile: θ jumps by π across domain wall
    center = n_sites // 2
    for i in range(n_sites):
        distance = min(abs(i - center), n_sites - abs(i - center))  # Ring topology
        theta_profile[i] = amplitude * np.tanh((distance - width) / width)
    
    return theta_profile
```

## Related

- CIP: 0005 (MEPP Thermalization)
- Theory: Quantum field theory, soliton physics
- Files: `mepp.py`, `mepp_demo.ipynb`
- Dependencies: Multi-information tracking for correlation directions

## Progress Updates

### 2025-07-25
Task created with Proposed status. Normal mode theory and wave analysis framework established. 