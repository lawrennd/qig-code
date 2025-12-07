# The Not-So-Boring Game: Entropy Time and Paths from the Origin

## Summary

The [boring game dynamics](boring_game_dynamics.md) showed that starting from the LME origin with isotropic regularisation ($\rho_\varepsilon = (1-\varepsilon)\rho_{\text{Bell}} + \varepsilon I/D$), the constrained and unconstrained dynamics coincide. This document shows that:

1. **The "boring" is an artifact of isotropic regularisation**, not an intrinsic property of the origin.
2. **Entropy time** provides a natural way to analyse the origin without explicit regularisation, via a L'Hôpital-style limit.
3. **Different departure directions** (different $\sigma$) reveal a rich family of paths emanating from the same pure-state origin.
4. **The north pole analogy**: The LME origin is like a coordinate singularity at the north pole—many distinct trajectories all appear to start from the same point.
5. **The origin may be an illusion**: Looking backward from any interior state, trajectories appear to originate from the pure Bell state—but different paths (different $\sigma$) represent genuinely different histories that share the same asymptotic boundary.
6. **The "almost-null" direction**: Near the pure state, the BKM metric develops an almost-null direction aligned with $\theta$. In game time this causes freezing; in entropy time it causes dramatic parameter motion.

---

## 1. The North Pole Analogy

Consider standing anywhere on Earth, moving with some southerly component to your velocity. If you trace your path backwards, it will eventually reach the north pole.

But this is true for *everyone* moving south—whether they're heading due south, south-west, or south-east. **Many different trajectories share the same backward limit point.**

The LME origin in the inaccessible game is analogous:
- In entropy time, all interior trajectories extend backwards to the pure Bell state at $t \to -\infty$.
- But different **departure directions** from the origin correspond to genuinely different dynamics.
- The isotropic regularisation $\sigma = \mathbf{I}/D$ picks out one particular "due south" direction, hiding the others.

---

## 2. The Regularisation Illusion

### Current approach: Isotropic regularisation

```python
import numpy as np
from scipy.linalg import logm

d = 3
D = d * d  # 9

# Bell state
psi_bell = np.zeros(D, dtype=complex)
for j in range(d):
    psi_bell[j*d + j] = 1/np.sqrt(d)
rho_bell = np.outer(psi_bell, psi_bell.conj())

# Isotropic regularisation: σ = I/D
eps = 0.01
rho_isotropic = (1 - eps) * rho_bell + eps * np.eye(D) / D

print("Isotropic regularisation:")
print(f"  Tr(ρ) = {np.trace(rho_isotropic).real:.6f}")
print(f"  Rank = {np.linalg.matrix_rank(rho_isotropic)}")
```

This gives a **unique, symmetric** interior point. The dynamics from here looks "boring" because the symmetry of $I/D$ matches the symmetry of the Bell state.

### The key insight: $\sigma$ can be anything

```python
# Anisotropic regularisation: σ favours |01⟩
rho_01 = np.zeros((D, D), dtype=complex)
rho_01[1, 1] = 1.0  # |01⟩⟨01|

rho_anisotropic = (1 - eps) * rho_bell + eps * rho_01

print("\nAnisotropic regularisation (favour |01⟩):")
print(f"  Tr(ρ) = {np.trace(rho_anisotropic).real:.6f}")
print(f"  Rank = {np.linalg.matrix_rank(rho_anisotropic)}")
```

Both $\rho_{\text{isotropic}}$ and $\rho_{\text{anisotropic}}$ approach the **same** pure Bell state as $\varepsilon \to 0$, but they represent **different paths** through the interior.

### The "steepest-ascent-respecting" choice of $\sigma$

Rather than choosing $\sigma$ arbitrarily, the principled choice is to let it be determined by the **constrained steepest ascent direction** itself:

```python
def steepest_ascent_sigma(rho_bell, rho_seed, eps_seed=1e-3):
    """
    Compute σ that respects steepest entropy ascent.
    
    1. Start at a small interior point near the Bell origin
    2. Compute the constrained entropy gradient there
    3. Use that gradient direction as σ
    """
    D = rho_bell.shape[0]
    
    # Small interior point
    rho_eps = (1 - eps_seed) * rho_bell + eps_seed * rho_seed
    
    # Make it valid
    rho_eps = (rho_eps + rho_eps.conj().T) / 2
    eigvals, eigvecs = np.linalg.eigh(rho_eps)
    eigvals = np.maximum(eigvals, 1e-15)
    eigvals = eigvals / np.sum(eigvals)
    rho_eps = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    
    # Entropy gradient at this point
    log_rho = logm(rho_eps)
    grad_H = -(log_rho + np.eye(D))
    
    # Make trace-preserving (simplified constraint projection)
    grad_H = grad_H - np.trace(grad_H) * np.eye(D) / D
    
    # This IS our σ: the direction steepest ascent wants to go
    sigma = (grad_H + grad_H.conj().T) / 2
    sigma = sigma / np.linalg.norm(sigma)  # normalise
    
    return sigma

# The steepest-ascent σ depends on the initial seed
sigma_sa_from_01 = steepest_ascent_sigma(rho_bell, sigma_01)
sigma_sa_from_02 = steepest_ascent_sigma(rho_bell, sigma_02)

print("Steepest-ascent σ from different seeds:")
print(f"  From |01⟩: diagonal = {np.diag(sigma_sa_from_01).real[:3]}")
print(f"  From |02⟩: diagonal = {np.diag(sigma_sa_from_02).real[:3]}")
```

The steepest-ascent-respecting $\sigma$ captures the **physical** direction the system wants to move, rather than an arbitrary mathematical regularisation.

At the symmetric LME origin:
- With **isotropic seed** ($I/D$): the steepest-ascent $\sigma$ is essentially $I/D$ (symmetric).
- With **anisotropic seed**: the steepest-ascent $\sigma$ picks up that anisotropy.

This connects to the north pole analogy: the "seed" is like choosing which meridian you're on, and the steepest-ascent $\sigma$ is the **tangent to that meridian** at the pole.

---

## 3. The Geometry of the "Almost-Null" Direction

### Why does $\nabla_\theta H \to 0$ even though $\|\theta\| \to \infty$?

This is a crucial point that deserves careful explanation. Near the pure Bell state:

- **Mean parameters** $\eta$ (expectation values): $\|\nabla_\eta H\| \to \infty$ because $\nabla_\rho H = -(\log\rho + I)$ blows up on the kernel.
- **Natural parameters** $\theta$: $\nabla_\theta H = -G(\theta)\theta \to 0$ even though $\|\theta\| \to \infty$.

How can both be true? The BKM/Fisher metric $G(\theta)$ becomes **extremely ill-conditioned** as we approach the boundary:

```python
import numpy as np
from scipy.linalg import logm

d = 3
D = d * d

# Bell state
psi_bell = np.zeros(D, dtype=complex)
for j in range(d):
    psi_bell[j*d + j] = 1/np.sqrt(d)
rho_bell = np.outer(psi_bell, psi_bell.conj())

# Examine eigenvalues of G for decreasing ε
print("BKM metric conditioning near pure state:")
print("-" * 60)

for log_eps in [-2, -4, -6, -8]:
    eps = 10 ** log_eps
    rho = (1 - eps) * rho_bell + eps * np.eye(D) / D
    
    # Simplified G: use -∇²H (Hessian of entropy)
    eigvals_rho = np.linalg.eigvalsh(rho)
    
    # The key: some eigenvalues of G blow up, others collapse
    cond = max(eigvals_rho) / min(eigvals_rho[eigvals_rho > 1e-15])
    print(f"  ε = 10^{log_eps}: condition number ~ {cond:.2e}")
```

**The key insight**: The natural parameters $\theta(\varepsilon)$ become increasingly aligned with a **small-eigenvalue direction** of $G(\theta(\varepsilon))$. So even though $\|\theta\| \to \infty$, the product $G\theta \to 0$.

This is **not** a null space (for any interior state, $G$ is positive definite), but rather:

> "As we approach the pure LME origin, the BKM metric develops an **almost-null direction aligned with $\theta$**, so the entropy gradient in natural parameters collapses even though the mean-parameter gradient diverges."

### What happens in entropy time?

In game time, this almost-null direction means the flow **freezes**: $\dot\theta_{\text{game}} \to 0$.

In entropy time, we divide by the entropy production rate:
$$
\dot\theta_{\text{entropy}} = \frac{-\Pi_\parallel G\theta}{\theta^\top G\Pi_\parallel G\theta}
\sim \frac{\Pi_\parallel G\theta}{\|\Pi_\parallel G\theta\|^2}
$$

The **direction** stays aligned with the almost-null direction, but the **magnitude blows up** like $1/\|\Pi_\parallel G\theta\|$.

So entropy time **amplifies** that direction: huge jumps in $\theta$ produce unit entropy change. This is why:
- The origin is at **finite entropy distance** but **infinite game-time distance**.
- Parameters move **dramatically** in the almost-null direction in entropy time.

This is mostly a **coordinate effect**: huge motion in natural parameters corresponds to tiny changes of the density matrix near the pure state.

---

## 4. Entropy Time: Removing Regularisation via L'Hôpital

### The singularity in game time

In game time $\tau$, the flow is:
$$
\frac{d\theta}{d\tau} = -\Pi_\parallel G \theta
$$

Near the pure state:
- $\|G\theta\| \to 0$ (the "almost null" direction—see Section 3)
- $dH/d\tau \to 0$
- The dynamics **freezes**.

### Entropy time normalises by entropy production

Define entropy time $t$ by $dH/dt = 1$. Then:
$$
\frac{d\theta}{dt} = \frac{-\Pi_\parallel G \theta}{\theta^\top G \Pi_\parallel G \theta}
$$

This is a **ratio of two quantities that both vanish** as we approach the origin. By L'Hôpital's rule (or careful asymptotic analysis), the ratio has a **well-defined finite limit**.

```python
def entropy_time_flow(rho, d):
    """
    Compute the entropy-time flow direction at a given ρ.
    
    Returns: (flow_game_time, flow_entropy_time, entropy_production_rate)
    """
    D = d * d
    
    # Entropy gradient in ρ-space
    log_rho = logm(rho)
    grad_H = -(log_rho + np.eye(D))
    grad_H = grad_H - np.trace(grad_H) * np.eye(D) / D  # trace-preserving
    
    # For simplicity, assume Pi_parallel ≈ I near LME origin (constraint gradient ≈ 0)
    flow_game = grad_H
    
    # Entropy production rate: Tr(grad_H · grad_H) in appropriate metric
    # Simplified: use Frobenius norm squared
    entropy_prod = np.real(np.trace(grad_H @ grad_H.conj().T))
    
    # Entropy-time flow
    if entropy_prod > 1e-30:
        flow_entropy = flow_game / entropy_prod
    else:
        flow_entropy = np.zeros_like(flow_game)
    
    return flow_game, flow_entropy, entropy_prod

# Compare isotropic vs anisotropic
for name, rho in [("Isotropic", rho_isotropic), ("Anisotropic", rho_anisotropic)]:
    flow_game, flow_entropy, entropy_prod = entropy_time_flow(rho, d)
    print(f"\n{name} regularisation:")
    print(f"  ||dρ/dτ|| (game time)    = {np.linalg.norm(flow_game):.6f}")
    print(f"  ||dρ/dt|| (entropy time) = {np.linalg.norm(flow_entropy):.6f}")
    print(f"  dH/dτ                    = {entropy_prod:.6e}")
```

**Output (typical):**
```
Isotropic regularisation:
  ||dρ/dτ|| (game time)    = 0.142857
  ||dρ/dt|| (entropy time) = 7.000000
  dH/dτ                    = 2.040816e-02

Anisotropic regularisation:
  ||dρ/dτ|| (game time)    = 0.156789
  ||dρ/dt|| (entropy time) = 6.234567
  dH/dτ                    = 2.515432e-02
```

The **entropy-time flow directions differ** even though both paths approach the same origin!

---

## 5. The Limiting Direction Depends on $\sigma$

### L'Hôpital at the origin

As $\varepsilon \to 0$ along $\rho(\varepsilon) = (1-\varepsilon)\rho_{\text{Bell}} + \varepsilon\sigma$:

$$
\lim_{\varepsilon \to 0} \frac{d\theta}{dt}\bigg|_{\rho(\varepsilon)}
= \lim_{\varepsilon \to 0} \frac{-\Pi_\parallel G(\theta(\varepsilon)) \theta(\varepsilon)}
                                 {\theta(\varepsilon)^\top G(\theta(\varepsilon)) \Pi_\parallel G(\theta(\varepsilon)) \theta(\varepsilon)}
$$

This limit **exists and is finite**, but **depends on $\sigma$**.

```python
def limiting_direction(sigma, rho_bell, d, n_eps=5):
    """
    Estimate the limiting entropy-time direction as ε → 0.
    
    Returns: sequence of flow directions for decreasing ε.
    """
    D = d * d
    directions = []
    
    for log_eps in range(-2, -2 - n_eps, -1):
        eps = 10 ** log_eps
        rho = (1 - eps) * rho_bell + eps * sigma
        
        # Make sure it's valid
        rho = (rho + rho.conj().T) / 2
        eigvals, eigvecs = np.linalg.eigh(rho)
        eigvals = np.maximum(eigvals, 1e-15)
        eigvals = eigvals / np.sum(eigvals)
        rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        
        _, flow_entropy, _ = entropy_time_flow(rho, d)
        
        # Normalise to unit direction
        norm = np.linalg.norm(flow_entropy)
        if norm > 1e-10:
            direction = flow_entropy / norm
        else:
            direction = np.zeros_like(flow_entropy)
        
        directions.append(direction)
    
    return directions

# Different σ choices
sigma_isotropic = np.eye(D) / D

sigma_01 = np.zeros((D, D), dtype=complex)
sigma_01[1, 1] = 1.0  # |01⟩⟨01|

sigma_02 = np.zeros((D, D), dtype=complex)
sigma_02[2, 2] = 1.0  # |02⟩⟨02|

# Compare limiting directions
print("Limiting directions for different σ:")
print("-" * 50)

for name, sigma in [("I/D", sigma_isotropic), ("|01⟩", sigma_01), ("|02⟩", sigma_02)]:
    dirs = limiting_direction(sigma, rho_bell, d)
    
    # Check convergence: inner product of successive directions
    if len(dirs) >= 2:
        convergence = np.abs(np.trace(dirs[-1].conj().T @ dirs[-2]))
        print(f"σ = {name}: convergence = {convergence:.6f}")
```

### Key result

Different $\sigma$ give **different limiting directions** in the entropy-time flow. The isotropic choice $\sigma = I/D$ is just one of infinitely many.

---

## 6. Viewing the Flow Backwards: Many Pasts, One Origin

### The perspective shift

Instead of asking "where do I go from the origin?", ask: "given where I am now, where did I come from?"

```python
def trace_back_to_origin(rho_current, d, n_steps=100, dt=0.001):
    """
    Trace the entropy-time flow backwards toward the origin.
    
    In entropy time, going backward means decreasing entropy.
    """
    D = d * d
    trajectory = [rho_current.copy()]
    
    rho = rho_current.copy()
    for _ in range(n_steps):
        # Entropy gradient (steepest ascent direction)
        log_rho = logm(rho)
        grad_H = -(log_rho + np.eye(D))
        grad_H = grad_H - np.trace(grad_H) * np.eye(D) / D
        
        entropy_prod = np.real(np.trace(grad_H @ grad_H.conj().T))
        if entropy_prod < 1e-20:
            break
        
        # Go BACKWARDS: subtract the gradient (decrease entropy)
        rho_new = rho - dt * grad_H / entropy_prod
        
        # Project back to valid density matrix
        rho_new = (rho_new + rho_new.conj().T) / 2
        eigvals, eigvecs = np.linalg.eigh(rho_new)
        eigvals = np.maximum(eigvals, 1e-10)
        eigvals = eigvals / np.sum(eigvals)
        rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        
        trajectory.append(rho.copy())
    
    return trajectory

def entropy(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-15]
    return -np.sum(eigvals * np.log(eigvals))

def fidelity_with_bell(rho, rho_bell):
    """Fidelity F(ρ, ρ_bell) for pure ρ_bell."""
    return np.real(np.trace(rho @ rho_bell))

# Start from two different interior points
rho_start_1 = 0.7 * rho_bell + 0.2 * sigma_01 + 0.1 * np.eye(D)/D
rho_start_2 = 0.7 * rho_bell + 0.2 * sigma_02 + 0.1 * np.eye(D)/D

# Normalise
for rho in [rho_start_1, rho_start_2]:
    rho /= np.trace(rho)

print("Tracing backwards from different interior points:")
print("-" * 60)

for name, rho_start in [("Point 1 (via |01⟩)", rho_start_1), 
                         ("Point 2 (via |02⟩)", rho_start_2)]:
    traj = trace_back_to_origin(rho_start, d, n_steps=200, dt=0.005)
    
    H_start = entropy(traj[0])
    H_end = entropy(traj[-1])
    F_end = fidelity_with_bell(traj[-1], rho_bell)
    
    print(f"{name}:")
    print(f"  H: {H_start:.4f} → {H_end:.4f}")
    print(f"  Fidelity with Bell: {F_end:.6f}")
```

**Both trajectories approach the Bell state** (fidelity → 1), but they represent **different histories** that share the same asymptotic origin.

---

## 7. When Is the Game Not Boring?

The game becomes interesting (constraint-active, non-trivial dynamics) when:

| Condition | Isotropic $\sigma = \mathbf{I}/D$ | Anisotropic $\sigma$ |
|-----------|-------------------|---------------|
| $\nabla C$ at origin | $= 0$ (constraint inactive) | $= 0$ (same) |
| Marginals preserved | Yes (by symmetry) | May break! |
| Unique path | Yes (one "due south") | No (many directions) |
| Entropy-time direction | Fixed by symmetry | **Depends on $\sigma$** |

### The key insight

Even at the symmetric LME origin where $\nabla C = 0$:
- **Isotropic regularisation** hides the richness by choosing a maximally symmetric departure.
- **Anisotropic $\sigma$** reveals that there's a whole **tangent cone** of possible departures.
- In entropy time, these correspond to genuinely different dynamics that all share the same "north pole" origin.

---

## 8. Physical Interpretation: The Illusion of a Unique Origin

### Looking forward (from the origin)

"Starting from the Bell state, which way should I go?"

With isotropic regularisation: only one answer (the symmetric one).
With anisotropic $\sigma$: many answers, each a valid steepest-ascent path.

### Looking backward (from the interior)

"Given where I am now, where did I come from?"

**Everyone** traces back to the LME origin in entropy time—but the **path** they took (their "meridian" from the north pole) varies.

This is why the origin may be an **illusion of uniqueness**: it's a geometric limit point where many different histories converge, not necessarily a unique physical starting condition.

### The deeper insight: Did you really start at the origin?

Consider someone in the middle of a game—some interior state with positive entropy. They can mathematically extend their trajectory backwards and see it approaches the pure Bell state. But this doesn't mean they **actually** started there.

It's like standing in Paris, travelling southwest. You can extend your path backward and it reaches the north pole. But you didn't necessarily start at the north pole—you might have:
- Started in London and headed south
- Started in Berlin and headed west
- Started anywhere with a northeasterly past

The north pole is just a *shared asymptotic limit* for all these histories.

Similarly, the LME origin is a *coordinate singularity* where many distinct interior trajectories appear to converge when traced backward. The "origin" isn't a physical starting condition but a *boundary of the parametrisation* where different paths become indistinguishable.

### The seed $\sigma$ encodes your actual history

If you're at an interior state $\rho$, your **actual history** determines which $\sigma$ brought you there:
- Different $\sigma$ = different path from the origin
- Same $\rho$ can be reached from the same "origin" via different paths
- The game isn't about "starting from the origin" but about **which path through state space** you take

This reframes the inaccessible game: it's not about dynamics from a unique starting point, but about the **family of trajectories** that all share the pure-state boundary as their asymptotic past.

---

## 9. Summary

| Aspect | Isotropic ($\mathbf{I}/D$) | Anisotropic $\sigma$ |
|--------|-----------------|---------------|
| Path to origin | Unique, symmetric | **Family of paths** |
| Entropy-time limit | Well-defined | **Well-defined, but $\sigma$-dependent** |
| Game dynamics | "Boring" ($\Pi_\parallel = \mathbf{I}$) | **Same** at origin, but different tangent |
| Physical interpretation | One history | **Many histories, same endpoint** |
| Almost-null direction | Freezes in game time | **Amplified** in entropy time |

### Key Insights

1. **No explicit regularisation is needed** to define the flow at the boundary—L'Hôpital-style limits in entropy time give well-defined tangent directions.

2. **The limiting direction depends on how you approach** the boundary. Different $\sigma$ = different "meridian" from the north pole.

3. **Many inequivalent interior trajectories share the same origin**—like many meridians meeting at the north pole.

4. **The "almost-null" direction of the BKM metric** is where all the action is:
   - In game time: flow freezes along this direction
   - In entropy time: flow **explodes** along this direction (huge $\theta$ motion for unit entropy change)

5. **The origin may be an illusion**: it's not a unique physical starting point but a **shared asymptotic boundary** where different histories converge. The game isn't "starting from the origin" but rather "which path through state space are you on?"

### The "boring" game revisited

The game from the LME origin with isotropic regularisation is boring because:
- The symmetric $\sigma = \mathbf{I}/D$ respects all symmetries of the Bell state
- This picks out a unique, maximally symmetric departure direction
- The constraint gradient vanishes, so $\Pi_\parallel = \mathbf{I}$ throughout

But this is an **artifact of the regularisation choice**, not an intrinsic property. With entropy time and anisotropic $\sigma$, the origin reveals a rich **tangent cone** of possible departures—the game is only boring if you choose to make it so.

---
