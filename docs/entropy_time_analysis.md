# The Inaccessible Game: Dynamics in Entropy Time

## Executive Summary

This document synthesises our analysis of the inaccessible game dynamics, particularly how the choice of **entropy time** $t$ (rather than affine game time $\tau$) fundamentally changes the picture. The key findings are.

1. **In game time τ**: The dynamics appears to "freeze" near the origin (pure Bell states)
2. **In entropy time t**: The dynamics accelerates toward the origin, but the origin remains at infinite entropy-time distance
3. **Multi-pair coupling**: Pairs interact through the **shared entropy budget**, not through the constraint gradient
4. **Antisymmetric differentiation**: Different Bell states have orthogonal Hamiltonian (antisymmetric) dynamics, providing the strategic degrees of freedom

---

## 1. The Origin State

The "origin" of the inaccessible game is a product of maximally entangled Bell states for qutrit pairs:

$$|\Psi_{\text{origin}}\rangle = \bigotimes_{k=1}^{n} |\Phi_k\rangle$$

where each pair is in a generalised Bell state:

$$|\Phi_m\rangle = \frac{1}{\sqrt{d}} \sum_{j=0}^{d-1} |j, (j+m) \mod d\rangle$$

For qutrits ($d=3$), there are three orthogonal Bell states:
- **k=0**: $\frac{1}{\sqrt{3}}(|0,0\rangle + |1,1\rangle + |2,2\rangle)$ — perfect correlation
- **k=1**: $\frac{1}{\sqrt{3}}(|0,1\rangle + |1,2\rangle + |2,0\rangle)$ — cyclic correlation  
- **k=2**: $\frac{1}{\sqrt{3}}(|0,2\rangle + |1,0\rangle + |2,1\rangle)$ — anti-cyclic correlation

All three have:
- **Same entropy**: $H = 0$ (pure states)
- **Same marginals**: $\rho_A = \rho_B = I/3$
- **Same entanglement**: $\log 3$ ebits
- **Different correlation structure**: Which is what distinguishes them dynamically

---

## 2. The Constraint and Its Gradient

The inaccessible game imposes the constraint that the **sum of marginal entropies** is conserved:

$$C = \sum_i h_i = \text{constant}$$

where $h_i = -\text{tr}(\rho_i \log \rho_i)$ is the von Neumann entropy of the $i$-th marginal.

### At the Origin

For a Bell state, both marginals are maximally mixed:
$$\rho_A = \rho_B = \frac{I}{d}$$

This means:
- Each marginal entropy: $h_i = \log d$ (maximum)
- Constraint value: $C = 2n \log d$ (sum over all marginals)
- **Constraint gradient**: $\nabla C = a \approx 0$

The constraint gradient vanishes because any small perturbation that changes one marginal's entropy is balanced by changes in other correlations. The Bell state sits at a **saddle point** of the marginal entropy landscape.

### Verification

```python
from qig.exponential_family import QuantumExponentialFamily

d = 3
qef = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
theta_bell = qef.get_bell_state_parameters(log_epsilon=-10)

C, a = qef.marginal_entropy_constraint(theta_bell)
print(f"||a|| = {np.linalg.norm(a):.2e}")
print(f"||a|| / ||θ|| = {np.linalg.norm(a)/np.linalg.norm(theta_bell):.2e}")
# Result: ||a||/||θ|| ≈ 10⁻¹⁷ — essentially zero
```

### Implication

When $a \approx 0$:
- The projection $\Pi = I - aa^\top/\|a\|^2 \approx \mathbf{I}$ (no projection needed)
- The Lagrange multiplier $\nu = (a^\top G \theta)/(a^\top a) \to 0/0$ (indeterminate)
- **No coupling through the constraint** at the origin!

---

## 3. Two Time Parametrisations

### Game Time ($\tau$) — Affine Parameter

The natural parameter flow in game time is.

$$\frac{\text{d}\theta}{\text{d}\tau} = -\Pi_\parallel G \theta$$

**Behaviour at origin**:

- $\|\dot{\theta}\| \to 0$ as $\epsilon \to 0$ (regularisation parameter)
- $\text{d}H/\text{d}\tau \to 0$
- System appears to **freeze**

| $\log(\epsilon)$ | $\|\text{d}\theta/\text{d}\tau\|$ | $\text{d}H/\text{d}\tau$ |
|--------|-----------|-------|
| -5  | 6.4×10⁻² | 4.1×10⁻³ |
| -10 | 7.4×10⁻⁴ | 5.5×10⁻⁷ |
| -15 | 7.0×10⁻⁶ | 4.9×10⁻¹¹ |
| -20 | 6.1×10⁻⁸ | 3.7×10⁻¹⁵ |

### Entropy Time ($t$)

Define entropy time by normalising entropy production to unity.
$$\frac{\text{d}H}{\text{d}t} = 1$$
This gives the reparametrised flow
$$\frac{\text{d}\theta}{\text{d}t} = \frac{-\Pi_\parallel G \theta}{\theta^\top G \Pi_\parallel G \theta}$$

**Behaviour at origin**
- Numerator $\to 0$
- Denominator $\to 0$  
- **Ratio remains finite** (l'Hôpital-like behaviour)
- $\|\dot{\theta}\|$ **diverges** as $\epsilon \to 0$!

| $\log(\epsilon)$ | $\|\text{d}\theta/\text{d}t\|$ |
|--------|-------------|
| -5  | 15.6 |
| -10 | 1,354 |
| -15 | 142,567 |
| -20 | 16,392,761 |

### Verification

```python
import numpy as np
from qig.exponential_family import QuantumExponentialFamily

d = 3
qef = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)

for log_eps in [-5, -10, -15, -20]:
    theta = qef.get_bell_state_parameters(log_epsilon=log_eps)
    G = qef.fisher_information(theta)
    C, a = qef.marginal_entropy_constraint(theta)
    
    # Projection (handle a ≈ 0)
    a_norm_sq = np.dot(a, a)
    Pi = np.eye(len(theta)) - np.outer(a, a) / a_norm_sq if a_norm_sq > 1e-20 else np.eye(len(theta))
    
    # Game time flow: dθ/dτ = -Π G θ
    flow_game = -Pi @ G @ theta
    
    # Entropy production rate: dH/dτ = θᵀ G Π G θ
    entropy_production = theta @ G @ Pi @ G @ theta
    
    # Entropy time flow: dθ/dt = flow_game / entropy_production
    flow_entropy = flow_game / entropy_production if entropy_production > 1e-30 else np.zeros_like(flow_game)
    
    print(f"log_ε={log_eps}: ||dθ/dτ||={np.linalg.norm(flow_game):.2e}, ||dθ/dt||={np.linalg.norm(flow_entropy):.2f}")
```

### Physical Interpretation

The dynamics flows **away** from the origin (entropy production). In **entropy time**:
- The system always evolves at unit entropy rate
- Near the pure state, there's very little entropy to "spend"
- Therefore, **huge changes in θ** are needed to produce unit entropy
- Looking backward: the origin is at **finite entropy distance** but **infinite game time distance**

This resolves the apparent contradiction:
- $\nabla_\theta H = -G\theta \to 0$ (vanishes in natural parameters)
- $\nabla_\eta H = -\theta \to \infty$ (diverges in mean parameters)
- The dynamics in entropy time captures the divergent behaviour correctly

---

## 4. The GENERIC Decomposition: $M = S + A$

The Jacobian of the flow decomposes into symmetric and antisymmetric parts,
$$
M = S + A,
$$
where:
- **S** (symmetric): Generates dissipation/entropy production
- **A** (antisymmetric): Generates Hamiltonian/unitary rotations

### At the Origin

For a regularised Bell state.

| Component | Magnitude | Physical Role |
|-----------|-----------|---------------|
| $\|S \cdot \theta\|$ | ~10⁶ | Dissipative tendency |
| $\|A \cdot \theta\|$ | ~10⁶ | Hamiltonian rotation |
| $\|(S+A) \cdot \theta\|$ | ~10⁰ | **Net flow** (mostly cancelled!) |

The key observation: **$S \cdot \theta$ and $A \cdot \theta$ nearly cancel**!

The system has a huge dissipative tendency (toward higher entropy) but this is almost exactly counterbalanced by the Hamiltonian rotation (on iso-entropy surfaces).

### Verification

```python
import numpy as np
from qig.exponential_family import QuantumExponentialFamily

d = 3
qef = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
theta_bell = qef.get_bell_state_parameters(log_epsilon=-10)

S = qef.symmetric_part(theta_bell)
A = qef.antisymmetric_part(theta_bell)

S_theta = S @ theta_bell
A_theta = A @ theta_bell
net = S_theta + A_theta

print(f"||S·θ|| = {np.linalg.norm(S_theta):.2e}")
print(f"||A·θ|| = {np.linalg.norm(A_theta):.2e}")
print(f"||S·θ + A·θ|| = {np.linalg.norm(net):.2e}")
print(f"Cancellation: {100*(1 - np.linalg.norm(net)/np.linalg.norm(S_theta)):.4f}%")
# Result: >99.9999% cancellation
```

### The Degeneracy Condition

The antisymmetric part satisfies:

$$A \cdot \nabla H = 0$$

This means $A \cdot \theta$ is **orthogonal to the entropy gradient** — it generates rotations that conserve entropy.

### Verification

```python
G = qef.fisher_information(theta_bell)
grad_H = -G @ theta_bell  # ∇H in natural params
A_theta = qef.antisymmetric_part(theta_bell) @ theta_bell

cos_angle = np.dot(A_theta, grad_H) / (np.linalg.norm(A_theta) * np.linalg.norm(grad_H))
print(f"cos(A·θ, ∇H) = {cos_angle:.2e}")
# Result: ≈ 10⁻⁸ — orthogonal as expected
```

---

## 5. Different Bell States = Different Strategies

The three Bell states ($k=0, 1, 2$) have:

| $k$ | $\|S \cdot \theta\|$ | $\|A \cdot \theta\|$ | $\|S+A \cdot \theta\|$ | $A \cdot \theta$ direction |
|---|--------|--------|-----------|---------------|
| 0 | 1.07×10⁶ | 1.07×10⁶ | **0.44** | → |
| 1 | 1.05×10⁶ | 1.05×10⁶ | **0.06** | ↗ (orthogonal to $k=0$) |
| 2 | 2.69×10⁵ | 2.69×10⁵ | **0.19** | ↖ (orthogonal to $k=0,1$) |

### Key Finding

- All have **same entropy production rate** (equally pure)
- All have **similar magnitude** dissipative and Hamiltonian tendencies
- But **different net flow** after $S$-$A$ cancellation!
- The antisymmetric flows are **mutually orthogonal**

**$k=1$ achieves the best cancellation** — smallest net $\|S+A \cdot \theta\|$.

### Verification

```python
import numpy as np
from qig.exponential_family import QuantumExponentialFamily
from scipy.linalg import logm

d = 3
qef = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)

def make_bell_state(d, k):
    """Make Bell state |Φ_k⟩ = (1/√d) Σ_j |j,(j+k)mod d⟩"""
    psi = np.zeros(d**2, dtype=complex)
    for j in range(d):
        psi[j * d + ((j + k) % d)] = 1.0 / np.sqrt(d)
    return psi

def get_theta_for_bell(qef, k, eps=1e-10):
    psi = make_bell_state(d, k)
    rho = np.outer(psi, psi.conj())
    rho_reg = (1 - eps) * rho + eps * np.eye(d**2) / d**2
    log_rho = logm(rho_reg)
    theta = np.zeros(qef.n_params)
    for a, F_a in enumerate(qef.operators):
        num = np.real(np.trace(log_rho @ F_a))
        denom = np.real(np.trace(F_a @ F_a))
        if denom > 1e-12:
            theta[a] = num / denom
    return theta

# Compare different Bell states
A_thetas = {}
for k in range(d):
    theta_k = get_theta_for_bell(qef, k)
    S_k = qef.symmetric_part(theta_k)
    A_k = qef.antisymmetric_part(theta_k)
    net_k = S_k @ theta_k + A_k @ theta_k
    A_thetas[k] = A_k @ theta_k
    print(f"k={k}: ||S+A·θ|| = {np.linalg.norm(net_k):.4f}")

# Check orthogonality of A·θ between different k
for i in range(d):
    for j in range(i+1, d):
        cos_ij = np.dot(A_thetas[i], A_thetas[j]) / (np.linalg.norm(A_thetas[i]) * np.linalg.norm(A_thetas[j]))
        print(f"cos(A_{i}·θ, A_{j}·θ) = {cos_ij:.6f}")
# Result: All cross-terms ≈ 0 — orthogonal!
```

### Interpretation as Strategy

The antisymmetric part $A$ encodes a "strategy":
- Which direction to rotate on the iso-entropy surface
- Different rotations cancel differently with dissipation
- **Efficient strategies** minimise parameter motion while producing entropy

---

## 6. Multi-Pair Interactions

### Not Through the Constraint

At the origin:
- $\nabla C = a \approx 0$ for each pair
- Lagrange multiplier $\nu \to 0/0$
- **No direct coupling** through constraint enforcement

### Through the Entropy Budget

In entropy time, the total entropy production is normalised
$$\frac{\text{d}H_{\text{total}}}{\text{d}t} = 1$$
For $n$ pairs:
$$\frac{\text{d}H_{\text{total}}}{\text{d}\tau} = \sum_{k=1}^n \left(\frac{\text{d}H_k}{\text{d}\tau}\right)$$

Each pair's flow gets scaled by:
$$\frac{1}{\sum_k (\text{d}H_k/\text{d}\tau)}$$

### The Competition

| $n$ pairs | Entropy budget per pair | Evolution speed per pair |
|---------|------------------------|-------------------------|
| 1 | 100% | 1× |
| 2 | 50% | 0.5× |
| 10 | 10% | 0.1× |

**More pairs → slower evolution per pair** (in entropy time).

If pairs have different entropy production rates:
- The "faster" pair dominates the budget
- "Slower" pairs get diluted
- **Natural selection for efficient entropy producers**

### Verification

```python
import numpy as np
from qig.exponential_family import QuantumExponentialFamily

d = 3
qef = QuantumExponentialFamily(n_pairs=1, d=d, pair_basis=True)
theta_bell = qef.get_bell_state_parameters(log_epsilon=-10)
G = qef.fisher_information(theta_bell)

# Single pair entropy production rate
entropy_prod_single = theta_bell @ G @ theta_bell

print("Effect of n pairs on per-pair evolution speed in entropy time:")
for n in [1, 2, 5, 10, 100]:
    total_entropy_prod = n * entropy_prod_single
    flow_per_pair = np.linalg.norm(-G @ theta_bell) / total_entropy_prod
    print(f"  n={n:3d}: ||dθ/dt|| per pair = {flow_per_pair:.4f}")
# Result: Linear slowdown with n
```

---

## 7. The Game Mechanics

### Setup
- $n$ qutrit pairs, each starting in a Bell state
- Different pairs may have different Bell states ($k=0, 1$, or $2$)
- Dynamics governed by constrained maximum entropy production

### The Rules
1. **Total entropy production = 1** (in entropy time)
2. **Marginal entropy sum conserved** (but trivially satisfied at origin)
3. **GENERIC structure**: $M = S + A$ decomposition

### The Strategic Elements

1. **Entropy Budget Sharing**: Pairs compete for the fixed entropy budget. More pairs = less budget per pair.

2. **Antisymmetric Differentiation**: Different Bell states have orthogonal $A \cdot \theta$ directions. These represent different "strategies" for balancing dissipation and rotation.

3. **Cancellation Efficiency**: The net flow $\|S+A\cdot \theta\|$ depends on how well dissipation and rotation cancel. $k=1$ is most efficient.

4. **Parameter Space Motion**: In entropy time, evolution in $\theta$-space can be fast ($k=0$) or slow ($k=1$), even though entropy production is identical.

### Geometry Near the Origin

- The dynamics flows **away** from the origin (entropy increases)
- Looking backward: the pure Bell state is at **infinite game time** behind any interior point
- But only at **finite entropy distance** (the entropy you've gained since starting)
- The origin is a **boundary of the state space** — a pure state
- Near the origin, the dynamics in entropy time is extremely fast (huge $\|\text{d}\theta/\text{d}t\|$) but produces little entropy per unit game time

---

## 8. Open Questions

1. **Selection dynamics**: Do certain Bell states ($k$ values) get selected over time? Does $k=1$'s efficiency give it an advantage?

2. **Symmetry breaking**: If all pairs start identical, does noise or the dynamics break this symmetry?

3. **Interaction beyond budget**: When pairs move away from the origin, does $a \neq 0$ create additional coupling through the constraint?

4. **Hamiltonian interpretation**: Can we extract the effective Hamiltonian from the antisymmetric part and interpret it physically?

5. **Thermodynamic limit**: What happens as $n \to \infty$? Does the per-pair dynamics simplify?

---

## 9. Summary

| Aspect | Game Time τ | Entropy Time t |
|--------|-------------|--------------|
| Flow speed near origin | $\|\text{d}\theta/\text{d}\tau\| \to 0$ | $\|\text{d}\theta/\text{d}t\| \to \infty$ |
| Entropy rate near origin | $\text{d}H/\text{d}\tau \to 0$ | $\text{d}H/\text{d}t = 1$ (by definition) |
| Backward distance to origin | Infinite | Finite (= current entropy $H$) |
| Multi-pair coupling | None ($a \approx 0$) | Through shared budget |
| Strategic element | Hidden | Explicit ($A \cdot \theta$ direction) |

The inaccessible game is **not boring** — the dynamics is rich when viewed in entropy time. The coupling between pairs comes not from the constraint (which is trivially satisfied at the origin) but from the **shared entropy clock** that forces pairs to compete for the evolution budget.

Different Bell states provide different "strategies" via their antisymmetric parts, and the balance between dissipation ($S$) and rotation ($A$) determines which strategies are efficient.

---

*Document generated from analysis session, December 2024*
