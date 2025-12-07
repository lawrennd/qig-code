::: {.cell .markdown colab_type="text" id="view-in-github"}
`<a href="https://colab.research.google.com/github/lawrennd/qig-code/blob/main/examples/boring_game_dynamics.ipynb" target="_parent">`{=html}`<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>`{=html}`</a>`{=html}
:::

::: {.cell .markdown id="\"0\""}

# The "Boring" Game: Dynamics from the LME Origin

### Neil D. Lawrence

### December 2025
:::

::: {.cell .code id="aTgg9NdI4zIQ"}
``` {.python}
# Auto-install QIG package if not available
import os

try:
    import qig
except ImportError:
    print("📦 Installing QIG package...")
    %pip install -q git+https://github.com/lawrennd/qig-code.git
    print("✓ QIG package installed!")
```
:::

::: {.cell .code id="iFA6fiYc5gUq"}
``` {.python}
import numpy as np
import matplotlib.pyplot as plt
```
:::

::: {.cell .code id="TkHi03ye5M_N"}
``` {.python}
# Plot configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
big_wide_figsize = (10, 5)
big_figsize = (8, 6)
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'axes.labelsize': 18,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
})

# Create output directory
os.makedirs('./diagrams', exist_ok=True)
print("✓ Configuration complete")
```
:::

::: {.cell .markdown id="1"}
## Summary

When the inaccessible game starts from the Locally Maximally Entangled (LME) origin—a product of Bell states—the marginal entropy constraint is **automatically satisfied** along the entire gradient flow. The game is "boring" in the sense that the constrained and unconstrained dynamics are identical: the constraint projection $\Pi_\parallel = \mathbf{I}$ throughout.

This document explains why this happens and when the game becomes non-trivial.
:::

::: {.cell .markdown id="2"}
## Setup

### The Bell State

For a pair of qutrits ($d=3$), the Bell state is
$$
|\Phi\rangle = \frac{1}{\sqrt{3}}(|00\rangle + |11\rangle + |22\rangle)
$$
This is a **maximally entangled** state with:
- Joint entropy: $H = 0$ (pure state)
- Marginal entropies: $h_A = h_B = \log 3$ (marginals are $\mathbf{I}/3$)
- Mutual information: $I(A:B) = 2 \log 3$ (maximum)

### The Constraint

The marginal entropy constraint is
$$
C = \sum_i h_i = h_A + h_B
$$
For the Bell state: $C = 2 \log 3$.

### The Target State

The maximum entropy state is $\mathbf{I}/9$ (maximally mixed), with:
- Joint entropy: $H = \log 9$
- Marginal entropies: $h_A = h_B = \log 3$ (marginals are *also* $\mathbf{I}/3$!)
- Mutual information: $I(A:B) = 0$
:::

::: {.cell .markdown id="3"}
## Key Observation: Both Endpoints Have the Same Marginals

```python
import numpy as np

d = 3
D = d * d  # 9

# Bell state
psi_bell = np.zeros(D, dtype=complex)
for j in range(d):
    psi_bell[j*d + j] = 1/np.sqrt(d)
rho_bell = np.outer(psi_bell, psi_bell.conj())

# Maximally mixed state
I_D = np.eye(D) / D

def partial_trace_B(rho, d):
    """Trace out subsystem B, keep A"""
    rho_A = np.zeros((d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                rho_A[i, j] += rho[i*d + k, j*d + k]
    return rho_A

# Compute marginals
rho_A_bell = partial_trace_B(rho_bell, d)
rho_A_mixed = partial_trace_B(I_D, d)

print("Marginal of Bell state:")
print(np.round(rho_A_bell.real, 4))

print("\nMarginal of I/9:")
print(np.round(rho_A_mixed.real, 4))

print(f"\nBoth are I/3: {np.allclose(rho_A_bell, np.eye(d)/d) and np.allclose(rho_A_mixed, np.eye(d)/d)}")
```

**Output:**
```
Marginal of Bell state:
[[0.3333 0.     0.    ]
 [0.     0.3333 0.    ]
 [0.     0.     0.3333]]

Marginal of I/9:
[[0.3333 0.     0.    ]
 [0.     0.3333 0.    ]
 [0.     0.     0.3333]]

Both are I/3: True
```

**Why?**
- Bell state: $\rho_A = \mathrm{Tr}_B(|\Phi\rangle\langle\Phi|) = \mathbf{I}/3$ (defining property of maximal entanglement)
- $\mathbf{I}/9 = (\mathbf{I}\otimes\mathbf{I})/9$: $\rho_A = \mathrm{Tr}_B(\mathbf{I}\otimes\mathbf{I}/9) = \mathbf{I}\cdot\mathrm{Tr}(\mathbf{I})/9 = \mathbf{I}\cdot 3/9 = \mathbf{I}/3$

## The Gradient Flow Preserves C Exactly

The steepest entropy ascent dynamics naturally preserve the constraint:

```python
import numpy as np

d = 3
D = 9

# Bell state
psi_bell = np.zeros(D, dtype=complex)
for j in range(d):
    psi_bell[j*d + j] = 1/np.sqrt(d)
rho_bell = np.outer(psi_bell, psi_bell.conj())

def partial_trace_B(rho, d):
    rho_A = np.zeros((d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                rho_A[i, j] += rho[i*d + k, j*d + k]
    return rho_A

def entropy(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-15]
    return -np.sum(eigvals * np.log(eigvals))

def constraint_C(rho, d):
    rho_A = partial_trace_B(rho, d)
    return 2 * entropy(rho_A)  # Symmetric state: h_A = h_B

# Regularize to avoid pure state singularity
eps = 0.01
rho = (1-eps) * rho_bell + eps * np.eye(D)/D

# Gradient flow simulation
dt = 0.001
n_steps = 300
C_initial = constraint_C(rho, d)

print(f"{'Step':<8} {'H':<12} {'C':<12} {'dC':<14}")
print("-" * 46)

for step in range(n_steps + 1):
    H = entropy(rho)
    C = constraint_C(rho, d)
    
    if step % 50 == 0:
        print(f"{step:<8} {H:<12.6f} {C:<12.6f} {C - C_initial:<+14.10f}")
    
    if step < n_steps:
        # Entropy gradient
        eigvals, eigvecs = np.linalg.eigh(rho)
        eigvals = np.maximum(eigvals, 1e-15)
        log_rho = eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T.conj()
        
        grad = -log_rho - np.eye(D)
        grad = grad - np.trace(grad) * np.eye(D) / D  # Trace-preserving
        
        # Update and project to valid density matrix
        rho_new = rho + dt * grad
        rho_new = (rho_new + rho_new.T.conj()) / 2
        eigvals_new, eigvecs_new = np.linalg.eigh(rho_new)
        eigvals_new = np.maximum(eigvals_new, 0)
        eigvals_new = eigvals_new / np.sum(eigvals_new)
        rho = eigvecs_new @ np.diag(eigvals_new) @ eigvecs_new.T.conj()
```

**Output:**
```
Step     H            C            dC            
----------------------------------------------
0        0.069315     2.197225     +0.0000000000 
50       0.658147     2.197225     +0.0000000000 
100      1.027185     2.197225     +0.0000000000 
150      1.298177     2.197225     +0.0000000000 
200      1.504272     2.197225     +0.0000000000 
250      1.663553     2.197225     +0.0000000000 
300      1.788283     2.197225     +0.0000000000 
```

The constraint $C = 2 \log 3$ is preserved to machine precision at every step!

## Multiple Pairs: Same Result

For $n$ pairs, the product of Bell states $|\Phi_1\rangle\otimes|\Phi_2\rangle\otimes\cdots\otimes|\Phi_n\rangle$ evolves toward $\mathbf{I}/D$ where $D = 9^n$. Both have all marginals $= \mathbf{I}/3$.

```python
import numpy as np

d = 3
n_pairs = 2
D = d ** (2 * n_pairs)  # 81
dims = [d, d, d, d]  # A1, B1, A2, B2

# Product of Bell states
psi_bell = np.zeros(9, dtype=complex)
for j in range(d):
    psi_bell[j*d + j] = 1/np.sqrt(d)
psi_product = np.kron(psi_bell, psi_bell)
rho_product = np.outer(psi_product, psi_product.conj())

def marginal_entropy(rho, dims, keep_idx):
    """Compute entropy of subsystem keep_idx"""
    n = len(dims)
    d_keep = dims[keep_idx]
    d_after = int(np.prod(dims[keep_idx+1:]))
    d_before = int(np.prod(dims[:keep_idx]))
    
    rho_reduced = np.zeros((d_keep, d_keep), dtype=complex)
    for i in range(d_keep):
        for j in range(d_keep):
            val = 0
            for before in range(d_before):
                for after in range(d_after):
                    idx_i = before * (d_keep * d_after) + i * d_after + after
                    idx_j = before * (d_keep * d_after) + j * d_after + after
                    val += rho[idx_i, idx_j]
            rho_reduced[i, j] = val
    
    eigvals = np.linalg.eigvalsh(rho_reduced)
    eigvals = eigvals[eigvals > 1e-15]
    return -np.sum(eigvals * np.log(eigvals))

def total_C(rho):
    return sum(marginal_entropy(rho, dims, i) for i in range(4))

def joint_entropy(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-15]
    return -np.sum(eigvals * np.log(eigvals))

# Regularize with whole-system regularization
eps = 0.01
rho = (1-eps) * rho_product + eps * np.eye(D)/D

# Gradient flow
dt = 0.0002
n_steps = 150
C_initial = total_C(rho)

print(f"2 PAIRS: Bell⊗Bell -> I/81")
print(f"{'Step':<8} {'H':<12} {'C':<12} {'dC':<14}")
print("-" * 46)

for step in range(n_steps + 1):
    H = joint_entropy(rho)
    C = total_C(rho)
    
    if step % 30 == 0:
        print(f"{step:<8} {H:<12.6f} {C:<12.6f} {C - C_initial:<+14.10f}")
    
    if step < n_steps:
        eigvals, eigvecs = np.linalg.eigh(rho)
        eigvals = np.maximum(eigvals, 1e-15)
        log_rho = eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T.conj()
        
        grad = -log_rho - np.eye(D)
        grad = grad - np.trace(grad) * np.eye(D) / D
        
        rho_new = rho + dt * grad
        rho_new = (rho_new + rho_new.T.conj()) / 2
        eigvals_new, eigvecs_new = np.linalg.eigh(rho_new)
        eigvals_new = np.maximum(eigvals_new, 0)
        eigvals_new = eigvals_new / np.sum(eigvals_new)
        rho = eigvecs_new @ np.diag(eigvals_new) @ eigvecs_new.T.conj()

print(f"\nTarget: H = log 81 = {np.log(81):.6f}, C = 4 log 3 = {4*np.log(3):.6f}")
```

**Output:**
```
2 PAIRS: Bell⊗Bell -> I/81
Step     H            C            dC            
----------------------------------------------
0        0.098713     4.394449     +0.0000000000 
30       0.470614     4.394449     +0.0000000000 
60       0.781082     4.394449     +0.0000000000 
90       1.055112     4.394449     +0.0000000000 
120      1.299948     4.394449     +0.0000000000 
150      1.520100     4.394449     +0.0000000000 

Target: H = log 81 = 4.394449, C = 4 log 3 = 4.394449
```

## Intuition: What's Happening to the Entanglement?

The Bell state density matrix has a specific structure:

```python
import numpy as np

d = 3
psi_bell = np.zeros(9, dtype=complex)
for j in range(d):
    psi_bell[j*d + j] = 1/np.sqrt(d)
rho_bell = np.outer(psi_bell, psi_bell.conj())

print("Bell state rho:")
print("Rows/cols: |00>, |01>, |02>, |10>, |11>, |12>, |20>, |21>, |22>")
print(np.round(rho_bell.real, 3))
```

**Output:**
```
Bell state rho:
Rows/cols: |00>, |01>, |02>, |10>, |11>, |12>, |20>, |21>, |22>
[[0.333 0.    0.    0.    0.333 0.    0.    0.    0.333]
 [0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
 [0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
 [0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
 [0.333 0.    0.    0.    0.333 0.    0.    0.    0.333]
 [0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
 [0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
 [0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
 [0.333 0.    0.    0.    0.333 0.    0.    0.    0.333]]
```

**Two components:**
- **Diagonal** (positions 0,4,8): The classical correlation—only $|00\rangle$, $|11\rangle$, $|22\rangle$ are occupied
- **Off-diagonal** ($|00\rangle\langle 11|$, $|00\rangle\langle 22|$, $|11\rangle\langle 22|$): The quantum coherence = entanglement

**During the gradient flow:**
1. Off-diagonal coherences **decay** → entanglement destroyed
2. Diagonal **spreads** $\rightarrow$ all 9 states become equally likely ($1/9$)
3. Marginals **stay $\mathbf{I}/3$** throughout

```python
import numpy as np

d = 3
D = 9

psi_bell = np.zeros(D, dtype=complex)
for j in range(d):
    psi_bell[j*d + j] = 1/np.sqrt(d)
rho_bell = np.outer(psi_bell, psi_bell.conj())

def partial_trace_B(rho, d):
    rho_A = np.zeros((d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                rho_A[i, j] += rho[i*d + k, j*d + k]
    return rho_A

def entropy(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-15]
    return -np.sum(eigvals * np.log(eigvals))

def mutual_information(rho, d):
    rho_A = partial_trace_B(rho, d)
    H_A = entropy(rho_A)
    H_AB = entropy(rho)
    return 2 * H_A - H_AB

eps = 0.01
rho = (1-eps) * rho_bell + eps * np.eye(D)/D

dt = 0.001
n_steps = 400

print(f"{'Step':<6} {'H':<10} {'I(A:B)':<10} {'Off-diag':<12} {'h_A':<10}")
print("-" * 50)

for step in range(n_steps + 1):
    H = entropy(rho)
    I_AB = mutual_information(rho, d)
    h_A = entropy(partial_trace_B(rho, d))
    
    # Off-diagonal magnitude
    off_diag = np.sqrt(sum(np.abs(rho[i,j])**2 for i in range(D) for j in range(D) if i != j))
    
    if step % 80 == 0:
        print(f"{step:<6} {H:<10.4f} {I_AB:<10.4f} {off_diag:<12.6f} {h_A:<10.4f}")
    
    if step < n_steps:
        eigvals, eigvecs = np.linalg.eigh(rho)
        eigvals = np.maximum(eigvals, 1e-15)
        log_rho = eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T.conj()
        grad = -log_rho - np.eye(D)
        grad = grad - np.trace(grad) * np.eye(D) / D
        rho_new = rho + dt * grad
        rho_new = (rho_new + rho_new.T.conj()) / 2
        eigvals_new, eigvecs_new = np.linalg.eigh(rho_new)
        eigvals_new = np.maximum(eigvals_new, 0)
        eigvals_new = eigvals_new / np.sum(eigvals_new)
        rho = eigvecs_new @ np.diag(eigvals_new) @ eigvecs_new.T.conj()
```

**Output:**
```
Step   H          I(A:B)     Off-diag     h_A       
--------------------------------------------------
0      0.0693     2.1279     0.808332     1.0986    
80     1.1112     0.8541     0.505835     1.0986    
160    1.6845     0.3791     0.317953     1.0986    
240    1.9684     0.1502     0.187556     1.0986    
320    2.1092     0.0495     0.101329     1.0986    
400    2.1724     0.0131     0.049724     1.0986    
```

| Quantity | Changes | Direction |
|----------|---------|-----------|
| $H$ (joint entropy) | ✓ | $0 \to \log 9$ |
| $I(A:B)$ (mutual information) | ✓ | $2 \log 3 \to 0$ |
| Off-diagonal coherences | ✓ | Decay to 0 |
| $h_A = h_B$ (marginal entropies) | ✗ | Constant at $\log 3$ |
| $C = h_A + h_B$ | ✗ | Constant at $2 \log 3$ |

## When Does the Game Become Interesting?

The constraint only engages when starting from states where marginals $\neq \mathbf{I}/d$:

```python
import numpy as np

d = 3
n_pairs = 2
D = 81
dims = [d, d, d, d]

# Bell state (pair 1) x |00><00| (pair 2)
psi_bell = np.zeros(9, dtype=complex)
for j in range(d):
    psi_bell[j*d + j] = 1/np.sqrt(d)
rho_bell = np.outer(psi_bell, psi_bell.conj())

rho_00 = np.zeros((9, 9))
rho_00[0, 0] = 1  # |00><00|

# Mixed marginals: Bell (h=log3) x |00> (h=0)
rho_mixed = np.kron(rho_bell, rho_00)

def marginal_entropy(rho, dims, keep_idx):
    n = len(dims)
    d_keep = dims[keep_idx]
    d_after = int(np.prod(dims[keep_idx+1:]))
    d_before = int(np.prod(dims[:keep_idx]))
    
    rho_reduced = np.zeros((d_keep, d_keep), dtype=complex)
    for i in range(d_keep):
        for j in range(d_keep):
            val = 0
            for before in range(d_before):
                for after in range(d_after):
                    idx_i = before * (d_keep * d_after) + i * d_after + after
                    idx_j = before * (d_keep * d_after) + j * d_after + after
                    val += rho[idx_i, idx_j]
            rho_reduced[i, j] = val
    
    eigvals = np.linalg.eigvalsh(rho_reduced)
    eigvals = eigvals[eigvals > 1e-15]
    return -np.sum(eigvals * np.log(eigvals))

def total_C(rho):
    return sum(marginal_entropy(rho, dims, i) for i in range(4))

print("State: Bell ⊗ |00><00|")
print(f"Marginal entropies: h_A1={marginal_entropy(rho_mixed, dims, 0):.4f}, "
      f"h_B1={marginal_entropy(rho_mixed, dims, 1):.4f}, "
      f"h_A2={marginal_entropy(rho_mixed, dims, 2):.4f}, "
      f"h_B2={marginal_entropy(rho_mixed, dims, 3):.4f}")
print(f"C = {total_C(rho_mixed):.4f}")
print(f"Target (I/81): C = {total_C(np.eye(D)/D):.4f}")

# Regularize and take one step
eps = 0.01
rho = (1-eps) * rho_mixed + eps * np.eye(D)/D

eigvals, eigvecs = np.linalg.eigh(rho)
log_rho = eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T.conj()
grad = -log_rho - np.eye(D)
grad = grad - np.trace(grad) * np.eye(D) / D

rho_new = rho + 0.001 * grad
rho_new = (rho_new + rho_new.T.conj()) / 2

print(f"\nAfter one unconstrained step:")
print(f"C_before = {total_C(rho):.6f}")
print(f"C_after  = {total_C(rho_new):.6f}")
print(f"dC = {total_C(rho_new) - total_C(rho):.6f}")
print("\n=> Constraint would ENGAGE to project out this change!")
```

**Output:**
```
State: Bell ⊗ |00><00|
Marginal entropies: h_A1=1.0986, h_B1=1.0986, h_A2=0.0000, h_B2=0.0000
C = 2.1972
Target (I/81): C = 4.3944

After one unconstrained step:
C_before = 2.286564
C_after  = 2.350563
dC = 0.063999

=> Constraint would ENGAGE to project out this change!
```

## Summary

| Initial State | $C$ | Target $C$ | Constraint Active? |
|---------------|-----|------------|-------------------|
| $|\Phi\rangle \otimes |\Phi\rangle \otimes \cdots$ | $n \times 2 \log d$ | $n \times 2 \log d$ | **No** |
| Mixed marginals | $< n \times 2 \log d$ | $n \times 2 \log d$ | **Yes** |

The "boring" game from the LME origin:
1. All marginals are $\mathbf{I}/d$, and stay $\mathbf{I}/d$ along the flow
2. The constraint is automatically satisfied
3. Constrained = unconstrained dynamics
4. Entanglement dissipates while marginal structure is preserved

The game becomes interesting when:
1. Different subsystems have different marginal structures
2. The unconstrained flow would change $C$
3. The constraint projection actively shapes the dynamics
