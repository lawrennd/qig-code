---
id: "000E"
title: "Generalised Bell Basis and Near-Bell Testbed Constructors"
status: "Proposed"
created: "2026-06-13"
last_updated: "2026-06-13"
author: "Neil D. Lawrence"
related_requirements: []
related_cips: ["000C", "000D"]
tags: ["pair-operators", "lme", "bell-states", "testbed"]
---

# CIP-000E: Generalised Bell Basis and Near-Bell Testbed Constructors

## Status
- [x] Proposed
- [ ] Accepted
- [ ] Implemented
- [ ] Closed

## Summary

Add three functions to `qig/pair_operators.py` that construct the full
generalised Bell basis for d-level bipartite systems, a near-Bell Hamiltonian
diagonal in that basis, and a convenience constructor for the corresponding
`GibbsLockedFrame`. These fill a gap exposed by CIP-000D: the companion
notebook currently rebuilds the generalised Bell basis inline rather than
calling library code.

## Motivation

The LME origin identified in Lawrence-origin26 is the vicinity of a pure
maximally entangled (Bell) state. To study the Gibbs-locked frame near this
origin we need:

1. The **full generalised Bell basis** — all d² states
   `|Φ_{mn}⟩ = (1/√d) Σ_k ω^{km} |k, (k+n) mod d⟩` (ω = e^{2πi/d}).
   The existing `bell_state(d, k)` in `pair_operators.py` only covers the
   cyclic-shift sub-family (m = 0); it cannot span the full Bell basis.

2. A **near-Bell Hamiltonian** — diagonal in the generalised Bell basis with
   a spectrum that puts the ground state at the reference Bell state
   `|Φ_{00}⟩` and one neighbouring level at a distinct eigenvalue δ.
   Any mixture of Bell states has exactly maximally mixed marginals
   (ρ_A = I/d, ρ_B = I/d), so the Gibbs state is exactly LME for all β,
   approaching the pure Bell boundary as β → ∞.

3. A **convenience frame constructor** — wraps (2) and `GibbsLockedFrame`
   so that examples and tests can set up the canonical near-Bell testbed in
   one call.

Without these, every notebook and test that wants the near-Bell testbed must
inline the Bell-basis construction, risking inconsistency.

## Implementation

All three additions go in `qig/pair_operators.py`.

### `generalised_bell_basis(d)`

Return the d² × d² unitary matrix whose columns are the generalised Bell
states ordered by (m, n) with column index m·d + n:

```
U_bell[:, m*d+n] = |Φ_{mn}⟩
```

### `near_bell_hamiltonian(d, delta=0.1)`

Return a d²×d² Hermitian matrix H diagonal in the generalised Bell basis:

- `|Φ_{00}⟩`  eigenvalue 0      (ground state)
- `|Φ_{01}⟩`  eigenvalue delta  (one distinct level)
- all others  eigenvalue 1

The Gibbs state `ρ_0 = exp(-βH)/Z` is:
- exactly LME (marginals = I/d) for every β — any Bell mixture has uniform marginals
- near-pure-Bell for large β
- Gibbs-locked: `[K_0, H] = 0` trivially since K_0 = βH

### `near_bell_gibbs_frame(d, delta=0.1, beta=5.0)`

Convenience wrapper:

```python
H = near_bell_hamiltonian(d, delta)
return GibbsLockedFrame(H, beta=beta, dims=[d, d])
```

### Export

Add all three to `qig/__init__.py` and to the `__all__` list in
`pair_operators.py`.

### Notebook update

Once implemented, replace the inline Bell-basis construction in
`examples/gibbs_lock_hamiltonian_extraction.ipynb` (Section 1, cell 6) with:

```python
from qig.pair_operators import near_bell_gibbs_frame, generalised_bell_basis

frame = near_bell_gibbs_frame(d=3, delta=0.1, beta=5.0)
U_bell = generalised_bell_basis(d=3)
```

## Backward Compatibility

New functions only; no existing API changes.

## Testing Strategy

Add to `tests/test_pair_operators.py` (or a new `tests/test_bell_basis.py`):

- `generalised_bell_basis(d)` returns a unitary: `U† U = I`
- All d² columns have exactly maximally mixed marginals: `Tr_B(|Φ_{mn}⟩⟨Φ_{mn}|) = I/d`
- `near_bell_hamiltonian` is Hermitian, real, diagonal in the Bell basis with the correct spectrum
- `near_bell_gibbs_frame` produces a `GibbsLockedFrame` with `gibbs_lock_residual() < 1e-12` and `||ρ_A - I/d||_F < 1e-12`

## Implementation Status
- [ ] Add `generalised_bell_basis(d)` to `pair_operators.py`
- [ ] Add `near_bell_hamiltonian(d, delta)` to `pair_operators.py`
- [ ] Add `near_bell_gibbs_frame(d, delta, beta)` to `pair_operators.py`
- [ ] Export from `qig/__init__.py`
- [ ] Write tests
- [ ] Update companion notebook to call library functions

## References

- `qig/pair_operators.py` — existing `bell_state`, `bell_state_density_matrix`
- `qig/gibbs_lock.py` — `GibbsLockedFrame` (CIP-000C)
- `examples/gibbs_lock_hamiltonian_extraction.ipynb` — CIP-000D companion notebook
- Lawrence-origin26 — LME origin as the axiomatically distinguished game origin
