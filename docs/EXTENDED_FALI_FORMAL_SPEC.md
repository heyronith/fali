# Extended / Causal FALI — Formal Specification (EF-0)

Mathematical specification only. No empirical model claims. No novelty claims.

Branch: `extended-fali/formalization` (from `pre-extended-fali-audited`).

---

## 1. Research question

When primary name-movers are knocked out, how much **candidate-head causal usefulness**
changes, and how much of that change is attributable to structured interventions on
target attention score (T), competing scores (R), other candidate-head scores (O),
candidate values (V), and surrounding context (C)?

---

## 2. Estimand

For candidate head *h* and prompt instance, define endpoint IO−S values (see §3) and
**candidate-head causal usefulness**:

\[
U(x) = \text{IO−S}(\text{context } x,\; h\ \text{injected}) - \text{IO−S}(\text{same context},\; h\ \text{zeroed})
\]

where injection/ zeroing follow the hybrid construction in §6–§7.

The primary estimand is:

\[
\Delta_U = U(B\text{-state}) - U(A\text{-state}) = \texttt{conditional\_gain}
\]

which equals `conditional_effect − intact_effect` at the two complete endpoints
(A_h, D_h, B_h, C_h notation from `docs/CURRENT_DEFINITIONS.md`).

Extended FALI attributes **Δ_U**, not raw model IO−S performance.

---

## 3. A/B endpoint notation

| Symbol | Meaning |
|--------|---------|
| **A** | Intact context (primary movers not ablated) |
| **B** | Primary-damaged context (L9H6+L9H9+L10H0 ablated) |
| **A_h** | IO−S with candidate present under A |
| **D_h** | IO−S with candidate zeroed under A |
| **B_h** | IO−S with candidate present under B |
| **C_h** | IO−S with candidate zeroed under B |

`intact_effect = A_h − D_h`, `conditional_effect = B_h − C_h`,
`conditional_gain = conditional_effect − intact_effect`.

---

## 4. T / R / O / V / C player definitions

Five **binary intervention factors** (0 = A-state, 1 = B-state):

| Player | Intervention |
|--------|----------------|
| **T** | Target attention score at final task query → IO key |
| **R** | All other **causally valid** competing scores in that same final-query row |
| **O** | All remaining causally valid attention-score entries for this candidate head at **other** query positions |
| **V** | Candidate head value vectors at all source positions |
| **C** | Surrounding-network / context state (intact A vs primary-damaged B) |

**T ∪ R ∪ O** partitions every causally valid score coordinate of the candidate head
exactly once. Masked / future coordinates are structural constants, not players.

---

## 5. Mask semantics

Causal validity follows the same rule as revision code: at query position *q*, only
keys *k ≤ q* with finite scores are causally valid. Positions with −inf / non-finite
scores are mask constants copied from the live reference row; they are never assigned
to T, R, or O as players.

---

## 6. Hybrid-score construction

For binary state (T, R, O) ∈ {0,1}³:

1. Build full pre-softmax score tensor **S(T,R,O)** by injecting A- or B-state scores
   per player on their coordinate blocks.
2. Preserve the exact causal/padding mask.
3. Compute **P(T,R,O) = masked_softmax(S(T,R,O))** on causally valid entries only.

When an entire block is switched (all of SCORE at 0 or all at 1), this reduces to
the cached A-pattern or B-pattern used in Phase-3 factorial cells.

---

## 7. Hybrid-z reconstruction

For (T,R,O,V):

1. Select value tensor V_A or V_B according to V.
2. Reconstruct candidate output at each query *q*:

\[
z(T,R,O,V)[q] = \sum_k P(T,R,O)[q,k]\; V[k]
\]

3. Inject this **full** candidate z at all query positions under context C.
4. The zero-candidate comparison for U uses the **same** context C.

No live candidate computation may leak into an injected hybrid cell.

---

## 8. Utility U

For each of the 2⁵ = **32** binary cells, run the forward pass with the hybrid
candidate head and compute U as in §2. Define cooperative-game values:

\[
v(S) = U\big(\text{state}(S)\big),\quad v(\emptyset) = U(0,0,0,0,0)
\]

so that attributions sum to **Δ_U = v(N) − v(∅)**.

---

## 9. Coalition structure (frozen)

\[
\Pi = \{\;\text{SCORE}=\{T,R,O\},\; \text{VALUE}=\{V\},\; \text{CONTEXT}=\{C\}\;\}
\]

Order is fixed: **SCORE ≺ VALUE ≺ CONTEXT**. This hierarchy is prospective and must
not be rearranged based on empirical results.

---

## 10. Exact Owen formula (hierarchical, fixed order)

Permutations are uniform over:

- any order within SCORE (3!),
- then V,
- then C.

For player *i* in group *G_k*, all players from earlier groups are always in the
predecessor coalition; within *G_k*, predecessors range over subsets of *G_k \ {i}*
with factorial weights. Two independent implementations are required:

- **A.** subset-weight enumeration (`owen_value_subset`);
- **B.** brute-force hierarchical permutation enumeration (`owen_value_permutation`).

See `src/extended_fali_math.py`.

---

## 11. Global efficiency theorem

**Proposition (efficiency).** For any v and fixed-order hierarchical Owen attributions
φ_i:

\[
\phi_T + \phi_R + \phi_O + \phi_V + \phi_C = v(N) - v(\emptyset) = \Delta_U.
\]

*Proof sketch.* In each admissible permutation, sequential marginal contributions
telescope from v(∅) to v(N). Averaging preserves telescoping. ∎

Verified synthetically at tolerance 1e−12 (`verify_efficiency`).

---

## 12. Quotient / group consistency theorem

Define the **quotient game** on meta-players {SCORE, VALUE, CONTEXT} by activating
all members of a selected group. Because meta-order is frozen (not all 3! meta
permutations), use **fixed-order meta marginals**:

\[
\Phi_{\text{SCORE}} = v(\{T,R,O\}) - v(\emptyset)
\]
\[
\Phi_{\text{VALUE}} = v(\{T,R,O,V\}) - v(\{T,R,O\})
\]
\[
\Phi_{\text{CONTEXT}} = v(N) - v(\{T,R,O,V\})
\]

**Proposition (group consistency).** Under fixed-order hierarchical Owen:

\[
\phi_T+\phi_R+\phi_O = \Phi_{\text{SCORE}},\quad
\phi_V = \Phi_{\text{VALUE}},\quad
\phi_C = \Phi_{\text{CONTEXT}}.
\]

*Proof sketch.* Players in SCORE are processed before VALUE/C in every admissible
permutation; their Owen marginals telescope to v(TRO)−v(∅). Single-player groups
V and C have unique predecessor sets. ∎

**Note:** This differs from full 3-player **Shapley** on the quotient game, which
averages over all meta permutations. The frozen hierarchy uses fixed-order meta
marginals (`quotient_shapley_fixed_order`).

---

## 13. Bridge to Phase-3 P×V×C

When the entire SCORE block switches together:

- SCORE=A ⇒ injected pattern equals P_A;
- SCORE=B ⇒ injected pattern equals P_B;

provided cached patterns equal masked_softmax(cached scores) (future numerical check).

The 3-group quotient game {SCORE, V, C} then matches the Phase-3 factorial over
(P, V, C) at matched endpoints. **Expected future numerical correspondence** (not
assumed here):

\[
\Phi_{\text{SCORE}} \approx \phi_{\text{pattern}},\quad
\phi_V \approx \phi_{\text{value}},\quad
\phi_C \approx \phi_{\text{context}}
\]

within predefined FP32 tolerances after model implementation.

---

## 14. Original-FALI special case

**Do not** identify old FALI numbers with Extended FALI causal attributions.

**Special case.** If utility is changed from U to target attention probability p_IO;
O, V, C fixed at A; only T and R vary; then the two-player Shapley values on
(T,R) equal the path-stable **target-score / competing-score** FALI equations in
`revision_fali.fali_shapley`. Extended FALI generalizes the intervention partition
and changes the primary utility to causal usefulness Δ_U.

---

## 15. What φ_T, φ_R, φ_O, φ_V, φ_C **do** mean

Hierarchical Owen allocations of **Δ_U** to each intervention factor, respecting
frozen SCORE ≺ VALUE ≺ CONTEXT precedence, including interaction effects (§19).

---

## 16. What they **do not** mean

- Not guaranteed equal to old descriptive FALI target/competing attributions on
  attention probability.
- Not purely “direct effects”; interactions are embedded in Owen allocations.
- Not claims about physical mechanisms until model-level validation passes preregistered
  gates.

---

## 17. Off-manifold counterfactual limitation

Hybrid injections construct off-manifold states. Claims are conditional on this
intervention algebra and may not correspond to any single natural forward pass except
at the two complete endpoints (all-0 and all-1 cells).

---

## 18. Context C limitation

C captures differences in surrounding computation between A and B **outside** the
explicitly injected candidate P/V state. It is not reducible to downstream sensitivity
alone (see `docs/CURRENT_DEFINITIONS.md`).

---

## 19. Interaction interpretation

Owen values allocate cooperative interactions among players. A positive φ_T may reflect
T main effects, interactions with R/O inside SCORE, or cross-group interactions (e.g.
T×C) depending on join order. No separate interaction taxonomy is defined in EF-0.

---

## 20. Falsification criteria (synthetic + future model)

**Synthetic (EF-0, now):**

- Owen subset vs permutation max discrepancy < 1e−12 on random games;
- efficiency residual < 1e−12;
- fixed-order quotient residuals < 1e−12;
- original-FALI special-case algebra matches `fali_shapley`.

**Future model (not EF-0):**

- score→pattern→z reconstruction errors ≤ predefined FP32 tolerance;
- endpoint Δ_U matches measured `conditional_gain`;
- Φ_SCORE/V/C matches Phase-3 P×V×C decomposition within tolerance;
- no masked/future coordinate drift under interventions.

Failure of any preregistered future criterion falsifies the corresponding hypothesis
(M1–M4 in preregistration), not the historical Phase 1–4 record.
