# Functional taxonomy (Phase-1C)

These labels separate **helpfulness under knockout** from **change relative to intact**.

## Quantities

- **Intact effect** `A − D_h`: how much the head helps IOI in the clean model.
- **Conditional effect** `B − C_h`: how much the head helps IOI after primary Name Movers are ablated.
- **Conditional gain** `(B − C_h) − (A − D_h)`: how much that helpfulness *increases* after the knockout.

## Classes

### Strong positive functional recruitment
The head is helpful under the corrected primary knockout **and** becomes more helpful after that knockout, with stable seed means and bootstrap CIs.

### Suggestive positive functional recruitment
Same directional pattern (helpful + more helpful), but CI consistency is weaker.

### Supportive but downregulated
The head remains helpful under knockout, but less helpful than in the intact model (negative conditional gain). Attention may still rise.

### Less suppressive after knockout
The head still hurts IOI under knockout (negative conditional effect), but hurts *less* than when intact (positive conditional gain). **This is not recruitment.**

### More suppressive after knockout
The head hurts IOI under knockout and hurts even more than when intact.

### Supportive / suppressive with no stable modulation
Direction of conditional effect is stable, but conditional gain is not.

### Unstable / mixed
Seed-level signs disagree; no robust claim.

## Sign flips

- `strong_suppressive_to_supportive`: mean intact < 0, mean conditional > 0, CI excludes 0 in matching direction for >=3/4 seeds.
- `suggestive_suppressive_to_supportive`: mean signs flip and seed means agree (>=3/4), but strong CI requirement fails.
- Analogous `strong_` / `suggestive_supportive_to_suppressive` for the opposite flip.
- `none` / `unstable`: no robust flip.
