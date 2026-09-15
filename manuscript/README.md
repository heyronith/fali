# Manuscript directory

## ACTIVE MANUSCRIPT (current repository state)

```text
ACTIVE MANUSCRIPT:
manuscript/machine_learning/

TARGET JOURNAL (as currently built):
Machine Learning (Springer Nature) — sn-jnl

INTENDED NEXT TARGET (research-lead instruction):
Neural Networks (Elsevier)
```

**Canonical entry point today:**

```text
manuscript/machine_learning/main.tex
```

## NEURAL NETWORKS STATUS

```text
NEURAL NETWORKS TEMPLATE NOT FOUND
```

No verified Elsevier / `elsarticle` / CAS package is present in this repository.  
Therefore **`manuscript/neural_networks/` was not created** on this audit branch (Case C: do not invent a journal template).

After the official Neural Networks template package is added, the intended active layout will be:

```text
ACTIVE MANUSCRIPT:
manuscript/neural_networks/

TARGET JOURNAL:
Neural Networks (Elsevier)

Entry point:
manuscript/neural_networks/main.tex
```

## Historical / non-active manuscript locations

| Location | Notes |
|---|---|
| `archive/manuscripts/tmlr_rejected_*` | Frozen Attention Logit Intervention — TMLR rejected archive |
| `submission_tmlr/` | Full historical TMLR submission tree |
| `tmlr-style-file-main/` | Upstream TMLR style package / examples |
| `paper.tex` | Historical generic-article Frozen Attention draft |
| `tools/springer-template/` | Untracked Springer package extract/cache |

## Audit documents (this branch)

- `manuscript/TEMPLATE_AUDIT_BASELINE.md`
- `manuscript/TEMPLATE_INVENTORY.md`
- `manuscript/CURRENT_MANUSCRIPT_DIAGNOSIS.md`
- `manuscript/SCIENTIFIC_SOURCE_MANIFEST.json`
- `manuscript/NEURAL_NETWORKS_TEMPLATE_COMPARISON.md`
- `manuscript/CONTENT_MIGRATION_MAP.md`
- `manuscript/MANUSCRIPT_CONFLICT_REPORT.md`

## Cleanup policy on this branch

Destructive archive/delete of venue trees was **not** performed (Case C).  
Scientific prose remains in place; Git history remains the recovery path for any future cleanup.
