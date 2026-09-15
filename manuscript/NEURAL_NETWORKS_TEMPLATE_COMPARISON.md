# NEURAL NETWORKS TEMPLATE COMPARISON

**Status:** **NOT APPLICABLE — TEMPLATE ABSENT**

```text
NEURAL NETWORKS TEMPLATE NOT FOUND
```

## What was compared

Per instructions, comparison requires a verified downloaded Neural Networks / Elsevier template versus a new active shell.

| Item | Result |
|---|---|
| `elsarticle.cls` | NOT FOUND |
| `elsarticle-template*.tex` | NOT FOUND |
| `cas-sc.cls` / `cas-dc.cls` | NOT FOUND |
| Elsevier package README / zip in repo | NOT FOUND |
| `manuscript/neural_networks/` | NOT CREATED (Case C: do not invent) |

## Closest existing journal template (for awareness only)

`manuscript/machine_learning/` uses Springer `sn-jnl`. That is a **different venue**. It must **not** be treated as a Neural Networks template.

## Divergence table

No Elsevier baseline exists in-repo, so every Elsevier-specific checklist item is:

| Check | Classification |
|---|---|
| `\documentclass` / options | UNKNOWN / VERIFY — need official package |
| packages | UNKNOWN / VERIFY |
| title/front matter | UNKNOWN / VERIFY |
| authors/affiliations | UNKNOWN / VERIFY |
| corresponding-author fields | UNKNOWN / VERIFY |
| abstract / keywords / highlights | UNKNOWN / VERIFY |
| graphical abstract hooks | UNKNOWN / VERIFY |
| figure/table syntax | UNKNOWN / VERIFY |
| bibliography style | UNKNOWN / VERIFY |
| acknowledgments / funding / CRediT / COI | UNKNOWN / VERIFY |
| appendix/supplement handling | UNKNOWN / VERIFY |

**REQUIRED next step (outside this audit STOP):** obtain the official Elsevier Neural Networks / `elsarticle` or CAS template package from the journal author site, add it under a verified path, and re-run comparison.

## Compile note

Canonical Neural Networks shell compile was **not** attempted, because creating a shell without the official class would violate “Do NOT install random alternative templates merely to obtain a successful compile.”
