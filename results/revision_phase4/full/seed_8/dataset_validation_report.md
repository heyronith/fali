# Dataset validation report (Phase-4A smoke)

- N=900 = 30 × 30 strata
- seed=8
- usable single-token names: 99
- rejected names: 0
- usable places: 8 (rejected 0)
- usable objects: 8 (rejected 0)

## Balance
- template IDs present: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
- nunique template IDs: 15 (expect 15)
- pattern types: ['ABBA', 'BABA']
- n strata: 30 (expect 30)
- counts per stratum: [30]

## Token/index assertions
- All StandardIOIDataset.validate() checks passed at construction.
- end never on padding; S×2 / IO×1; S≠IO; decoded prompts contain names.

## Explicit tokenization rejects
- none (all canonical names/places/objects single-token under GPT-2 Small).
