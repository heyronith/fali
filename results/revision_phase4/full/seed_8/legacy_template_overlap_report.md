# Legacy template overlap report

Historical revision templates (exact):
- A: `When {S} and {IO} went to the {place}, {S} gave a {object} to`
- B: `Then, {S} and {IO} went to the {place}. {S} gave a {object} to`

Rewritten with Wang [A]=IO, [B]=S placeholders (BABA first-clause order):
- A↔ `When [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to`
- B↔ `Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to`

## Exact string matches against canonical BABA (prefix before final [A])

### Legacy A
- No exact match to any canonical BABA prefix.

### Legacy B
- EXACT match to BABA template_id=0: `Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]`

## Notes
- Overlap is reported only for exact (or whitespace-identical) string equality.
- No subjective 'similar template' labels.
- ABBA variants of exact matches are also exact ABBA transforms of those BABA strings.
