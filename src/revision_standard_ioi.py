"""
Canonical Wang et al. / Redwood Easy-Transformer IOI templates for revision Phase 4.

Does NOT modify the historical 2-template IOIDataset.

Experiment name:
  "canonical-template / public-implementation lexical distribution"

- The 15 low-level BABA templates match paper Appendix E (arXiv:2211.00593).
- Lexical inventory follows the *public Easy-Transformer implementation*:
  99 names, 8 places, 8 objects.
- Paper text describes ~100 names and ~20 total common place/object words;
  this module does **not** claim exact reproduction of every lexical detail
  of the paper distribution — only the public-implementation lists + Appendix E templates.

Source:
  https://github.com/redwoodresearch/Easy-Transformer
  easy_transformer/ioi_dataset.py (BABA_TEMPLATES × ABBA transform)
  Paper appendix E (arXiv:2211.00593) lists the same 15 BABA strings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

# Exact BABA strings from Easy-Transformer ioi_dataset.py (commit e168301aa0…)
BABA_TEMPLATES: List[str] = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

assert len(BABA_TEMPLATES) == 15


def make_abba_template(baba: str) -> str:
    """
    Exact Easy-Transformer ABBA transform: swap first-clause [B]↔[A] only.
    """
    first_clause = True
    chars = list(baba)
    # Operate on string with same logic as Easy-Transformer (index into original)
    out = baba
    i = 1
    while i < len(out) - 1:
        if out[i - 1 : i + 2] == "[B]" and first_clause:
            out = out[:i] + "A" + out[i + 1 :]
        elif out[i - 1 : i + 2] == "[A]" and first_clause:
            first_clause = False
            out = out[:i] + "B" + out[i + 1 :]
        i += 1
    return out


ABBA_TEMPLATES: List[str] = [make_abba_template(t) for t in BABA_TEMPLATES]

# Exact lists from Easy-Transformer ioi_dataset.py
NAMES: List[str] = [
    'Michael',
    'Christopher',
    'Jessica',
    'Matthew',
    'Ashley',
    'Jennifer',
    'Joshua',
    'Amanda',
    'Daniel',
    'David',
    'James',
    'Robert',
    'John',
    'Joseph',
    'Andrew',
    'Ryan',
    'Brandon',
    'Jason',
    'Justin',
    'Sarah',
    'William',
    'Jonathan',
    'Stephanie',
    'Brian',
    'Nicole',
    'Nicholas',
    'Anthony',
    'Heather',
    'Eric',
    'Elizabeth',
    'Adam',
    'Megan',
    'Melissa',
    'Kevin',
    'Steven',
    'Thomas',
    'Timothy',
    'Christina',
    'Kyle',
    'Rachel',
    'Laura',
    'Lauren',
    'Amber',
    'Brittany',
    'Danielle',
    'Richard',
    'Kimberly',
    'Jeffrey',
    'Amy',
    'Crystal',
    'Michelle',
    'Tiffany',
    'Jeremy',
    'Benjamin',
    'Mark',
    'Emily',
    'Aaron',
    'Charles',
    'Rebecca',
    'Jacob',
    'Stephen',
    'Patrick',
    'Sean',
    'Erin',
    'Jamie',
    'Kelly',
    'Samantha',
    'Nathan',
    'Sara',
    'Dustin',
    'Paul',
    'Angela',
    'Tyler',
    'Scott',
    'Katherine',
    'Andrea',
    'Gregory',
    'Erica',
    'Mary',
    'Travis',
    'Lisa',
    'Kenneth',
    'Bryan',
    'Lindsey',
    'Kristen',
    'Jose',
    'Alexander',
    'Jesse',
    'Katie',
    'Lindsay',
    'Shannon',
    'Vanessa',
    'Courtney',
    'Christine',
    'Alicia',
    'Cody',
    'Allison',
    'Bradley',
    'Samuel',
]

PLACES: List[str] = [
    "store",
    "garden",
    "restaurant",
    "school",
    "hospital",
    "office",
    "house",
    "station",
]

OBJECTS: List[str] = [
    "ring",
    "kiss",
    "bone",
    "basketball",
    "computer",
    "necklace",
    "drink",
    "snack",
]

def _dedupe(seq: Sequence[str]) -> List[str]:
    """Preserve order; drop duplicates if present in a source list."""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# Easy-Transformer currently ships 99 unique names; dedupe is defensive.
NAMES_DEDUPED = _dedupe(NAMES)


def stratum_name(template_id: int, pattern_type: str) -> str:
    return f"T{template_id:02d}_{pattern_type}"


def fill_template(template: str, A: str, B: str, place: str, obj: str) -> str:
    s = template.replace("[A]", A).replace("[B]", B)
    s = s.replace("[PLACE]", place).replace("[OBJECT]", obj)
    assert "[A]" not in s and "[B]" not in s and "[PLACE]" not in s and "[OBJECT]" not in s
    return s


def filter_single_token_words(
    tokenizer,
    words: Sequence[str],
    *,
    prefix: str = " ",
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Return (usable, rejected_records). Never silent."""
    usable: List[str] = []
    rejected: List[Dict[str, Any]] = []
    for w in words:
        ids = tokenizer.encode(prefix + w)
        if len(ids) == 1:
            usable.append(w)
        else:
            rejected.append(
                {
                    "word": w,
                    "prefix": prefix,
                    "n_tokens": len(ids),
                    "token_ids": ids,
                    "decoded": [tokenizer.decode([i]) for i in ids],
                }
            )
    return usable, rejected


@dataclass
class Provenance:
    source_repo: str
    source_file_url: str
    source_commit_sha: str
    paper_arxiv: str
    retrieval_date_utc: str
    n_low_level_templates: int
    n_pattern_types: int
    n_strata: int
    n_names_raw: int
    n_names_deduped: int
    n_places: int
    n_objects: int
    baba_templates: List[str]
    abba_templates: List[str]
    names: List[str]
    places: List[str]
    objects: List[str]
    notes: List[str]


def build_provenance(
    *,
    commit_sha: str,
    retrieval_date_utc: Optional[str] = None,
) -> Provenance:
    return Provenance(
        source_repo="https://github.com/redwoodresearch/Easy-Transformer",
        source_file_url=(
            "https://raw.githubusercontent.com/redwoodresearch/Easy-Transformer/"
            f"{commit_sha}/easy_transformer/ioi_dataset.py"
        ),
        source_commit_sha=commit_sha,
        paper_arxiv="https://arxiv.org/abs/2211.00593",
        retrieval_date_utc=retrieval_date_utc
        or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        n_low_level_templates=15,
        n_pattern_types=2,
        n_strata=30,
        n_names_raw=len(NAMES),
        n_names_deduped=len(NAMES_DEDUPED),
        n_places=len(PLACES),
        n_objects=len(OBJECTS),
        baba_templates=list(BABA_TEMPLATES),
        abba_templates=list(ABBA_TEMPLATES),
        names=list(NAMES_DEDUPED),
        places=list(PLACES),
        objects=list(OBJECTS),
        notes=[
            "Experiment: canonical-template / public-implementation lexical distribution.",
            "BABA_TEMPLATES copied verbatim from Easy-Transformer ioi_dataset.py; match paper Appendix E.",
            "ABBA_TEMPLATES produced with the same first-clause [B]↔[A] swap as Easy-Transformer.",
            "Lexical inventory = public Easy-Transformer implementation (99 names, 8 places, 8 objects).",
            "Paper text describes ~100 names and ~20 total common place/object words; "
            "we do not claim exact reproduction of every lexical detail of the paper distribution.",
            "Names are defensively deduplicated for sampling (no removals; source list already unique).",
        ],
    )


class StandardIOIDataset:
    """
    Balanced canonical IOI dataset over 15×2=30 template strata.

    Interface mirrors revision-critical fields of historical IOIDataset:
      clean_tokens, end, IO, S1, S2, io_token_ids, s_token_ids, batch_size, samples
    """

    def __init__(
        self,
        model,
        *,
        num_per_stratum: int = 2,
        seed: int = 42,
        names: Optional[Sequence[str]] = None,
        places: Optional[Sequence[str]] = None,
        objects: Optional[Sequence[str]] = None,
    ):
        self.model = model
        self.tokenizer = model.tokenizer
        self.num_per_stratum = int(num_per_stratum)
        self.seed = int(seed)
        self.n_strata = 30
        self.num_samples = self.num_per_stratum * self.n_strata

        names_in = list(names) if names is not None else list(NAMES_DEDUPED)
        places_in = list(places) if places is not None else list(PLACES)
        objects_in = list(objects) if objects is not None else list(OBJECTS)

        self.names, self.rejected_names = filter_single_token_words(
            self.tokenizer, names_in
        )
        self.places, self.rejected_places = filter_single_token_words(
            self.tokenizer, places_in
        )
        self.objects, self.rejected_objects = filter_single_token_words(
            self.tokenizer, objects_in
        )
        if len(self.names) < 3:
            raise RuntimeError(
                f"Need ≥3 single-token names; usable={len(self.names)}; "
                f"rejected={self.rejected_names}"
            )
        if not self.places or not self.objects:
            raise RuntimeError(
                f"Empty places/objects after tokenization filter. "
                f"rejected_places={self.rejected_places} rejected_objects={self.rejected_objects}"
            )

        self.rng = np.random.default_rng(self.seed)
        self.samples = self._generate_samples()
        self._materialize_tensors()
        self.validate()

    def _generate_samples(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        example_idx = 0
        for tid in range(15):
            for pattern in ("BABA", "ABBA"):
                template = BABA_TEMPLATES[tid] if pattern == "BABA" else ABBA_TEMPLATES[tid]
                for _ in range(self.num_per_stratum):
                    # Draw distinct S, IO
                    io, s = self.rng.choice(self.names, size=2, replace=False)
                    place = str(self.rng.choice(self.places))
                    obj = str(self.rng.choice(self.objects))
                    # In templates: [A]=IO, [B]=S (subject)
                    full = fill_template(template, A=str(io), B=str(s), place=place, obj=obj)
                    samples.append(
                        {
                            "example_idx": example_idx,
                            "low_level_template_id": tid,
                            "pattern_type": pattern,
                            "template_stratum": stratum_name(tid, pattern),
                            "template_string": template,
                            "full_prompt_with_io": full,
                            "S": str(s),
                            "IO": str(io),
                            "place": place,
                            "object": obj,
                            "seed": self.seed,
                        }
                    )
                    example_idx += 1
        assert example_idx == self.num_samples
        return samples

    def _materialize_tensors(self) -> None:
        tokenizer = self.tokenizer
        device = self.model.cfg.device
        prompt_token_lists: List[List[int]] = []
        meta_rows: List[Dict[str, Any]] = []

        bos = tokenizer.bos_token_id
        assert bos is not None

        for sample in self.samples:
            full = sample["full_prompt_with_io"]
            s_name, io_name = sample["S"], sample["IO"]
            s_id = tokenizer.encode(" " + s_name)
            io_id = tokenizer.encode(" " + io_name)
            assert len(s_id) == 1 and len(io_id) == 1
            s_id, io_id = s_id[0], io_id[0]

            # Tokenize without special tokens then prepend BOS (matches HookedTransformer.to_tokens)
            body = tokenizer.encode(full, add_special_tokens=False)
            full_ids = [bos] + body
            assert full_ids[-1] == io_id, (
                f"Final token must be IO for example {sample['example_idx']}: "
                f"got {full_ids[-1]} expected {io_id} prompt={full!r}"
            )
            # Model input excludes final IO
            prompt_ids = full_ids[:-1]
            end_pos = len(prompt_ids) - 1

            # Positions among prompt_ids (which still contain S1,S2,IO in first clause)
            s_positions = [i for i, t in enumerate(prompt_ids) if t == s_id]
            io_positions = [i for i, t in enumerate(prompt_ids) if t == io_id]
            assert len(s_positions) == 2, (
                f"S must appear twice in prompt tokens; got {s_positions} for {full!r}"
            )
            assert len(io_positions) == 1, (
                f"IO must appear once in truncated prompt; got {io_positions} for {full!r}"
            )
            s1, s2 = s_positions[0], s_positions[1]
            io_pos = io_positions[0]
            assert s1 < s2
            # Semantic order: BABA = S IO S; ABBA = IO S S (after stripping final IO)
            if sample["pattern_type"] == "BABA":
                assert s1 < io_pos < s2, (
                    f"BABA expected S1 < IO < S2; got {s1},{io_pos},{s2} for {full!r}"
                )
            else:
                assert io_pos < s1 < s2, (
                    f"ABBA expected IO < S1 < S2; got {io_pos},{s1},{s2} for {full!r}"
                )
            assert end_pos > max(s1, s2, io_pos)
            assert prompt_ids[end_pos] != tokenizer.eos_token_id

            clean_prompt = tokenizer.decode(prompt_ids[1:])  # drop BOS for readable string
            # Ensure names reconstruct
            assert s_name in clean_prompt and io_name in clean_prompt

            sample["clean_prompt"] = clean_prompt
            sample["end_position"] = end_pos
            sample["S1_position"] = s1
            sample["S2_position"] = s2
            sample["IO_position"] = io_pos
            sample["s_token_id"] = s_id
            sample["io_token_id"] = io_id

            prompt_token_lists.append(prompt_ids)
            meta_rows.append(sample)

        # Pad
        max_len = max(len(x) for x in prompt_token_lists)
        eos = tokenizer.eos_token_id
        padded = []
        for ids in prompt_token_lists:
            pad = [eos] * (max_len - len(ids))
            padded.append(ids + pad)

        self.clean_tokens = torch.tensor(padded, dtype=torch.long, device=device)
        self.batch_size, self.seq_len = self.clean_tokens.shape
        self.end = torch.tensor(
            [s["end_position"] for s in self.samples], dtype=torch.long, device=device
        )
        self.S1 = torch.tensor(
            [s["S1_position"] for s in self.samples], dtype=torch.long, device=device
        )
        self.S2 = torch.tensor(
            [s["S2_position"] for s in self.samples], dtype=torch.long, device=device
        )
        self.IO = torch.tensor(
            [s["IO_position"] for s in self.samples], dtype=torch.long, device=device
        )
        self.s_token_ids = torch.tensor(
            [s["s_token_id"] for s in self.samples], dtype=torch.long, device=device
        )
        self.io_token_ids = torch.tensor(
            [s["io_token_id"] for s in self.samples], dtype=torch.long, device=device
        )
        self.metadata_df = pd.DataFrame(
            [
                {
                    "example_idx": s["example_idx"],
                    "low_level_template_id": s["low_level_template_id"],
                    "pattern_type": s["pattern_type"],
                    "template_stratum": s["template_stratum"],
                    "prompt": s["clean_prompt"],
                    "S": s["S"],
                    "IO": s["IO"],
                    "S1_position": s["S1_position"],
                    "S2_position": s["S2_position"],
                    "IO_position": s["IO_position"],
                    "end_position": s["end_position"],
                    "place": s["place"],
                    "object": s["object"],
                    "seed": s["seed"],
                }
                for s in self.samples
            ]
        )

    def validate(self) -> None:
        meta = self.metadata_df
        assert len(meta) == self.num_samples
        assert meta["low_level_template_id"].nunique() == 15
        assert set(meta["pattern_type"]) == {"BABA", "ABBA"}
        counts = meta.groupby("template_stratum").size()
        assert len(counts) == 30
        assert (counts == self.num_per_stratum).all(), counts.value_counts().to_dict()
        # Per-template both patterns
        for tid in range(15):
            sub = meta[meta["low_level_template_id"] == tid]
            assert set(sub["pattern_type"]) == {"BABA", "ABBA"}

        eos = self.tokenizer.eos_token_id
        for i in range(self.batch_size):
            end = int(self.end[i].item())
            assert self.clean_tokens[i, end].item() != eos
            s_id = int(self.s_token_ids[i].item())
            io_id = int(self.io_token_ids[i].item())
            assert s_id != io_id
            toks = self.clean_tokens[i, : end + 1]
            assert (toks == s_id).sum().item() == 2
            assert (toks == io_id).sum().item() == 1
            assert int(self.IO[i].item()) < end
            assert int(self.S1[i].item()) < int(self.S2[i].item())

    def tokenization_filter_report(self) -> Dict[str, Any]:
        return {
            "usable_names": len(self.names),
            "rejected_names": self.rejected_names,
            "usable_places": len(self.places),
            "rejected_places": self.rejected_places,
            "usable_objects": len(self.objects),
            "rejected_objects": self.rejected_objects,
        }


def legacy_template_overlap_report() -> str:
    """Exact string/structural overlap vs historical 2-template IOIDataset."""
    legacy = {
        "A": "When {S} and {IO} went to the {place}, {S} gave a {object} to",
        "B": "Then, {S} and {IO} went to the {place}. {S} gave a {object} to",
    }
    # Map legacy to Wang tokens: S↔B, IO↔A for BABA-like first clause order S then IO = B then A? 
    # Legacy: "When {S} and {IO} ..." is S then IO = B then A in Wang notation = BABA first clause.
    legacy_as_wang = {
        "A": "When [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to",
        "B": "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    }
    lines = [
        "# Legacy template overlap report",
        "",
        "Historical revision templates (exact):",
        f"- A: `{legacy['A']}`",
        f"- B: `{legacy['B']}`",
        "",
        "Rewritten with Wang [A]=IO, [B]=S placeholders (BABA first-clause order):",
        f"- A↔ `{legacy_as_wang['A']}`",
        f"- B↔ `{legacy_as_wang['B']}`",
        "",
        "## Exact string matches against canonical BABA (prefix before final [A])",
        "",
    ]
    for lid, lstr in legacy_as_wang.items():
        hits = []
        for tid, baba in enumerate(BABA_TEMPLATES):
            # Canonical full ends with [A]; legacy has no final name
            prefix = baba[: -len(" [A]")] if baba.endswith(" [A]") else baba
            # Also try without trailing [A] only
            if baba.endswith("[A]"):
                prefix2 = baba[: -len("[A]")].rstrip()
            else:
                prefix2 = baba
            exact = lstr == prefix2 or lstr == prefix
            hits.append((tid, exact, baba, prefix2))
        exact_ids = [tid for tid, exact, _, _ in hits if exact]
        lines.append(f"### Legacy {lid}")
        if exact_ids:
            for tid in exact_ids:
                lines.append(f"- EXACT match to BABA template_id={tid}: `{BABA_TEMPLATES[tid]}`")
        else:
            lines.append("- No exact match to any canonical BABA prefix.")
            # Report closest by equality of normalized whitespace only — still exactness focused
            for tid, exact, baba, prefix2 in hits:
                if lstr.replace(" ", "") == prefix2.replace(" ", ""):
                    lines.append(
                        f"- Whitespace-only difference vs T{tid:02d}: canonical=`{baba}`"
                    )
        lines.append("")

    lines += [
        "## Notes",
        "- Overlap is reported only for exact (or whitespace-identical) string equality.",
        "- No subjective 'similar template' labels.",
        "- ABBA variants of exact matches are also exact ABBA transforms of those BABA strings.",
        "",
    ]
    return "\n".join(lines)


def template_inventory_df() -> pd.DataFrame:
    rows = []
    for tid, baba in enumerate(BABA_TEMPLATES):
        abba = ABBA_TEMPLATES[tid]
        rows.append(
            {
                "low_level_template_id": tid,
                "pattern_type": "BABA",
                "template_stratum": stratum_name(tid, "BABA"),
                "template_string": baba,
            }
        )
        rows.append(
            {
                "low_level_template_id": tid,
                "pattern_type": "ABBA",
                "template_stratum": stratum_name(tid, "ABBA"),
                "template_string": abba,
            }
        )
    return pd.DataFrame(rows)
