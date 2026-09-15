#!/usr/bin/env python3
"""Enforce Neural Networks highlight length: 3--5 items, each <=85 chars."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "highlights_items.tex"
text = path.read_text()
items = re.findall(r"\\item\s+(.*)", text)
ok = True
if not (3 <= len(items) <= 5):
    print(f"FAIL: expected 3--5 highlights, found {len(items)}")
    ok = False
for i, item in enumerate(items, 1):
    n = len(item.strip())
    status = "OK" if n <= 85 else "FAIL"
    if n > 85:
        ok = False
    print(f"{status} [{n:2d}/85] H{i}: {item.strip()}")
sys.exit(0 if ok else 1)
