"""Enforce separate coverage percentages using raw counts, without rounding."""
import json
from pathlib import Path

totals = json.loads(Path("coverage.json").read_text())["totals"]
for title, hit, total, minimum in (
    ("statements", "covered_lines", "num_statements", 95),
    ("branches", "covered_branches", "num_branches", 90),
):
    print(f"{title}: {totals[hit]}/{totals[total]} ({100*totals[hit]/totals[total]:.3f}%)")
    if 100 * totals[hit] < minimum * totals[total]:
        raise SystemExit(f"{title} below {minimum}%")
