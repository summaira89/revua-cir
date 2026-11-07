"""Regression parity of a re-run against a stored run (v3.1 §19.1, condition 4).

The tolerance was declared before any test: bit-identical values. Every row of
the run's JSONL output must equal the stored row exactly (same keys, same
numbers as written), and results.json must equal the stored one except the
provenance fields `manifest`, `wall_seconds` and `output_files`.

  python scripts/parity_compare.py --stored ../bbnv_runs/windows_cirr_v1 --rerun ../edir_work/parity/windows_cirr
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence

PROVENANCE = ("manifest", "wall_seconds", "output_files")


def compare(stored: str, rerun: str) -> List[str]:
    problems = []
    rs, rr = (json.load(open(os.path.join(d, "results.json"), encoding="utf-8")) for d in (stored, rerun))
    for k in sorted(set(rs) | set(rr)):
        if k not in PROVENANCE and rs.get(k, "<absent>") != rr.get(k, "<absent>"):
            problems.append("results.json field %r differs" % k)
    names = sorted(n for n in rs.get("output_files", {}) if n.endswith(".jsonl"))
    if names != sorted(n for n in rr.get("output_files", {}) if n.endswith(".jsonl")):
        return problems + ["the runs list different JSONL outputs"]
    for name in names:
        with open(os.path.join(stored, name), encoding="utf-8") as a, open(os.path.join(rerun, name), encoding="utf-8") as b:
            la, lb = a.read().splitlines(), b.read().splitlines()
        if len(la) != len(lb):
            problems.append("%s: %d rows stored, %d re-run" % (name, len(la), len(lb)))
        for i, (x, y) in enumerate(zip(la, lb)):
            if json.loads(x) != json.loads(y):
                problems.append("%s: row %d differs (first difference)" % (name, i))
                break
    return problems


def compare_features(stored: str, rerun: str, cos_tol: float = 1e-4) -> List[str]:
    """GPU-stage parity of two feature runs (cluster vs laptop): every non-float
    array (row pointers, ids, paths, texts) equal exactly; every embedding
    within cos_tol (1 - cosine), the declared COS_TOL."""
    import numpy as np
    problems = []
    with np.load(os.path.join(stored, "features.npz"), allow_pickle=False) as a, \
            np.load(os.path.join(rerun, "features.npz"), allow_pickle=False) as b:
        if sorted(a.files) != sorted(b.files):
            return ["the runs store different arrays: %s vs %s" % (sorted(a.files), sorted(b.files))]
        for k in sorted(a.files):
            x, y = a[k], b[k]
            if x.shape != y.shape:
                problems.append("%s: shape %s vs %s" % (k, x.shape, y.shape))
            elif x.dtype.kind == "f":
                x64, y64 = x.astype(np.float64), y.astype(np.float64)
                worst = float((1.0 - np.einsum("nd,nd->n", x64, y64)
                               / (np.linalg.norm(x64, axis=1) * np.linalg.norm(y64, axis=1))).max())
                print("%s: max 1-cos %.3g" % (k, worst))
                if worst > cos_tol:
                    problems.append("%s: max 1-cos %.3g > %.0e" % (k, worst, cos_tol))
            elif k.endswith("_paths"):
                # identity is the file name; the folder prefix depends on each machine's layout
                base = lambda arr: [str(p).replace("\\", "/").rsplit("/", 1)[-1] for p in arr]  # noqa: E731
                if base(x) != base(y):
                    problems.append("%s: file names differ" % k)
                elif not np.array_equal(x, y):
                    print("%s: same files, folder prefix differs (%s vs %s)" % (k, x[0], y[0]))
            elif not np.array_equal(x, y):
                problems.append("%s differs" % k)
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stored", required=True)
    ap.add_argument("--rerun", required=True)
    ap.add_argument("--features", action="store_true", help="compare two feature runs (GPU-stage parity)")
    args = ap.parse_args(argv)
    if args.features:
        problems = compare_features(args.stored, args.rerun)
        for p in problems:
            print(p)
        print("FEATURE PARITY: within tolerance" if not problems else "FEATURE PARITY FAILED: %d" % len(problems))
        return 0 if not problems else 1
    problems = compare(args.stored, args.rerun)
    for p in problems:
        print(p)
    n = sum(1 for _ in open(os.path.join(args.rerun, "windows.jsonl"), encoding="utf-8")) \
        if os.path.exists(os.path.join(args.rerun, "windows.jsonl")) else 0
    print("PARITY: bit-identical (%d rows)" % n if not problems else "PARITY FAILED: %d problem(s)" % len(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
