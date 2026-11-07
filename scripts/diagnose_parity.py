"""Read-only diagnostic for the deployed-score parity check of
bank_normalized_verification.py (deployed_scores). CPU only; writes nothing.

For every query of each cache it recomputes the deployed verification score
from the cached terms with the analysis's own code path (corrected_scores with
zero correction) and compares it with the cached score. For each query whose
verifier order or full bounded-fusion permutation differs it prints the
candidates that change place, their cached and recomputed scores, the gaps
between them, and whether fused top-1 or Hit@1 changes. It also prints, per
cache, the largest |recomputed - cached| and how many adjacent pairs in the
cached verifier order are closer than 1e-6 and 1e-7 (near-ties).

Usage
-----
  python scripts/diagnose_parity.py CACHE NPZ [CACHE NPZ ...]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apply_frozen_policy import _order, hit_at  # noqa: E402
from bank_normalized_verification import (K_PRIMARY, LAMBDA, SPLIT_SEED, TERMS,  # noqa: E402
                                          corrected_scores)
from fusion_controls import load_records, relevant_set, split_indices  # noqa: E402
from fusion_suite import r_bounded  # noqa: E402


def diagnose(cache: str, npz: str) -> int:
    with open(npz[:-4] + ".manifest.json", encoding="utf-8") as fh:
        man = json.load(fh)
    recs, multi = load_records(cache)
    n, k = len(recs), man["k"]
    idx = list(range(n))
    zero = {t: {K_PRIMARY: np.zeros((n, k))} for t in TERMS}
    rec = corrected_scores(recs, idx, zero, man)
    first, second = split_indices(n, SPLIT_SEED)
    half = {i: "first" for i in first}
    half.update({i: "second (report/eval)" for i in second})
    diffs = [abs(x - y) for i in idx for x, y in zip(rec[i], recs[i]["verification_score"])]
    gaps = []
    for i in idx:
        c = recs[i]["verification_score"]
        o = _order(c)
        gaps += [c[o[j]] - c[o[j + 1]] for j in range(k - 1)]
    gaps = np.array(gaps)
    print("== %s/%s | %d queries | max |recomputed - cached| %.3g | adjacent cached gaps"
          " < 1e-6: %d, < 1e-7: %d, == 0: %d"
          % (man["dataset"], man.get("category") or "val", n, max(diffs),
             int((gaps < 1e-6).sum()), int((gaps < 1e-7).sum()), int((gaps == 0).sum())))
    bad = 0
    for i in idx:
        r = recs[i]
        cached, recomp = r["verification_score"], rec[i]
        oc, orc = _order(cached), _order(recomp)
        fc = _order(r_bounded(r["base_score"], cached, LAMBDA))
        fr = _order(r_bounded(r["base_score"], recomp, LAMBDA))
        if oc == orc and fc == fr:
            continue
        bad += 1
        rel = relevant_set(r, multi)
        moved = sorted({oc[j] for j in range(k) if oc[j] != orc[j]}
                       | {fc[j] for j in range(k) if fc[j] != fr[j]})
        print("query %d (%s half): verifier order %s, fused permutation %s"
              % (i, half[i], "differs" if oc != orc else "same",
                 "differs" if fc != fr else "same"))
        for j in moved:
            print("   cand %d: cached %.10f recomputed %.10f (diff %.2e) base %.6f relevant %s"
                  % (j, cached[j], recomp[j], recomp[j] - cached[j], r["base_score"][j], j in rel))
        if len(moved) >= 2:
            a, b = moved[0], moved[1]
            print("   cached gap %.3e | recomputed gap %.3e"
                  % (cached[a] - cached[b], recomp[a] - recomp[b]))
        print("   fused top-1 cached %d recomputed %d | Hit@1 cached %d recomputed %d"
              % (fc[0], fr[0], hit_at(fc, rel, 1), hit_at(fr, rel, 1)))
    print("   %d of %d queries differ" % (bad, n))
    return bad


def main(argv) -> int:
    if len(argv) < 2 or len(argv) % 2:
        print(__doc__, file=sys.stderr)
        return 2
    for cache, npz in zip(argv[0::2], argv[1::2]):
        diagnose(cache, npz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
