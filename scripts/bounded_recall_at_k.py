"""R@k of the bounded-fusion ranking, from the cached CIRR scores.

Why
---
Table 3 reported bounded fusion's R@1 only. Fusion reorders the top ten and
nothing else, so R@10 and R@50 are unchanged by construction, but R@5 can
move: a target can enter or leave the top five when the window is
reordered. This computes it offline -- no GPU.

Self-check before reporting
---------------------------
The same cache produced the published numbers, so the script first
recomputes four of them and refuses to report anything if one disagrees:

    base R@1    1550 / 4181  (37.072, Table 3 FC+AEP+RV)
    base R@5    3024 / 4181  (72.327)
    base R@10   3536 / 4181  (84.573, the coverage of the window)
    bounded R@1 1646 / 4181  (39.369, lambda = 1.0 frozen)

Anchored on query counts, not rounded percentages: the paper once printed
bounded R@1 as 39.370, which is 39.37 padded, while 1646/4181 = 39.3686.

A mismatch means the cache, the ranking rule or the tie-breaking differs
from what the paper used, and any R@5 it printed would be meaningless.

Usage
-----
  python scripts/bounded_recall_at_k.py rv_score_cache_fc_aep_rv.jsonl --lam 1.0
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fusion_controls import (  # noqa: E402
    CacheError, load_records, relevant_set, rule_base, rule_bounded,
)

N_QUERIES = 4181
# Published hit counts this cache must reproduce, out of 4181 queries.
ANCHORS = {("base", 1): 1550, ("base", 5): 3024, ("base", 10): 3536,
           ("bounded", 1): 1646}
KS = (1, 5, 10)


def hit_at_k(scores: Sequence[float], rel: set, k: int) -> int:
    """1 if any relevant candidate is among the k highest scores.

    Ties are broken by original position, exactly as fusion_controls.hit_at_1
    does, so k=1 reproduces the published R@1.
    """
    if not rel:
        return 0
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return 1 if any(i in rel for i in order[:k]) else 0


def recall_table(recs: Sequence[dict], multipositive: bool,
                 lam: float) -> dict:
    out = {}
    for name, rule in (("base", rule_base), ("bounded", rule_bounded)):
        for k in KS:
            hits = sum(hit_at_k(rule(r["base_score"], r["verification_score"], lam),
                                relevant_set(r, multipositive), k)
                       for r in recs)
            out[(name, k)] = (hits, 100.0 * hits / len(recs))
    return out


def check_anchors(table: dict) -> List[str]:
    bad = []
    for key, want in ANCHORS.items():
        got = table[key][0]
        if got != want:
            bad.append("%s R@%d = %d hits, published %d" % (key[0], key[1], got, want))
    return bad


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache")
    ap.add_argument("--lam", type=float, required=True,
                    help="the frozen bounded-fusion weight (1.0)")
    args = ap.parse_args(argv)

    try:
        recs, multipositive = load_records(args.cache)
    except CacheError as exc:
        print("CACHE ERROR: %s" % exc, file=sys.stderr)
        return 1
    if multipositive:
        print("ERROR: multi-positive (CIRCO) cache; Table 3 is CIRR.",
              file=sys.stderr)
        return 1
    if len(recs) != N_QUERIES:
        print("ERROR: %d records, expected all %d CIRR validation queries"
              % (len(recs), N_QUERIES), file=sys.stderr)
        return 1

    table = recall_table(recs, multipositive, args.lam)
    print("%-8s %4s %6s %8s" % ("ranking", "k", "hits", "R@k"))
    for (name, k), (hits, pct) in sorted(table.items()):
        print("%-8s %4d %6d %8.3f" % (name, k, hits, pct))

    bad = check_anchors(table)
    print()
    if bad:
        print("SELF-CHECK FAILED -- not reporting bounded R@5:")
        for b in bad:
            print("  " + b)
        return 1
    print("Self-check passed: all four published values reproduced.")
    print("bounded R@10 equals base R@10 by construction: %s"
          % (table[("bounded", 10)][0] == table[("base", 10)][0]))
    print("RESULT  bounded-fusion R@5 = %.3f  (%d / %d)"
          % (table[("bounded", 5)][1], table[("bounded", 5)][0], len(recs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
