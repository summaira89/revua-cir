"""Which queries each fusion rule rescues and breaks, as sets rather than counts.

The questions
-------------
An external review raised three objections to Section 5.4 as written:

  (3) The manuscript says bounded and rank fusion rescue "the same 246"
      queries. Equal counts do not establish equal sets. The intersection,
      and each rule's exclusive rescues and breaks, have to be reported
      before that claim can be made.

  (4) The both-wrong category conflates two different failures: the target
      was never in the retrieved window, so no reranker could have helped,
      and the target was in the window but neither rule promoted it. Only
      the second is a reranking failure.

  (5) Table 9 uses rank 0 without stating the convention.

This script answers (3) and (4) from the cached scores, and fixes (5) by
stating the convention it uses and printing both numbering schemes.

Why the distinction in (3) matters
----------------------------------
Two rules that rescue the same number of queries by rescuing different
queries are not equivalent: their agreement bounds how much of the gain is
attributable to the verification signal as opposed to the rule. On the
220-query CIRCO cache the two rules' rescue sets overlap at Jaccard 0.333
despite comparable counts, which is what prompted building this.

Rank convention
---------------
Ranks here are **0-based**: rank 0 is the top-ranked candidate, matching the
project's `_rank_fuse`, which computes `argsort(descending).argsort()` so the
best candidate gets 0. Every rank printed below is 0-based and labelled; the
1-based equivalent is shown alongside so a reader cannot mistake one for the
other.

Usage
-----
  python -m revua_cir.scripts.rescue_set_analysis CACHE.jsonl --lam 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fusion_controls import (  # noqa: E402
    CacheError, load_records, relevant_set, rule_base, rule_bounded,
    rule_rank, hit_at_1,
)


def ranked_order(scores: Sequence[float]) -> List[int]:
    """Candidate indices best-first, ties by index. Position 0 is rank 0."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def best_relevant_rank(scores: Sequence[float], rel: set) -> Optional[int]:
    """0-based rank of the best-placed relevant candidate, or None.

    None means no relevant candidate is in this window at all, which is a
    different condition from "present but ranked last" and is counted
    separately throughout.
    """
    if not rel:
        return None
    order = ranked_order(scores)
    for pos, i in enumerate(order):
        if i in rel:
            return pos
    return None


def outcome_sets(recs: Sequence[dict], rel: Sequence[set],
                 lam: float) -> Dict[str, Dict[str, set]]:
    """Per-rule sets of rescued and broken query indices.

    Rescued: base wrong at rank 0, rule right. Broken: the reverse. These are
    sets of query indices, not counts, which is the whole point.
    """
    out: Dict[str, Dict[str, set]] = {}
    base_hits = [hit_at_1(rule_base(r["base_score"], r["verification_score"],
                                    0.0), s)
                 for r, s in zip(recs, rel)]
    for name, fn, use_lam in (("bounded", rule_bounded, lam),
                              ("rank", rule_rank, 0.0)):
        hits = [hit_at_1(fn(r["base_score"], r["verification_score"], use_lam),
                         s) for r, s in zip(recs, rel)]
        out[name] = {
            "rescued": {i for i in range(len(recs))
                        if base_hits[i] == 0 and hits[i] == 1},
            "broken": {i for i in range(len(recs))
                       if base_hits[i] == 1 and hits[i] == 0},
            "hits": hits,
        }
    out["_base_hits"] = base_hits  # type: ignore[assignment]
    return out


def jaccard(a: set, b: set) -> float:
    """|a and b| / |a or b|; 1.0 for two empty sets, which is the useful
    convention here (two rules that rescue nothing agree completely)."""
    u = a | b
    return 1.0 if not u else len(a & b) / len(u)


def both_wrong_breakdown(recs: Sequence[dict], rel: Sequence[set],
                         sets: Dict[str, Dict[str, set]]) -> dict:
    """Split the both-wrong queries into reachable and unreachable.

    A query neither rule gets right divides into:
      * target absent from the window -- no reranking could have helped, so
        this is a first-stage failure, not a verification failure;
      * target present but neither rule promoted it to rank 0 -- this one is
        a genuine reranking failure.
    The manuscript's both-wrong category currently merges the two.
    """
    base_hits = sets["_base_hits"]
    bnd, rnk = sets["bounded"]["hits"], sets["rank"]["hits"]
    absent = present_not_promoted = 0
    present_ranks: List[int] = []
    for i, rec in enumerate(recs):
        if base_hits[i] or bnd[i] or rnk[i]:
            continue
        r = best_relevant_rank(rec["base_score"], rel[i])
        if r is None:
            absent += 1
        else:
            present_not_promoted += 1
            present_ranks.append(r)
    present_ranks.sort()
    med = (present_ranks[len(present_ranks) // 2] if present_ranks else None)
    return {
        "both_wrong_total": absent + present_not_promoted,
        "target_absent_from_window": absent,
        "target_present_not_promoted": present_not_promoted,
        "median_base_rank_0based": med,
        "median_base_rank_1based": None if med is None else med + 1,
    }


def analyse(path: str, lam: float) -> dict:
    recs, multipositive = load_records(path)
    rel = [relevant_set(r, multipositive) for r in recs]
    sets = outcome_sets(recs, rel, lam)

    rb, br = sets["bounded"]["rescued"], sets["bounded"]["broken"]
    rr, kr = sets["rank"]["rescued"], sets["rank"]["broken"]

    coverage = sum(1 for i, rec in enumerate(recs)
                   if best_relevant_rank(rec["base_score"], rel[i]) is not None)

    return {
        "path": path,
        "n_queries": len(recs),
        "lam": lam,
        "multipositive": multipositive,
        "coverage": coverage,
        "rescued": {
            "bounded": len(rb), "rank": len(rr),
            "both": len(rb & rr),
            "bounded_only": len(rb - rr), "rank_only": len(rr - rb),
            "jaccard": jaccard(rb, rr),
        },
        "broken": {
            "bounded": len(br), "rank": len(kr),
            "both": len(br & kr),
            "bounded_only": len(br - kr), "rank_only": len(kr - br),
            "jaccard": jaccard(br, kr),
        },
        "both_wrong": both_wrong_breakdown(recs, rel, sets),
    }


def report(res: dict) -> None:
    print("=" * 76)
    print("RESCUE AND BREAK SETS, NOT COUNTS")
    print("=" * 76)
    print("cache     : %s" % os.path.basename(res["path"]))
    print("queries   : %d   (target in window: %d)"
          % (res["n_queries"], res["coverage"]))
    print("lambda    : %s   (bounded rule; rank fusion has no parameter)"
          % res["lam"])
    print("relevance : %s" % ("multi-positive" if res["multipositive"]
                              else "single-target"))
    print()
    print("Ranks below are 0-BASED: rank 0 is the top candidate, matching")
    print("the project's _rank_fuse. The 1-based equivalent is shown too.")
    print()

    for label in ("rescued", "broken"):
        d = res[label]
        print("QUERIES %s" % label.upper())
        print("   bounded            %5d" % d["bounded"])
        print("   rank               %5d" % d["rank"])
        print("   by both            %5d" % d["both"])
        print("   bounded only       %5d" % d["bounded_only"])
        print("   rank only          %5d" % d["rank_only"])
        print("   Jaccard overlap    %5.3f" % d["jaccard"])
        print()

    print("  Equal counts would not imply equal sets. The overlap above is")
    print("  what licenses or refuses a claim that two rules act alike.")
    print()

    bw = res["both_wrong"]
    print("QUERIES NEITHER RULE GETS RIGHT")
    print("   total                         %5d" % bw["both_wrong_total"])
    print("   target absent from window     %5d   (no reranker could help)"
          % bw["target_absent_from_window"])
    print("   target present, not promoted  %5d   (a reranking failure)"
          % bw["target_present_not_promoted"])
    if bw["median_base_rank_0based"] is not None:
        print("   median base rank of those     %5d (0-based) = %d (1-based)"
              % (bw["median_base_rank_0based"], bw["median_base_rank_1based"]))
    print()
    print("  The manuscript's both-wrong category merges these two. Only the")
    print("  second is attributable to verification.")
    print("=" * 76)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache")
    ap.add_argument("--lam", type=float, required=True,
                    help="frozen bounded-fusion weight")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)

    if args.lam < 0:
        print("error: --lam must be >= 0", file=sys.stderr)
        return 2
    try:
        res = analyse(args.cache, args.lam)
    except CacheError as exc:
        print("CACHE ERROR: %s" % exc, file=sys.stderr)
        return 1

    report(res)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print()
        print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
