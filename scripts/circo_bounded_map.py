"""Bounded-fusion mAP on CIRCO, computed offline from the cached rankings.

What this answers
-----------------
Section 5.6 reports CIRCO transfer under rank fusion, but the bounded rule is
the one the paper deploys, and its mAP was missing. An external review asked
for it, noting correctly that mAP@25 and mAP@50 can move even when top-10
membership does not, because reordering inside the window changes precision
at the ranks where relevant images sit.

No GPU is needed. The cache already stores `base_top50_gallery_ids` for every
query, so the full top-50 ranking can be rebuilt offline: rerank the first
ten by bounded fusion, leave positions 11--50 in their base order, and score.

The metric
----------
Reimplemented from the official CIRCO evaluator
(`miccunifi/CIRCO`, `src/evaluation.py`), which computes

    ap_labels  = isin(predictions, gt_img_ids)
    precisions = cumsum(ap_labels) * ap_labels / arange(1, n+1)
    AP@K       = sum(precisions[:K]) / min(len(gt_img_ids), K)

Two details matter and are easy to get wrong. The relevance indicator zeroes
non-relevant positions, so only ranks holding a ground-truth image
contribute. And the denominator is `min(n_gt, K)`, not `n_gt`: with four
ground truths and K=5 a query can reach AP 1.0, which it could not if the
denominator were fixed at the number of ground truths.

`test_ap_matches_the_official_formula` pins both against hand-computed cases,
and `test_real_cache_base_map5_matches_published` checks the base-ranking
mAP@5 against the 24.49 already reported, which is the anchor that says this
reimplementation agrees with the evaluator that produced the published
number.

Scope
-----
This is CIRCO **validation** (220 queries). The official test split has 800
queries and server-held labels; the two are not comparable and are never
mixed here. Lambda is the value frozen on CIRR, not retuned on CIRCO: a
CIRCO-tuned weight would be a different experiment and is not run.

Usage
-----
  python -m revua_cir.scripts.circo_bounded_map CACHE.jsonl --lam 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fusion_controls import (  # noqa: E402
    CacheError, _zscore, rule_rank,
)

RANKS: Tuple[int, ...] = (5, 10, 25, 50)
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 54321


def load_circo_records(path: str) -> List[dict]:
    """Load, requiring the fields mAP needs rather than only the fusion ones.

    A CIRR cache has no `base_top50_gallery_ids` and no
    `relevant_gallery_indices`, so pointing this script at one must fail
    loudly instead of silently scoring a 10-wide window as if it were 50.
    """
    recs: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CacheError("%s:%d: malformed JSON (%s)"
                                 % (os.path.basename(path), lineno, exc))
            for f in ("base_top50_gallery_ids", "relevant_gallery_indices",
                      "candidate_gallery_ids", "base_score",
                      "verification_score"):
                if f not in rec:
                    raise CacheError(
                        "%s:%d: missing %r -- this script needs a CIRCO cache "
                        "with full top-50 rankings and multi-positive "
                        "annotations" % (os.path.basename(path), lineno, f))
            n = len(rec["candidate_gallery_ids"])
            for f in ("base_score", "verification_score"):
                if len(rec[f]) != n:
                    raise CacheError("%s:%d: %s has %d entries for %d "
                                     "candidates"
                                     % (os.path.basename(path), lineno, f,
                                        len(rec[f]), n))
            if len(set(rec["base_top50_gallery_ids"])) != \
                    len(rec["base_top50_gallery_ids"]):
                raise CacheError("%s:%d: duplicate ids in the top-50 ranking"
                                 % (os.path.basename(path), lineno))
            recs.append(rec)
    if not recs:
        raise CacheError("%s: no records" % path)
    return recs


def average_precision_at_k(ranked_ids: Sequence[int], gt_ids: Sequence[int],
                           k: int) -> float:
    """AP@k exactly as the official CIRCO evaluator computes it."""
    gt = set(gt_ids)
    if not gt:
        return 0.0
    hits = 0
    total = 0.0
    for pos, img in enumerate(ranked_ids[:k], start=1):
        if img in gt:
            hits += 1
            total += hits / pos
    return total / min(len(gt), k)


def bounded_rerank_top50(rec: dict, lam: float) -> List[int]:
    """Top-50 gallery ids with the first ten reordered by bounded fusion.

    Positions 11--50 keep their base order, which is what a deployed system
    would do: verification is only computed for the retrieved window.
    """
    top50 = list(rec["base_top50_gallery_ids"])
    cand = list(rec["candidate_gallery_ids"])
    k = len(cand)

    zb = _zscore(rec["base_score"])
    zr = _zscore(rec["verification_score"])
    fused = [zb[i] + lam * math.tanh(zr[i]) for i in range(k)]
    order = sorted(range(k), key=lambda i: (-fused[i], i))
    reranked_window = [cand[i] for i in order]

    # The cached window is the base top-k, so it must be the head of the
    # top-50. Asserting rather than assuming: if the two ever disagree the
    # splice below would silently duplicate or drop ids.
    if top50[:k] != cand:
        raise CacheError("the cached candidate window is not the head of "
                         "base_top50_gallery_ids; the two were produced by "
                         "different rankings and cannot be spliced")
    return reranked_window + top50[k:]


def rank_rerank_top50(rec: dict) -> List[int]:
    """The same splice under the project's rank-fusion rule, for comparison."""
    top50 = list(rec["base_top50_gallery_ids"])
    cand = list(rec["candidate_gallery_ids"])
    k = len(cand)
    fused = rule_rank(rec["base_score"], rec["verification_score"], 0.0)
    order = sorted(range(k), key=lambda i: (-fused[i], i))
    if top50[:k] != cand:
        raise CacheError("candidate window is not the head of the top-50")
    return [cand[i] for i in order] + top50[k:]


def paired_bootstrap(a: Sequence[float], b: Sequence[float],
                     n_rep: int = N_BOOTSTRAP,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    """CI for mean(a) - mean(b) over per-query AP, resampling queries.

    The review asked for uncertainty on per-query AP differences rather than
    only a hit@1 McNemar test, because mAP is the metric CIRCO reports.
    """
    n = len(a)
    if n != len(b):
        raise ValueError("arms differ in length (%d vs %d)" % (n, len(b)))
    obs = (sum(a) - sum(b)) / n
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_rep):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(a[i] - b[i] for i in idx) / n)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]
    tail = (sum(1 for d in deltas if d <= 0.0) if obs >= 0
            else sum(1 for d in deltas if d >= 0.0))
    p = min(1.0, 2.0 * (tail + 1) / (len(deltas) + 1))
    return {"delta": 100.0 * obs, "ci_low": 100.0 * lo,
            "ci_high": 100.0 * hi, "p_two_sided": p}


def analyse(path: str, lam: float, n_rep: int = N_BOOTSTRAP) -> dict:
    recs = load_circo_records(path)

    rankings = {
        "base": [list(r["base_top50_gallery_ids"]) for r in recs],
        "bounded": [bounded_rerank_top50(r, lam) for r in recs],
        "rank": [rank_rerank_top50(r) for r in recs],
    }
    gts = [r["relevant_gallery_indices"] for r in recs]

    per_query: Dict[str, Dict[int, List[float]]] = {}
    maps: Dict[str, Dict[int, float]] = {}
    for name, ranked in rankings.items():
        per_query[name] = {}
        maps[name] = {}
        for k in RANKS:
            aps = [average_precision_at_k(ranked[i], gts[i], k)
                   for i in range(len(recs))]
            per_query[name][k] = aps
            maps[name][k] = 100.0 * sum(aps) / len(aps)

    contrasts = {}
    for name in ("bounded", "rank"):
        contrasts[name] = {
            k: paired_bootstrap(per_query[name][k], per_query["base"][k],
                                n_rep)
            for k in RANKS
        }

    # How often does reranking change the top-10 membership at all? If it
    # never does, any mAP@25/50 movement is purely reordering, which is the
    # effect the review pointed out.
    changed = sum(1 for i, r in enumerate(recs)
                  if rankings["bounded"][i][:10] != rankings["base"][i][:10])

    return {
        "path": path, "lam": lam, "n_queries": len(recs),
        "mean_gt_per_query": sum(len(g) for g in gts) / len(gts),
        "maps": maps, "contrasts": contrasts,
        "queries_with_reordered_top10": changed,
        "n_bootstrap": n_rep,
    }


def report(res: dict) -> None:
    print("=" * 74)
    print("CIRCO BOUNDED-FUSION mAP  (validation split, offline from cache)")
    print("=" * 74)
    print("cache      : %s" % os.path.basename(res["path"]))
    print("queries    : %d   mean ground truths: %.2f"
          % (res["n_queries"], res["mean_gt_per_query"]))
    print("lambda     : %s   (frozen on CIRR, NOT retuned on CIRCO)"
          % res["lam"])
    print("top-10 reordered in %d of %d queries"
          % (res["queries_with_reordered_top10"], res["n_queries"]))
    print()
    print("  %-10s" % "ranking" + "".join("%10s" % ("mAP@%d" % k)
                                          for k in RANKS))
    for name in ("base", "bounded", "rank"):
        row = "  %-10s" % name
        for k in RANKS:
            row += "%10.2f" % res["maps"][name][k]
        print(row)
    print()
    print("PAIRED CONTRASTS vs BASE (per-query AP, bootstrap over queries)")
    print("  %-10s %6s %9s %-22s %8s" % ("ranking", "K", "delta", "95% CI",
                                         "p"))
    for name in ("bounded", "rank"):
        for k in RANKS:
            d = res["contrasts"][name][k]
            print("  %-10s %6d %+9.2f [%+7.2f, %+7.2f] %8.4f"
                  % (name, k, d["delta"], d["ci_low"], d["ci_high"],
                     d["p_two_sided"]))
    print()
    print("  Positions 11-50 keep their base order; only the top-10 window")
    print("  is reranked. mAP@25 and mAP@50 can still move, because")
    print("  reordering inside the window changes precision at the ranks")
    print("  where relevant images sit.")
    print("=" * 74)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache")
    ap.add_argument("--lam", type=float, required=True,
                    help="the CIRR-frozen bounded-fusion weight")
    ap.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)

    if args.lam < 0:
        print("error: --lam must be >= 0", file=sys.stderr)
        return 2
    try:
        res = analyse(args.cache, args.lam, args.bootstrap)
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
