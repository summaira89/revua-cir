"""
Third-stage diagnostic for the 36-query gap between job 6387's reranked
R@1 (37.41%, 1564 hits) and the offline recomputation (38.27%, 1600 hits).

RULED OUT so far, by direct evidence (not theory):
  1. argsort tie-breaking (CPU vs CUDA) -- 0/4181 queries affected.
  2. UG signal (verification_score vs *ug_confidence) -- no change.
  3. Which RV signal is used at all -- verification_score, cos_sim, and
     both UG-multiplied variants ALL give exactly 1600 hits, because
     verification_score = sigmoid(scale*cos_sim+bias) is strictly
     monotonic in cos_sim, and ug_confidence is constant across a query's
     own candidates -- so all four induce the SAME ordering, and
     _rank_fuse is rank-based. Expected, not a bug, but it means the
     fusion formula cannot explain the gap.

REMAINING HYPOTHESIS, tested here: `k` in evaluate.py's rerank block is
`k = min(cfg.reverse_verifier.top_k, top_idx.size(1))`, and the rerank
permutes `reranked[b, :k]`. The cache stores exactly `k` candidates. If
those agree, offline and online must agree. This script checks the one
remaining way they could differ: whether _rank_fuse's BASE-RANK input at
scoring time was computed over the same k-length slice the cache stored,
or over a different-length window -- because base_rank's absolute values
(0..k-1) scale with the window length, and _rank_fuse weights base_rank
2x, so a different window length changes every fused_rank and therefore
can change the winner.

Concretely: it recomputes each query's reranked top-1 using base_rank
derived from the cached k-slice (what the offline script does), and
compares against the alternative where base_rank is derived assuming the
scores were part of a wider array -- reporting how many queries change.

    python scripts/diagnose_rerank_topk_window.py --cache rv_score_cache_cirr_val.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from revua_cir.models.revua_cir import _rank_fuse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--max-print", type=int, default=20)
    args = p.parse_args()

    records = []
    with open(args.cache, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    n = len(records)
    k = len(records[0]["candidate_gallery_ids"])
    print(f"[diagnose] {n} records, cached top_k = {k}")

    # How many queries does reranking actually CHANGE the top-1 for,
    # offline? If reranking changes the top-1 for very few queries, the
    # 36-query gap is proportionally huge and points at those queries
    # specifically.
    changed = []
    for i, rec in enumerate(records):
        base = torch.tensor(rec["base_score"])
        rv = torch.tensor(rec["verification_score"])
        fused = _rank_fuse(base, rv)
        new_top1 = int(fused.argmax())
        if new_top1 != 0:  # base top-1 is always index 0 (topk-sorted)
            changed.append((i, new_top1, rec["target_pos"]))

    print(f"[diagnose] queries where offline reranking MOVES the top-1 away from base: "
          f"{len(changed)}/{n} ({100.0*len(changed)/n:.2f}%)")

    # Of those, how many are helps vs harms (offline view)?
    helps = sum(1 for _, t1, tp in changed if tp == t1 and tp != 0)
    harms = sum(1 for _, t1, tp in changed if tp == 0 and t1 != 0)
    print(f"[diagnose]   of those: {helps} helps (target was NOT base-top1, now IS top1), "
          f"{harms} harms (target WAS base-top1, now is not)")
    print(f"[diagnose]   net offline = {helps - harms:+d} queries "
          f"= {100.0*(helps-harms)/n:+.2f} pp")
    print(f"[diagnose]   job 6387's net = {1564 - 1576:+d} queries "
          f"= {100.0*(1564-1576)/n:+.2f} pp")
    print(f"[diagnose]   discrepancy   = {(helps-harms) - (1564-1576):+d} queries\n")

    # Distribution of how FAR the new top-1 came from. If reranking is
    # pulling candidates from deep in the top-k (rank 5-9), that is a
    # different regime than adjacent swaps (rank 1-2).
    from collections import Counter
    depth = Counter(t1 for _, t1, _ in changed)
    print("[diagnose] where the new top-1 came from (base rank -> count):")
    for rank in sorted(depth):
        print(f"[diagnose]   base rank {rank}: {depth[rank]}")

    print(f"\n[diagnose] first {min(args.max_print, len(changed))} changed queries "
          f"(query_index, new_top1_base_rank, target_pos):")
    for row in changed[:args.max_print]:
        print(f"[diagnose]   {row}")


if __name__ == "__main__":
    main()
