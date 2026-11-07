"""
Targeted diagnostic for the REAL, still-unresolved discrepancy between job
6387's on-cluster reranking (RERANK R@1=37.41) and rv_cache_analysis.py's
offline recomputation from the same cache (R@1=38.27) -- a difference of
exactly ~36 queries out of 4181.

Two hypotheses have already been TESTED AND REJECTED by direct evidence,
so this script does not re-test them:
  1. argsort tie-breaking (CPU vs CUDA): 0/4181 queries change top-1 under
     stable-vs-unstable argsort. Ruled out.
  2. UG signal (verification_score vs verification_score*ug_confidence):
     applying the real production signal changed nothing in Step 1. Ruled
     out.

This script instead isolates the arithmetic directly: it reports the exact
hit COUNT (not rate) under every plausible reconstruction of the reranked
top-1, so the 36-query gap can be attributed to a specific, identifiable
code path rather than theorized about a fourth time.

    python scripts/diagnose_cache_reproduction_gap.py --cache rv_score_cache_cirr_val.jsonl
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
    p.add_argument("--max-print", type=int, default=15)
    args = p.parse_args()

    records = []
    with open(args.cache, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n = len(records)
    print(f"[diagnose] loaded {n} records")
    print(f"[diagnose] job 6387 reported RERANK R@1=37.41% -> {round(0.3741*n)} hits")
    print(f"[diagnose] offline recompute gives  R@1=38.27% -> {round(0.3827*n)} hits")
    print(f"[diagnose] gap to explain: ~36 queries\n")

    has_ug = "ug_confidence" in records[0]
    print(f"[diagnose] cache has ug_confidence: {has_ug}")

    # Count target_pos == -1 (target not in base top-k at all). These can
    # NEVER be a rerank hit, but evaluate.py may or may not include them in
    # its denominator -- checking how many there are bounds that effect.
    n_missing = sum(1 for r in records if r["target_pos"] < 0)
    print(f"[diagnose] queries whose target is NOT in the cached top-k (target_pos=-1): "
          f"{n_missing}/{n} ({100.0*n_missing/n:.2f}%)")
    print(f"[diagnose]   -> base R@10 from cache = {100.0*(n-n_missing)/n:.2f}% "
          f"(compare to job 6387's R@10=84.81)\n")

    # Hit counts under several candidate reconstructions of the rerank.
    variants = {}

    def count_hits(signal_key: str, use_ug: bool, label: str):
        hits1 = 0
        for rec in records:
            tp = rec["target_pos"]
            if tp < 0:
                continue
            base = torch.tensor(rec["base_score"])
            rv = torch.tensor(rec[signal_key])
            if use_ug and has_ug:
                rv = rv * torch.tensor(rec["ug_confidence"])
            fused = _rank_fuse(base, rv)
            if int(fused.argmax()) == tp:
                hits1 += 1
        variants[label] = hits1
        print(f"[diagnose] {label:<52} top-1 hits = {hits1:>5}  "
              f"(R@1 = {100.0*hits1/n:.2f}%)")

    count_hits("verification_score", False, "rank_fuse(base, verification_score)")
    count_hits("verification_score", True, "rank_fuse(base, verification_score*ug_confidence)")
    if "cos_sim" in records[0]:
        count_hits("cos_sim", False, "rank_fuse(base, cos_sim)")
        count_hits("cos_sim", True, "rank_fuse(base, cos_sim*ug_confidence)")

    # Base-only (no rerank) for reference.
    base_hits = sum(1 for r in records if r["target_pos"] == 0)
    print(f"[diagnose] {'base only (no rerank)':<52} top-1 hits = {base_hits:>5}  "
          f"(R@1 = {100.0*base_hits/n:.2f}%)")

    target_hits = round(0.3741 * n)
    print(f"\n[diagnose] job 6387's implied hit count: {target_hits}")
    for label, h in variants.items():
        print(f"[diagnose]   {label:<52} differs by {h - target_hits:+d}")

    print(f"\n[diagnose] If NO variant matches {target_hits}, the difference is NOT in the fusion "
          f"formula at all -- it would mean evaluate.py's reranked ranking is affected by "
          f"something absent from the cache (e.g. the rerank writes into a WIDER array than "
          f"top_k and R@1 is read from a different position than assumed, or the cached "
          f"target_pos does not correspond to the same candidate ordering used at scoring time).")


if __name__ == "__main__":
    main()
