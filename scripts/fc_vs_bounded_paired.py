"""Paired test: FC vs FC+AEP+RV with bounded fusion, end to end (Table 3).

Why
---
Table 3 reports the end-to-end gain of the deployed system over FC
(+3.372pp, 1646 vs 1505 of 4181) with no p-value or interval: a paired test
needs FC's per-query outcomes, which were never saved. This produces them
with one FC evaluation pass and pairs them with the bounded-fusion outcomes
already in the CIRR score cache.

What is reused, unchanged
-------------------------
The evaluation (`_evaluate`), the per-query hit vector (`_hit_vector`) and
the statistics (`compare`: exact McNemar, mid-p, 10,000-replicate paired
bootstrap) are imported from scripts/ablation_significance.py, which
produced the published +1.076pp test. The fusion rule and tie-breaking are
imported from scripts/fusion_controls.py, which produced the published
39.369.

Pairing is proved, not assumed
------------------------------
The cache was written by evaluate_cirr.py over the same dataset with the
same gallery builder (build_cirr_full_gallery), and its query_index is the
dataset position. Before any statistic is computed the script checks:
  * the cache holds exactly query_index 0..N-1;
  * for EVERY query, the cache's target_gallery_index equals the target the
    new FC pass saw -- a per-query identity check, far stronger than
    matching totals;
  * FC reproduces its published 1505/4181, and the cache reproduces base
    1550 and bounded 1646.
Any failure aborts with no result.

Outputs: a JSON summary and FC's per-query hits (small; reused by the B6.4
evaluation so FC is not evaluated twice). No checkpoints, no caches.

Usage
-----
  python scripts/fc_vs_bounded_paired.py \\
      --config configs/stage1_finetune_full_unfreeze_gradcache.yaml \\
      --checkpoint-fc runs/ablation_collective_v3/fc/epoch003.pt \\
      --cache rv_score_cache_fc_aep_rv.jsonl --lam 1.0 \\
      --json-out ../fc_vs_bounded_paired.json \\
      --hits-out ../fc_per_query_hits.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fusion_controls import (  # noqa: E402
    CacheError, hit_at_1, load_records, relevant_set, rule_base, rule_bounded,
)

# Published counts out of 4181 that the inputs must reproduce.
N_QUERIES = 4181
FC_HITS = 1505
BASE_HITS = 1550
BOUNDED_HITS = 1646


class PairingError(RuntimeError):
    pass


def cache_outcomes(recs: Sequence[dict], lam: float):
    """Per-query base and bounded hit@1, plus target indices, in query order."""
    by_index = {}
    for r in recs:
        qi = r.get("query_index")
        if qi is None or qi in by_index:
            raise PairingError("cache query_index missing or duplicated (%r)" % qi)
        by_index[qi] = r
    if sorted(by_index) != list(range(len(recs))):
        raise PairingError("cache query_index is not exactly 0..%d" % (len(recs) - 1))
    base, bounded, targets = [], [], []
    for i in range(len(recs)):
        r = by_index[i]
        rel = relevant_set(r, False)
        base.append(hit_at_1(rule_base(r["base_score"], r["verification_score"], lam), rel))
        bounded.append(hit_at_1(rule_bounded(r["base_score"], r["verification_score"], lam), rel))
        targets.append(int(r["target_gallery_index"]))
    return base, bounded, targets


def check_alignment(eval_targets: Sequence[int], cache_targets: Sequence[int]) -> None:
    if len(eval_targets) != len(cache_targets):
        raise PairingError("query counts differ: eval %d vs cache %d"
                           % (len(eval_targets), len(cache_targets)))
    bad = [i for i, (a, b) in enumerate(zip(eval_targets, cache_targets)) if a != b]
    if bad:
        raise PairingError("target index differs on %d queries (first: query %d, "
                           "eval %d vs cache %d) -- not the same query order or "
                           "gallery, so no paired test is valid"
                           % (len(bad), bad[0], eval_targets[bad[0]],
                              cache_targets[bad[0]]))


def check_anchors(fc_hits: List[int], base: List[int], bounded: List[int]) -> None:
    got = {"FC": sum(fc_hits), "FC+AEP+RV base": sum(base),
           "bounded": sum(bounded)}
    want = {"FC": FC_HITS, "FC+AEP+RV base": BASE_HITS, "bounded": BOUNDED_HITS}
    bad = ["%s %d (published %d)" % (k, got[k], want[k]) for k in want if got[k] != want[k]]
    if bad:
        raise PairingError("published counts not reproduced: " + "; ".join(bad))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-fc", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--hits-out", required=True)
    args = ap.parse_args(argv)

    # Cache first: it is cheap, so a bad cache fails before the GPU pass.
    try:
        recs, multipositive = load_records(args.cache)
    except CacheError as exc:
        print("CACHE ERROR: %s" % exc, file=sys.stderr)
        return 1
    if multipositive or len(recs) != N_QUERIES:
        print("ERROR: need the single-target CIRR cache with %d records" % N_QUERIES,
              file=sys.stderr)
        return 1
    try:
        base, bounded, cache_targets = cache_outcomes(recs, args.lam)
    except PairingError as exc:
        print("ABORT: %s" % exc, file=sys.stderr)
        return 1
    print("cache: base %d, bounded %d of %d" % (sum(base), sum(bounded), len(recs)))

    import torch
    from revua_cir.config import ReVUAConfig
    from revua_cir.data.datasets import DATASET_REGISTRY
    from revua_cir.data.transforms import build_image_transform
    from ablation_significance import _evaluate, _hit_vector, compare
    from evaluate_cirr import build_cirr_full_gallery

    cfg = ReVUAConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DATASET_REGISTRY["cirr"](
        root=os.path.join(cfg.data.root, "cirr"), split="val",
        transform=build_image_transform(cfg.data.image_size, train=False))
    full_gallery = build_cirr_full_gallery(dataset)
    print("device %s | queries %d | gallery %d" % (device, len(dataset), len(full_gallery)))

    res = _evaluate(cfg, args.checkpoint_fc, "fc", dataset, full_gallery, device)
    fc_hits, fc_valid = _hit_vector(res["per_query"])
    eval_targets = [int(t) for t in res["per_query"]["targets"]]

    try:
        check_alignment(eval_targets, cache_targets)
        check_anchors(fc_hits, base, bounded)
    except PairingError as exc:
        print("ABORT: %s" % exc, file=sys.stderr)
        return 1
    print("pairing verified on all %d queries; published counts reproduced"
          % len(fc_hits))

    out = compare(fc_hits, bounded, fc_valid, fc_valid,
                  "FC", "FC+AEP+RV bounded fusion")
    print()
    print("FC %.3f -> bounded %.3f  delta %+.3f pp"
          % (out["r1_a_pct"], out["r1_b_pct"], out["delta_pp"]))
    print("fixed %d, broken %d, discordant %d" % (out["help"], out["harm"], out["discordant"]))
    print("McNemar exact p = %.3g   mid-p = %.3g" % (out["mcnemar_exact_p"], out["mcnemar_midp_p"]))
    print("bootstrap 95%% CI [%+.3f, %+.3f] pp" % (out["ci95_low_pp"], out["ci95_high_pp"]))

    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump({"comparison": out, "checkpoint_fc": args.checkpoint_fc,
                   "cache": args.cache, "lam": args.lam}, fh, indent=2)
    with open(args.hits_out, "w", encoding="utf-8") as fh:
        json.dump({"checkpoint": args.checkpoint_fc, "query_ids": res["per_query"].get("query_ids"),
                   "targets": eval_targets, "hit_at_1": fc_hits}, fh)
    print("\nwrote %s and %s" % (args.json_out, args.hits_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
