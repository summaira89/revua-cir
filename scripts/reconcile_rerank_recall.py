"""
Reconciliation job: resolves the 36-query gap between job 6387's reported
reranked R@1 (37.41%, 1564 hits) and the offline recomputation from its own
cache (38.27%, 1600 hits).

WHY A CLUSTER JOB IS NEEDED: five hypotheses were tested and ruled out
purely offline, by direct evidence rather than argument --
  1. argsort tie-breaking (CPU vs CUDA): 0/4181 queries change top-1.
  2. UG signal (verification_score vs *ug_confidence): no change in Step 1.
  3. Which RV signal is used: verification_score, cos_sim and both
     UG-multiplied variants ALL give exactly 1600 hits, because
     verification_score = sigmoid(a*cos_sim+b) is strictly monotonic in
     cos_sim and _rank_fuse is rank-based, so all four induce the same
     ordering.
  4. Candidate window: cache and fusion both slice [:k] with the same k.
  5. Target resolution: `all_targets` and `batch_target_idx` both resolve
     via dataset.triplets[i]["target_path"] -> path_to_idx.get(p, -1),
     i.e. identical source and mapping.
The remaining suspect is inside evaluate_dataset's own recall extraction
(`all_ranked` / `valid_mask` / recall_at_k), which operates on state the
cache never captured. This script captures exactly that state.

WHAT IT DOES: runs ONE real evaluation (rerank=True) and dumps, per query,
the exact quantities recall_at_k consumes -- captured via evaluate_dataset's
`per_query_out` hook at the precise point recall is computed, not
reconstructed afterwards. Each row carries the target gallery index, the
reranked top-1, the target's rank in the reranked list, and the valid-mask
flag. The script then recomputes R@1 from those rows and asserts it matches
the reported metric, so the dump is self-validating: if the dump and the
metric disagree, that disagreement IS the bug.

Comparing these rows against the existing cache pins the divergence to a
specific query set instead of another round of theory. The base pass is
OFF by default (--include-base) because base retrieval already agrees
between job 6387 and the offline recomputation (both 37.69) -- it is not
in dispute, and running it costs ~19 min of GPU for nothing.

Deliberately does NOT change any fusion formula, add gating, or tune
lambda -- per the standing instruction not to contaminate the research
before this is resolved. It only OBSERVES.

    python scripts/reconcile_rerank_recall.py \
        --config configs/stage1_finetune_full_unfreeze_gradcache.yaml \
        --checkpoint runs/stage1_finetune_full_unfreeze_gradcache_v3/epoch005.pt \
        --module-combo fc+aep+rv+ug \
        --out reconcile_dump.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from revua_cir.config import ReVUAConfig
from revua_cir.data.datasets import CIRRDataset, DATASET_REGISTRY
from revua_cir.data.tokenizer import build_tokenizer
from revua_cir.data.transforms import build_image_transform
from revua_cir.evaluate import evaluate_dataset
from revua_cir.models.revua_cir import ReVUACIR
from revua_cir.utils import load_checkpoint

# Reuse evaluate_cirr.py's own full-gallery builder so this job scores over
# EXACTLY the same gallery job 6387 did (2297 images) -- a different gallery
# would make the comparison meaningless, which is the whole point here.
sys.path.insert(0, os.path.dirname(__file__))
from evaluate_cirr import build_cirr_full_gallery  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--module-combo", type=str, required=True)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--eval-split", type=str, default="val")
    p.add_argument("--out", type=str, required=True,
                   help="JSONL dump: one line of run metadata, then ONE LINE PER QUERY with the "
                        "exact inputs recall_at_k consumes (target gallery index, base top-1, "
                        "reranked top-1, target's rank in the reranked list, valid-mask flag). "
                        "Exclusive-create by default so a re-run cannot silently append.")
    p.add_argument("--include-base", action="store_true",
                   help="also run the rerank=False pass. OFF by default: base retrieval already "
                        "matches between job 6387 and the offline recomputation (both 37.69), so "
                        "the base pass costs ~19 min of GPU for a number that is not in dispute. "
                        "Pass this only if you specifically want base metrics in the same job.")
    p.add_argument("--overwrite", action="store_true",
                   help="allow overwriting an existing --out file. Default is exclusive-create "
                        "(fail loudly), since silently appending/overwriting a reconciliation "
                        "dump would corrupt exactly the evidence this job exists to produce.")
    args = p.parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"[reconcile] FATAL: {args.out} already exists -- move/delete it, or pass "
                          f"--overwrite (exclusive-create by default so a re-run cannot silently "
                          f"clobber prior evidence).")

    cfg = ReVUAConfig.from_yaml(args.config)
    if args.data_root:
        cfg.data.root = args.data_root
    cfg = cfg.apply_module_ablation(args.module_combo)

    eval_transform = build_image_transform(cfg.data.image_size, train=False)
    tokenizer = build_tokenizer(cfg.backbone)
    cirr_root = os.path.join(cfg.data.root, "cirr")
    dataset: CIRRDataset = DATASET_REGISTRY["cirr"](
        root=cirr_root, split=args.eval_split, transform=eval_transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[reconcile] CIRR {args.eval_split}: {len(dataset)} queries")
    full_gallery = build_cirr_full_gallery(dataset)
    print(f"[reconcile] full official gallery: {len(full_gallery)} images")

    model = ReVUACIR(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    print(f"[reconcile] loaded {args.checkpoint} (module_combo={args.module_combo})")

    common = dict(
        ks=(1, 5, 10, 50),
        use_map=False,
        use_group_recall=True,
        group_ks=(1, 2, 3),
        full_gallery_paths=full_gallery,
    )

    # --- The run whose number IS the question -----------------------------
    base = None
    base_pq: dict = {}
    if args.include_base:
        t0 = time.time()
        base = evaluate_dataset(model, dataset, cfg, tokenizer, eval_transform, device,
                                 rerank=False, per_query_out=base_pq, **common)
        print(f"[reconcile] base eval done in {time.time() - t0:.1f}s")
        print(f"[reconcile] BASE   R@1={base.get('recall_at_1'):.2f} "
              f"R@5={base.get('recall_at_5'):.2f} R@10={base.get('recall_at_10'):.2f}")
    else:
        print("[reconcile] base pass SKIPPED (--include-base not set): base retrieval already "
              "agrees between job 6387 and the offline recomputation (both 37.69), so it is not "
              "in dispute and would cost ~19 min of GPU for nothing.")

    rr_pq: dict = {}
    t0 = time.time()
    rr = evaluate_dataset(model, dataset, cfg, tokenizer, eval_transform, device,
                           rerank=True, per_query_out=rr_pq, **common)
    print(f"[reconcile] rerank eval done in {time.time() - t0:.1f}s")
    print(f"[reconcile] RERANK R@1={rr.get('recall_at_1'):.2f} R@5={rr.get('recall_at_5'):.2f} "
          f"R@10={rr.get('recall_at_10'):.2f}")
    if base is not None:
        print(f"[reconcile] GAIN R@1: {rr.get('recall_at_1') - base.get('recall_at_1'):+.2f}")

    n = len(dataset)
    # The two reference numbers below are FIXED CONSTANTS from the real
    # 4181-query CIRR val run (job 6387) and its offline recomputation --
    # NOT rescaled to this run's query count, which would be meaningless
    # (and actively misleading on a small smoke fixture). Only compare
    # against them when n == 4181.
    JOB_6387_RERANK_R1 = 37.41
    OFFLINE_CACHE_RERANK_R1 = 38.27
    REFERENCE_N = 4181

    print(f"\n[reconcile] this run: RERANK R@1={rr.get('recall_at_1'):.2f} over {n} queries")
    if n == REFERENCE_N:
        print(f"[reconcile] job 6387 reported   RERANK R@1={JOB_6387_RERANK_R1} "
              f"({round(JOB_6387_RERANK_R1/100*REFERENCE_N)} hits)")
        print(f"[reconcile] offline cache gave  RERANK R@1={OFFLINE_CACHE_RERANK_R1} "
              f"({round(OFFLINE_CACHE_RERANK_R1/100*REFERENCE_N)} hits)")
    else:
        print(f"[reconcile] NOTE: n={n} != {REFERENCE_N}, so the job-6387 / offline-cache "
              f"reference numbers do NOT apply to this run -- comparison skipped.")

    # --- Per-query dump: the actual reconciliation evidence ---------------
    # Aggregate metrics alone can only say "the numbers differ"; these rows
    # say WHICH queries differ and how, so a mismatch is localized in one
    # run instead of triggering another investigation cycle.
    ranked = rr_pq["ranked"]          # (n, rank_width), POST-rerank
    targets = rr_pq["targets"]        # (n,), target gallery index, -1 if unresolved
    valid = rr_pq["valid_mask"]       # (n,), which queries enter the denominator
    base_ranked = base_pq.get("ranked")  # only present with --include-base

    mode = "w" if args.overwrite else "x"
    n_hits_at1 = 0
    with open(args.out, mode, encoding="utf-8", buffering=1) as f:
        f.write(json.dumps({
            "_meta": {
                "n_queries": n,
                "gallery_size": len(full_gallery),
                "include_base": bool(args.include_base),
                "base_metrics": ({k: v for k, v in base.items() if isinstance(v, (int, float))}
                                  if base is not None else None),
                "rerank_metrics": {k: v for k, v in rr.items() if isinstance(v, (int, float))},
                "checkpoint": args.checkpoint,
                "config": args.config,
            }
        }) + "\n")

        for i in range(n):
            tgt = int(targets[i])
            row_i = ranked[i]
            rerank_top1 = int(row_i[0])
            # Target's rank in the POST-rerank list (-1 if it never appears
            # in the ranked window at all, which is also how recall_at_k
            # effectively treats it -- a miss at every k).
            hit = (row_i == tgt).nonzero(as_tuple=True)[0]
            target_rank = int(hit[0]) if tgt >= 0 and hit.numel() > 0 else -1
            is_valid = bool(valid[i])
            if is_valid and target_rank == 0:
                n_hits_at1 += 1
            rec = {
                "query_id": i,
                "target_gallery_index": tgt,
                "rerank_top1": rerank_top1,
                "target_rank": target_rank,
                "valid": is_valid,
                "hit_at_1": bool(is_valid and target_rank == 0),
            }
            if base_ranked is not None:
                rec["base_top1"] = int(base_ranked[i][0])
            f.write(json.dumps(rec) + "\n")

    # Denominator MUST be the valid count, not n (fixed 2026-09-12, external
    # review -- a real bug in this self-check): recall_at_k is called as
    # `recall_at_k(ranked[valid_mask], targets[valid_mask], ks)` and computes
    # `hits[:, :k].any(dim=1).float().mean()`, so its denominator is the
    # number of VALID rows it was handed, not the dataset size. The numerator
    # here (`n_hits_at1`) was already filtered by `is_valid`, so dividing it
    # by `n` mixed a filtered numerator with an unfiltered denominator and
    # would have under-reported the recomputed R@1 -- making this self-check
    # cry "mismatch" on a perfectly correct evaluation, i.e. manufacturing a
    # phantom bug in the exact run meant to settle a real one.
    valid_count = int(valid.sum())
    recomputed_r1 = 100.0 * n_hits_at1 / valid_count if valid_count else float("nan")
    reported_r1 = rr.get("recall_at_1")

    print(f"\n[reconcile] wrote 1 metadata line + {n} per-query rows to {args.out}")
    print(f"[reconcile] valid queries (targets resolved in gallery): {valid_count}/{n}")
    print(f"[reconcile] SELF-CHECK: recomputed R@1 from the dump = {n_hits_at1}/{valid_count} "
          f"= {recomputed_r1:.2f}%  vs reported RERANK R@1 = {reported_r1:.2f}%")
    if abs(recomputed_r1 - reported_r1) < 0.005:
        print(f"[reconcile] SELF-CHECK PASSED -- the dump faithfully reproduces the reported "
              f"metric, so per-query rows can be trusted to localize any remaining discrepancy.")
    else:
        print(f"[reconcile] SELF-CHECK FAILED -- the dump and the metric disagree by "
              f"{recomputed_r1 - reported_r1:+.2f} pp. That disagreement is itself a bug and must "
              f"be resolved before drawing any conclusion from these rows.")
    if n == REFERENCE_N:
        got = rr.get("recall_at_1")
        print(f"[reconcile] KEY COMPARISON:")
        print(f"[reconcile]   if RERANK R@1 == {JOB_6387_RERANK_R1} -> job 6387 is reproducible; "
              f"the offline cache recomputation ({OFFLINE_CACHE_RERANK_R1}) is the outlier, and "
              f"the bug is in rv_cache_analysis.py's reconstruction.")
        print(f"[reconcile]   if RERANK R@1 == {OFFLINE_CACHE_RERANK_R1} -> the OFFLINE number was "
              f"right all along; job 6387 ran against different code/state, and its -0.29 'RV "
              f"hurts' result is the artifact that needs retracting.")
        print(f"[reconcile]   if NEITHER -> a third state exists; compare _meta against both.")
        if abs(got - JOB_6387_RERANK_R1) < 0.005:
            print(f"[reconcile]   >>> VERDICT: matches job 6387 ({got:.2f}).")
        elif abs(got - OFFLINE_CACHE_RERANK_R1) < 0.005:
            print(f"[reconcile]   >>> VERDICT: matches the OFFLINE recomputation ({got:.2f}).")
        else:
            print(f"[reconcile]   >>> VERDICT: matches NEITHER ({got:.2f}) -- third state.")


if __name__ == "__main__":
    main()
