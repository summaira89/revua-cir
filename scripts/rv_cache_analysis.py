"""
Offline RV/UG fusion analysis (per ReVUA_CIR_Offline_RV_Improvement_Plan.md
Steps 1-2, verified before implementing -- see that document's own
verification notes): reads the RV/UG score cache produced by
`scripts/evaluate_cirr.py --rerank --cache-path ...` (one real inference
pass over the frozen checkpoint, already run -- job 6387,
rv_score_cache_cirr_val.jsonl, 4181 records) and runs every subsequent
experiment on CPU, with ZERO further GPU time.

This script does NOT modify the model, retrain anything, or touch
`_rank_fuse` (the production reranking formula) -- per the low-GPU-strategy
document's explicit "do not change _rank_fuse yet" instruction, honored
this session. It ONLY reads the cache and computes metrics/experiments on
it, offline.

**Fusion-experiment choice, verified before implementing (not taken from
the offline-improvement-plan document as given):** that document
"recommends" a demotion-only formula (S = base - lambda*ReLU(tau-RV)) as
specifically matching the observed symptom (RV helps R@5, hurts R@1), but
cites no precedent for that specific claim. A dedicated research pass
found: (1) bounded/tanh-normalized additive score fusion IS a real,
standard IR technique (Jain & Dubes tanh-estimator; CombSUM/CombMNZ-style
normalize-then-sum fusion) -- kept here as `bounded_additive_fusion`; (2)
the demotion-only formula has ZERO citable precedent anywhere in IR/
reranking literature -- it reads as a curve-fit to this project's own
symptom, not a grounded method, so it is NOT implemented here as
"recommended"; (3) the one real, directly-on-point paper found for this
EXACT problem -- protecting a strong first-stage top-1 from a weaker
second-stage reranker -- is CoVR-R (arXiv:2606.01097, a 2026 CIR-adjacent
challenge report), which solves it with a conservative GATED 1-vs-1
comparison (only replace the current top-1 if the challenger passes a
direct check), not a pointwise fusion formula. Implemented here as
`gated_rerank` and treated as the PRIMARY candidate, with
`bounded_additive_fusion` kept as a real-precedent comparison baseline --
not the uncited demotion-only formula.

Step 1 (this file's `reproduce_baseline` function): recompute R@1/R@5/R@10
directly from the cache's own base_score and verification_score fields,
and confirm they match the real numbers already measured on-cluster (job
6387: base R@1=37.69/R@5=72.47/R@10=84.81, current rank-fusion RV
R@1=37.41/R@5=73.93/R@10=84.81) -- a sanity check that the cache itself is
correct and self-consistent BEFORE trusting any further analysis built on
top of it. Note: this recomputes recall only over the CACHED top-k window
(cfg.reverse_verifier.top_k=10), which is why R@10 recomputed here should
match job 6387's R@10 exactly, while R@50 cannot be recomputed from this
cache at all (RV/UG were never computed beyond top-k -- only CIRCO records
carry a base_top50_gallery_ids field for exactly this reason, and this is
a CIRR cache, not CIRCO).

Step 2 (`help_harm_analysis`): per-query classification of whether
reranking (current production `_rank_fuse`, reproduced identically here
from the cached base_score/verification_score) helped (base top-1 wrong,
reranked top-1 right) or harmed (base top-1 right, reranked top-1 wrong)
the top-1 decision, plus the net effect (help_rate - harm_rate) -- the
"reliability metric" both the low-GPU-strategy and offline-improvement-
plan documents flag as worth reporting.

    python scripts/rv_cache_analysis.py \
        --cache rv_score_cache_cirr_val.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from revua_cir.models.revua_cir import _rank_fuse


def load_cache(cache_path: str) -> list[dict]:
    records = []
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _effective_rv_signal(rec: dict) -> torch.Tensor:
    """The REAL per-candidate signal `_rank_fuse` was fed in production for
    this query (fixed 2026-09-12 -- a real, found-not-guessed bug: the
    original reproduce_baseline used `rec["verification_score"]` directly
    for `_rank_fuse`'s second argument, but evaluate.py's actual rerank
    block (confirmed by reading it directly) computes `adj = verification_
    score`, then -- ONLY IF `model.ug is not None` -- overwrites `adj =
    ug_adj = adj * ug_b["confidence"]` BEFORE calling `_rank_fuse`. This
    project's real training config has `uncertainty_gate.enabled: true`
    (confirmed: configs/stage1_finetune_full_unfreeze_gradcache.yaml), so
    job 6387's actual reranking used verification_score*ug_confidence, NOT
    verification_score alone -- a genuinely different formula, which is
    what caused the ~0.86-point R@1 discrepancy this function's callers
    were chasing under the (wrong) tie-breaking hypothesis before this fix.
    `ug_confidence` is only present in the cache when UG was actually
    enabled for that run (see evaluate.py's cache-writer, item #4 of the
    low-GPU-strategy document) -- falls back to verification_score alone
    when absent, matching evaluate.py's own `if model.ug is not None:`
    gating exactly.
    """
    rv = torch.tensor(rec["verification_score"])
    if "ug_confidence" in rec:
        return rv * torch.tensor(rec["ug_confidence"])
    return rv


def _recall_at_k(records: list[dict], order_key: str, ks: tuple[int, ...]) -> dict[int, float]:
    """order_key: a callable-free field name whose per-record list gives the
    ALREADY-SORTED (or to-be-sorted) candidate order to check `target_pos`
    against. For 'base' and 'rv_rank_fusion' the actual sorting happens in
    the caller, which passes precomputed `rank_positions` instead -- see
    call sites below. `target_pos` is -1 (base ranker's own top-k missed
    the target entirely) for a query that cannot possibly contribute a hit
    at ANY k, by construction -- included in the denominator (matches
    revua_cir.metrics.recall_at_k's own semantics: a query whose target
    never appears in `ranked_indices` at all is correctly counted as a
    miss, not excluded), so a lower topk_target_present_rate silently
    caps the achievable recall here, exactly as it should.
    """
    hits = {k: 0 for k in ks}
    n = len(records)
    for rec in records:
        pos = rec[order_key]
        for k in ks:
            if 0 <= pos < k:
                hits[k] += 1
    return {k: 100.0 * hits[k] / n for k in ks}


def reproduce_baseline(records: list[dict], top_k: int) -> None:
    """Step 1: recompute R@1/R@5/R@10 (within the cached top-k window)
    directly from the cache, for BOTH the base ranking (order = descending
    base_score) and the CURRENT PRODUCTION rank-fusion reranking (real
    `_rank_fuse`, imported directly from revua_cir.models.revua_cir so this
    can never silently drift from the actual deployed formula, fed the real
    production signal via `_effective_rv_signal` -- which is
    verification_score*ug_confidence when UG is enabled, NOT
    verification_score alone; see that helper's own docstring for the real
    bug this fixed). Prints both sets of numbers so they can be checked by
    eye against job 6387's real, on-cluster measurement before trusting
    anything built on top of this cache.
    """
    ks = tuple(k for k in (1, 5, 10) if k <= top_k)

    base_target_pos = []
    rerank_target_pos = []
    for rec in records:
        base_scores = torch.tensor(rec["base_score"])
        rv_scores = _effective_rv_signal(rec)
        target_pos = rec["target_pos"]  # -1 if base top-k missed the target

        # `target_pos` (from the cache) IS ALREADY the target's rank under
        # the base ordering -- SIMPLIFIED 2026-09-12, external review: no
        # need to re-derive it via argsort. `candidate_gallery_ids`/
        # `base_score` were written from `top_idx[b,:k]`/`top_scores[b,:k]`
        # (evaluate.py), which `.topk()` already returns sorted descending
        # by construction, so `base_scores.argsort(descending=True)` is
        # provably the identity permutation [0,1,...,k-1] here -- confirmed
        # directly (not assumed) before removing the redundant computation.
        base_target_pos.append(target_pos)

        if target_pos >= 0:
            fused = _rank_fuse(base_scores, rv_scores)
            # stable=True (fixed 2026-09-12, real bug found via this exact
            # script's own CPU-side discrepancy against job 6387's GPU
            # numbers -- see _rank_fuse's own docstring in revua_cir.py for
            # the full root-cause trace): matches every real production
            # call site's own fix, so this offline recomputation now
            # actually reproduces the same tie-breaking convention.
            fused_order = fused.argsort(descending=True, stable=True).tolist()
            rerank_rank_of_target = fused_order.index(target_pos)
        else:
            rerank_rank_of_target = -1
        rerank_target_pos.append(rerank_rank_of_target)

    base_recall = _recall_at_k(
        [{"pos": p} for p in base_target_pos], "pos", ks)
    rerank_recall = _recall_at_k(
        [{"pos": p} for p in rerank_target_pos], "pos", ks)

    print(f"[rv_cache_analysis] Step 1 -- reproduced from cache ({len(records)} queries, "
          f"top_k={top_k} window):")
    print(f"[rv_cache_analysis]   BASE:   " + " ".join(f"R@{k}={base_recall[k]:.2f}" for k in ks))
    print(f"[rv_cache_analysis]   RERANK: " + " ".join(f"R@{k}={rerank_recall[k]:.2f}" for k in ks))
    print(f"[rv_cache_analysis]   Compare against job 6387's real measurement: "
          f"BASE R@1=37.69 R@5=72.47 R@10=84.81 | RERANK R@1=37.41 R@5=73.93 R@10=84.81")
    print(f"[rv_cache_analysis]   R@1/R@5/R@10 should reproduce EXACTLY (CORRECTED 2026-09-12, "
          f"external review): traced evaluate.py directly -- reranking (evaluate_dataset's `if "
          f"rerank:` block) only overwrites `reranked[b, :k]`, the first top_k=10 positions;"
          f" R@1/R@5/R@10 (all <= top_k) therefore depend ONLY on (a) whether the target is in "
          f"the base top-10 at all and (b) its position within that top-10 after reranking -- "
          f"both fully determined by the cache's own base_score/verification_score/target_pos "
          f"fields. If these do NOT match exactly, that indicates a real bug (in this script or "
          f"in how the cache was generated), not expected noise. R@50 CANNOT be reproduced from "
          f"this cache -- it needs positions 10-49, which were never computed/stored beyond the "
          f"reranking top-k window.")


def help_harm_analysis(records: list[dict]) -> dict:
    """Step 2: per-query help/harm classification for the CURRENT PRODUCTION
    rank-fusion reranking (real _rank_fuse, not a reimplementation) vs the
    base ranking, restricted to top-1 correctness (matches both source
    documents' own definition):
      Help = base top-1 wrong, reranked top-1 right
      Harm = base top-1 right, reranked top-1 wrong
    net_effect = help_rate - harm_rate (a signed reliability metric: >0
    means reranking is a net positive for top-1 decisions, <0 a net
    negative -- exactly the number both source documents flag as worth
    reporting in the paper).
    """
    n = len(records)
    help_count = 0
    harm_count = 0
    unchanged_correct = 0
    unchanged_incorrect = 0
    hh_base_hits: list[int] = []
    hh_rerank_hits: list[int] = []
    for rec in records:
        target_pos = rec["target_pos"]
        base_scores = torch.tensor(rec["base_score"])
        # The REAL production signal (verification_score*ug_confidence when
        # UG is enabled) -- this function characterizes what the DEPLOYED
        # reranking actually does, so it must use the same signal
        # evaluate.py feeds _rank_fuse, not verification_score alone.
        rv_scores = _effective_rv_signal(rec)

        base_top1 = int(base_scores.argmax())
        base_correct = (target_pos == base_top1)

        fused = _rank_fuse(base_scores, rv_scores)
        rerank_top1 = int(fused.argmax())
        rerank_correct = (target_pos == rerank_top1)
        # Retained per query (aligned with `records`) so the consolidated
        # report below can attach a paired bootstrap CI to the headline
        # delta -- a p-value alone gives no effect-size interval.
        hh_base_hits.append(1 if base_correct else 0)
        hh_rerank_hits.append(1 if rerank_correct else 0)

        if not base_correct and rerank_correct:
            help_count += 1
        elif base_correct and not rerank_correct:
            harm_count += 1
        elif base_correct and rerank_correct:
            unchanged_correct += 1
        else:
            unchanged_incorrect += 1

    help_rate = 100.0 * help_count / n
    harm_rate = 100.0 * harm_count / n
    net_effect = help_rate - harm_rate

    print(f"[rv_cache_analysis] Step 2 -- help/harm analysis ({n} queries, current production "
          f"rank-fusion reranking, top-1 only):")
    print(f"[rv_cache_analysis]   Help (base wrong -> RV right): {help_count} ({help_rate:.2f}%)")
    print(f"[rv_cache_analysis]   Harm (base right -> RV wrong): {harm_count} ({harm_rate:.2f}%)")
    print(f"[rv_cache_analysis]   Unchanged, still correct:      {unchanged_correct} "
          f"({100.0*unchanged_correct/n:.2f}%)")
    print(f"[rv_cache_analysis]   Unchanged, still incorrect:    {unchanged_incorrect} "
          f"({100.0*unchanged_incorrect/n:.2f}%)")
    print(f"[rv_cache_analysis]   NET EFFECT (help - harm): {net_effect:+.2f} percentage points")
    print(f"[rv_cache_analysis]   Sanity check: net_effect should equal the measured R@1 delta "
          f"(-0.29 from job 6387) to within rounding -- help_count-harm_count over n queries IS "
          f"the R@1 delta by construction (each help is a +1 hit, each harm a -1 hit, unchanged "
          f"queries contribute 0 to the delta).")
    implied_r1_delta = 100.0 * (help_count - harm_count) / n
    print(f"[rv_cache_analysis]   Implied R@1 delta from help/harm counts: {implied_r1_delta:+.2f} "
          f"(compare to job 6387's measured -0.29)")

    # Significance of the production reranking effect itself (added
    # 2026-09-12). This number is reported as the project's headline RV
    # result, so it must carry a p-value: help/harm counts of comparable
    # size are exactly what chance produces, and reporting a +0.58pp delta
    # without testing it would overstate what the data supports.
    from revua_cir.significance import mcnemar_exact, mcnemar_midp
    p_exact = mcnemar_exact(help_count, harm_count)
    p_midp = mcnemar_midp(help_count, harm_count)
    print(f"[rv_cache_analysis]   McNemar exact p={p_exact:.4g} (mid-p={p_midp:.4g}) on "
          f"{help_count + harm_count} discordant pairs")
    if p_exact < 0.05:
        print(f"[rv_cache_analysis]   -> SIGNIFICANT at alpha=0.05: the reranking effect is "
              f"unlikely to be chance.")
    else:
        print(f"[rv_cache_analysis]   -> NOT SIGNIFICANT at alpha=0.05: {help_count} helps vs "
              f"{harm_count} harms is within what chance produces. The R@1 delta should NOT be "
              f"reported as a demonstrated improvement without further evidence.")

    # Consolidated, paper-ready block (external review, 2026-09-12): a
    # reviewer needs the delta, the discordant counts behind it, the
    # p-value AND an effect-size interval together -- scattered across
    # separate lines they are easy to quote selectively.
    from revua_cir.significance import paired_bootstrap_ci
    boot = paired_bootstrap_ci(hh_base_hits, hh_rerank_hits, n_replicates=1000, seed=12345)
    base_r1 = 100.0 * sum(hh_base_hits) / n
    rerank_r1 = 100.0 * sum(hh_rerank_hits) / n
    print(f"[rv_cache_analysis]   ---- PAPER-READY SUMMARY (production RV reranking) ----")
    print(f"[rv_cache_analysis]     R@1: {base_r1:.2f} -> {rerank_r1:.2f}  "
          f"(delta {rerank_r1 - base_r1:+.2f} pp)")
    print(f"[rv_cache_analysis]     queries: n={n}  help={help_count}  harm={harm_count}  "
          f"net={help_count - harm_count:+d}")
    print(f"[rv_cache_analysis]     McNemar exact p={p_exact:.4g}  (mid-p={p_midp:.4g})")
    print(f"[rv_cache_analysis]     paired bootstrap 95% CI on delta: "
          f"[{boot['ci_low']:+.2f}, {boot['ci_high']:+.2f}] pp "
          f"({'INCLUDES' if boot['straddles_zero'] else 'excludes'} zero)")
    print(f"[rv_cache_analysis]     NOTE: this is the FULL-SET number (tuning-free, so no "
          f"selection bias) -- distinct from the held-out fusion result in Step 4, and distinct "
          f"from any synthetic positive-control magnitude, which must never be quoted together "
          f"with it.")

    return {"help_count": help_count, "harm_count": harm_count, "help_rate": help_rate,
            "harm_rate": harm_rate, "net_effect": net_effect, "n": n}


def _evaluate_scoring_fn(records: list[dict], score_fn, ks: tuple[int, ...]) -> dict:
    """Runs `score_fn(base_scores, rv_scores) -> fused_scores` (argmax/
    argsort convention: higher is better, same as _rank_fuse's own output
    convention) over every cached query and returns {k: recall_at_k, plus
    help_count/harm_count vs the BASE ranking (not vs current production
    rank-fusion -- comparing every candidate fusion rule against the same
    base-only anchor keeps the three offline experiments directly
    comparable to each other, not just to the production baseline)}.
    """
    n = len(records)
    hits = {k: 0 for k in ks}
    help_count = 0
    harm_count = 0
    base_hits: list[int] = []
    new_hits: list[int] = []
    for rec in records:
        target_pos = rec["target_pos"]
        base_scores = torch.tensor(rec["base_score"])
        rv_scores = torch.tensor(rec["verification_score"])
        base_top1 = int(base_scores.argmax())
        base_correct = (target_pos == base_top1)

        fused = score_fn(base_scores, rv_scores)
        # stable=True: same reasoning as reproduce_baseline's own fix above
        # -- z-scored/tanh-bounded fusion scores can still produce genuine
        # ties (especially at lambda=0 or when RV's own values are tied).
        order = fused.argsort(descending=True, stable=True).tolist()
        top1 = order[0]
        rank_of_target = order.index(target_pos) if target_pos >= 0 else -1
        for k in ks:
            if 0 <= rank_of_target < k:
                hits[k] += 1
        top1_correct = (target_pos == top1)
        # `target_pos < 0` (target absent from the cached top-k) makes BOTH
        # flags False, so such a query is a miss for both systems and can
        # never become a discordant pair -- verified explicitly, since a
        # bug here would silently corrupt every McNemar count.
        base_hits.append(1 if base_correct else 0)
        new_hits.append(1 if top1_correct else 0)
        if not base_correct and top1_correct:
            help_count += 1
        elif base_correct and not top1_correct:
            harm_count += 1

    return {
        "recall": {k: 100.0 * hits[k] / n for k in ks},
        "help_count": help_count, "harm_count": harm_count,
        "net_effect": 100.0 * (help_count - harm_count) / n,
        # Per-query 0/1 R@1 outcomes, ALIGNED BY QUERY with `records`, for
        # the paired bootstrap (added 2026-09-12). Aggregate counts alone
        # cannot produce a confidence interval on the effect size -- the
        # bootstrap must resample queries with their (base, new) pair kept
        # together, which needs these vectors, not just totals.
        "base_hits": base_hits,
        "new_hits": new_hits,
    }


def _evaluate_ordering_fn(records: list[dict], order_fn, ks: tuple[int, ...]) -> dict:
    """Same purpose as `_evaluate_scoring_fn`, but for experiments that only
    ever REORDER candidates (never invent new score magnitudes) --
    CORRECTED 2026-09-12, external review: the gated rerank experiment
    used to synthesize artificial score values purely to satisfy an
    argsort-based interface, which works for ordering but risks a future
    caller accidentally computing margins/calibration/fusion weights on
    those made-up numbers. `order_fn(base_scores, rv_scores) -> list[int]`
    returns the reordered list of POSITIONS (indices into the original
    top-k candidate list, base_top1-first convention same as before) --
    no scores invented, ever.
    """
    n = len(records)
    hits = {k: 0 for k in ks}
    help_count = 0
    harm_count = 0
    for rec in records:
        target_pos = rec["target_pos"]
        base_scores = torch.tensor(rec["base_score"])
        rv_scores = torch.tensor(rec["verification_score"])
        base_top1 = int(base_scores.argmax())
        base_correct = (target_pos == base_top1)

        order = order_fn(base_scores, rv_scores)
        top1 = order[0]
        rank_of_target = order.index(target_pos) if target_pos >= 0 else -1
        for k in ks:
            if 0 <= rank_of_target < k:
                hits[k] += 1
        top1_correct = (target_pos == top1)
        if not base_correct and top1_correct:
            help_count += 1
        elif base_correct and not top1_correct:
            harm_count += 1

    return {
        "recall": {k: 100.0 * hits[k] / n for k in ks},
        "help_count": help_count, "harm_count": harm_count,
        "net_effect": 100.0 * (help_count - harm_count) / n,
    }


def bounded_additive_fusion_sweep(records: list[dict], top_k: int,
                                   lambdas: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5),
                                   rv_field: str = "verification_score") -> None:
    """Step 3, experiment A: S = z(base) + lambda*tanh(z(RV)), z() = per-query
    z-score normalization over the query's own top-k candidates (the only
    population available offline -- a global/dev-set-fit normalization
    would need its own held-out split, out of scope for this first pass;
    flagged explicitly in the printed output so this limitation is not
    silently lost). Real, standard IR technique (Jain & Dubes tanh-
    estimator; CombSUM/CombMNZ-style normalize-then-sum score fusion) --
    see this module's own docstring for the verification trail.

    `rv_field` (added 2026-09-12, external-review suggestion): which cached
    RV signal to use as "RV" in the formula above -- either the sigmoid-
    calibrated `verification_score` (default, what production `_rank_fuse`
    also uses) or the raw, pre-calibration `cos_sim`. Worth comparing
    because `PlattCalibrator.forward()`'s sigmoid (reverse_verifier.py) can
    compress ranking-relevant score DIFFERENCES between candidates even
    when it preserves their relative ORDER -- a monotonic transform changes
    argsort results only through how differently it compresses different
    regions of the score range, which matters here because this formula's
    tanh-bounded correction depends on RELATIVE magnitude (via z-scoring),
    not just order. Costs zero extra GPU: `cos_sim` was already cached
    (evaluate.py's cache-writer, item #3 of the low-GPU-strategy document).
    """
    ks = tuple(k for k in (1, 5, 10) if k <= top_k)
    print(f"[rv_cache_analysis] Step 3a -- bounded additive fusion sweep "
          f"(S = z(base) + lambda*tanh(z({rv_field})), per-query z-score, {len(records)} queries):")
    for lam in lambdas:
        def score_fn(base_scores, rv_scores, lam=lam):
            def z(x):
                std = x.std(unbiased=False)
                return (x - x.mean()) / (std + 1e-8)
            return z(base_scores) + lam * torch.tanh(z(rv_scores))
        result = _evaluate_scoring_fn(
            [{**rec, "verification_score": rec[rv_field]} for rec in records], score_fn, ks)
        recall_str = " ".join(f"R@{k}={result['recall'][k]:.2f}" for k in ks)
        print(f"[rv_cache_analysis]   lambda={lam:.2f}: {recall_str}  "
              f"help={result['help_count']} harm={result['harm_count']} "
              f"net={result['net_effect']:+.2f}")


def rv_only_baseline(records: list[dict], top_k: int, rv_field: str = "effective_rv") -> None:
    """Rank the cached top-k candidates by RV score ALONE, discarding the
    base retrieval score entirely.

    **Why this ablation matters** (added 2026-09-12): every other
    experiment here combines RV with the base ranking, so none of them can
    distinguish "RV supplies complementary evidence" from "RV is simply a
    better ranker and the base score is dead weight". Those imply very
    different papers. This baseline separates them.

    Note the result is already latent in the gated sweep: margin=0.00
    ("always let RV's top pick win") is RV-only ranking within the base's
    own top-10, and it scored R@1=25.66 against a 37.69 base. But that
    number sits in a row of a margin sweep where nobody would recognise it
    as the RV-only ablation, so it is computed and labelled explicitly
    here.

    IMPORTANT scope limit, stated so the number is not over-read: this
    ranks only the top-k candidates the BASE retriever already selected.
    It is therefore NOT "RV as a standalone retriever over the full
    gallery" -- RV never sees the other ~2287 gallery images, and scoring
    them all would need a fresh GPU pass. The honest claim this supports
    is "RV cannot reorder the base's own shortlist better than the base
    does", which is precisely the complementarity question at issue.
    """
    from revua_cir.significance import mcnemar_exact, paired_bootstrap_ci

    ks = tuple(k for k in (1, 5, 10) if k <= top_k)
    n = len(records)
    hits = {k: 0 for k in ks}
    base_hits: list[int] = []
    rv_hits: list[int] = []
    help_count = harm_count = 0

    for rec in records:
        target_pos = rec["target_pos"]
        rv_scores = torch.tensor(rec[rv_field])
        order = rv_scores.argsort(descending=True, stable=True).tolist()
        rank_of_target = order.index(target_pos) if target_pos >= 0 else -1
        for k in ks:
            if 0 <= rank_of_target < k:
                hits[k] += 1
        # Base top-1 is always index 0: cached candidates come from
        # `.topk()`, i.e. already sorted by descending base score.
        base_correct = (target_pos == 0)
        rv_correct = (target_pos == order[0])
        base_hits.append(1 if base_correct else 0)
        rv_hits.append(1 if rv_correct else 0)
        if not base_correct and rv_correct:
            help_count += 1
        elif base_correct and not rv_correct:
            harm_count += 1

    recall_str = " ".join(f"R@{k}={100.0 * hits[k] / n:.2f}" for k in ks)
    base_r1 = 100.0 * sum(base_hits) / n
    rv_r1 = 100.0 * sum(rv_hits) / n
    p = mcnemar_exact(help_count, harm_count)
    boot = paired_bootstrap_ci(base_hits, rv_hits, n_replicates=1000, seed=12345)

    print(f"[rv_cache_analysis] Step 5 -- RV-ONLY baseline (rank by {rv_field} alone, base score "
          f"discarded; {n} queries, within the base retriever's own top-{top_k}):")
    print(f"[rv_cache_analysis]   {recall_str}")
    print(f"[rv_cache_analysis]   vs base R@1={base_r1:.2f}  ->  delta {rv_r1 - base_r1:+.2f} pp")
    print(f"[rv_cache_analysis]   help={help_count} harm={harm_count}  McNemar exact p={p:.4g}  "
          f"95% CI=[{boot['ci_low']:+.2f}, {boot['ci_high']:+.2f}]")
    if rv_r1 < base_r1:
        print(f"[rv_cache_analysis]   -> RV alone is WORSE than the base ranking it was given. "
              f"This is the evidence that RV is COMPLEMENTARY evidence, not a superior ranker: "
              f"any gain from fusion comes from combining the two signals, not from RV replacing "
              f"the retriever. Report this alongside the fusion result.")
    else:
        print(f"[rv_cache_analysis]   -> RV alone MATCHES OR BEATS the base ranking. That would "
              f"substantially change the paper's claim -- the contribution would be a better "
              f"ranker, not complementary evidence. Verify carefully before relying on it.")


def _bounded_additive_score_fn(lam: float):
    """The bounded additive fusion scorer, factored out so the sweep and the
    tune/report protocol provably use the SAME function -- two copies of
    this formula could silently drift apart and make the held-out number
    describe a different method than the sweep that selected it.
    """
    def score_fn(base_scores, rv_scores):
        def z(x):
            return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)
        return z(base_scores) + lam * torch.tanh(z(rv_scores))
    return score_fn


def tuned_and_tested_fusion(records: list[dict], top_k: int,
                             lambdas: tuple[float, ...],
                             rv_field: str = "verification_score",
                             tune_frac: float = 0.5,
                             seed: int = 12345,
                             n_bootstrap: int = 1000) -> None:
    """Selection-bias-safe evaluation of the bounded additive fusion sweep.

    **Why this exists.** The post-reconciliation document proposed accepting
    a fusion rule if "R@1 > 38.27" -- i.e. selecting lambda by sweeping on
    the same 4181 queries the final number is reported on. That criterion
    is not sound, and this was verified empirically rather than argued:
    sweeping 10 lambdas over a PURE NOISE RV signal (carrying zero
    information) produced an apparent +0.60pp R@1 "gain" -- the same order
    as the +0.67pp the real candidate showed. A criterion that accepts pure
    noise cannot validate a real effect. See Jensen & Cohen, "Multiple
    Comparisons in Induction Algorithms", Machine Learning 38(3), 2000:
    taking the maximum over many evaluated options is itself the mechanism
    that manufactures optimistic bias.

    **Protocol.** Split the queries once, deterministically, into TUNE and
    REPORT halves. Select lambda by R@1 on TUNE only. Then evaluate that
    single frozen lambda on REPORT only, and test it there with McNemar's
    exact test (paired binary outcomes) plus a paired bootstrap CI on the
    effect size. The reported number and the number that chose lambda come
    from disjoint queries, so the selection cannot inflate the result.

    Also prints the FULL sweep curve on TUNE. A smooth, peaked curve is
    evidence of real signal; a jagged one is evidence of noise-picking --
    a diagnostic the argmax alone hides. Holm-Bonferroni across the sweep
    is reported as a supplementary check, never as a substitute for the
    split (a correction bounds family-wise error but does not undo
    selection bias).
    """
    from revua_cir.significance import (holm_bonferroni, mcnemar_exact, mcnemar_midp,
                                         paired_bootstrap_ci, split_indices)

    ks = tuple(k for k in (1, 5, 10) if k <= top_k)
    n = len(records)
    tune_idx, report_idx = split_indices(n, tune_frac=tune_frac, seed=seed)
    tune_recs = [{**records[i], "verification_score": records[i][rv_field]} for i in tune_idx]
    report_recs = [{**records[i], "verification_score": records[i][rv_field]} for i in report_idx]

    print(f"[rv_cache_analysis] Step 4 -- selection-bias-safe fusion evaluation "
          f"(rv_field={rv_field})")
    print(f"[rv_cache_analysis]   split: tune={len(tune_recs)} report={len(report_recs)} "
          f"(disjoint, deterministic seed={seed})")
    print(f"[rv_cache_analysis]   -- full sweep on TUNE (curve shape matters, not just argmax) --")

    tune_r1: list[float] = []
    tune_pvals: list[float] = []
    for lam in lambdas:
        res = _evaluate_scoring_fn(tune_recs, _bounded_additive_score_fn(lam), ks)
        r1 = res["recall"][1]
        tune_r1.append(r1)
        tune_pvals.append(mcnemar_exact(res["help_count"], res["harm_count"]))
        print(f"[rv_cache_analysis]     lambda={lam:.2f}: R@1={r1:.2f}  "
              f"help={res['help_count']} harm={res['harm_count']}  "
              f"McNemar p={tune_pvals[-1]:.4g}")

    rejected = holm_bonferroni(tune_pvals, alpha=0.05)
    n_sig = sum(rejected)
    print(f"[rv_cache_analysis]   Holm-Bonferroni across {len(lambdas)} lambdas (TUNE): "
          f"{n_sig}/{len(lambdas)} remain significant at family-wise alpha=0.05")

    # PRE-DECLARED SELECTION RULE (made explicit 2026-09-12, external
    # review §6 -- it must be stated and frozen BEFORE any report-half
    # result is inspected, not left implicit in an argmax):
    #
    #   "Select the candidate lambda with the highest R@1 on the tuning
    #    data. If two or more candidates tie, select the SMALLEST such
    #    lambda."
    #
    # The tie-break toward smaller lambda is the conservative choice: a
    # smaller weight leans less on the weaker RV signal, matching the
    # one-standard-error rule's parsimony principle (Hastie, Tibshirani &
    # Friedman, Elements of Statistical Learning 2nd ed., sec. 7.10).
    # Enforced explicitly below rather than relying on `max` returning the
    # first maximal index AND on `lambdas` happening to arrive in
    # ascending order -- both true today, neither guaranteed by contract.
    best_r1 = max(tune_r1)
    best_lam = min(lam for lam, r1 in zip(lambdas, tune_r1) if r1 == best_r1)
    best_i = list(lambdas).index(best_lam)
    print(f"[rv_cache_analysis]   selected lambda={best_lam:.2f} "
          f"(TUNE R@1={tune_r1[best_i]:.2f}) -- FROZEN, now evaluated on REPORT only")
    if best_i in (0, len(lambdas) - 1):
        print(f"[rv_cache_analysis]   WARNING: the selected lambda sits at the EDGE of the swept "
              f"range -- the true optimum is likely outside it and this sweep has not located it. "
              f"Widen `lambdas` before treating this value as optimal.")
    if best_lam >= 2.0:
        # Measured directly (2026-09-12): tanh saturates at +/-1, so the RV
        # term's magnitude is ~lambda while z(base) stays ~N(0,1). At
        # lambda=2 the fused top-1 already agrees with RV's own top-1 more
        # often than with base's (37.7% vs 35.8%); by lambda=8 it is 67.7%
        # vs 20.2%. A large selected lambda is therefore NOT "better
        # fusion" -- it means the base retriever is being progressively
        # discarded and the system is approaching RV-only ranking.
        print(f"[rv_cache_analysis]   CAUTION: lambda={best_lam:.2f} is large enough that the "
              f"fused ranking is driven more by RV than by the base retriever (tanh saturates, so "
              f"the RV term scales with lambda while z(base) stays ~N(0,1)). Such a result should "
              f"NOT be described as RV 'refining' the base ranking -- at this weight it is closer "
              f"to replacing it, which is a different and much stronger claim requiring its own "
              f"justification.")

    res = _evaluate_scoring_fn(report_recs, _bounded_additive_score_fn(best_lam), ks)
    recall_str = " ".join(f"R@{k}={res['recall'][k]:.2f}" for k in ks)
    p_exact = mcnemar_exact(res["help_count"], res["harm_count"])
    p_midp = mcnemar_midp(res["help_count"], res["harm_count"])
    boot = paired_bootstrap_ci(res["base_hits"], res["new_hits"],
                               n_replicates=n_bootstrap, seed=seed)

    base_r1_report = 100.0 * sum(res["base_hits"]) / len(res["base_hits"])
    print(f"[rv_cache_analysis]   -- HELD-OUT RESULT (REPORT half, lambda={best_lam:.2f}) --")
    print(f"[rv_cache_analysis]     base R@1={base_r1_report:.2f}  ->  fused {recall_str}")
    print(f"[rv_cache_analysis]     help={res['help_count']} harm={res['harm_count']}  "
          f"McNemar exact p={p_exact:.4g} (mid-p={p_midp:.4g})")
    print(f"[rv_cache_analysis]     paired bootstrap ({boot['n_replicates']} reps): "
          f"diff={boot['observed_diff']:+.2f}pp  "
          f"95% CI=[{boot['ci_low']:+.2f}, {boot['ci_high']:+.2f}]")
    if p_exact < 0.05 and not boot["straddles_zero"]:
        print(f"[rv_cache_analysis]     VERDICT: SIGNIFICANT on held-out data "
              f"(p<0.05 and CI excludes 0). This is a defensible result.")
    else:
        print(f"[rv_cache_analysis]     VERDICT: NOT significant on held-out data "
              f"(p={p_exact:.4g}, CI {'straddles' if boot['straddles_zero'] else 'excludes'} 0). "
              f"Do NOT report this as an improvement.")


def gated_rerank_sweep(records: list[dict], top_k: int,
                        margins: tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)) -> None:
    """Step 3, experiment B (PRIMARY candidate -- see this module's own
    docstring for why this replaces the document's uncited "demotion-only"
    recommendation): a conservative GATED 1-vs-1 comparison, following
    CoVR-R's (arXiv:2606.01097) approach to the same problem (protecting a
    strong first-stage top-1 from a weaker second-stage reranker) -- only
    let RV's top-scoring candidate REPLACE the base ranker's own top-1 if
    RV's verification_score for that challenger beats the base top-1's own
    verification_score by at least `margin`. If the gate does not clear,
    the base ranking is used UNCHANGED (not just for the top-1 -- the
    whole base order is kept, since RV is being distrusted entirely for
    this query, not partially blended in). At margin=0.0 this reduces to
    "always let RV's best candidate replace the top-1 if RV disagrees with
    base at all" (the least conservative gate); larger margins make the
    gate progressively harder to clear, approaching "never override base"
    as margin grows.
    """
    ks = tuple(k for k in (1, 5, 10) if k <= top_k)
    print(f"[rv_cache_analysis] Step 3b -- gated 1-vs-1 rerank sweep (CoVR-R-style; "
          f"only override base top-1 if RV's best candidate beats base-top-1's own RV score "
          f"by >= margin, else keep base ranking unchanged; {len(records)} queries):")
    for margin in margins:
        def order_fn(base_scores, rv_scores, margin=margin):
            # Index-based ordering, no invented score magnitudes
            # (CORRECTED 2026-09-12, external review): a future caller
            # computing margins/calibration/fusion weights on this
            # experiment's output must never be able to accidentally read
            # a made-up score as if it were real -- this returns positions
            # only, via _evaluate_ordering_fn, not _evaluate_scoring_fn.
            base_order = base_scores.argsort(descending=True, stable=True).tolist()
            base_top1 = base_order[0]
            rv_challenger = int(rv_scores.argmax())
            if rv_challenger != base_top1 and (rv_scores[rv_challenger] - rv_scores[base_top1]) >= margin:
                # Gate clears: swap the challenger into rank 0 and the
                # former top-1 into the challenger's old rank, keeping
                # every other position exactly as the base order had it --
                # a single swap, not a full re-ranking by RV, matching
                # CoVR-R's "replace only if the challenger wins" semantics
                # rather than blending RV into every candidate's score.
                order = base_order.copy()
                i, j = order.index(rv_challenger), order.index(base_top1)
                order[i], order[j] = order[j], order[i]
                return order
            return base_order
        # Use the REAL production signal (effective_rv = verification_score
        # * ug_confidence when UG is enabled, materialized in main()) so
        # this gate is evaluated on the same signal production reranking
        # actually sees -- otherwise the gate's margins would be calibrated
        # against a signal the deployed system never uses.
        result = _evaluate_ordering_fn(
            [{**rec, "verification_score": rec.get("effective_rv", rec["verification_score"])}
             for rec in records], order_fn, ks)
        recall_str = " ".join(f"R@{k}={result['recall'][k]:.2f}" for k in ks)
        print(f"[rv_cache_analysis]   margin={margin:.2f}: {recall_str}  "
              f"help={result['help_count']} harm={result['harm_count']} "
              f"net={result['net_effect']:+.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=str, required=True,
                   help="path to a rv_score_cache_*.jsonl file produced by "
                        "scripts/evaluate_cirr.py or evaluate_circo.py --rerank --cache-path")
    p.add_argument("--tune-frac", type=float, default=0.5,
                   help="fraction of queries used to SELECT lambda; the rest are held out for "
                        "the reported number and its significance test (default 0.5)")
    p.add_argument("--split-seed", type=int, default=12345,
                   help="seed for the deterministic tune/report split. Fixed by default so the "
                        "split cannot be re-rolled until the reported number improves -- which "
                        "would reintroduce exactly the selection bias the split prevents.")
    p.add_argument("--bootstrap-reps", type=int, default=1000,
                   help="paired bootstrap replicates for the effect-size CI (default 1000)")
    args = p.parse_args()

    records = load_cache(args.cache)
    if not records:
        raise SystemExit(f"[rv_cache_analysis] FATAL: {args.cache} contains zero records")
    top_k = len(records[0]["candidate_gallery_ids"])
    print(f"[rv_cache_analysis] loaded {len(records)} records from {args.cache} (top_k={top_k})")

    # Materialize the REAL production signal (verification_score *
    # ug_confidence when UG is enabled) as an explicit cache field, so the
    # fusion sweeps can test it directly and be compared like-for-like
    # against production -- added 2026-09-12 alongside the _effective_rv_
    # signal fix. Without this, every sweep tested a signal production
    # never actually used (raw verification_score), making "does bounded
    # fusion beat production?" an apples-to-oranges question.
    has_ug = "ug_confidence" in records[0]
    for rec in records:
        rec["effective_rv"] = _effective_rv_signal(rec).tolist()

    reproduce_baseline(records, top_k)
    print()
    help_harm_analysis(records)
    print()
    if has_ug:
        # The production signal first -- the only sweep directly comparable
        # to job 6387's own reranking numbers.
        bounded_additive_fusion_sweep(records, top_k, rv_field="effective_rv")
        print()
    bounded_additive_fusion_sweep(records, top_k, rv_field="verification_score")
    if "cos_sim" in records[0]:
        print()
        bounded_additive_fusion_sweep(records, top_k, rv_field="cos_sim")
    print()
    gated_rerank_sweep(records, top_k)
    print()
    rv_only_baseline(records, top_k, rv_field="effective_rv" if has_ug else "verification_score")

    # Extended lambda range: the original sweep's R@1 rose MONOTONICALLY to
    # its boundary (0.05->0.50: 37.89/38.12/38.34/38.67/38.94), so lambda=
    # 0.50 was not an optimum -- it was simply where the sweep stopped. The
    # range below extends past it so the curve can actually turn over; the
    # edge-of-range warning in tuned_and_tested_fusion fires if it still
    # does not.
    # WIDENED 2026-09-12 after the first real run: R@1 rose MONOTONICALLY
    # across the entire 0.05-1.00 range (37.99 -> 39.47 on TUNE) and was
    # still climbing at the boundary, so lambda=1.00 was a floor, not an
    # optimum -- the edge-of-range warning fired. This range extends far
    # enough that the curve can actually turn over. If the warning STILL
    # fires at 8.0, that is itself the finding: it would mean the base
    # score contributes almost nothing and the fusion is effectively
    # ranking by RV alone, which is a different (and much stronger) claim
    # than "RV refines the base ranking" -- and one this data would then
    # need to be re-examined for.
    extended = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.50, 2.00, 3.00, 5.00, 8.00)
    print()
    tuned_and_tested_fusion(records, top_k, lambdas=extended,
                             rv_field="effective_rv" if has_ug else "verification_score",
                             tune_frac=args.tune_frac, seed=args.split_seed,
                             n_bootstrap=args.bootstrap_reps)
    if "cos_sim" in records[0]:
        # Raw vs calibrated comparison, held-out. Meaningful ONLY under this
        # score-based fusion: under production's rank-based _rank_fuse the
        # comparison is provably vacuous, since verification_score =
        # sigmoid(a*cos_sim+b) is strictly monotonic in cos_sim and so
        # induces an identical ranking (verified: 0/500 random trials
        # differed under _rank_fuse, vs 304/500 under this formula).
        print()
        tuned_and_tested_fusion(records, top_k, lambdas=extended, rv_field="cos_sim",
                                 tune_frac=args.tune_frac, seed=args.split_seed,
                                 n_bootstrap=args.bootstrap_reps)


if __name__ == "__main__":
    main()
