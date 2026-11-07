"""
Evaluation pipeline (Section 6 / 6.1 of the paper framework).

Builds a gallery (candidate pool) for a given val/test dataset object, embeds
it once with the current model, then for every query in the dataset:
  1. ranks the full gallery by forward-composer cosine similarity,
  2. optionally reverse-verifies + uncertainty-reranks the top-K
     (`cfg.reverse_verifier.top_k`) before scoring,
  3. reports Recall@K (FashionIQ/CIRR/LaSCo), Group-Recall@K (CIRR, if
     group members are present in `meta`), mAP@K (CIRCO, multi-target), plus
     the diagnostic metrics from Section 6.1 (verification agreement rate,
     ECE, mean added inference latency vs. a forward-only pass).

Gallery construction default: the union of every reference + target image
appearing in the split's annotation file (fast, works out-of-the-box for a
quick/iterative validation loop during training). For the *official*,
publication-grade protocol (CIRR's full open-domain corpus, CIRCO's full
COCO2017-unlabeled gallery, etc.) pass `full_gallery_paths=` explicitly --
see `scripts/evaluate.py --full-gallery` for the CLI switch.
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from revua_cir.config import ReVUAConfig
from revua_cir.data.collate import build_collate_fn
from revua_cir.data.transforms import safe_load_transform
from revua_cir.metrics import (
    expected_calibration_error, mean_average_precision_at_k,
    recall_at_k, verification_agreement_rate,
    verification_auroc, verification_pairwise_win_rate,
)
from revua_cir.models.revua_cir import _rank_fuse


def _spearman_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation between two equal-length 1-D tensors --
    implemented directly (no scipy dependency) as Pearson correlation of
    the two rank vectors, the standard, exact definition of Spearman's rho
    for data with no ties (ties get the average-rank treatment below, the
    same convention scipy.stats.spearmanr uses).

    Avoids `from scipy.stats import spearmanr`: importing scipy.stats in
    the same process as torch + scikit-learn (both already imported
    elsewhere in this codebase) triggered a real, fatal `OMP: Error #15`
    (duplicate OpenMP runtime initialization) crash on this project's
    Windows dev environment the first time this function was added and
    used in a real test run -- not a hypothetical risk, an actually
    observed crash. A ~10-line rank-correlation implementation avoids
    depending on scipy's own OpenMP-linked BLAS/LAPACK backend entirely for
    what is a simple, well-defined statistic.

    Returns `float('nan')` if either input is constant (rank variance zero
    -- correlation undefined), matching scipy's own behavior for that case.
    """
    def _ranks(x: torch.Tensor) -> torch.Tensor:
        order = x.argsort()
        ranks = torch.empty_like(order, dtype=torch.float64)
        ranks[order] = torch.arange(len(x), dtype=torch.float64)
        # Average tied ranks (standard convention).
        sorted_x = x[order]
        i = 0
        n = len(sorted_x)
        while i < n:
            j = i
            while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
                j += 1
            if j > i:
                avg_rank = ranks[order[i:j + 1]].mean()
                ranks[order[i:j + 1]] = avg_rank
            i = j + 1
        return ranks

    ra, rb = _ranks(a.double()), _ranks(b.double())
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    if denom == 0:
        return float("nan")
    return float((ra @ rb) / denom)


def build_default_gallery(dataset) -> List[str]:
    paths = set()
    for t in dataset.triplets:
        paths.add(t["ref_path"])
        paths.add(t["target_path"])
    return sorted(paths)


@torch.no_grad()
def encode_gallery_paths(model, image_paths: Sequence[str], transform, device, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    embeds = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        pixel_values = torch.stack([safe_load_transform(p, transform) for p in batch_paths]).to(device)
        out = model.encode_gallery(pixel_values)
        embeds.append(out["pooled"].cpu())
    return torch.cat(embeds, dim=0) if embeds else torch.empty(0)


@torch.no_grad()
def evaluate_dataset(model, dataset, cfg: ReVUAConfig, tokenizer, transform, device,
                      ks: Sequence[int] = (1, 5, 10, 50), use_map: bool = False,
                      use_group_recall: bool = False, full_gallery_paths: Optional[List[str]] = None,
                      max_queries: Optional[int] = None,
                      group_ks: Sequence[int] = (1, 2, 3),
                      rerank: bool = False,
                      cache_path: Optional[str] = None,
                      per_query_out: Optional[dict] = None) -> Dict[str, float]:
    """`rerank=True` applies the Reverse Verifier / Uncertainty Gate rescaling to
    the top-k before metrics are computed -- i.e. it actually exercises the
    inference-time mechanism Section 3 claims as the contribution. With
    `rerank=False` (default, and how training-time validation runs) RV/UG only
    populate diagnostic metrics and the ranking is pure embedding similarity.
    Report both: the difference between them *is* the reranking gain.

    `group_ks` is deliberately separate from `ks`.

    CIRR provides a six-member image set containing the reference image;
    after excluding the reference, subset retrieval ranks the remaining
    **5 candidates** (fixed wording 2026-09-09, code-audit finding #2 --
    the reference is structurally excluded from ranking, not merely a
    6th candidate that happens to score low, see the group-recall block
    below). So any k >= 5 saturates: reusing the global ks=(1,5,10,50)
    made group_recall@5/@10/@50 report the identical number every epoch.
    The official CIRR protocol reports subset R@1/R@2/R@3 -- those are the
    values that belong in a paper.

    **Latency numbers here (base_retrieval_latency_sec, rv_reranking_latency_
    sec, total_pipeline_latency_sec) are CUDA-synchronized (fixed 2026-09-09,
    code-audit finding #11) and therefore correct as wall-clock elapsed time,
    but are NOT publication-grade benchmark numbers**: this function runs
    several diagnostic-only candidate-encode passes (top-1 pass, the
    discrimination pair, the within-top-k spread pass) BEFORE the timed
    rerank block, which can warm CUDA kernels/OS file cache/image-decode
    paths and bias later-batch timings; there is no warmup-iteration
    discard, no repeated-measurement averaging beyond simple per-query mean,
    and no p50/p95 reporting. Fine for a quick sanity check of relative
    latency (e.g. "did reranking get 10x slower after this change"); a
    dedicated benchmark script with a clean, diagnostics-free inference path
    is needed before citing these numbers in a paper (deferred per the
    code-audit document's own priority list -- not required to unblock
    training).

    `cache_path`, only meaningful when `rerank=True` (there is no rerank block
    to cache from otherwise): writes one JSON line per query to this path,
    with the per-candidate values the rerank block below already computes in
    memory for the top-k (base similarity, RV's sub-scores, the fused
    verification_score, and the target candidate's position, if present).
    Written to support OFFLINE experimentation with alternative RV/UG fusion
    rules (score-based bounded correction, a tiny cached-feature reranker, a
    help/harm selective-trust gate) without a second GPU pass per variant
    tried -- those experiments then cost CPU time on this cache instead.

    The file is created with "x" (exclusive-create) -- a cache run FAILS if
    `cache_path` already exists, rather than silently appending a second
    copy of every query behind whatever is already there. This is
    deliberate: offline analysis reading duplicate (query_index, candidate)
    rows would be silently and non-obviously wrong, not an obvious crash.
    Delete/move the old file (or pass a fresh path) to regenerate a cache.
    """
    # Genuinely a no-op (does not even create the file) when rerank=False --
    # there is no rerank block below to cache from, so opening the file here
    # unconditionally would silently create an empty/truncated-looking file
    # for a caller who only meant to run the base (non-rerank) pass.
    #
    # "x" (exclusive create), not "a" (append) -- CORRECTED 2026-09-11: a
    # normal cache-generation run must FAIL LOUDLY if cache_path already
    # exists, not silently append a second copy of every query behind the
    # first. Append-mode's original intent (a resumed/retried run keeps
    # adding to the same file) is a real use case, but making it the DEFAULT
    # meant a simple re-run (e.g. after fixing an unrelated bug, or just
    # re-invoking the same command by habit) would silently double every
    # record -- and offline fusion-rule analysis reading duplicate
    # (query_index, candidate) rows would be silently and non-obviously
    # wrong (not a crash, just corrupted statistics). If/when resuming a
    # partial cache run becomes a real need, add it as an explicit, separate
    # opt-in (e.g. `--cache-append`), not the default.
    #
    # buffering=1 (line-buffered) -- each `cache_file.write(...)` call below
    # writes one complete JSON line, so line buffering flushes every record
    # to the OS as soon as it's written, not just when Python's internal
    # buffer fills or the file is closed. A full CIRR/CIRCO eval pass runs
    # for a long time (real images, real gallery, thousands of queries) on a
    # cluster where the job can be killed (wall-clock limit, preemption, a
    # crash) without Python getting a chance to run its own cleanup --
    # without line buffering, a large amount of already-computed cache data
    # sitting in an unflushed buffer would be lost with it. Deliberately NOT
    # calling fsync() after every query: that forces a real disk write (not
    # just a kernel-buffer handoff) on every single query, which is far
    # more I/O overhead than this needs -- line buffering already protects
    # against a Python-level loss (unflushed language-level buffer); an
    # OS-level/power-loss guarantee is not what this is trying to solve.
    #
    # Opened here but closed in a `finally` around the WHOLE reranking-batch
    # loop below (not just at the bottom of the happy path): an exception
    # partway through (OOM, a corrupted image, a transient I/O error) must
    # still close (and thus flush any remaining OS-level buffering of) the
    # handle -- the file staying merely *open* isn't itself the risk once
    # every write is already line-flushed, but the explicit close is a
    # cheap, correct belt-and-suspenders guarantee, not just relying on GC.
    cache_file = open(cache_path, "x", encoding="utf-8", buffering=1) if (cache_path and rerank) else None
    try:
        return _evaluate_dataset_body(
            model, dataset, cfg, tokenizer, transform, device, ks, use_map, use_group_recall,
            full_gallery_paths, max_queries, group_ks, rerank, cache_file, per_query_out)
    finally:
        if cache_file is not None:
            cache_file.close()


def _evaluate_dataset_body(model, dataset, cfg: ReVUAConfig, tokenizer, transform, device,
                            ks, use_map, use_group_recall, full_gallery_paths,
                            max_queries, group_ks, rerank, cache_file,
                            per_query_out: Optional[dict] = None) -> Dict[str, float]:
    """The actual body of `evaluate_dataset`, factored out only so the
    `cache_file` handle above can be guaranteed-closed via `finally` without
    re-indenting this whole (already long) function under a `try:` block.
    Not meant to be called directly -- use `evaluate_dataset()`.
    """
    model.eval()
    gallery_paths = full_gallery_paths or build_default_gallery(dataset)
    path_to_idx = {p: i for i, p in enumerate(gallery_paths)}
    # Built once (fixed 2026-09-09, code-audit finding): group-member lookup
    # below used to rescan the ENTIRE gallery with _extract_id() per query
    # (`[path_to_idx[p] for p in gallery_paths if _extract_id(p) in ...]`),
    # an O(gallery_size) scan repeated for every one of the ~thousands of
    # queries in an eval pass. This dict is O(gallery_size) once, then every
    # query's lookup is O(group_size).
    #
    # Two different ID schemes exist across datasets, so this can't uniformly
    # use `_extract_id(path)`:
    #   * CIRR's `img_set.members` (-> meta["group_members"]) holds CIRR's OWN
    #     alphanumeric image ID strings (e.g. "dev-244-0-img0"), the exact key
    #     `dataset.id_to_path` (CIRRDataset.__init__, loaded from
    #     split.rc2.<split>.json) uses to resolve a path. Those IDs are NOT
    #     recoverable by parsing digits out of the resolved filename --
    #     `_extract_id` doing that here caused every CIRR group lookup to
    #     silently miss (confirmed via test_cirr_full_gallery_and_incomplete_
    #     group_handling: group_queries_complete measured 0 instead of the
    #     expected 1 before this fix), invisibly collapsing subset recall to
    #     0/deflating it. When `dataset.id_to_path` exists (CIRR), use it
    #     directly as the ground truth for id -> path, resolved through the
    #     SAME per-image fallback CIRRDataset._img_path already applies (so a
    #     mirror's dev_img/-suffixed layout is handled identically here).
    #   * CIRCO's `target_img_ids` are plain zero-padded-filename-derivable
    #     COCO ids, which `_extract_id` genuinely does recover correctly (see
    #     its existing, working use at the mAP `rel = {...}` line below) --
    #     CIRCODataset has no `id_to_path` attribute, so this falls back to
    #     the filename-based scheme for it and anything else unrecognized.
    if hasattr(dataset, "id_to_path") and hasattr(dataset, "_img_path"):
        id_to_idx = {}
        for image_id in dataset.id_to_path:
            p = dataset._img_path(image_id)
            if p in path_to_idx:
                id_to_idx[str(image_id)] = path_to_idx[p]
    else:
        id_to_idx = {_extract_id(p): i for i, p in enumerate(gallery_paths)}
    gallery_embeds = encode_gallery_paths(model, gallery_paths, transform, device, cfg.data.eval_batch_size).to(device)

    collate_fn = build_collate_fn(tokenizer)
    n = len(dataset) if max_queries is None else min(max_queries, len(dataset))
    loader = DataLoader(torch.utils.data.Subset(dataset, range(n)), batch_size=cfg.data.eval_batch_size,
                         shuffle=False, collate_fn=collate_fn, num_workers=0)

    all_ranked, all_targets, all_target_sets = [], [], []
    all_query_ids: list[str] = []
    all_verif_scores, all_confidences, all_correct = [], [], []
    all_verif_pos_scores, all_verif_neg_scores = [], []
    all_topk_spread, all_topk_std = [], []
    all_topk_target_present, all_topk_target_is_argmax = [], []
    # UG ranking-impact diagnostics (added 2026-09-09, novelty-review finding
    # #9/#10): UG's non-RV inputs (query-uncertainty, image-text
    # coordination) are constant across a query's own top-k candidates --
    # only RV's per-candidate disagreement varies -- so UG's confidence
    # scaling of `adj = rv_score * ug_confidence` COULD collapse to a
    # near-identity rank transform of rv_score alone, in which case UG would
    # be functioning as a calibration/confidence layer rather than an actual
    # reranking layer, weakening (not invalidating) the claim that UG changes
    # candidate ordering. Measured directly here rather than assumed either
    # way -- see this function's docstring for the full argument and why the
    # document's own simplified monotonicity sketch is not by itself
    # sufficient to settle the question (verified separately: even under the
    # document's own assumptions, s*C(s) is not guaranteed monotonic in s
    # purely because C(s) itself varies with s -- this does NOT require s to
    # be negative. CORRECTED 2026-09-11: an earlier version of this comment
    # claimed RV's verification_score "frequently" goes negative to justify
    # the non-monotonicity concern -- wrong, verification_score is
    # `PlattCalibrator.forward()`'s `torch.sigmoid(...)` output and is
    # therefore always in (0, 1); only the raw, pre-calibration `cos_sim`
    # (reverse_verifier.py) can be negative. `adj` here is verification_score,
    # not cos_sim, so the non-monotonicity concern stands on its own -- s*C(s)
    # need not be monotonic for s confined to (0, 1) either, since C(s) is
    # not itself constant).
    # Two genuinely different comparisons, added 2026-09-10 per external
    # review of the real dry-run (job 6298) diagnostics: "adjustment order"
    # (below) compares rv_score alone vs rv_score*ug_confidence -- i.e.
    # whether UG changes the pre-fusion RV signal. But the ACTUAL observable
    # output ranking comes from `_rank_fuse(top_scores, adj)`, which mixes
    # `adj` with the base retrieval score via rank fusion (2*base_rank +
    # adj_rank) -- so a change in adj's own order does not necessarily
    # survive being fused with base_rank, and conversely adj's order could
    # be unchanged while still shifting the fused rank differently due to
    # floating-point tie-breaking. "final order" (ug_changes_final_order_*)
    # answers the question that actually matters operationally: does adding
    # UG change what a user would SEE at the end of the whole pipeline,
    # compared to RV-only reranking with no UG at all.
    all_ug_order_changed, all_ug_target_rank_delta, all_ug_spearman = [], [], []
    all_ug_final_order_changed, all_ug_target_final_rank_delta = [], []
    latencies = []
    # Reranking-stage latency (added 2026-09-09, code-audit finding B.10):
    # only populated when `rerank=True` -- see the `if rerank:` block below
    # for exactly what's timed and why RV/UG aren't split into two separate
    # numbers.
    rerank_latencies = []
    # Group-recall hit counts, computed DIRECTLY from each query's full,
    # untruncated `scores` row inside the batch loop below (fixed
    # 2026-09-08, real bug found via code-audit document review -- see the
    # loop body for the full explanation). One counter per k in group_ks.
    group_recall_hits = {k: 0 for k in group_ks}
    group_recall_total = 0
    # Completeness tracking (added 2026-09-09, code-audit finding): CIRR's
    # official subset protocol defines a group as its full, fixed 6-member
    # set. If the gallery used for this eval is missing one or more of a
    # query's listed group_members (e.g. a reduced dev gallery, per this
    # function's own `rank_width`/development-gallery caveat elsewhere), the
    # recall computed over the remaining members is not the official metric
    # and must not be silently reported as if it were.
    group_queries_total = 0
    group_queries_complete = 0
    group_queries_incomplete = 0

    query_offset = 0
    for batch in loader:
        reference = batch["reference_pixel_values"].to(device)
        input_ids = batch["edit_input_ids"].to(device)
        attn = batch["edit_attention_mask"].to(device)

        # CUDA-synchronized wall-clock timing (fixed 2026-09-09, code-audit
        # finding #11): CUDA kernel launches are ASYNCHRONOUS -- `time.time()`
        # around GPU work without a sync point measures only how long it took
        # the CPU to *enqueue* the kernels, not how long they took to actually
        # run, silently underestimating real latency (invisible on this
        # project's CPU-only smoke tests, which is exactly why it went
        # uncaught this long). `torch.cuda.synchronize()` blocks until all
        # enqueued work on `device` completes before the clock starts/stops.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        q = model.encode_query(reference, input_ids, attn)
        query = q["query"]
        scores = query @ gallery_embeds.t()

        # Standard CIRR/FashionIQ protocol: the reference image is removed from
        # the retrieval corpus for its own query. Without this the composed query
        # -- which is built *from* the reference -- retrieves the reference itself
        # at rank 1 on essentially every query, pinning R@1 at ~0 and making the
        # metric measure nothing. (Observed exactly that before this masking.)
        ref_rows = []
        for i in range(reference.size(0)):
            ref_path = dataset.triplets[query_offset + i]["ref_path"]
            ref_rows.append(path_to_idx.get(ref_path, -1))
        for i, r in enumerate(ref_rows):
            if r >= 0:
                scores[i, r] = float("-inf")

        # Rank a little wider than max(k): group-recall restricts to a subset, so
        # a list truncated at exactly max(ks) can miss group members that fall
        # just outside it and silently flatten R@50 onto R@10.
        rank_width = min(max(max(ks) * 2, max(ks) + 50), gallery_embeds.size(0))
        top_scores, top_idx = scores.topk(rank_width, dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - t0) / max(reference.size(0), 1))

        all_ranked.append(top_idx.cpu())

        if per_query_out is not None:
            # Stable per-query identity, for callers that pair two separate
            # evaluation runs query-by-query (scripts/ablation_significance.py).
            # Comparing target vectors alone cannot distinguish "same queries in
            # the same order" from "different queries that happen to share a
            # target sequence" -- targets repeat across queries, so that is a
            # real (if currently unreachable) loophole. CIRR carries `pairid`,
            # CIRCO carries `query_id`; falling back to the reference id keeps
            # this working for any dataset that supplies neither.
            all_query_ids.extend(
                str(m.get("pairid", m.get("query_id", m.get("reference_id", ""))))
                for m in batch["meta"]
            )

        for b_local, meta in enumerate(batch["meta"]):
            if use_map:
                tgt_ids = meta.get("target_ids", [meta.get("target_id")])
                rel = {path_to_idx[p] for p in gallery_paths if _extract_id(p) in {str(t) for t in tgt_ids}}
                if not rel:
                    # With the correct full gallery (all candidate images present,
                    # e.g. CIRCO's full COCO2017-unlabeled set) this should not
                    # happen for a well-formed query -- every gt_img_ids entry
                    # should resolve to a real file. If it does, mean_average_
                    # precision_at_k silently SKIPS this query (`if not rel:
                    # continue`) rather than counting it as a failure, which
                    # previously hid a wrong-gallery bug (see README.md item #4 /
                    # scripts/evaluate_circo.py) as a merely slightly-lower mAP
                    # instead of a loud error. Fall back to the query's own
                    # primary target path so it still counts as a
                    # single-relevant-item query rather than vanishing, and warn
                    # so a real mismatch (bad image ID, wrong gallery) is visible.
                    primary = dataset.triplets[query_offset].get("target_path")
                    if primary is not None and primary in path_to_idx:
                        rel = {path_to_idx[primary]}
                    print(f"[evaluate_dataset] WARNING: no gallery match for target_ids={tgt_ids} "
                          f"(query meta={meta}); falling back to primary target only. If this "
                          f"appears often, the gallery is likely missing images or wrong.")
                all_target_sets.append(rel)
            target_path = dataset.triplets[query_offset]["target_path"]
            target_idx = path_to_idx.get(target_path, -1)
            all_targets.append(target_idx)
            if use_group_recall:
                # Fixed 2026-09-08 (real bug found via code-audit document
                # review): group recall used to be computed from the
                # GLOBALLY-TRUNCATED `top_idx` (rank_width ~100), filtered
                # down to the group's members after the fact. CIRR's group
                # members are chosen for VISUAL similarity to each other,
                # not because they score highly against the query overall --
                # a group's true best-scoring member (by design, what CIRR's
                # subset protocol measures) can easily rank outside the
                # global top-100 while still being the unambiguous #1 among
                # its own group's 5 non-reference candidates. When that happens, the old
                # implementation silently reported a miss even though the
                # correct subset-only ranking would have hit. Confirmed via
                # a constructed reproduction: a target that is the clear #1
                # among its group but ranks ~150th globally was scored as a
                # group_recall@1 MISS under the old code, HIT under the
                # correct protocol.
                #
                # Fixed by ranking directly from `scores[b_local]` (this
                # query's full, untruncated per-gallery-item score row,
                # still in scope here, before any topk truncation) restricted
                # to just the group's own member indices -- exactly what
                # CIRR's official subset protocol specifies, never touching
                # the globally-truncated top_idx/all_ranked at all for this
                # metric.
                # Strict official 6-member/5-candidate semantics (fixed
                # 2026-09-09, code-audit finding #9): CIRR's `img_set.members`
                # is the full 6-image visually-similar group INCLUDING the
                # reference image itself -- the official subset protocol
                # ranks only the 5 NON-reference members against the query (a
                # query cannot correctly retrieve its own reference image as
                # "the edited target"). Previously all 6 raw members were fed
                # into member_idxs/group_scores unfiltered: the reference
                # happened to score -inf (from the earlier global
                # reference-masking step) so it was very unlikely to ever
                # rank #1-3 among 6, but it still occupied a slot in the
                # "complete group" denominator and, more importantly, the
                # code was checking "all 6 listed members resolve", not the
                # official "5 non-reference candidates resolve" -- a subtly
                # different, looser requirement than the true protocol.
                members = meta.get("group_members", [])
                ref_id = str(meta.get("reference_id", ""))
                all_member_ids = {str(m) for m in members}
                candidate_ids = {m for m in all_member_ids if m != ref_id}
                candidate_idxs = [id_to_idx[m] for m in candidate_ids if m in id_to_idx]
                # Official CIRR groups are exactly 6 members (5 candidates + 1
                # reference); anything else is malformed metadata, not a
                # valid group to evaluate at all. Computed unconditionally
                # (not just inside `if members:`) so it's always defined for
                # the group_recall_hits check below regardless of whether
                # this particular query has group metadata at all.
                is_well_formed = len(all_member_ids) == 6 and ref_id in all_member_ids
                if members:
                    group_queries_total += 1
                    if is_well_formed and len(candidate_idxs) == len(candidate_ids) == 5:
                        group_queries_complete += 1
                    else:
                        # Either malformed (not a real 6-member/1-reference
                        # group) or one/more of the 5 non-reference candidates
                        # aren't present in this gallery (typical of a
                        # reduced development gallery). Ranking over a
                        # partial or malformed group is NOT the official
                        # CIRR subset protocol, so this query is excluded
                        # from group_recall_hits/total below rather than
                        # silently scored over whatever candidates happen to
                        # be present -- fixed 2026-09-09, code-audit finding.
                        group_queries_incomplete += 1
                if (candidate_idxs and is_well_formed and len(candidate_idxs) == len(candidate_ids) == 5
                        and target_idx in candidate_idxs):
                    group_scores = scores[b_local, candidate_idxs]
                    group_order_local = group_scores.argsort(descending=True).tolist()
                    group_ranked_ids = [candidate_idxs[i] for i in group_order_local]
                    group_recall_total += 1
                    for k in group_ks:
                        if target_idx in group_ranked_ids[:k]:
                            group_recall_hits[k] += 1
                # A query whose target isn't among its own 5 non-reference
                # candidates (malformed metadata), with no group members at
                # all, or with an incomplete/malformed group (above) is
                # skipped from the group-recall denominator entirely --
                # silently forcing every such query into "counted as a hit"
                # would inflate the metric with degenerate queries instead of
                # reporting an honest, possibly-lower recall.
            query_offset += 1

        if model.rv is not None:
            k = min(cfg.reverse_verifier.top_k, top_idx.size(1))
            invariant = q["ref_out"].pooled
            if model.aep is not None:
                invariant = model.aep(q["text_out"].pooled, q["ref_out"].pooled)["invariant_embed"]

            # Top-1 pass: the diagnostic signals (verification agreement, ECE)
            # are defined on the rank-1 candidate.
            cand_pixels = torch.stack([safe_load_transform(gallery_paths[i], transform) for i in top_idx[:, 0].cpu().tolist()]).to(device)
            cand_out = model.backbone.encode_image(cand_pixels)
            cand_pooled = F.normalize(cand_out.pooled, dim=-1)
            rv_out = model.rv(cand_out.tokens, cand_pooled, q["text_out"].pooled, invariant)
            all_verif_scores.append(rv_out["verification_score"].cpu())

            # Discrimination pass: score the *known-correct* target and a
            # *known-wrong* candidate for the same query, so
            # verification_auroc/verification_pairwise_win_rate can
            # measure whether RV actually separates them. top_idx[:, 0] alone
            # cannot answer this -- it is always the base ranker's own best
            # guess, so it is systematically biased toward "plausible"
            # regardless of RV's discrimination ability (see
            # metrics.verification_agreement_rate docstring).
            batch_target_paths = [dataset.triplets[i]["target_path"]
                                   for i in range(query_offset - reference.size(0), query_offset)]
            batch_target_idx = [path_to_idx.get(p, -1) for p in batch_target_paths]
            neg_idx = top_idx[:, 0].cpu().tolist()
            # Deterministic negative selection (fixed 2026-09-09, code-audit
            # finding B.9): previously fell back to `torch.randint(...)` when
            # top-1 was already correct, making this diagnostic
            # non-reproducible run-to-run (same checkpoint, same data, two
            # different verification_auroc numbers) and picking an
            # arbitrary, often-easy random negative rather than a meaningful
            # one. Use the highest-ranked INCORRECT candidate instead --
            # deterministic, and a harder (more informative) negative than a
            # uniformly random gallery draw, since it is by construction the
            # candidate RV is most likely to be fooled by.
            for i, (tgt, neg) in enumerate(zip(batch_target_idx, neg_idx)):
                if neg == tgt:
                    row = top_idx[i].cpu().tolist()
                    next_best = next((c for c in row if c != tgt), None)
                    if next_best is None:
                        # Every ranked candidate in this row equals tgt (a
                        # degenerate gallery of size 1, or a duplicate-path
                        # gallery) -- no valid negative exists for this
                        # query. Sentinel -1, excluded explicitly by `valid`
                        # below.
                        neg_idx[i] = -1
                    else:
                        neg_idx[i] = next_best

            valid = [i for i, t in enumerate(batch_target_idx) if t >= 0 and neg_idx[i] >= 0]
            if valid:
                pos_paths = [gallery_paths[batch_target_idx[i]] for i in valid]
                neg_paths = [gallery_paths[neg_idx[i]] for i in valid]
                pos_pixels = torch.stack([safe_load_transform(p, transform) for p in pos_paths]).to(device)
                neg_pixels = torch.stack([safe_load_transform(p, transform) for p in neg_paths]).to(device)
                pos_out = model.backbone.encode_image(pos_pixels)
                neg_out = model.backbone.encode_image(neg_pixels)
                pos_pooled = F.normalize(pos_out.pooled, dim=-1)
                neg_pooled = F.normalize(neg_out.pooled, dim=-1)
                text_v = q["text_out"].pooled[valid]
                inv_v = invariant[valid]
                rv_pos = model.rv(pos_out.tokens, pos_pooled, text_v, inv_v)
                rv_neg = model.rv(neg_out.tokens, neg_pooled, text_v, inv_v)
                all_verif_pos_scores.append(rv_pos["verification_score"].cpu())
                all_verif_neg_scores.append(rv_neg["verification_score"].cpu())

            # Within-top-k spread pass: verification_auroc tests
            # true-target vs the highest-ranked incorrect candidate -- still
            # an easy-ish pair relative to what reranking needs. What reranking
            # actually needs is separation
            # *among the top-k*, which are all already plausible near-misses
            # from the same base ranker -- a much harder pool. This measures
            # RV's score std-dev across each query's own top-k and, when the
            # true target is present in the top-k, whether it gets the
            # highest RV score among the k candidates ("top-k RV hit rate").
            #
            # REUSED for reranking below when rerank=True (fixed 2026-09-11,
            # low-GPU-strategy document item #8): this loop used to compute
            # RV on exactly this same top-k candidate set for diagnostics
            # ONLY, and the `if rerank:` block below then re-encoded the
            # SAME images and called `model.rv(...)` a SECOND time on the
            # SAME inputs to get a result that was mathematically identical
            # (verified: same cand_ids, same q["text_out"].pooled[b:b+1]
            # .expand(...), same invariant[b:b+1].expand(...) -- deterministic
            # given eval-mode dropout is off). `rv_by_query`/`c_out_by_query`
            # cache this loop's own results so the rerank block can index
            # into them instead of recomputing -- real GPU savings whenever
            # rerank=True, with NO change to what gets measured (same
            # verification_score values feed both the diagnostics above and
            # the reranking below).
            #
            # Timing (also moved here from the old rerank-only block, so
            # rv_reranking_latency_sec still counts the RV encode+forward
            # cost exactly once, just attributed to where that work now
            # actually happens -- CUDA-synchronized per finding #11, same
            # reasoning as the base-retrieval `t0` above): when rerank=True
            # this loop's own image-encode + RV forward IS the real
            # candidate-verification cost the metric is meant to report, so
            # the rerank timer now starts here rather than at the top of the
            # (now RV-free) `if rerank:` block below.
            if rerank and device.type == "cuda":
                torch.cuda.synchronize(device)
            rerank_t0 = time.perf_counter() if rerank else None
            cand_ids_k = top_idx[:, :k].cpu()
            rv_by_query: list = []
            c_out_by_query: list = []
            for b in range(cand_ids_k.size(0)):
                ids = cand_ids_k[b].tolist()
                pix = torch.stack([safe_load_transform(gallery_paths[i], transform) for i in ids]).to(device)
                c_out = model.backbone.encode_image(pix)
                c_pooled = F.normalize(c_out.pooled, dim=-1)
                rv_k = model.rv(
                    candidate_tokens=c_out.tokens, candidate_pooled=c_pooled,
                    forward_text_pooled=q["text_out"].pooled[b : b + 1].expand(len(ids), -1),
                    invariant_embed=invariant[b : b + 1].expand(len(ids), -1),
                )
                if rerank:
                    # Only kept when rerank=True -- these hold real tensors
                    # (activations) for the whole batch otherwise unused
                    # after the diagnostics below, so avoid the memory cost
                    # on every rerank=False call (the far more common
                    # training-time-validation path).
                    rv_by_query.append(rv_k)
                    c_out_by_query.append(c_out)
                scores_k = rv_k["verification_score"].cpu()
                all_topk_spread.append((scores_k.max() - scores_k.min()).item())
                all_topk_std.append(scores_k.std(unbiased=False).item())

                tgt = batch_target_idx[b]
                if tgt in ids:
                    pos_in_k = ids.index(tgt)
                    all_topk_target_present.append(1)
                    all_topk_target_is_argmax.append(1 if pos_in_k == int(scores_k.argmax()) else 0)
                else:
                    all_topk_target_present.append(0)

            if rerank:
                # Section 3's novelty claim is candidate-level reverse verification
                # *at inference*. Ranking by `query @ gallery.T` alone never
                # exercises it, so recall could not show the module's benefit --
                # the +rv/+ug rows would look flat for a measurement reason rather
                # than a modelling one. This applies the same rescaling as
                # ReVUACIR.retrieve_and_rerank() to the top-k of each query.
                #
                # RV is NOT recomputed here (fixed 2026-09-11, see the
                # diagnostics loop above's own comment for the full
                # explanation) -- `rv_by_query[b]`/`c_out_by_query[b]` are
                # the exact same result the old code got from a second,
                # redundant encode+RV forward on the identical candidate set.
                # UG still runs here (its `reverse_disagreement` input needs
                # THIS query's RV output, but UG itself was never duplicated).
                reranked = top_idx.clone()
                B = top_idx.size(0)
                for b in range(B):
                    cand_ids = top_idx[b, :k].cpu().tolist()
                    rv_b = rv_by_query[b]
                    adj = rv_b["verification_score"]

                    # UG runs BEFORE the cache write below (reordered
                    # 2026-09-11 -- was previously only computed after) so
                    # its already-computed outputs (confidence, log_var,
                    # coordination) can be cached alongside RV's in the same
                    # record, with NO extra model call: `ug_b` here is the
                    # exact same forward pass the rerank fusion below already
                    # needed regardless of caching.
                    ug_b = None
                    if model.ug is not None:
                        ug_b = model.ug(
                            query[b : b + 1].expand(k, -1),
                            q["ref_out"].pooled[b : b + 1].expand(k, -1),
                            q["text_out"].pooled[b : b + 1].expand(k, -1),
                            reverse_disagreement=rv_b["disagreement"],
                        )

                    if cache_file is not None:
                        # One JSON line per query: everything an OFFLINE
                        # fusion-rule experiment needs, using only tensors
                        # already computed above for the real rerank this
                        # query is about to get -- no extra forward pass.
                        # `target_pos` is the target's index WITHIN cand_ids
                        # (i.e. its position among this query's own top-k),
                        # not a gallery index -- -1 if the base ranker's
                        # top-k missed it entirely (can't be "fixed" by any
                        # reranking of this same top-k in that case).
                        tgt_b = batch_target_idx[b]
                        target_pos = cand_ids.index(tgt_b) if tgt_b in cand_ids else -1
                        query_idx_abs = query_offset - B + b
                        record = {
                            # Stable identifiers (added 2026-09-11) so offline
                            # analysis can join/debug without re-deriving
                            # anything from dataset internals: query_id is
                            # this query's own triplet index (stable across
                            # re-runs of the SAME dataset/split -- not a
                            # gallery index), target_gallery_index is the
                            # target's actual index into gallery_paths (-1 if
                            # the target path never resolved in the gallery
                            # at all -- a different failure mode than
                            # target_pos=-1, which means it resolved but
                            # missed the top-k).
                            "query_id": query_idx_abs,
                            "query_index": query_idx_abs,
                            "target_gallery_index": int(tgt_b),
                            "candidate_gallery_ids": cand_ids,
                            "base_score": top_scores[b, :k].detach().cpu().tolist(),
                            # Raw, pre-calibration cosine similarity (added
                            # 2026-09-11) alongside the already-cached
                            # sigmoid-calibrated verification_score -- see
                            # reverse_verifier.py's PlattCalibrator: cos_sim
                            # is `(1-w)*(0.5*recon_cos+0.5*invariant_cos) +
                            # w*edit_cos` and CAN be negative (verification_
                            # score, being a sigmoid output, cannot). Cached
                            # so offline experiments can compare raw RV score
                            # vs calibrated score vs the individual sub-terms
                            # without a second GPU pass.
                            "cos_sim": rv_b["cos_sim"].detach().cpu().tolist(),
                            "recon_cos": rv_b["recon_cos"].detach().cpu().tolist(),
                            "invariant_cos": rv_b["invariant_cos"].detach().cpu().tolist(),
                            "edit_cos": rv_b["edit_cos"].detach().cpu().tolist(),
                            "verification_score": adj.detach().cpu().tolist(),
                            "target_pos": target_pos,
                        }
                        if ug_b is not None:
                            # ug_confidence varies per candidate (depends on
                            # RV's per-candidate disagreement -- see
                            # UncertaintyGate.forward), so it's a length-k
                            # list like the RV sub-scores above. query_log_var
                            # and coordination are constant across this
                            # query's own top-k (UncertaintyGate.forward is
                            # called with query_pooled/image_pooled/
                            # text_pooled all `.expand(k, -1)`-broadcast from
                            # a single query-level value -- confirmed
                            # directly at this call site, only
                            # reverse_disagreement actually varies by
                            # candidate) -- stored ONCE per query, not
                            # duplicated k times, per the low-GPU-strategy
                            # document's own instruction.
                            record["ug_confidence"] = ug_b["confidence"].detach().cpu().tolist()
                            record["query_log_var"] = float(ug_b["log_var"][0].detach().cpu())
                            record["coordination"] = float(ug_b["coordination"][0].detach().cpu())
                        if use_map:
                            # CIRCO multi-positive awareness (added
                            # 2026-09-11): CIRCO queries can have MULTIPLE
                            # valid target images, unlike CIRR's single
                            # target_path -- `all_target_sets` (populated
                            # earlier in this same batch's query loop, before
                            # this rerank block runs, at the same per-query
                            # index) already holds exactly this set of
                            # relevant gallery indices per evaluate_dataset's
                            # own mAP computation, so relying only on the
                            # single-target `target_pos` above would silently
                            # under-report relevance for any CIRCO query with
                            # more than one correct answer.
                            rel_set = all_target_sets[query_idx_abs] if query_idx_abs < len(all_target_sets) else set()
                            record["relevant_gallery_indices"] = sorted(rel_set)
                            record["relevant_positions_in_candidates"] = sorted(
                                i for i, cid in enumerate(cand_ids) if cid in rel_set)
                            # Base ranking out to top-50 (added 2026-09-11,
                            # CIRCO reports mAP up to @50 but RV/the cache
                            # above only covers top-k=cfg.reverse_verifier.
                            # top_k candidates) -- `top_idx` is already
                            # computed to width `rank_width` (>= max(ks)+50
                            # whenever the gallery is large enough, see this
                            # function's own rank_width derivation above) for
                            # the base ranking itself, so this is a slice of
                            # an existing tensor, not new model inference.
                            record["base_top50_gallery_ids"] = (
                                top_idx[b, :min(50, top_idx.size(1))].detach().cpu().tolist())
                        cache_file.write(json.dumps(record) + "\n")
                    if ug_b is not None:
                        ug_adj = adj * ug_b["confidence"]
                        # UG ranking-impact diagnostics (novelty-review
                        # finding #9/#10): compares the candidate ORDER
                        # implied by rv_score alone against the order
                        # implied by rv_score*ug_confidence, for THIS
                        # query's own top-k -- exactly the comparison
                        # _rank_fuse itself will use downstream (both go
                        # through argsort, matching _rank_fuse's own
                        # rank-based fusion rather than the raw-magnitude
                        # comparison the review document's simplified sketch
                        # used). Named "adjustment" (not "final") order,
                        # 2026-09-10: this compares the pre-fusion RV/UG
                        # signal only, NOT the actual output ranking a user
                        # would see -- see all_ug_final_order_changed below
                        # for that comparison.
                        rv_order = adj.argsort(descending=True)
                        ug_order = ug_adj.argsort(descending=True)
                        all_ug_order_changed.append(0 if torch.equal(rv_order, ug_order) else 1)
                        tgt = batch_target_idx[b]
                        if tgt in cand_ids:
                            rv_rank = (adj.argsort(descending=True) == cand_ids.index(tgt)).nonzero(as_tuple=True)[0].item()
                            ug_rank = (ug_adj.argsort(descending=True) == cand_ids.index(tgt)).nonzero(as_tuple=True)[0].item()
                            all_ug_target_rank_delta.append(ug_rank - rv_rank)
                        if k >= 2:
                            rho = _spearman_corr(adj.cpu(), ug_adj.cpu())
                            if np.isfinite(rho):
                                all_ug_spearman.append(float(rho))

                        # Final-order diagnostics (added 2026-09-10, external
                        # review of the real dry-run diagnostics): the
                        # "adjustment order" comparison above never actually
                        # runs `_rank_fuse` -- it compares adj's own argsort,
                        # not the fused rank a real query's output ranking
                        # would show. _rank_fuse mixes `adj` with the base
                        # retrieval score (2*base_rank + adj_rank), so a
                        # change in adj's order does not necessarily survive
                        # fusion, and a real answer to "does UG change what
                        # the user sees" requires computing the RV-only
                        # fused order as an explicit counterfactual and
                        # comparing it to the RV+UG fused order actually
                        # used below.
                        # `ug_fused` below is exactly what `new_scores` will
                        # equal once `adj = ug_adj` is set (this IS the real
                        # rerank output for this query) -- computed once and
                        # reused, not redundantly recomputed after this block.
                        # stable=True (fixed 2026-09-12, same reasoning as
                        # _rank_fuse's own stable=True fix -- see that
                        # function's docstring): _rank_fuse's raw output can
                        # itself contain ties (multiple candidates landing
                        # on the same fused_rank), so THIS argsort -- which
                        # turns those fused values into a final order --
                        # needs the same CPU/CUDA-consistent tie-breaking,
                        # or two devices could disagree on final_order even
                        # though _rank_fuse's own output already agrees.
                        rv_only_final_order = _rank_fuse(top_scores[b, :k], adj).argsort(descending=True, stable=True)
                        ug_fused = _rank_fuse(top_scores[b, :k], ug_adj)
                        ug_final_order = ug_fused.argsort(descending=True, stable=True)
                        all_ug_final_order_changed.append(
                            0 if torch.equal(rv_only_final_order, ug_final_order) else 1)
                        if tgt in cand_ids:
                            rv_only_final_rank = (rv_only_final_order == cand_ids.index(tgt)).nonzero(as_tuple=True)[0].item()
                            ug_final_rank = (ug_final_order == cand_ids.index(tgt)).nonzero(as_tuple=True)[0].item()
                            all_ug_target_final_rank_delta.append(ug_final_rank - rv_only_final_rank)

                        adj = ug_adj
                        new_scores = ug_fused
                    else:
                        new_scores = _rank_fuse(top_scores[b, :k], adj)
                    order = new_scores.argsort(descending=True, stable=True)
                    reranked[b, :k] = top_idx[b, :k][order]
                all_ranked[-1] = reranked.cpu()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                rerank_latencies.append((time.perf_counter() - rerank_t0) / max(B, 1))
            if model.ug is not None:
                ug_out = model.ug(query, q["ref_out"].pooled, q["text_out"].pooled, reverse_disagreement=rv_out["disagreement"])
                all_confidences.append(ug_out["confidence"].cpu())
                correct = (top_idx[:, 0].cpu() == torch.tensor([path_to_idx.get(dataset.triplets[i]["target_path"], -1)
                                                                  for i in range(query_offset - reference.size(0), query_offset)]))
                all_correct.append(correct)

    ranked = torch.cat(all_ranked, dim=0)
    targets = torch.tensor(all_targets)
    results: Dict[str, float] = {}

    valid_mask = targets >= 0
    if per_query_out is not None:
        # Capture the EXACT three tensors recall_at_k consumes, at the exact
        # point it consumes them (added 2026-09-12 for the job-6387
        # reconciliation): `ranked` (post-rerank when rerank=True, since the
        # rerank block overwrites all_ranked[-1] per batch), `targets` (the
        # per-query target gallery index, -1 when the target path never
        # resolved in the gallery), and `valid_mask` (which queries enter
        # the recall denominator at all). Aggregate metrics alone cannot
        # localize a per-query discrepancy -- these can. Only populated when
        # a caller explicitly passes a dict, so every existing call site is
        # completely unaffected.
        per_query_out["ranked"] = ranked.cpu()
        per_query_out["targets"] = targets.cpu()
        per_query_out["valid_mask"] = valid_mask.cpu()
        per_query_out["query_ids"] = list(all_query_ids)
    if valid_mask.any():
        results.update(recall_at_k(ranked[valid_mask], targets[valid_mask], ks))
    if use_map and all_target_sets:
        results.update(mean_average_precision_at_k(ranked, all_target_sets, ks))
    if use_group_recall:
        # Computed directly per-query inside the batch loop above (fixed
        # 2026-09-08 -- see that loop's own comment for the full bug
        # explanation: the old post-hoc group_recall_at_k(ranked, ...)
        # filtered a GLOBALLY-TRUNCATED ranking down to group members,
        # which could miss a group's true top candidate entirely if it
        # didn't crack the global top-100). `group_recall_total == 0` here
        # (no query had a well-formed group + a target that's actually a
        # member of its own group) is surfaced explicitly, matching the
        # spirit of the old code's "must be visible, not silently skipped"
        # comment -- a paper table missing this metric should be obvious,
        # not a quiet empty dict.
        results["group_queries_total"] = float(group_queries_total)
        results["group_queries_complete"] = float(group_queries_complete)
        results["group_queries_incomplete"] = float(group_queries_incomplete)
        if group_recall_total > 0:
            for k in group_ks:
                results[f"group_recall_at_{k}"] = 100.0 * group_recall_hits[k] / group_recall_total
        else:
            results["group_recall_error"] = (
                "no query had both group_members metadata and a target that is a member of its "
                "own group's 5 non-reference candidates -- CIRR's subset protocol needs the "
                "official six-member image set file (5 candidates + the reference)")

    if all_verif_scores:
        results["verification_agreement_rate"] = verification_agreement_rate(torch.cat(all_verif_scores))
    if all_verif_pos_scores and all_verif_neg_scores:
        pos_cat, neg_cat = torch.cat(all_verif_pos_scores), torch.cat(all_verif_neg_scores)
        # Fixed 2026-09-09 (code-audit finding B.8): the old single
        # `verification_discrimination_auc` name/metric was actually a
        # per-query matched-pair win rate, not pooled ROC-AUC -- see
        # metrics.verification_pairwise_win_rate's docstring for the exact
        # distinction. Report both under honest names: the pooled AUROC is
        # what a paper table claiming "AUROC" should cite; the matched-pair
        # rate is kept as the same diagnostic it always was, just correctly
        # named.
        results["verification_auroc"] = verification_auroc(pos_cat, neg_cat)
        results["verification_pairwise_win_rate"] = verification_pairwise_win_rate(pos_cat, neg_cat)
    if all_topk_spread:
        # Mean spread/std of RV's verification_score across each query's own
        # top-k -- this is the separation _rank_fuse's RANK-based reranking
        # (revua_cir.py's _rank_fuse, fused by 2*base_rank + adj_rank, not
        # the older magnitude-based score * (0.5 + 0.5*adj) formula it
        # replaced -- comment corrected 2026-09-08) has to work with. A small
        # value here means RV assigns nearly the same score to every top-k
        # candidate for a given query, even if verification_auroc
        # (a *global* target-vs-highest-ranked-incorrect comparison) is
        # healthy -- discriminating an easy global pair does not imply
        # separating a hard local one.
        results["topk_verification_score_spread"] = float(np.mean(all_topk_spread))
        results["topk_verification_score_std"] = float(np.mean(all_topk_std))
    if all_topk_target_present:
        results["topk_target_present_rate"] = float(np.mean(all_topk_target_present))
        if all_topk_target_is_argmax:
            results["topk_target_rv_argmax_rate"] = float(np.mean(all_topk_target_is_argmax))
    if all_confidences and all_correct:
        results["ece"] = expected_calibration_error(torch.cat(all_confidences), torch.cat(all_correct))
    if all_ug_order_changed:
        # UG ranking-impact diagnostics (added 2026-09-09, novelty-review
        # findings #9/#10; split into adjustment-order vs. final-order
        # metrics 2026-09-10 per external review): only populated when
        # rerank=True and both RV and UG are enabled -- see the `if rerank:`
        # block above for exactly what's compared.
        #
        # "adjustment" metrics compare rv_score alone vs rv_score*
        # ug_confidence -- the pre-fusion signal, NOT the observable output
        # ranking (that still goes through _rank_fuse against the base
        # retrieval score afterward).
        # "final" metrics compare the ACTUAL fused output ranking with UG
        # vs. an RV-only counterfactual (same _rank_fuse call, RV score
        # only) -- this is the operationally meaningful question: does
        # adding UG change what a user would see, not just RV's own
        # pre-fusion ordering.
        #
        # Interpretation: rates near 0% and spearman near 1.0 mean UG is
        # functioning mainly as a calibration/confidence layer on this data
        # (the ranking survives UG's rescaling almost unchanged); values
        # further from 0%/1.0 mean UG is genuinely reordering candidates.
        # This is an empirical measurement, not an assumption -- collect it
        # before deciding whether UG needs redesigning into a selective
        # reliability gate.
        results["ug_changes_adjustment_order_rate"] = float(np.mean(all_ug_order_changed)) * 100.0
        if all_ug_target_rank_delta:
            results["ug_target_rank_delta_mean"] = float(np.mean(all_ug_target_rank_delta))
        if all_ug_spearman:
            results["ug_rv_order_spearman"] = float(np.mean(all_ug_spearman))
        if all_ug_final_order_changed:
            results["ug_changes_final_order_rate"] = float(np.mean(all_ug_final_order_changed)) * 100.0
        if all_ug_target_final_rank_delta:
            results["ug_target_final_rank_delta_mean"] = float(np.mean(all_ug_target_final_rank_delta))
    if latencies:
        # Renamed with an explicit "base_retrieval" prefix (fixed 2026-09-09,
        # code-audit finding B.10): the old name `mean_query_latency_sec` was
        # ambiguous about WHICH stage it measured -- easy to misread as "the
        # full ReVUA-CIR pipeline latency" when it only ever covered
        # encode_query + score + topk (see the `t0 = time.time()` /
        # `latencies.append(...)` pair above, which sits entirely before the
        # `if model.rv is not None:` block). Kept under the old key too, for
        # any existing log-parsing code, but base_retrieval_latency_sec is
        # the name that should be used going forward.
        results["mean_query_latency_sec"] = sum(latencies) / len(latencies)
        results["base_retrieval_latency_sec"] = results["mean_query_latency_sec"]
    if rerank_latencies:
        # rv_reranking_latency_sec necessarily includes UG's contribution
        # when model.ug is not None -- see the `if rerank:` block's own
        # comment for why RV and UG are not timed as two separate numbers.
        # UPDATED 2026-09-11 (item #8, redundant-RV-forward elimination):
        # rerank_t0 now starts at the top-k candidate encode+RV pass (moved
        # into the diagnostics loop above, which is now reused for
        # reranking instead of recomputed) rather than at the top of this
        # `if rerank:` block -- so this still measures the real, complete
        # candidate-encode + RV + UG + fusion cost exactly once, just
        # timed from where that work now actually starts.
        results["rv_reranking_latency_sec"] = sum(rerank_latencies) / len(rerank_latencies)
        if latencies:
            results["total_pipeline_latency_sec"] = (
                results["base_retrieval_latency_sec"] + results["rv_reranking_latency_sec"])

    return results


def _extract_id(path: str) -> str:
    import os
    return os.path.splitext(os.path.basename(path))[0].lstrip("0") or "0"
