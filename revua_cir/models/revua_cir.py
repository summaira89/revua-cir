"""
ReVUACIR: the full four-module architecture (Section 4) wired together with
the connective joint loss (Section 4.5):

    L = L_retrieval + lambda1 * L_disentangle + lambda2 * L_reverse_consistency
        + lambda3 * L_uncertainty_calibration

Every module is individually toggleable via `ReVUAConfig` (see
`ReVUAConfig.apply_module_ablation` and Section 8 / Ablation #1), which is
what the training/eval scripts use to run the progressive module ablation
that the paper framework calls "the single most important table in the
paper".
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from revua_cir.config import ReVUAConfig
from revua_cir.losses import symmetric_info_nce, attribute_bce_loss
from revua_cir.models.attribute_parser import AttributeEditParserHead
from revua_cir.models.backbone import build_backbone
from revua_cir.models.fusion import ForwardComposer
from revua_cir.models.reverse_verifier import ReverseVerifier
from revua_cir.models.uncertainty_gate import UncertaintyGate


def _rank_fuse(base_scores: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """Combine base-ranker scores with RV/UG's adjustment signal by RANK, not
    magnitude, and return a score whose argsort reproduces the fused order.

    Diagnostic measurement (scripts/diagnose_rv.py,
    scripts/verify_rv_hardneg.py, 2026-09) found RV/UG's verification_score
    spread across a query's own top-k collapses to ~0.00007 after the
    hard-negative-mining fix -- tiny relative to typical base cosine-score
    gaps between adjacent top-k ranks (~0.005-0.05). The previous formula,
    `base_score * (0.5 + 0.5*adj)`, is a magnitude-based rescale: with a
    spread this small the multiplier is nearly constant across candidates
    (0.5 + 0.5*adj varies by <0.0001), so it could not move the base ranking
    at all -- even though `adj`'s *ordering* (topk_target_rv_argmax_rate,
    16%->23% after the hard-negative fix, chance=10%) is demonstrably useful
    signal. Fusing by rank instead of magnitude uses exactly that ordering,
    independent of `adj`'s absolute scale.

    `base_scores`, `adj`: (k,), aligned by candidate. Higher is better in
    both. Returns (k,) fused scores; argsort(descending=True) gives the final
    order. Base rank keeps a 2x weight so a candidate RV is not confident
    about (near-tied adj) still falls back mostly to the (validated) base
    similarity ranking, rather than being reordered by noise.

    `stable=True` (fixed 2026-09-12, external review of an offline analysis
    script's cache-reproduction discrepancy): `adj`'s spread across a
    query's own top-k is documented above as tiny (~0.00007), meaning real
    ties (or near-ties that round to the same float32/JSON value) are
    common, not a rare edge case. Plain (unstable) argsort's tie-breaking
    is NOT guaranteed consistent across backends -- confirmed directly:
    `torch.tensor([0.5,0.5,0.5,0.3,0.3]).argsort(descending=True)` returns
    `[0,1,2,3,4]` on CPU but `[2,1,0,4,3]` on CUDA for the identical input.
    Without `stable=True` this meant the SAME cached scores could rerank
    differently depending on which device evaluated them -- a real
    reproducibility bug (not just an offline-analysis quirk): production
    inference always runs on CUDA, but any CPU-side reprocessing of the
    same scores (offline analysis, a CPU fallback deployment, a future
    unit test) could silently disagree with the original GPU ranking on
    tied candidates. `stable=True` makes both backends agree (confirmed:
    both preserve original index order among ties) without changing the
    ranking of any non-tied pair.

    IMPORTANT for callers: this fixes _rank_fuse's OWN internal tie-
    breaking, but its RETURNED `fused_rank` values can still contain ties
    across candidates (multiple candidates landing on the same fused
    rank) -- confirmed directly, e.g. base_scores=[0.5,0.5,0.5,0.3,0.3],
    adj=[0.1,0.1,0.2,0.4,0.4] returns fused_rank=[-3,-6,-6,-6,-9]. Every
    caller that does its OWN `.argsort(descending=True)` on this
    function's output (to get a final candidate order) must ALSO pass
    `stable=True` to that call, or the same CPU/CUDA disagreement
    reappears one level up -- fixed at all three real call sites
    (evaluate.py's two `if rerank:` block argsorts, and this module's own
    retrieve_and_rerank) as of 2026-09-12.
    """
    base_rank = base_scores.argsort(descending=True, stable=True).argsort().float()   # 0 = best
    adj_rank = adj.argsort(descending=True, stable=True).argsort().float()            # 0 = best
    fused_rank = 2.0 * base_rank + adj_rank
    return -fused_rank  # higher fused score = better, for argsort(descending=True)


class ReVUACIR(nn.Module):
    def __init__(self, cfg: ReVUAConfig):
        super().__init__()
        self.cfg = cfg
        embed_dim = cfg.backbone.embed_dim

        self.backbone = build_backbone(cfg.backbone)
        self.composer = ForwardComposer(embed_dim, cfg.fusion)

        self.aep: Optional[AttributeEditParserHead] = (
            AttributeEditParserHead(embed_dim, cfg.attribute_parser) if cfg.attribute_parser.enabled else None
        )
        self.rv: Optional[ReverseVerifier] = None
        if cfg.reverse_verifier.enabled:
            shared = self.composer if cfg.reverse_verifier.tier == "tier1_inverse_consistency" else None
            self.rv = ReverseVerifier(embed_dim, cfg.reverse_verifier, shared_forward_composer=shared)
        self.ug: Optional[UncertaintyGate] = (
            UncertaintyGate(embed_dim, cfg.uncertainty_gate) if cfg.uncertainty_gate.enabled else None
        )

    # ------------------------------------------------------------------ #
    # Encoding primitives (shared by train/eval/inference)
    # ------------------------------------------------------------------ #
    def encode_query(self, reference_pixel_values, edit_input_ids, edit_attention_mask):
        ref_out = self.backbone.encode_image(reference_pixel_values)
        text_out = self.backbone.encode_text(edit_input_ids, edit_attention_mask)
        query, text_attended, composer_diagnostics = self.composer(
            image_tokens=ref_out.tokens, image_pooled=ref_out.pooled,
            text_tokens=text_out.tokens, text_pooled=text_out.pooled, text_mask=text_out.mask,
        )
        # `text_attended` (CAGF's cross-attention-fused text, already mixed
        # with image content) is intentionally NOT what AEP decomposes below
        # -- AEP is fed `text_out.pooled` (CLIP's raw, un-mixed pretrained
        # text embedding) instead, on purpose (clarified 2026-09-08, see
        # AttributeEditParserHead's own docstring): AEP's job is to split the
        # EDIT INSTRUCTION into editable/invariant sub-spaces, and feeding it
        # an already-image-mixed representation would risk circularity (it
        # would be decomposing something CAGF already blended with the very
        # image AEP's invariant branch is meant to be compared against by
        # RV). `text_attended` is returned here only as a diagnostic/
        # inspection value (e.g. for a future ablation or debugging), not
        # consumed anywhere downstream by design.
        return {"query": query, "ref_out": ref_out, "text_out": text_out, "text_attended": text_attended,
                "composer_diagnostics": composer_diagnostics}

    def encode_gallery(self, pixel_values):
        out = self.backbone.encode_image(pixel_values)
        return {"pooled": F.normalize(out.pooled, dim=-1), "tokens": out.tokens, "raw_pooled": out.pooled}

    # ------------------------------------------------------------------ #
    # Training forward pass -- in-batch contrastive + connective joint loss
    # ------------------------------------------------------------------ #
    def forward(self, batch: dict, full_batch_context: Optional[dict] = None) -> dict:
        """`full_batch_context`, when given, is
        `{"query": Q, "target_pooled": T, "chunk_offset": int}` -- the FULL
        logical batch's cached, detached, no-grad query/target embeddings
        (built by `gradcache.gradcache_step`'s pass 2) plus this call's
        starting offset into them. Only ever passed from gradcache's
        per-chunk auxiliary-loss recompute; the plain (non-gradcache)
        training path never sets it, so `full_batch_context is None` there
        and every code path below is byte-for-byte what it always was.

        Why this exists: two mechanisms need FULL-BATCH statistics, not a
        chunk's -- RV's semi-hard negative search (needs the whole batch's
        similarity matrix to find "hardest negative below the positive") and
        UG's neighborhood-deviation loss (needs the whole batch's k-NN
        distances and log-variance mean/std). Computed from `batch` alone
        (a small gradcache chunk, e.g. size 2), both silently degrade to
        near-meaningless statistics over just that chunk -- measured
        (2026-09) at up to 195x gradient error on `ug.variance_head` and
        ~40% on `rv.candidate_decomp`/`rv.reverse_gen`, discovered via
        `tests/test_smoke.py::test_gradcache_trainable_backbone_matches_plain_step`
        (extended to `fc+aep+rv+ug`) when gradcache was extended to support a
        trainable backbone. `full_batch_context`'s tensors are used ONLY to
        select which chunk-local computation to run / what target statistics
        to regress toward -- never as the tensor gradient actually flows
        through, so pass 3's per-chunk backward() still only ever touches
        this chunk's own activations (peak-memory guarantee unchanged).
        """
        q = self.encode_query(batch["reference_pixel_values"], batch["edit_input_ids"], batch["edit_attention_mask"])
        query, ref_out, text_out = q["query"], q["ref_out"], q["text_out"]

        target_out = self.backbone.encode_image(batch["target_pixel_values"])
        target_pooled = F.normalize(target_out.pooled, dim=-1)

        retrieval = symmetric_info_nce(
            query, target_pooled, self.composer.temperature,
            label_smoothing=self.cfg.loss.label_smoothing, hard_negative_weight=self.cfg.loss.hard_negative_weight,
        )
        losses = {"retrieval": retrieval["loss"]}
        total = retrieval["loss"]
        outputs = {"query": query, "target_pooled": target_pooled, "logits": retrieval["logits"], "labels": retrieval["labels"]}

        # Residual-composer monitoring (added 2026-09-09, code-audit finding
        # #14): diagnostic-only, never contributes to `total` -- picked up
        # automatically by train.py's generic per-step meter logging (the
        # same mechanism `diag_rv_cos_gap` below already uses) since it's a
        # `diag_`-prefixed key in `losses`. residual_alpha is trainable and
        # unbounded; watching these across training is how a real drift
        # problem (vs. normal fluctuation) would actually be noticed.
        for k, v in q["composer_diagnostics"].items():
            losses[f"diag_composer_{k}"] = v

        invariant_embed = ref_out.pooled  # fallback if AEP disabled: RV checks against raw reference embedding
        if self.aep is not None:
            aep_out = self.aep(text_out.pooled, ref_out.pooled)
            l_disentangle = self.aep.orthogonality_loss(aep_out["editable_embed"], aep_out["invariant_embed"])
            if "attribute_targets" in batch and batch["attribute_targets"] is not None:
                l_disentangle = l_disentangle + attribute_bce_loss(aep_out["attribute_logits"], batch["attribute_targets"])
            losses["disentangle"] = l_disentangle
            losses["disentangle_weighted"] = self.cfg.loss.lambda_disentangle * l_disentangle
            total = total + losses["disentangle_weighted"]
            invariant_embed = aep_out["invariant_embed"]
            outputs["aep"] = aep_out

        reverse_disagreement = None
        if self.rv is not None:
            rv_out = self.rv(
                candidate_tokens=target_out.tokens, candidate_pooled=target_pooled,
                forward_text_pooled=text_out.pooled, invariant_embed=invariant_embed,
            )

            # In-batch SEMI-HARD negatives: for each query, pick the other
            # target in the batch the base ranker scores highest AMONG THOSE
            # STILL BELOW the query's own true-target similarity (the
            # diagonal) -- i.e. hard, but never harder than the positive.
            # Reuses retrieval["logits"] (already computed, shape (B, B), row
            # i = query i's similarity to every target in the batch), so no
            # extra encoding.
            #
            # This replaced a `torch.roll` shuffle (easy, ~random negative,
            # AUC ~0.66 global but useless for reranking) with a pure
            # `argmax` hard-negative (2026-09-01), which measurably improved
            # early-training top-k discrimination (16%->23-24% at 6-8 epochs)
            # but was found (scripts/verify_rv_hardneg.py, full 20-24 epoch
            # convergence) to COLLAPSE that same metric to ~13-14% (worse
            # than the original 16%) and to HURT end-task reranking by -5.3
            # to -5.6 R@1 (scripts/verify_rerank_gain.py, 2026-09-05/06,
            # confirmed twice on 2000-query CIRR val).
            #
            # Root cause (literature-grounded, 2026-09-06 survey): pure
            # argmax selects the in-batch item most similar to the query
            # under the SAME embedding space currently being optimized. As
            # that space converges/tightens, "most similar wrong item" and
            # "the actual right item" become progressively harder to tell
            # apart by construction -- RocketQA (Qu et al., NAACL 2021)
            # measured ~70% of a converged retriever's own hardest mined
            # negatives are secretly true positives. FaceNet (Schroff et
            # al., CVPR 2015) hit an identical failure with pure
            # hardest-triplet mining ("collapsed solutions" at scale) and
            # fixed it with exactly this semi-hard rule: take the hardest
            # negative that is still easier than the positive, never harder.
            # Bounding selection below the positive's own score structurally
            # prevents the negative from ever converging onto the positive,
            # which a raw argmax has no mechanism to prevent.
            # gradcache note: this selection needs a real similarity matrix to
            # search over. `retrieval["logits"]` is chunk-local under
            # gradcache's per-chunk aux-loss recompute (e.g. a 2x2 matrix for
            # a chunk of size 2) -- searching a chunk this tiny for "the
            # hardest negative below the positive" finds close to nothing
            # real, silently degrading the whole mechanism (measured up to
            # ~40% gradient error on rv.candidate_decomp/rv.reverse_gen via
            # tests/test_smoke.py::test_gradcache_trainable_backbone_matches_plain_step
            # extended to fc+aep+rv+ug, 2026-09). When `full_batch_context` is
            # given, search the FULL batch's cached (detached, no-grad)
            # target embeddings instead, for a correct positive-similarity
            # threshold and a real candidate pool -- but the selected
            # candidate's OWN embedding still has to come from a tensor that
            # actually carries gradient this call, which only exists for
            # targets in the CURRENT chunk (gradcache never holds another
            # chunk's activations in memory at the same time -- that's the
            # whole memory-saving point). So: use the full batch to find the
            # right threshold and to rank chunk-local candidates against the
            # TRUE (not chunk-local) positive similarity, but only ever pick
            # among this chunk's own candidates.
            negative_cos = None
            chunk_size = target_out.tokens.size(0)
            if full_batch_context is not None and self.rv.xbm is not None and self.rv.xbm.is_ready:
                # gradcache + XBM together: search the XBM queue (past-step
                # targets, decoupled from the live/converging embedding
                # space) instead of the chunk-restricted in-batch pool below.
                # BUG this branch fixes (found 2026-09, dry_run_full_pipeline.py's
                # first real run): before this branch existed, `full_batch_context
                # is not None` was always True under gradcache (every real training
                # step), so control flow NEVER reached the XBM branch or the
                # warm-up enqueue below it -- XBM was silently dead code whenever
                # gradcache was enabled, regardless of use_xbm. Caught by a cheap
                # 5-step real-data dry run before any full training run, exactly
                # the class of bug this project's dry-run/smoke-test discipline
                # exists to catch before it costs GPU-hours.
                # Fixed 2026-09-08: this branch previously mined negatives by
                # TARGET-to-target similarity (target_pooled @ queue.t()) --
                # i.e. "which queued target looks like this step's target",
                # not "which queued target looks like a plausible match for
                # THIS QUERY" -- the wrong signal for a query-conditioned
                # semi-hard search (a negative should be hard *for the
                # query*, not merely similar to the positive target). It also
                # compared that raw cosine similarity against pos_sim taken
                # from retrieval["logits"] (logit_scale-multiplied, not raw
                # cosine) -- comparing two different scales in the same
                # semi_hard_negatives() threshold check. Both fixed by using
                # `query` (this chunk's own composed query, already computed
                # above regardless of gradcache) for both similarities, at
                # the same raw (unscaled) cosine magnitude.
                with torch.no_grad():
                    pos_sim = (query * target_pooled).sum(dim=-1, keepdim=True)
                    sim_to_queue = query @ self.rv.xbm.pooled_queue.t()
                    queue_idx = self.rv.xbm.semi_hard_negatives(sim_to_queue, pos_sim)
                rv_neg = self.rv(
                    candidate_tokens=self.rv.xbm.tokens_queue[queue_idx],
                    candidate_pooled=self.rv.xbm.pooled_queue[queue_idx],
                    forward_text_pooled=text_out.pooled, invariant_embed=invariant_embed,
                )
                negative_cos = rv_neg["cos_sim"]
                outputs["rv_negative"] = rv_neg
                self.rv.xbm.enqueue(target_pooled, target_out.tokens)
            elif full_batch_context is not None:
                full_q, full_t = full_batch_context["query"], full_batch_context["target_pooled"]
                off = full_batch_context["chunk_offset"]
                with torch.no_grad():
                    # This chunk's queries' similarity to the FULL batch's targets:
                    # (chunk_size, full_B). Correct positive-similarity threshold
                    # per query, and a real (if chunk-restricted) candidate pool.
                    full_sim_rows = full_q[off:off + chunk_size] @ full_t.t()
                    pos_sim = full_sim_rows[:, off:off + chunk_size].diagonal().unsqueeze(1)
                    # Restrict the search to THIS chunk's own targets (the only
                    # ones with a gradient-carrying tensor available this call).
                    chunk_sim = full_sim_rows[:, off:off + chunk_size].clone()
                    not_self = ~torch.eye(chunk_size, dtype=torch.bool, device=chunk_sim.device)
                    semi_hard_mask = (chunk_sim < pos_sim) & not_self
                    semi_hard_sim = chunk_sim.masked_fill(~semi_hard_mask, float("-inf"))
                    hard_idx = semi_hard_sim.argmax(dim=-1)
                    no_semi_hard = ~semi_hard_mask.any(dim=-1)
                    if no_semi_hard.any():
                        fallback_idx = chunk_sim.masked_fill(~not_self, float("-inf")).argmax(dim=-1)
                        hard_idx = torch.where(no_semi_hard, fallback_idx, hard_idx)
                rv_neg = self.rv(
                    candidate_tokens=target_out.tokens[hard_idx], candidate_pooled=target_pooled[hard_idx],
                    forward_text_pooled=text_out.pooled, invariant_embed=invariant_embed,
                )
                negative_cos = rv_neg["cos_sim"]
                # BUG (found 2026-09, via test_gradcache_trainable_backbone_matches_plain_step
                # extended to fc+aep+rv+ug): this assignment was missing from this
                # branch entirely, so under gradcache UG's negative-pair pass
                # (`if self.rv is not None and "rv_negative" in outputs:` below)
                # silently never ran -- UG only ever saw the positive-pair
                # calibration target under gradcache, reproducing exactly the
                # degenerate pre-fix behavior the negative-pair pass was added to
                # prevent. Measured effect: sign-flipped, ~10x-magnitude-wrong
                # gradients on ug.fusion/ug.log_temperature/rv.calibrator.
                outputs["rv_negative"] = rv_neg
                if self.rv.xbm is not None:
                    # XBM enabled but not yet ready (still warming up) -- still
                    # enqueue this chunk's targets so the queue fills and
                    # `steps_seen` advances toward `warmup_steps`, exactly
                    # mirroring the plain-path warm-up enqueue below. Without
                    # this, a warming-up XBM under gradcache would never
                    # accumulate steps and `is_ready` would never become True.
                    self.rv.xbm.enqueue(target_pooled, target_out.tokens)
            elif self.rv.xbm is not None and self.rv.xbm.is_ready:
                # Cross-Batch Memory (Wang et al., CVPR 2020, arXiv:1912.06798):
                # search a FIFO queue of PAST steps' target embeddings instead
                # of only the current batch's ~B-1 others. Decouples the
                # negative source from the live, converging embedding space --
                # see CrossBatchMemory's docstring in reverse_verifier.py for
                # the full rationale and the semi-hard in-batch mining
                # collapse this was built to fix (verify_semihard_negative.sbatch,
                # 2026-09-07: topk_target_rv_argmax_rate settled ~14-16% at
                # full convergence, barely above the 10% chance baseline).
                # Same query-vs-target-similarity and scale fix as the
                # gradcache+XBM branch above -- see that branch's comment for
                # the full rationale.
                with torch.no_grad():
                    pos_sim = (query * target_pooled).sum(dim=-1, keepdim=True)
                    sim_to_queue = query @ self.rv.xbm.pooled_queue.t()
                    queue_idx = self.rv.xbm.semi_hard_negatives(sim_to_queue, pos_sim)
                # Queued tokens/pooled are used as-is (no gradient flows into
                # PAST steps' activations -- they were detached on enqueue).
                # Only this step's own rv(...) call on them carries gradient,
                # through the CURRENT rv module weights being scored against
                # a fixed historical candidate -- exactly the decoupling XBM
                # is meant to provide.
                rv_neg = self.rv(
                    candidate_tokens=self.rv.xbm.tokens_queue[queue_idx],
                    candidate_pooled=self.rv.xbm.pooled_queue[queue_idx],
                    forward_text_pooled=text_out.pooled, invariant_embed=invariant_embed,
                )
                negative_cos = rv_neg["cos_sim"]
                outputs["rv_negative"] = rv_neg
                self.rv.xbm.enqueue(target_pooled, target_out.tokens)
            elif chunk_size > 1:
                # In-batch SEMI-HARD negatives: for each query, pick the other
                # target in the batch the base ranker scores highest AMONG
                # THOSE STILL BELOW the query's own true-target similarity
                # (the diagonal) -- i.e. hard, but never harder than the
                # positive. Reuses retrieval["logits"] (already computed,
                # shape (B, B), row i = query i's similarity to every target
                # in the batch), so no extra encoding. This is the plain
                # (non-gradcache) path, where `batch` already IS the full
                # logical batch, so `retrieval["logits"]` needs no
                # substitution.
                #
                # This replaced a `torch.roll` shuffle (easy, ~random
                # negative, AUC ~0.66 global but useless for reranking) with
                # a pure `argmax` hard-negative (2026-09-01), which
                # measurably improved early-training top-k discrimination
                # (16%->23-24% at 6-8 epochs) but was found
                # (scripts/verify_rv_hardneg.py, full 20-24 epoch
                # convergence) to COLLAPSE that same metric to ~13-14%
                # (worse than the original 16%) and to HURT end-task
                # reranking by -5.3 to -5.6 R@1
                # (scripts/verify_rerank_gain.py, 2026-09-05/06, confirmed
                # twice on 2000-query CIRR val).
                #
                # Root cause (literature-grounded, 2026-09-06 survey): pure
                # argmax selects the in-batch item most similar to the query
                # under the SAME embedding space currently being optimized.
                # As that space converges/tightens, "most similar wrong
                # item" and "the actual right item" become progressively
                # harder to tell apart by construction -- RocketQA (Qu et
                # al., NAACL 2021) measured ~70% of a converged retriever's
                # own hardest mined negatives are secretly true positives.
                # FaceNet (Schroff et al., CVPR 2015) hit an identical
                # failure with pure hardest-triplet mining ("collapsed
                # solutions" at scale) and fixed it with exactly this
                # semi-hard rule: take the hardest negative that is still
                # easier than the positive, never harder. Bounding selection
                # below the positive's own score structurally prevents the
                # negative from ever converging onto the positive, which a
                # raw argmax has no mechanism to prevent.
                with torch.no_grad():
                    sim = retrieval["logits"].clone()
                    pos_sim = torch.diagonal(sim).unsqueeze(1)          # (B, 1): each query's own positive
                    sim.fill_diagonal_(float("-inf"))
                    # Explicit identity mask, NOT reliance on "-inf < pos_sim":
                    # that comparison is always True (a query's own masked-out
                    # diagonal entry is "easier than the positive" by
                    # definition, since -inf < anything), so without this
                    # exclusion a row with no real semi-hard candidate would
                    # have argmax return the diagonal index itself -- training
                    # RV to treat the query's own true target as its own
                    # negative. Caught in local testing before ever training
                    # on it (2026-09-06).
                    not_self = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
                    semi_hard_mask = (sim < pos_sim) & not_self          # candidates easier than the positive, excluding self
                    semi_hard_sim = sim.masked_fill(~semi_hard_mask, float("-inf"))
                    semi_hard_idx = semi_hard_sim.argmax(dim=-1)
                    # Fallback to plain hardest-in-batch when a row has NO
                    # candidate easier than its positive (small batches,
                    # very early training, or a genuinely tight cluster) --
                    # better than crashing or silently picking the diagonal.
                    no_semi_hard = ~semi_hard_mask.any(dim=-1)
                    if no_semi_hard.any():
                        hard_idx = sim.argmax(dim=-1)
                        semi_hard_idx = torch.where(no_semi_hard, hard_idx, semi_hard_idx)
                    hard_idx = semi_hard_idx
                rv_neg = self.rv(
                    candidate_tokens=target_out.tokens[hard_idx], candidate_pooled=target_pooled[hard_idx],
                    forward_text_pooled=text_out.pooled, invariant_embed=invariant_embed,
                )
                negative_cos = rv_neg["cos_sim"]
                outputs["rv_negative"] = rv_neg
                if self.rv.xbm is not None:
                    # XBM enabled but still warming up (`is_ready` False) --
                    # fall back to in-batch semi-hard mining for the actual
                    # training signal (above), but still enqueue this step's
                    # targets so the queue is populated and `steps_seen`
                    # advances toward `warmup_steps`. Without this, a
                    # warming-up XBM would never accumulate steps and
                    # `is_ready` would never become True.
                    self.rv.xbm.enqueue(target_pooled, target_out.tokens)

            l_reverse = self.rv.reverse_consistency_loss(
                rv_out["cos_sim"], negative_cos_sim=negative_cos,
                margin=self.cfg.loss.reverse_negative_margin,
            )
            losses["reverse"] = l_reverse
            losses["reverse_weighted"] = self.cfg.loss.lambda_reverse * l_reverse
            total = total + losses["reverse_weighted"]

            # RV calibrator training (fixed 2026-09-08): `reverse_consistency_loss`
            # above trains everything upstream of the calibrator via `cos_sim`
            # directly (candidate_decomp, backbone, etc.), but never actually
            # calls `self.calibrator` itself -- and the only other consumer of
            # its output, `disagreement`, is now correctly `.detach()`'d before
            # reaching UG (see the fix below). Verified directly: without this,
            # `rv.calibrator.scale.grad` / `.bias.grad` are None after a real
            # backward pass -- the calibrator is trainable but structurally
            # dead, so `verification_score` (what inference/reranking actually
            # reads) stays stuck at its init (scale=1.0, bias=0.0) forever.
            # Fixed with an explicit positive/negative BCE on RE-DETACHED
            # cos_sim values run back through the calibrator: this trains
            # ONLY calibrator.scale/bias (the graph starts fresh at the
            # detach point), deliberately not re-perturbing candidate_decomp/
            # backbone a second time -- that's already `l_reverse`'s job.
            if negative_cos is not None:
                cal_pos_logit = self.rv.calibrator.logit(rv_out["cos_sim"].detach())
                cal_neg_logit = self.rv.calibrator.logit(negative_cos.detach())
                l_calibration = 0.5 * (
                    F.binary_cross_entropy_with_logits(cal_pos_logit, torch.ones_like(cal_pos_logit))
                    + F.binary_cross_entropy_with_logits(cal_neg_logit, torch.zeros_like(cal_neg_logit))
                )
                losses["rv_calibration"] = l_calibration
                losses["rv_calibration_weighted"] = self.cfg.loss.lambda_reverse * l_calibration
                total = total + losses["rv_calibration_weighted"]
            # Detached (fixed 2026-09-08): UG's own module comment below
            # states the design intent -- "UG is trained detached from the
            # retrieval/backbone graph" -- but this specific input was
            # missed. rv_out["disagreement"] = 1 - verification_score, which
            # traces back through RV's candidate_decomp/calibrator to
            # candidate_pooled/forward_text_pooled, i.e. the SHARED BACKBONE.
            # Undetached, UG's calibration loss (weight lambda_uncertainty)
            # had a live gradient path into RV's parameters and the shared
            # trunk via this one input, even though UG's other three inputs
            # (query/ref/text below) were already correctly detached --
            # exactly the "unintended gradient path... destabilizing the
            # trunk" the comment already claims was fixed.
            reverse_disagreement = rv_out["disagreement"].detach()
            outputs["rv"] = rv_out

            # Diagnostic only (not part of `total`, never backpropped): the
            # positive-vs-negative cos_sim gap the margin loss is actually
            # trying to grow. Logged every step via the same meters/logging
            # path as the real losses (train.py logs every key in
            # out["losses"]) so this is visible without a full eval pass --
            # verification_agreement_rate could not show whether the
            # contrastive loss is working (it never samples a known-wrong
            # candidate), so training was blind to this until now.
            if negative_cos is not None:
                with torch.no_grad():
                    losses["diag_rv_cos_gap"] = (rv_out["cos_sim"].mean() - negative_cos.mean()).detach()

        if self.ug is not None:
            # UG is trained detached from the retrieval/backbone graph: it is a
            # confidence *estimator* over the composer+RV's existing output, not
            # a component that should be reshaping the shared embeddings via its
            # own (noisy, indirect) calibration gradient. Fixes an unintended
            # gradient path that was destabilizing the trunk. ALL FOUR of UG's
            # inputs are detached to enforce this: query/ref/text here, and
            # reverse_disagreement below at its own assignment site (fixed
            # 2026-09-08 -- that one was missed initially, leaving a live
            # gradient path from UG's calibration loss through RV into the
            # shared backbone; see test_ug_calibration_loss_does_not_leak_
            # into_rv_or_backbone in tests/test_smoke.py for the verification).
            ug_query, ug_ref, ug_text = query.detach(), ref_out.pooled.detach(), text_out.pooled.detach()

            # Positive-pair pass: this candidate is the correct target, so its
            # reverse_disagreement is whatever the RV computed above --
            # .detach()'d at its own assignment site a few lines up, same
            # detach-at-the-source pattern as ug_query/ug_ref/ug_text.
            ug_out = self.ug(ug_query, ug_ref, ug_text, reverse_disagreement=reverse_disagreement)

            if full_batch_context is not None and "log_var" in full_batch_context:
                l_nd = self.ug.neighborhood_deviation_loss(
                    ug_query, ug_out["log_var"], self.cfg.uncertainty_gate.neighborhood_k,
                    full_batch_query_pooled=full_batch_context["query"],
                    full_batch_log_var=full_batch_context["log_var"],
                    chunk_offset=full_batch_context["chunk_offset"],
                )
            else:
                l_nd = self.ug.neighborhood_deviation_loss(ug_query, ug_out["log_var"], self.cfg.uncertainty_gate.neighborhood_k)
            l_cal = self.ug.calibration_loss(ug_out["confidence_logit"], torch.ones_like(ug_out["confidence_logit"]))

            # Negative-pair pass: at inference, retrieve_and_rerank() scores every
            # top-k candidate (mostly wrong ones) through this exact call, gating
            # rv_out["verification_score"] by ug_out["confidence"]. Training UG
            # only on the true target's disagreement (as before) meant
            # reverse_disagreement never varied with candidate correctness during
            # training -- the fusion layer had no signal to learn "wrong candidate
            # -> lower confidence" from, mirroring the RV cos_sim collapse this
            # was meant to help fix. Reuses the already-computed rv_neg pass
            # (negative_cos / outputs["rv_negative"]) so this costs one extra UG
            # forward, no extra encoding.
            if self.rv is not None and "rv_negative" in outputs:
                rv_neg_out = outputs["rv_negative"]
                # Detached for the same reason as the positive-pair
                # reverse_disagreement above -- same leak, same fix.
                ug_neg_out = self.ug(ug_query, ug_ref, ug_text,
                                     reverse_disagreement=rv_neg_out["disagreement"].detach())
                l_cal = l_cal + self.ug.calibration_loss(
                    ug_neg_out["confidence_logit"], torch.zeros_like(ug_neg_out["confidence_logit"])
                )
                l_cal = l_cal * 0.5

            l_uncertainty = l_nd + l_cal
            losses["uncertainty"] = l_uncertainty
            losses["uncertainty_weighted"] = self.cfg.loss.lambda_uncertainty * l_uncertainty
            total = total + losses["uncertainty_weighted"]
            outputs["ug"] = ug_out

        # `total` accumulated the lambda-WEIGHTED terms all along (each
        # `losses["<name>_weighted"]` above); this final assignment does not
        # re-sum anything, it just publishes the running total under the key
        # train.py's logger reads. Kept as a separate line (not folded into
        # the last `total = total + ...` above) so `total` stays correct even
        # when RV/UG are disabled and that line never executes.
        losses["total"] = total
        outputs["losses"] = losses
        return outputs

    # ------------------------------------------------------------------ #
    # Inference: forward retrieval + optional top-K reverse-verification /
    # uncertainty-gated reranking (Section 4, full pipeline at test time).
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def retrieve_and_rerank(self, reference_pixel_values, edit_input_ids, edit_attention_mask,
                             gallery_pooled: torch.Tensor, gallery_image_fn=None,
                             top_n: int = 50):
        """gallery_pooled: (G, D) precomputed, L2-normalized gallery embeddings.
        gallery_image_fn: optional callable(indices) -> pixel_values for the top-K candidates,
                           required if the Reverse Verifier is enabled (needs token-level candidate features).
        Returns ranked indices (B, top_n) after optional RV/UG reranking of the top-K.
        """
        q = self.encode_query(reference_pixel_values, edit_input_ids, edit_attention_mask)
        query, ref_out, text_out = q["query"], q["ref_out"], q["text_out"]

        base_scores = query @ gallery_pooled.t()                      # (B, G)
        top_scores, top_idx = base_scores.topk(min(top_n, gallery_pooled.size(0)), dim=-1)

        if self.rv is None or gallery_image_fn is None:
            return top_idx, top_scores

        k = min(self.cfg.reverse_verifier.top_k, top_idx.size(1))
        invariant_embed = ref_out.pooled
        if self.aep is not None:
            aep_out = self.aep(text_out.pooled, ref_out.pooled)
            invariant_embed = aep_out["invariant_embed"]

        # Fixed 2026-09-08 (real bug found via code audit, confirmed to have
        # never affected any reported result since this API had no callers
        # outside its own definition and an unused smoke test):
        # `rerank_scores` used to start as a full clone of `top_scores` (raw
        # cosine similarities, roughly [0.2, 0.8]) for ALL top_n candidates,
        # then only the first k entries were overwritten with `_rank_fuse`'s
        # output -- `-fused_rank`, a small negative integer in [-2k+1, 0].
        # A single `rerank_scores.argsort()` over the WHOLE top_n array then
        # mixed these two incompatible score domains: every untouched
        # tail candidate (score ~0.2-0.8) always outranked every reranked
        # top-k candidate (score <= 0), pushing the true top-k results down
        # past rank k. Fixed by reordering ONLY the first k positions (same
        # pattern evaluate.py's own, separately-implemented reranking path
        # already used correctly) and leaving `top_idx[:, k:]` /
        # `top_scores[:, k:]` untouched.
        reranked_idx = top_idx.clone()
        reranked_scores = top_scores.clone()
        for b in range(top_idx.size(0)):
            cand_ids = top_idx[b, :k]
            cand_pixels = gallery_image_fn(cand_ids)
            cand_out = self.backbone.encode_image(cand_pixels)
            cand_pooled = F.normalize(cand_out.pooled, dim=-1)
            rv_out = self.rv(
                candidate_tokens=cand_out.tokens, candidate_pooled=cand_pooled,
                forward_text_pooled=text_out.pooled[b : b + 1].expand(k, -1),
                invariant_embed=invariant_embed[b : b + 1].expand(k, -1),
            )
            adj = rv_out["verification_score"]
            if self.ug is not None:
                ug_out = self.ug(
                    query[b : b + 1].expand(k, -1), ref_out.pooled[b : b + 1].expand(k, -1),
                    text_out.pooled[b : b + 1].expand(k, -1), reverse_disagreement=rv_out["disagreement"],
                )
                adj = adj * ug_out["confidence"]
            fused = _rank_fuse(top_scores[b, :k], adj)
            # stable=True (fixed 2026-09-12): _rank_fuse's own tie-breaking
            # is now CPU/CUDA-consistent (see its docstring), but its raw
            # output can still contain ties across candidates -- THIS
            # argsort (which produces the actual deployed ranking) needs
            # the same stability, or two devices could still disagree on
            # the final order despite _rank_fuse itself agreeing.
            order = fused.argsort(descending=True, stable=True)
            reranked_idx[b, :k] = top_idx[b, :k][order]
            reranked_scores[b, :k] = top_scores[b, :k][order]

        return reranked_idx, reranked_scores
