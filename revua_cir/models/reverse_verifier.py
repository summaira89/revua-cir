"""
Reverse Verifier (Section 4.3).

For each candidate, this module tests whether "reversing" the requested edit
on the candidate would plausibly reconstruct the reference image's invariant
attributes. Two tiers, matching Section 4.3 / 4.6 exactly:

  * Tier 1 (`tier1_inverse_consistency`, default): no dedicated trainable reconstruction
    network. A learned `ReverseInstructionGenerator` maps the forward edit's
    text embedding to an approximate *reverse*-instruction embedding, which
    is then composed with the candidate through the *same* Forward Composer
    weights (shared, no extra backbone pass) to obtain a reconstructed
    reference embedding. This mirrors the cheap, frozen-encoder,
    generative-likelihood spirit of VQAScore-style verification without
    requiring an actual VQA/LLM call inside the training loop -- swap
    `reverse_text_generator_fn` for a real captioning/LLM model at inference
    time if one is available (interface documented below).
  * Tier 2 (`tier2_reconstruction`, stretch goal / ablation #3): a dedicated,
    separately-parameterized cross-attention reconstruction head, trained
    jointly, that regresses a full reconstructed-reference embedding from
    (candidate tokens, reverse-instruction tokens).

Both tiers produce:
  - `reconstructed`: predicted reference embedding, (B, D)
  - `verification_score`: calibrated Platt-scaled probability in [0, 1] that
    the candidate reverse-reconstructs the reference, (B,)
  - `disagreement`: 1 - verification_score, consumed by the Uncertainty Gate
    (Section 4.5 connective mechanism)
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from revua_cir.config import ReverseVerifierConfig
from revua_cir.models.fusion import CrossAttentionBlock, ForwardComposer, masked_mean_pool


class CrossBatchMemory(nn.Module):
    """Cross-Batch Memory (Wang, Zhang, Huang & Scott, "Cross-Batch Memory for
    Embedding Learning," CVPR 2020, arXiv:1912.06798) adapted for the Reverse
    Verifier's negative-mining pool.

    **Why this exists**: in-batch semi-hard negative mining (the default,
    see `reverse_consistency_loss`'s docstring) was measured to collapse at
    full training convergence because the negative SELECTOR (argmax/semi-hard
    over the current batch) is entangled with the SAME embedding space it is
    selecting over -- as that space converges, "hardest in-batch item" and
    "the actual right item" become progressively harder to tell apart. XBM
    decouples the negative source: instead of searching only the current
    batch's ~B-1 other targets, search a much larger FIFO queue of embeddings
    computed at PAST training steps. The paper's own "slow drift" measurement
    (Section 3.2, Eq. 2) justifies why stale-by-a-few-hundred-to-thousand-
    steps embeddings are still trustworthy: drift stabilizes within ~3K
    iterations of training, so a queue refreshed one mini-batch per step
    (their own ablation found faster refresh brought no measurable gain)
    stays valid.

    **Adaptation forced by this architecture**: XBM's original design queues
    only pooled embeddings (single vectors). Our Reverse Verifier's
    reconstruction term needs per-patch TOKEN features
    (`candidate_tokens`), not just the pooled vector, to actually score a
    candidate -- token features are ~256x768 floats per image at ViT-L/14
    (~750KB fp32), too large to queue at the paper's own thousands-of-entries
    scale without materially eating into training memory. So this queues
    BOTH pooled (cheap, used for negative SELECTION/thresholding, matching
    the paper's design) and token features (for the smaller, actually-
    affordable window of entries that get used once selected) at the SAME
    queue size -- i.e. this is a bounded queue (default 1024 entries, not
    the paper's "great many thousands" scale), a deliberate cost/fidelity
    tradeoff given this is a small auxiliary module's negative pool, not the
    primary retrieval representation the original paper was queueing.
    """

    def __init__(self, queue_size: int, warmup_steps: int):
        super().__init__()
        self.queue_size = queue_size
        self.warmup_steps = warmup_steps
        # Lazily shaped on first enqueue(): num_tokens (patch count) and
        # embed_dim depend on the backbone's image size / patch size, which
        # this module has no visibility into at construction time (it is
        # built as `ReverseVerifier(embed_dim, cfg.reverse_verifier, ...)`,
        # before any image has ever been encoded). Registered as empty
        # buffers here so `state_dict()`/`.to(device)` work immediately;
        # `_ensure_initialized` resizes them to their real shape on the
        # first real enqueue() call.
        self.register_buffer("pooled_queue", torch.zeros(0))
        self.register_buffer("tokens_queue", torch.zeros(0))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("filled", torch.zeros(1, dtype=torch.long))
        self.register_buffer("steps_seen", torch.zeros(1, dtype=torch.long))
        self._shape_initialized = False

    def _ensure_initialized(self, pooled: torch.Tensor, tokens: torch.Tensor) -> None:
        if self._shape_initialized:
            return
        embed_dim = pooled.size(-1)
        num_tokens = tokens.size(1)
        device = pooled.device
        # Re-register (not plain attribute assignment): `register_buffer`
        # again on an existing buffer name correctly replaces it while
        # keeping it tracked as a buffer (state_dict, .to(device), etc.) --
        # a bare `self.pooled_queue = ...` would silently turn it into an
        # untracked plain tensor attribute instead.
        self.register_buffer("pooled_queue", torch.zeros(self.queue_size, embed_dim, device=device))
        self.register_buffer("tokens_queue", torch.zeros(self.queue_size, num_tokens, embed_dim, device=device))
        self._shape_initialized = True

    @property
    def is_ready(self) -> bool:
        """Paper's own warm-up protocol (their Algorithm 1): let the network
        train normally for `warmup_steps` iterations before drawing from the
        queue, so cached embeddings are past the early, fast-drift phase.

        NOTE on units under gradcache: `enqueue()` is called once per CHUNK
        (`gradcache_step` processes `chunks_per_step = batch_size //
        gradcache_chunk_size` chunks per logical optimizer step), so
        `steps_seen` advances `chunks_per_step`x faster than logical
        training steps -- `warmup_steps=1000` under gradcache reaches
        `is_ready` after roughly `1000 / chunks_per_step` logical steps, not
        1000 of them. This is a reasonable interpretation (the queue really
        has seen that many distinct sub-batches of samples), but size
        `warmup_steps` with this in mind when gradcache is enabled, not by
        assuming it counts logical steps."""
        return bool(self.steps_seen.item() >= self.warmup_steps)

    @torch.no_grad()
    def enqueue(self, pooled: torch.Tensor, tokens: torch.Tensor) -> None:
        """FIFO push of this step's (detached) target embeddings. One
        mini-batch in/out per step, matching the paper's own ablation
        showing faster refresh cadences bring no measurable gain."""
        self._ensure_initialized(pooled, tokens)
        n = pooled.size(0)
        pooled, tokens = pooled.detach().to(self.pooled_queue.dtype), tokens.detach().to(self.tokens_queue.dtype)
        ptr = int(self.ptr.item())
        if n >= self.queue_size:
            # Batch bigger than the whole queue -- keep only the most recent
            # queue_size samples rather than indexing out of bounds.
            self.pooled_queue.copy_(pooled[-self.queue_size:])
            self.tokens_queue.copy_(tokens[-self.queue_size:])
            self.ptr[0] = 0
        else:
            end = ptr + n
            if end <= self.queue_size:
                self.pooled_queue[ptr:end] = pooled
                self.tokens_queue[ptr:end] = tokens
            else:
                first = self.queue_size - ptr
                self.pooled_queue[ptr:] = pooled[:first]
                self.tokens_queue[ptr:] = tokens[:first]
                self.pooled_queue[: n - first] = pooled[first:]
                self.tokens_queue[: n - first] = tokens[first:]
            self.ptr[0] = end % self.queue_size
        self.filled[0] = min(int(self.filled.item()) + n, self.queue_size)
        self.steps_seen += 1

    def semi_hard_negatives(self, query_sim_to_queue: torch.Tensor, pos_sim: torch.Tensor) -> torch.Tensor:
        """Given each query's similarity to every queued target
        (query_sim_to_queue: (B, filled)) and each query's true-positive
        similarity (pos_sim: (B, 1)), return the queue index of the hardest
        entry still below the positive -- same semi-hard rule as the
        in-batch path (FaceNet, Schroff et al. CVPR 2015), just searched
        over the queue instead of the current batch."""
        filled = int(self.filled.item())
        sim = query_sim_to_queue[:, :filled]
        semi_hard_mask = sim < pos_sim
        semi_hard_sim = sim.masked_fill(~semi_hard_mask, float("-inf"))
        idx = semi_hard_sim.argmax(dim=-1)
        no_semi_hard = ~semi_hard_mask.any(dim=-1)
        if no_semi_hard.any():
            fallback_idx = sim.argmax(dim=-1)
            idx = torch.where(no_semi_hard, fallback_idx, idx)
        return idx


class ReverseInstructionGenerator(nn.Module):
    """Learned proxy for 'what edit undoes this forward edit' in embedding space.

    A real system may instead call an LLM/seq2seq model to produce an actual
    reverse-instruction string, re-tokenize it, and feed its text embedding
    in directly -- that path is supported via `reverse_text_generator_fn` in
    `ReverseVerifier.forward` (bypasses this module when supplied).
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, forward_edit_text_pooled: torch.Tensor) -> torch.Tensor:
        # negation prior: a "reverse" instruction should start from the negated
        # forward-edit direction, then be refined by a learned MLP.
        return F.normalize(self.net(-forward_edit_text_pooled) , dim=-1)


class PlattCalibrator(nn.Module):
    """Learned scale/bias turning a raw cosine similarity into a calibrated probability."""

    def __init__(self, init_scale: float = 10.0, init_bias: float = 0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def logit(self, cos_sim: torch.Tensor) -> torch.Tensor:
        """Pre-sigmoid logit -- use this (not forward()) when training the
        calibrator itself with BCE, since `F.binary_cross_entropy` on the
        post-sigmoid probability errors under torch.autocast, while the
        fused with-logits form is autocast-safe (same reasoning as
        UncertaintyGate.calibration_loss's confidence_logit)."""
        return self.scale * cos_sim + self.bias

    def forward(self, cos_sim: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logit(cos_sim))


class Tier2ReconstructionHead(nn.Module):
    """Dedicated cross-attention reconstruction network (Section 4.3, Tier 2)."""

    def __init__(self, embed_dim: int, cfg: ReverseVerifierConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads=8, ffn_mult=4, dropout=0.1) for _ in range(cfg.tier2_num_layers)
        ])
        self.pool_proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))

    def forward(self, candidate_tokens: torch.Tensor, reverse_text_tokens: torch.Tensor, reverse_text_mask=None):
        t, i = reverse_text_tokens, candidate_tokens
        kpm = None if reverse_text_mask is None else (reverse_text_mask == 0)
        for block in self.blocks:
            t, i = block(t, i, text_key_padding_mask=kpm)
        # pool both branches -- the image branch's own cross-attention weights
        # (i_attends_t) would otherwise never receive gradient from the final block.
        pooled = masked_mean_pool(t, reverse_text_mask) + i.mean(dim=1)
        return F.normalize(self.pool_proj(pooled), dim=-1)


class CandidateDecompositionHead(nn.Module):
    """Extracts a candidate's *own* editable/invariant decomposition.

    This is the fix for the verifier's collapse. The original score was
    `cos(reconstructed, invariant_embed)` where `invariant_embed = AEP(text, ref)`
    depends only on the reference and the edit text -- the comparison target was
    identical for every candidate, so the score could not discriminate (measured
    pairwise AUC 0.50, i.e. chance).

    Scoring the candidate on *both* sides is what every working reranker does
    (CASE attends to the reference-text-candidate triplet jointly; VQA4CIR asks
    questions about each candidate). We keep the paper's editable/invariant
    decomposition -- the §4.5 connective mechanism -- but apply it to the
    candidate too, so verification asks the question the framework actually
    states: *does this candidate preserve what the edit said to preserve, and
    change what it said to change?*
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.invariant_head = nn.Linear(hidden_dim, embed_dim)
        self.editable_head = nn.Linear(hidden_dim, embed_dim)

    def forward(self, candidate_pooled: torch.Tensor, text_pooled: torch.Tensor):
        h = self.shared(torch.cat([candidate_pooled, text_pooled], dim=-1))
        return (F.normalize(self.invariant_head(h), dim=-1),
                F.normalize(self.editable_head(h), dim=-1))


class ReverseVerifier(nn.Module):
    def __init__(self, embed_dim: int, cfg: ReverseVerifierConfig, shared_forward_composer: Optional[ForwardComposer] = None):
        super().__init__()
        self.cfg = cfg
        # scale 1.0, not 10.0: sigmoid(10*cos) saturates -- at cos ~0.86 it returns
        # 0.9998, collapsing a 0.09 cosine gap to 0.00007. The calibrator was
        # destroying whatever separation existed. Learnable, so it can sharpen if
        # the data supports it, but it no longer starts in saturation.
        self.calibrator = PlattCalibrator(init_scale=1.0, init_bias=0.0)
        self.candidate_decomp = CandidateDecompositionHead(embed_dim, cfg.hidden_dim)
        # Relative weight of the "edit was actually applied" term against the
        # "invariant content was preserved" term. Initialized to 0.0, not 0.5
        # (fixed 2026-09-08, code-audit finding): `w = sigmoid(edit_weight)`
        # below, and sigmoid(0.5) ~= 0.622, not 0.5 -- the parameter's own
        # value is a pre-sigmoid logit, not the weight itself, so
        # `torch.tensor(0.5)` did not give the balanced 50/50 starting point
        # its own name/comment implies. sigmoid(0.0) = 0.5 exactly.
        self.edit_weight = nn.Parameter(torch.tensor(0.0))
        if cfg.tier == "tier1_inverse_consistency":
            self.reverse_gen = ReverseInstructionGenerator(embed_dim, cfg.hidden_dim)
            self.shared_composer = shared_forward_composer  # weight-shared, no new backbone params
            self.tier2_head = None
        elif cfg.tier == "tier2_reconstruction":
            self.reverse_gen = ReverseInstructionGenerator(embed_dim, cfg.hidden_dim)
            self.tier2_head = Tier2ReconstructionHead(embed_dim, cfg)
            self.shared_composer = None
        else:
            raise ValueError(f"Unknown reverse verifier tier: {cfg.tier}")

        self.xbm: Optional[CrossBatchMemory] = (
            CrossBatchMemory(cfg.xbm_queue_size, cfg.xbm_warmup_steps) if cfg.use_xbm else None
        )

    def forward(self, candidate_tokens: torch.Tensor, candidate_pooled: torch.Tensor,
                forward_text_pooled: torch.Tensor, invariant_embed: torch.Tensor,
                reverse_text_generator_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        if reverse_text_generator_fn is not None:
            reverse_text_pooled = reverse_text_generator_fn(forward_text_pooled)
        else:
            reverse_text_pooled = self.reverse_gen(forward_text_pooled)

        if self.cfg.tier == "tier1_inverse_consistency":
            assert self.shared_composer is not None, "Tier-1 verifier needs a shared ForwardComposer instance."
            reverse_text_tokens = reverse_text_pooled.unsqueeze(1)
            reconstructed, _, _ = self.shared_composer(
                image_tokens=candidate_tokens, image_pooled=candidate_pooled,
                text_tokens=reverse_text_tokens, text_pooled=reverse_text_pooled, text_mask=None,
            )
        else:
            reverse_text_tokens = reverse_text_pooled.unsqueeze(1)
            reconstructed = self.tier2_head(candidate_tokens, reverse_text_tokens, reverse_text_mask=None)

        # --- Term 1: reverse reconstruction (the original signal) --------------
        # "Does undoing the edit on this candidate return us to the reference's
        # invariant content?" Kept because it is the paper's reverse-verification
        # idea, but it is no longer the only term: on its own it collapsed.
        recon_cos = F.cosine_similarity(reconstructed, invariant_embed, dim=-1)

        # --- Term 2: candidate-conditioned invariant preservation --------------
        # The candidate's *own* invariant sub-space, compared against what the
        # query says must be preserved. This is what makes the score depend on
        # the candidate at all.
        cand_invariant, cand_editable = self.candidate_decomp(candidate_pooled, forward_text_pooled)
        invariant_cos = F.cosine_similarity(cand_invariant, invariant_embed, dim=-1)

        # --- Term 3: edit application -----------------------------------------
        # The requested edit should be *present* in the candidate, so the
        # candidate's editable sub-space should align with the edit text. A
        # candidate that merely preserves the reference (the classic distractor,
        # and the reference image itself) scores high on term 2 and low here --
        # which is precisely the confusion the forward ranker cannot resolve.
        edit_cos = F.cosine_similarity(cand_editable, F.normalize(forward_text_pooled, dim=-1), dim=-1)

        w = torch.sigmoid(self.edit_weight)
        if self.cfg.use_invariant_term:
            cos_sim = (1.0 - w) * (0.5 * recon_cos + 0.5 * invariant_cos) + w * edit_cos
        else:
            # Retrained no-invariant ablation (B6.4.3): drop the term and
            # renormalise the remaining two in their trained ratio, exactly as
            # rv_component_ablation.component_score does post hoc. The
            # denominator is 0.5 + 0.5w, which lies in (0.5, 1) for any w, so
            # it never vanishes. invariant_cos is still computed and returned
            # so score caches keep the same fields; it simply carries no
            # weight, and the candidate-invariant head receives no gradient.
            recon_w = 0.5 * (1.0 - w)
            cos_sim = (recon_w * recon_cos + w * edit_cos) / (recon_w + w)

        verification_score = self.calibrator(cos_sim)
        disagreement = 1.0 - verification_score
        return {
            "reconstructed": reconstructed,
            "cos_sim": cos_sim,
            "recon_cos": recon_cos,
            "invariant_cos": invariant_cos,
            "edit_cos": edit_cos,
            "verification_score": verification_score,
            "disagreement": disagreement,
        }

    @staticmethod
    def reverse_consistency_loss(cos_sim: torch.Tensor,
                                 negative_cos_sim: Optional[torch.Tensor] = None,
                                 margin: float = 0.2) -> torch.Tensor:
        """Reverse-consistency objective.

        With `negative_cos_sim=None` this is the original positives-only form,
        `mean(1 - cos_sim)`. That form is **degenerate for reranking**: the
        verifier only ever sees ground-truth targets and is pushed to score every
        one of them ~1.0, so it has no reason to separate correct candidates from
        wrong ones. At inference the reranker feeds it the top-k *retrieved*
        candidates -- mostly wrong -- and needs exactly that separation. Training
        longer makes it worse, because as cos_sim -> 1 everywhere the reranking
        signal flattens toward a constant, and a constant rescale cannot reorder.

        Passing `negative_cos_sim` adds a hinge that pushes wrong candidates
        below the positives by `margin`, which is what makes the score
        discriminative and therefore usable for reranking:

            L = mean(1 - cos_pos) + mean(relu(cos_neg - cos_pos + margin))

        The margin term is only active while a negative scores within `margin`
        of its positive, so it stops penalising once separation is achieved.
        """
        loss = (1.0 - cos_sim).mean()
        if negative_cos_sim is not None:
            # Broadcast positives across however many negatives each query has.
            pos = cos_sim.unsqueeze(-1) if negative_cos_sim.dim() > cos_sim.dim() else cos_sim
            loss = loss + F.relu(negative_cos_sim - pos + margin).mean()
        return loss
