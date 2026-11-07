"""
Forward Composer (Section 4.1): Cross-Attentive Gated Fusion (CAGF).

Design rationale (paper framework, Section 4.1 + 4.6):
  1. Bidirectional multi-head cross-attention lets the edit text selectively
     attend to the reference image's spatial tokens (and vice versa) instead
     of collapsing both modalities to single vectors before fusing -- this is
     the "task-oriented multimodal composition" upgrade over naive
     concatenation/addition fusion used by older CIR baselines.
  2. A learned, per-dimension convex gate blends the cross-attended query
     representation with the raw reference-image embedding, generalizing the
     scalar residual gate popularized by Combiner-style architectures.
  3. A residual MLP projection head (GELU + LayerNorm) produces the final
     query embedding used for retrieval and consumed by the Attribute Edit
     Parser / Reverse Verifier / Uncertainty Gate downstream.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from revua_cir.config import FusionConfig


class CrossAttentionBlock(nn.Module):
    """Pre-norm bidirectional cross-attention block: text<-image and image<-text."""

    def __init__(self, dim: int, num_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.ln_t1 = nn.LayerNorm(dim)
        self.ln_i1 = nn.LayerNorm(dim)
        self.t_attends_i = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.i_attends_t = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_t2 = nn.LayerNorm(dim)
        self.ln_i2 = nn.LayerNorm(dim)
        self.ffn_t = nn.Sequential(nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * ffn_mult, dim))
        self.ffn_i = nn.Sequential(nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * ffn_mult, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, text_tok, image_tok, text_key_padding_mask=None):
        t = self.ln_t1(text_tok)
        i = self.ln_i1(image_tok)
        t_new, _ = self.t_attends_i(t, i, i, need_weights=False)
        i_new, _ = self.i_attends_t(i, t, t, key_padding_mask=text_key_padding_mask, need_weights=False)
        text_tok = text_tok + self.drop(t_new)
        image_tok = image_tok + self.drop(i_new)
        text_tok = text_tok + self.drop(self.ffn_t(self.ln_t2(text_tok)))
        image_tok = image_tok + self.drop(self.ffn_i(self.ln_i2(image_tok)))
        return text_tok, image_tok


def masked_mean_pool(tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return tokens.mean(dim=1)
    mask = mask.float().unsqueeze(-1)
    summed = (tokens * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


class ForwardComposer(nn.Module):
    """Fuses (reference image, edit text) backbone outputs into a single query embedding.

    **Residual composition (fixed 2026-09-08, `cfg.use_residual_composition`,
    default True).** A code audit found the previous CAGF-only path computed
    `fused = fused + text_pooled * 0.0` -- CLIP's own pretrained text
    embedding, multiplied by exactly zero, contributing NOTHING to either the
    forward value or the backward gradient (the chain rule multiplies by
    that 0.0 coefficient). The comment there claimed this "kept text_pooled
    in the graph for future use," but a literal `* 0.0` term is a true
    no-op, not a reduced-weight contribution -- so the query embedding that
    determines every retrieval R@1 number in this project never actually
    benefited from CLIP's contrastively pretrained, semantically-aligned
    text geometry. Because ReverseVerifier's Tier-1 path (the default) reuses
    this exact composer instance, RV's own reverse-reconstruction score was
    affected identically.

    Fixed by building a strong pretrained base query `q0 = norm(norm(v_ref) +
    norm(v_text))` (the same additive composition CLIP4Cir's own Stage 1
    uses) and letting the existing CAGF pipeline learn only a residual
    correction on top of it: `q = norm(q0 + alpha * CAGF_output)`, with
    `alpha` a learned scalar initialized near 0 (`cfg.residual_alpha_init`)
    so training starts from CLIP's already-aligned geometry rather than
    replacing it outright with a randomly-initialized fusion representation.
    `cfg.use_residual_composition=False` reproduces the old CAGF-only path
    (minus the dead `* 0.0` term, a behavior-preserving cleanup) for a
    controlled ablation against this fix.
    """

    def __init__(self, embed_dim: int, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, cfg.num_heads, cfg.ffn_mult, cfg.dropout) for _ in range(cfg.num_layers)
        ])
        gate_hidden = embed_dim * cfg.gate_hidden_mult
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, gate_hidden), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(gate_hidden, embed_dim)
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )
        self.final_ln = nn.LayerNorm(embed_dim)
        if cfg.use_residual_composition:
            self.residual_alpha = nn.Parameter(torch.tensor(float(cfg.residual_alpha_init)))
        if cfg.use_learned_temperature:
            self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / cfg.init_temperature))
        else:
            self.register_buffer("logit_scale", torch.ones([]) * math.log(1 / cfg.init_temperature))

    def forward(self, image_tokens, image_pooled, text_tokens, text_pooled, text_mask=None):
        """Returns (query_embedding [B,D] L2-normalized, attended_text_pooled [B,D]
        unnormalized -- used by AEP, diagnostics dict -- residual_alpha/
        cos_q0_q_final/pre_normalization_delta_norm/cos_q0_delta when
        use_residual_composition else {}, added 2026-09-09 for monitoring,
        keys revised same day -- see the residual branch below)."""
        text_kpm = None if text_mask is None else (text_mask == 0)
        t, i = text_tokens, image_tokens
        for block in self.blocks:
            t, i = block(t, i, text_key_padding_mask=text_kpm)
        text_attended = masked_mean_pool(t, text_mask)
        image_attended = i.mean(dim=1) if i.dim() == 3 else i
        # inject the text-grounded image signal (i_attends_t branch) back into the
        # pooled image representation, otherwise those cross-attention weights never
        # receive gradient (dead branch) -- this is the "bidirectional" half of CAGF.
        image_grounded = image_pooled + image_attended

        gate = torch.sigmoid(self.gate_mlp(torch.cat([image_grounded, text_attended], dim=-1)))
        fused = gate * text_attended + (1 - gate) * image_grounded
        residual = self.out_proj(fused)
        cagf_output = fused + residual

        diagnostics = {}
        if self.cfg.use_residual_composition:
            # Normalize the residual branch before scaling (fixed
            # 2026-09-08, code-audit finding): `cagf_output` is NOT unit
            # norm -- measured ~3.2 on this project's real data -- so
            # `alpha * cagf_output` was never actually a 5%-scale correction
            # despite `residual_alpha_init: 0.05`'s name/intent; the real
            # perturbation relative to q0 (always exactly unit norm by
            # construction) was ~16-21% of q0's own magnitude, over 3x what
            # the config value implied. Normalizing `delta` first makes
            # `||alpha * delta|| / ||q0|| ~= |alpha|` hold as intended, so
            # `residual_alpha_init: 0.05` genuinely means "a 5% correction."
            #
            # Also: `final_ln` (a LayerNorm with its own learned affine
            # scale/shift) used to apply AFTER combining with q0 --
            # LayerNorm's own rescaling can itself perturb q0's pretrained
            # CLIP geometry by an amount unrelated to alpha, partially
            # defeating the point of this fix. Applying it INSIDE the
            # residual branch (on `delta`, before the q0 combination) keeps
            # q0 untouched by any learned affine transform.
            q0 = F.normalize(F.normalize(image_pooled, dim=-1) + F.normalize(text_pooled, dim=-1), dim=-1)
            pre_norm_delta = self.final_ln(cagf_output)
            delta = F.normalize(pre_norm_delta, dim=-1)
            query = q0 + self.residual_alpha * delta
            # Monitoring diagnostics (added 2026-09-09, code-audit finding
            # #14; revised 2026-09-09, finding #7 of the follow-up document):
            # `residual_alpha` is trainable and unbounded -- nothing
            # currently stops it from growing well past its 0.05 init toward
            # a regime where the residual correction dominates q0's
            # pretrained CLIP geometry instead of gently correcting it. Not
            # constrained preemptively (per the document's own
            # recommendation: "do not constrain it unless training logs show
            # a real problem") -- these are surfaced so that judgment call
            # can be made from real training curves instead of guesswork.
            #
            # `delta_norm`/`alpha_delta_norm` (the original two diagnostics)
            # were dropped: `delta` is explicitly L2-normalized right above,
            # so `delta_norm` is ~1.0 by construction on every step and
            # `alpha_delta_norm` is just `|residual_alpha|` restated --
            # neither carries information the plain `residual_alpha` value
            # doesn't already give you. Replaced with two diagnostics that
            # DO carry new information:
            #   * `pre_normalization_delta_norm`: the raw scale of
            #     `cagf_output`, measured BEFORE `final_ln` (fixed 2026-09-10,
            #     external review of the real dry-run log): measuring it
            #     AFTER final_ln (as the first version of this diagnostic
            #     did) is dominated by LayerNorm's own per-element unit-
            #     variance normalization -- for embed_dim=D that gives an
            #     expected norm of sqrt(D) regardless of what CAGF actually
            #     learned (confirmed exactly: job 6298's real dry run on the
            #     real ViT-L/14 backbone, D=768, measured
            #     pre_normalization_delta_norm=27.7124 == sqrt(768)=27.7128
            #     to 4 significant figures -- the diagnostic was reporting a
            #     LayerNorm identity, not a property of CAGF). Measuring
            #     `cagf_output` directly, before final_ln, tracks how large
            #     CAGF's own RAW output is growing (independent of both
            #     alpha and LayerNorm's fixed rescaling), useful for
            #     noticing if the CAGF branch itself is drifting even though
            #     `delta`'s norm can never show that (it's always ~1
            #     post-normalize).
            #   * `cos_q0_delta`: cosine similarity between q0 and the
            #     (unit-norm, pre-alpha) residual direction `delta` --
            #     answers a genuinely different question than
            #     `cos_q0_q_final` (which measures the FINAL query's drift
            #     from q0, already implied by alpha's own magnitude once
            #     alpha is small). `cos_q0_delta` instead reveals what
            #     DIRECTION CAGF has learned to push in: near +1 means the
            #     learned correction reinforces q0 (CAGF has converged
            #     toward re-deriving the same simple sum-and-normalize
            #     signal), near 0 means it is learning something orthogonal
            #     (new, CAGF-specific information), near -1 means it is
            #     learning to actively counteract q0 -- a much more
            #     interpretable signal for "what is CAGF actually
            #     contributing" than the two norm-based diagnostics it
            #     replaces.
            with torch.no_grad():
                diagnostics["residual_alpha"] = self.residual_alpha.detach()
                diagnostics["cos_q0_q_final"] = F.cosine_similarity(
                    q0, F.normalize(query, dim=-1), dim=-1).mean().detach()
                diagnostics["pre_normalization_delta_norm"] = cagf_output.norm(dim=-1).mean().detach()
                diagnostics["cos_q0_delta"] = F.cosine_similarity(q0, delta, dim=-1).mean().detach()
        else:
            query = self.final_ln(cagf_output)
        query = F.normalize(query, dim=-1)
        return query, text_attended, diagnostics

    @property
    def temperature(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)
