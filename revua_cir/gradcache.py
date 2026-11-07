"""Gradient caching (Gao et al., arXiv:2101.06983) for large-batch contrastive
training on a small GPU.

**Why this exists.** With in-batch InfoNCE the batch size *is* the negative
count. Measured on this project: `batch_size: 64` gives 63 negatives, while
CLIP4Cir and comparable CIR baselines train with 511-1023. Contrastive retrieval
quality scales strongly with negatives, so a starved batch caps accuracy no
matter how long you train -- a plausible cause of results that plateau below
published numbers. Directly confirmed 2026-09: this project's own
full-unfreeze CIRR run, forced to batch_size=12 (11 negatives) by memory
limits at the time, scored R@1 21.90 -- *worse* than partial-freeze's
batch_size=24 run (R@1 29.35), despite training strictly more parameters.
CLIP4Cir's official reproduction (github.com/ABaldrati/CLIP4Cir, this
project's own baselines/CLIP4Cir/) trains the same scale of backbone
(RN50x4) at batch_size=96+ and reaches R@1 ~44 on the same dataset -- the
gap is consistent with a starved-batch effect, not an architecture ceiling.

**Interface.** `gradcache_step` takes a *list of small pre-chunked batches*
(each `optim.gradcache_chunk_size` samples), not one giant batch. `train.py`
builds these from a DataLoader whose own `batch_size` is the chunk size, so
Python/PIL never collates more than one chunk's worth of images at a time. An
earlier version collated the full logical batch (1024 samples, ~1.2 GB of
tensors) before chunking it internally, which crashed with "Couldn't open
shared file mapping ... error code: 1455" (Windows commit-charge exhaustion) --
the buffer size scales with the *logical* batch, not gradcache's internal
chunk size, so internal-only chunking never helped the loader.

**Trainable-backbone support (2026-09).** The original version of this
module REQUIRED `backbone.freeze_backbone == true`: pass 3 recomputed and
injected gradient only for the query side (`encode_query`), so a trainable
backbone received retrieval-loss gradient through the query path only --
the target-image encode path (`backbone.encode_image(target_pixel_values)`,
called directly, not through `encode_query`) got NO retrieval gradient at
all, silently. Verified via a tied-encoder dual-injection pattern confirmed
against the official GradCache reference implementation
(github.com/luyug/GradCache) and the original paper (Gao et al.,
arXiv:2101.06983): two independent forward passes through the SAME shared
module (one on the reference image via `encode_query`, one on the target
image via `encode_image` directly) each get their own cached gradient
injected via a separate `torch.autograd.backward()` call, and because
PyTorch accumulates gradient into shared leaf `Parameter.grad` by default
(no `zero_grad()` between the two calls), the two injections correctly sum
into the shared backbone's gradient -- this is exactly the mathematical
structure of `d(loss)/d(theta) = sum_i d(loss)/d(f(x_i)) * d(f(x_i))/d(theta)`
when `f` (the backbone) is evaluated on two different streams of `x_i`
(reference images, target images) that are summed into the same loss.
No `retain_graph=True` is needed: each forward() call builds a fresh,
independent autograd graph even when it reuses the same parameters, so the
two backward() calls never touch each other's graph.

**RNG/dropout consistency (fixed 2026-09).** Previously, neither pass 1
(cache-building, no_grad) nor pass 3 (recompute-with-grad) saved/restored RNG
state per chunk, so any dropout in the composer/backbone saw a DIFFERENT mask
in pass 1 (discarded) vs pass 3 (used for the real gradient) -- meaning the
injected gradient was technically the gradient of a different forward pass
than the one whose loss produced it. Confirmed via code inspection this was a
real, silent gap in every gradcache training run using `fusion.dropout > 0`
(every shipped config that actually trains with gradcache, including
`stage1_finetune_full_unfreeze_gradcache.yaml` at dropout=0.1 -- job 6258's
training run predates this fix). Fixed via `RandContext` (below), matching
the official reference implementation's mechanism
(github.com/luyug/GradCache, `grad_cache/context_managers.py`): each chunk's
pass-1 RNG state is captured before its forward and replayed in pass 3, so
`encode_query`/`encode_image`'s dropout draws the identical mask both times.
Scoped to the query/target-side forwards specifically (pass 1's exact
counterpart); the aux-module forward (`model.aep`/`rv`/`ug` via `model(sub)`
in pass 3) has no pass-1 equivalent to replay against, since pass 1 never
runs it -- that forward necessarily continues on the natural RNG stream, an
inherent asymmetry of caching only the retrieval-critical embeddings, not a
gap this fix leaves unaddressed within its actual scope.

See `gradcache_step`'s docstring for the three-pass mechanism and the measured
memory/throughput numbers.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from revua_cir.losses import symmetric_info_nce


class RandContext:
    """Captures CPU+CUDA RNG state on entry, restores it on exit -- the same
    mechanism the official GradCache reference implementation
    (github.com/luyug/GradCache, `grad_cache/context_managers.py`) uses to
    make a later recompute forward draw the IDENTICAL dropout mask as an
    earlier no_grad forward on the same input.

    Why this exists: pass 1 (embedding cache, no_grad) and pass 3 (recompute
    with grad) each call `model.encode_query`/`encode_image` once per chunk.
    Every dropout layer inside those calls consumes the global RNG stream, so
    without this, pass 3's dropout mask differs from pass 1's -- meaning the
    gradient injected via `torch.autograd.backward(cached_tensor, grad_cached)`
    is technically the gradient of a DIFFERENT forward pass than the one whose
    loss was used to compute `grad_cached`. This was a real, measured gap
    (not yet quantified in this project before now) whenever `fusion.dropout
    > 0` -- the shipped training configs (e.g. stage1_finetune_full_unfreeze_
    gradcache.yaml) all use dropout=0.1, so every gradcache run before this
    fix carried it.

    Usage: `ctx = RandContext()` right before a chunk's pass-1 forward (this
    captures the state the RNG stream is in at that moment); later,
    `with ctx.fork():` around that same chunk's pass-3 forward replays the
    stream from that exact captured point, so the two forwards draw the same
    dropout mask. `fork()` (not a bare restore) is deliberate: it snapshots
    the RNG state again on __exit__ so the stream continues moving forward
    normally after the `with` block, rather than getting stuck replaying the
    same state on every subsequent chunk.
    """

    def __init__(self):
        self.cpu_state = torch.get_rng_state()
        self.cuda_states = ([torch.cuda.get_rng_state(d) for d in range(torch.cuda.device_count())]
                             if torch.cuda.is_available() else [])

    def restore(self):
        torch.set_rng_state(self.cpu_state)
        if self.cuda_states:
            torch.cuda.set_rng_state_all(self.cuda_states)

    class _Fork:
        def __init__(self, outer: "RandContext"):
            self.outer = outer

        def __enter__(self):
            # Save the CURRENT (caller's) state so it can be put back after
            # the block, then jump the stream back to the captured point.
            self._caller_cpu = torch.get_rng_state()
            self._caller_cuda = ([torch.cuda.get_rng_state(d) for d in range(torch.cuda.device_count())]
                                  if torch.cuda.is_available() else [])
            self.outer.restore()

        def __exit__(self, *exc):
            # Re-capture so the NEXT fork() of this same RandContext replays
            # from where this block left off (matters if a chunk's forward is
            # ever entered more than once -- not currently the case, but
            # makes the class correct as a general reusable primitive rather
            # than a single-use one), then restore the caller's own state so
            # code after the `with` block sees an undisturbed RNG stream.
            self.outer.cpu_state = torch.get_rng_state()
            self.outer.cuda_states = ([torch.cuda.get_rng_state(d) for d in range(torch.cuda.device_count())]
                                       if torch.cuda.is_available() else [])
            torch.set_rng_state(self._caller_cpu)
            if self._caller_cuda:
                torch.cuda.set_rng_state_all(self._caller_cuda)

    def fork(self):
        return RandContext._Fork(self)


def _to_device(sub: dict, device) -> dict:
    out = {}
    for k, v in sub.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def gradcache_step(model, chunks: List[dict], optimizer, cfg, scaler=None) -> Dict[str, torch.Tensor]:
    """One optimizer step with gradient caching over pre-chunked sub-batches.

    `chunks` is a list of small batches (each `optim.gradcache_chunk_size` samples,
    already collated) that together make up one logical step of
    `data.batch_size` samples. Taking pre-chunked input rather than one giant
    batch is deliberate: the DataLoader then never has to collate, pin, or hold
    1024 images at once. Assembling that in one collate call was the actual
    cause of a Windows commit-charge crash at bs=1024 (`Couldn't open shared
    file mapping ... error code: 1455`) even with only 2 workers -- the buffer
    size scales with `batch_size`, not `num_workers`, once you look closely.

    Three passes over the chunks per optimizer step:

      1. Encode every chunk under ``no_grad`` and cache the pooled embeddings.
         The composer's cross-attention activations are freed after each chunk,
         so peak memory is set by the chunk, not the logical batch.
      2. Assemble the full ``B x B`` similarity matrix from the cached
         embeddings and backprop **only to the embeddings**. For B=1024 that
         matrix is ~4 MB.
      3. Recompute each chunk *with* grad and inject its cached
         embedding-gradient via ``torch.autograd.backward(tensors,
         grad_tensors)``. Only one chunk of activations is ever resident.

    The resulting parameter gradient is mathematically identical to a true
    large-batch step. Measured on an 8 GB RTX 4060: bs=1024 (chunk=32) peaks at
    **3.43 GB** at roughly half the throughput of an equivalent small-batch
    step, PROVIDED `loss.hard_negative_weight == 0`. That option reweights
    logits by each negative's closeness to the positive *within the batch*, so
    the weights a chunk sees are not the weights the full batch sees; mixing it
    with gradient caching gave a uniform 0.48x gradient. Since gradcache exists
    precisely to make the batch large, and a large batch already supplies
    abundant hard negatives, the two are redundant -- so this path asserts
    rather than silently disagreeing with the plain path.

    Auxiliary module losses (disentangle / reverse / uncertainty) are per-sample
    with no cross-batch coupling, so they are computed per chunk and averaged.
    """
    # Determine from ACTUAL parameters, not the coarse freeze_backbone flag
    # (fixed 2026-09-08): `freeze_backbone=True` only freezes the pretrained
    # trunk itself (e.g. HFClipBackbone's `self.clip.parameters()`) -- it does
    # NOT touch separate trainable adapter/projection layers that exist
    # whenever a backbone's native hidden/projection dim differs from
    # cfg.backbone.embed_dim (HFClipBackbone._adapter returns nn.Linear, not
    # nn.Identity, whenever in_dim != out_dim -- e.g. ViT-L/14's vision
    # hidden_size=1024 vs a typical embed_dim=768 makes image_out_proj a real
    # trainable nn.Linear even when freeze_backbone=True). The OLD `not
    # cfg.backbone.freeze_backbone` check would then skip injecting the
    # target-side gradient (see pass 3 below) into those very real, trainable
    # adapter parameters -- silently leaving them untrained under gradcache
    # specifically (the plain, non-gradcache path has no such gap, since it
    # just runs one ordinary backward() over the whole graph). Does not
    # affect any full-unfreeze config (trainable_backbone is already True
    # there regardless of this check); matters only for frozen-backbone
    # gradcache runs where in_dim != out_dim for at least one adapter.
    trainable_backbone = any(p.requires_grad for p in model.backbone.parameters())
    if cfg.loss.hard_negative_weight > 0:
        raise ValueError(
            "gradcache requires loss.hard_negative_weight == 0 (got "
            f"{cfg.loss.hard_negative_weight}). Its in-batch reweighting is not "
            "chunk-decomposable and silently scales gradients. With 511-1023 "
            "in-batch negatives the reweighting is unnecessary; set it to 0.")
    device = next(model.parameters()).device
    bsz = sum(c["reference_pixel_values"].size(0) for c in chunks)
    amp = cfg.optim.amp and device.type == "cuda"

    # ---- pass 1: cache embeddings without building a graph ------------------
    # Also cache UG's log_var (a no_grad, throwaway forward through the tiny
    # variance_head MLP -- cheap) when UG is enabled: neighborhood_deviation_loss
    # needs a FULL-BATCH k-NN target and mean/std normalization, not a
    # chunk's -- see revua_cir.py's forward() docstring for the measured
    # ~195x gradient error this fixes.
    q_cache: List[torch.Tensor] = []
    t_cache: List[torch.Tensor] = []
    log_var_cache: List[torch.Tensor] = []
    # One RandContext per chunk, captured immediately before that chunk's
    # forward -- pass 3 later replays each one so its recompute forward draws
    # the SAME dropout mask this pass-1 forward did (see RandContext's own
    # docstring above for why this matters).
    rand_contexts: List[RandContext] = []
    with torch.no_grad():
        for chunk_batch in chunks:
            rand_contexts.append(RandContext())
            sub = _to_device(chunk_batch, device)
            with torch.autocast("cuda", enabled=amp):
                q = model.encode_query(sub["reference_pixel_values"],
                                       sub["edit_input_ids"], sub["edit_attention_mask"])
                tgt = model.backbone.encode_image(sub["target_pixel_values"])
                if model.ug is not None:
                    log_var_cache.append(model.ug.variance_head(q["query"]).float())
            # composer output is already normalized; the target is not.
            q_cache.append(q["query"].float())
            t_cache.append(F.normalize(tgt.pooled, dim=-1).float())

    # ---- pass 2: full-batch InfoNCE, grad only w.r.t. the embeddings --------
    # Clear grads BEFORE this backward, not after: `model.composer.temperature` is
    # a live parameter, so this pass legitimately writes `logit_scale.grad`.
    # Zeroing afterwards silently discarded it and left the temperature untrained
    # (the equivalence check caught it as 62 params vs 63).
    optimizer.zero_grad(set_to_none=True)
    Q = torch.cat(q_cache).requires_grad_(True)
    # T only needs requires_grad when the backbone is trainable -- with a frozen
    # backbone the target-image encode has no trainable parameters upstream, so
    # d(loss)/d(T) would never be consumed and marking it just wastes autograd
    # bookkeeping. `symmetric_info_nce` is symmetric in Q/T (both i2t and t2i
    # terms), so T carries real gradient whenever the backbone can use it.
    T = torch.cat(t_cache).requires_grad_(trainable_backbone)
    retrieval = symmetric_info_nce(
        Q, T, model.composer.temperature,
        label_smoothing=cfg.loss.label_smoothing,
        hard_negative_weight=cfg.loss.hard_negative_weight,
    )
    # Scale pass 2's loss before backward (standard GradScaler usage) even
    # though this specific backward runs on fp32 Q/T with no fp16 tensors of
    # its own -- grad_q/grad_t are what pass 3 later injects INTO an
    # autocast-fp16 forward (encode_query/encode_image), and it is THAT
    # backward pass, through fp16 activations, where underflow is a real risk
    # -- exactly the failure mode GradScaler exists to prevent. Scaling here
    # means grad_q/grad_t themselves come out pre-scaled, so they arrive
    # already at a safe magnitude when pass 3 injects them.
    scaled_loss = scaler.scale(retrieval["loss"]) if scaler is not None else retrieval["loss"]
    scaled_loss.backward()
    grad_q = Q.grad.detach()
    grad_t = T.grad.detach() if trainable_backbone else None

    # Detached full-batch view for the per-chunk aux-loss recompute (pass 3
    # below) to use for statistics/selection that must be batch-wide, not
    # chunk-local -- RV's semi-hard negative search and UG's neighborhood-
    # deviation loss. Never used as a tensor gradient flows through (that
    # still happens strictly per-chunk in pass 3); see revua_cir.py's
    # forward() docstring.
    full_log_var = torch.cat(log_var_cache) if log_var_cache else None

    # ---- pass 3: recompute per chunk with grad, inject cached gradients -----
    # Call `encode_query` directly rather than `model(sub)`: the full forward also
    # computes a *chunk-level* InfoNCE, and backward through the resulting graph
    # double-counts the retrieval term that `grad_q` already carries. That was the
    # source of the uniform 0.5083x gradient discrepancy.
    #
    # Auxiliary module losses (disentangle / reverse / uncertainty) are per-sample
    # with no cross-batch coupling, so they are computed per chunk and scaled by
    # the chunk's share of the batch -- exactly reproducing a full-batch mean.
    aux_totals: Dict[str, float] = {}
    n_chunks = 0
    offset = 0
    for chunk_idx, chunk_batch in enumerate(chunks):
        n_in_chunk = chunk_batch["reference_pixel_values"].size(0)
        sl = slice(offset, offset + n_in_chunk)
        offset += n_in_chunk
        sub = _to_device(chunk_batch, device)

        # Replay this chunk's pass-1 RNG state so encode_query/encode_image's
        # dropout draws the SAME mask pass 1 used when it computed the
        # embedding that grad_q/grad_t are the gradient of -- without this,
        # the gradient injected below is technically the gradient of a
        # DIFFERENT forward pass (different dropout mask) than the one whose
        # loss produced grad_q/grad_t. Only covers the query/target-side
        # forwards, which are pass 1's exact counterpart; the aux module
        # forward below (model.aep/rv/ug) has no pass-1 equivalent to replay
        # against -- pass 1 never runs it -- so it necessarily continues on
        # the natural, still-advancing RNG stream instead.
        with rand_contexts[chunk_idx].fork():
            with torch.autocast("cuda", enabled=amp):
                q = model.encode_query(sub["reference_pixel_values"],
                                       sub["edit_input_ids"], sub["edit_attention_mask"])
        q_chunk = q["query"].float()
        if q_chunk.requires_grad:
            torch.autograd.backward([q_chunk], [grad_q[sl]])

        # Target-side injection: a SEPARATE forward through the same shared
        # backbone (tied-encoder pattern, confirmed against the official
        # GradCache reference implementation, github.com/luyug/GradCache,
        # "Use Tied Encoder?"). This forward() call builds its own independent
        # autograd graph even though it reuses `model.backbone`'s parameters,
        # so this backward() cannot interfere with the query-side backward()
        # just above -- no retain_graph needed, and PyTorch's default gradient
        # ACCUMULATION into the shared backbone parameters' .grad (no
        # zero_grad() between the two calls) is exactly what sums the two
        # injections into the correct total gradient, mirroring how pass 2's
        # loss is one sum over both the i2t and t2i terms.
        #
        # Only runs when the backbone is trainable: with a frozen backbone,
        # target_out.pooled has no trainable ancestor, so injecting grad_t
        # here would be a no-op walk with nothing to accumulate into -- skip
        # the extra forward pass entirely rather than pay for it and discard it.
        if trainable_backbone:
            # Second fork() on the SAME RandContext, not a fresh one: pass 1
            # ran encode_query THEN encode_image back-to-back under one
            # continuous RNG trajectory (see pass 1's loop above), so
            # encode_image's real dropout mask depends on the RNG state
            # AFTER encode_query already consumed some of it, not the state
            # from before either call. The query-side fork's __exit__
            # re-captured the context at exactly that post-query point (see
            # RandContext._Fork.__exit__), so this second fork continues
            # from there -- replaying the correct trajectory instead of
            # restarting encode_image as if it ran first.
            with rand_contexts[chunk_idx].fork():
                with torch.autocast("cuda", enabled=amp):
                    target_out = model.backbone.encode_image(sub["target_pixel_values"])
                    t_chunk = F.normalize(target_out.pooled, dim=-1).float()
            if t_chunk.requires_grad:
                torch.autograd.backward([t_chunk], [grad_t[sl]])

        # Module terms, if any are enabled, on their own small graph. NOTE:
        # `model(sub)` internally re-encodes the target image a THIRD time
        # this chunk (once more for `symmetric_info_nce`'s chunk-local
        # retrieval term, which is then subtracted out below) -- wasteful but
        # not incorrect, since it's yet another independent forward/graph and
        # `aux`'s backward only carries the disentangle/reverse/uncertainty
        # gradient (the chunk-local retrieval term cancels in the subtraction
        # and was already an existing property of this code, not introduced
        # by the target-side injection above). Left as-is: fixing the
        # redundant encode would need `revua_cir.py`'s forward() to accept
        # precomputed embeddings, out of scope for unblocking trainable-
        # backbone gradcache specifically.
        if model.aep is not None or model.rv is not None or model.ug is not None:
            full_batch_context = {"query": Q.detach(), "target_pooled": T.detach(), "chunk_offset": sl.start}
            if full_log_var is not None:
                full_batch_context["log_var"] = full_log_var
            with torch.autocast("cuda", enabled=amp):
                out = model(sub, full_batch_context=full_batch_context)
            aux = out["losses"]["total"] - out["losses"]["retrieval"]
            if torch.is_tensor(aux) and aux.requires_grad:
                aux_scaled = aux * (n_in_chunk / bsz)
                if scaler is not None:
                    aux_scaled = scaler.scale(aux_scaled)
                aux_scaled.backward()
            for k, v in out["losses"].items():
                # .detach() before float() (fixed 2026-09-09, code-audit
                # finding): `v` still carries requires_grad=True/its autograd
                # graph here even after aux_scaled.backward() above -- .backward()
                # does not clear that flag on the original tensor, only
                # populates .grad on the leaves. float() on a requires_grad
                # tensor triggers "Converting a tensor with requires_grad=True
                # to a scalar may lead to unexpected behavior" (seen in real
                # cluster logs, e.g. job 6258's dry run) -- harmless for a
                # logging-only value like this, but .detach() first is the
                # correct, warning-free way to pull out a Python float.
                aux_totals[k] = aux_totals.get(k, 0.0) + float(v.detach()) if torch.is_tensor(v) else v
        n_chunks += 1

    # AMP/GradScaler handling (fixed 2026-09-08): previously `scaler` was
    # accepted as a parameter but never used anywhere in this function --
    # optimizer.step() was called directly, bypassing scaler.step()/update()
    # entirely. That meant gradcache training got NONE of GradScaler's real
    # protections: no inf/NaN gradient detection (a corrupted fp16 forward
    # would silently corrupt the optimizer step instead of being caught and
    # skipped, exactly as the plain path's scaler.step() does via its
    # internal has_overflow check), and the scaler's adaptive scale factor
    # never updated during gradcache training, leaving it stuck wherever the
    # plain path (or the scaler's own init) last left it.
    #
    # Now mirrors the plain path's real sequence (train.py's non-gradcache
    # branch): unscale before clipping (so grad_clip_norm operates on
    # TRUE-magnitude gradients, not the scaled ones pass 2/pass 3 produced),
    # then scaler.step() (which internally skips optimizer.step() if any
    # unscaled gradient is inf/NaN -- the safety net gradcache never had),
    # then scaler.update() (adapts the scale factor for the next step based
    # on whether this step overflowed).
    step_skipped = False
    if scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip_norm)
        # Detect a skipped optimizer.step() (fixed 2026-09-10, real bug found
        # via external review of job 6300's real training log): `scaler.
        # step(optimizer)` silently SKIPS the underlying `optimizer.step()`
        # whenever any unscaled gradient this step is inf/NaN (the safety
        # net documented in the comment above) -- but train.py's caller
        # previously called `scheduler.step()` unconditionally right after,
        # with no way to know a step had been skipped. This produces
        # exactly the `UserWarning: Detected call of lr_scheduler.step()
        # before optimizer.step()` PyTorch emits, and is more than
        # cosmetic: on a skipped step the LR schedule still advances one
        # step even though no parameter update happened, silently
        # desynchronizing "step count" from "actual optimizer steps taken"
        # for the rest of training. `scaler.get_scale()` strictly decreases
        # only on a skipped step (the adaptive backoff after an overflow);
        # comparing before/after `update()` is GradScaler's own documented
        # way to detect this (see torch.cuda.amp docs, "Gradient scaling
        # examples").
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        step_skipped = scaler.get_scale() < scale_before
    else:
        # scaler=None means AMP is fully disabled for this run (or the
        # caller chose not to pass one) -- grad_q/grad_t/aux were never
        # scaled above either (see the `if scaler is not None` guards), so
        # they are already at true scale and clipping applies directly, same
        # as the always-true case before this fix.
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip_norm)
        optimizer.step()

    losses = {k: torch.tensor(v / max(n_chunks, 1))
              for k, v in aux_totals.items() if k not in ("retrieval", "total")}
    losses["retrieval"] = retrieval["loss"].detach()
    # Not a training loss term -- a plain bool the caller (train.py) reads
    # to decide whether to call scheduler.step() this iteration. Excluded
    # from AverageMeter logging the same way train.py already excludes any
    # non-numeric entry (see its own step_losses handling).
    losses["_optimizer_step_skipped"] = step_skipped
    # `out["losses"]` (revua_cir.py's forward()) carries BOTH the raw
    # per-term value (e.g. "disentangle") and its lambda-weighted counterpart
    # ("disentangle_weighted") for every enabled aux module -- see that
    # file's own header comment. `total` must sum only the *_weighted keys
    # (plus retrieval, which is already weight-1); summing the raw keys too
    # would double-count every aux term. Diagnostic-only entries (e.g.
    # "diag_rv_cos_gap") are never part of the optimized objective and stay
    # excluded either way.
    weighted_keys = [k for k in losses if k.endswith("_weighted")]
    losses["total"] = losses["retrieval"] + sum(losses[k] for k in weighted_keys)
    return losses
