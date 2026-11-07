"""
Training entrypoint.

    python -m revua_cir.train --config configs/default.yaml \
        --data-root /path/to/datasets --output-dir ./runs/exp1

Implements every "small thing" called out in the paper framework / user's
request on top of the four-module architecture:
  * discriminative learning rates (backbone trains `backbone_lr_mult` slower
    than the new heads -- standard for fine-tuning a pretrained CLIP),
  * linear warmup + cosine decay schedule,
  * gradient clipping,
  * mixed precision (AMP),
  * optional EMA of weights,
  * early stopping on a configurable validation metric (Section: Ablation
    plan / user's explicit request), with a warmup grace period so an
    unlucky first epoch cannot trigger it,
  * top-K checkpoint retention,
  * per-module loss logging so the connective-loss ablation (Section 8, #2)
    is directly readable from the training log without extra plumbing.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from revua_cir.config import ReVUAConfig
from revua_cir.data.collate import build_collate_fn
from revua_cir.data.datasets import DATASET_REGISTRY, ConcatTripletDataset
from revua_cir.gradcache import gradcache_step
from revua_cir.data.tokenizer import build_tokenizer
from revua_cir.data.transforms import build_image_transform
from revua_cir.evaluate import evaluate_dataset
from revua_cir.models.attribute_parser import build_attribute_extractor
from revua_cir.models.revua_cir import ReVUACIR
from revua_cir.utils import (
    AverageMeter, EarlyStopper, EMA, TopKCheckpointKeeper, get_logger, save_checkpoint, set_seed,
)


def build_train_dataset(cfg: ReVUAConfig, transform, llm_parser_fn=None):
    """`llm_parser_fn`: required when `cfg.attribute_parser.parser_backend`
    is "llm"/"hybrid" -- a real LLM call is a runtime integration point this
    codebase deliberately does not hardcode a specific provider for (see
    attribute_parser.py's LLMAttributeExtractor docstring), so it must be
    supplied by the caller. Previously (fixed 2026-09-08) this parameter did
    not exist at all: every dataset silently used RuleBasedAttributeExtractor
    regardless of parser_backend, so setting `parser_backend: llm` in a
    config had NO effect on the actual training data. Now the config value
    genuinely selects the extractor -- and, matching
    build_attribute_extractor's own existing behavior, raises a clear
    ValueError for parser_backend="llm" without llm_parser_fn, rather than
    silently falling back to rules."""
    attribute_extractor = (
        build_attribute_extractor(cfg.attribute_parser, llm_parser_fn=llm_parser_fn)
        if cfg.attribute_parser.enabled else None
    )
    datasets = []
    for name in cfg.data.train_datasets:
        cls = DATASET_REGISTRY[name]
        root = os.path.join(cfg.data.root, name)
        kwargs = dict(
            transform=transform, include_attribute_targets=cfg.attribute_parser.enabled,
            attribute_vocab_size=cfg.attribute_parser.attribute_vocab_size,
            max_attributes=cfg.attribute_parser.max_attributes,
            noise_injection_rate=cfg.data.noise_injection_rate, seed=cfg.run.seed,
            attribute_extractor=attribute_extractor,
        )
        datasets.append(cls(root=root, split="train", transform=transform, **{k: v for k, v in kwargs.items() if k != "transform"}))
    return ConcatTripletDataset(datasets) if len(datasets) > 1 else datasets[0]


def build_val_dataset(cfg: ReVUAConfig, transform):
    name = cfg.data.val_dataset
    cls = DATASET_REGISTRY[name]
    root = os.path.join(cfg.data.root, name)
    # CIRCO's test split ships without gt_img_ids (labels held server-side by
    # the dataset authors) -- mAP against it is meaningless, so validation
    # always uses "val" regardless of which dataset is selected. This used
    # to be written as a tautological `"val" if name != "circo" else "val"`
    # (dead conditional, cleaned up 2026-09-08) that always evaluated to
    # "val" anyway; written directly now that the reasoning is in a comment
    # instead of implied by a no-op branch.
    return cls(root=root, split="val", transform=transform)


def _iter_gradcache_steps(loader, chunks_per_step: int, skip_steps: int = 0):
    """Group a chunk-sized DataLoader's batches into logical gradcache steps.

    `loader` yields batches of `gradcache_chunk_size` samples; this regroups
    every `chunks_per_step` of them into the list `gradcache_step` expects, so
    one logical step covers `data.batch_size` samples. A short final group
    (from a non-divisible dataset length) is dropped, matching `drop_last=True`
    on the loader itself.

    `skip_steps` resumes a mid-epoch checkpoint: chunk batches for the first
    `skip_steps` logical steps are consumed and discarded without running the
    model, so the numbering (and therefore the LR schedule position and log
    output) continues correctly from where a run left off. This does still pay
    the CPU decode cost for skipped chunks -- cheap relative to the GPU compute
    it saves (one skipped epoch's remainder vs re-decoding a partial epoch's
    images), and simpler/more robust than a custom resumable sampler.
    """
    group: list = []
    step = 0
    for chunk_batch in loader:
        if step < skip_steps:
            group.append(chunk_batch)
            if len(group) == chunks_per_step:
                step += 1
                group = []
            continue
        group.append(chunk_batch)
        if len(group) == chunks_per_step:
            yield step, group
            step += 1
            group = []


def _iter_plain_steps(loader, skip_steps: int = 0):
    """Same resume semantics as `_iter_gradcache_steps`, for the non-gradcache
    training path (one DataLoader batch == one logical step already)."""
    for step, batch in enumerate(loader):
        if step < skip_steps:
            continue
        yield step, batch


def build_optimizer(model: ReVUACIR, cfg: ReVUAConfig):
    # Decay/no-decay split (fixed 2026-09-08, code-audit finding): weight_decay
    # was previously passed once to torch.optim.AdamW(groups, weight_decay=...),
    # applying uniformly to EVERY trainable parameter across both LR groups --
    # including biases, LayerNorm weights, and every learned scalar control
    # (composer.residual_alpha, composer.logit_scale, rv.edit_weight,
    # rv.calibrator.scale/.bias, ug.log_temperature). Standard practice (and
    # important here specifically): AdamW's decay term continuously pulls
    # every decayed parameter toward 0, which actively fights a scalar like
    # residual_alpha that training is trying to grow away from its small
    # near-zero init to learn a useful residual correction. `p.dim() <= 1`
    # is the standard, generic test (catches every bias, every LayerNorm/
    # scale-shift 1D weight, and every 0-dim scalar Parameter in this model
    # -- confirmed by enumerating every trainable param's shape directly)
    # rather than hardcoding a list of names that would silently miss a
    # future scalar parameter.
    decay_params, no_decay_params = [], []
    backbone_decay, backbone_no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        is_no_decay = p.dim() <= 1
        if is_backbone:
            (backbone_no_decay if is_no_decay else backbone_decay).append(p)
        else:
            (no_decay_params if is_no_decay else decay_params).append(p)
    backbone_lr = cfg.optim.lr * cfg.optim.backbone_lr_mult
    groups = [
        {"params": backbone_decay, "lr": backbone_lr, "weight_decay": cfg.optim.weight_decay},
        {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
        {"params": decay_params, "lr": cfg.optim.lr, "weight_decay": cfg.optim.weight_decay},
        {"params": no_decay_params, "lr": cfg.optim.lr, "weight_decay": 0.0},
    ]
    # Empty groups (e.g. no backbone params at all if frozen) are harmless to
    # AdamW but filtered out for a cleaner param_groups if ever inspected/logged.
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, betas=cfg.optim.betas, eps=cfg.optim.eps)


def build_scheduler(optimizer, cfg: ReVUAConfig, num_training_steps: int):
    warmup_steps = int(num_training_steps * cfg.optim.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        if cfg.optim.scheduler == "linear":
            return max(0.0, 1.0 - progress)
        if cfg.optim.scheduler == "constant":
            return 1.0
        import math
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine (default)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _warn_if_workers_exceed_memory(cfg: ReVUAConfig, logger) -> None:
    """Fail loudly at startup rather than ~150 steps in.

    On Windows each DataLoader worker is a spawned process holding its own copy
    of the dataset, and batches cross process boundaries via shared memory that
    counts against the system *commit limit*. Overshooting `num_workers` does
    not raise an out-of-memory error -- it surfaces as
    `RuntimeError: Couldn't open shared file mapping ... error code: 1455`
    from deep inside torch.multiprocessing, which reads like a code bug and
    costs a run's worth of GPU time to diagnose.
    """
    nw = cfg.data.num_workers
    if nw <= 2:
        return
    try:
        import psutil
        total_gb = psutil.virtual_memory().total / 1e9
    except Exception:
        return
    budget = max(2, int(total_gb // 6))  # ~1 worker per 6 GB, floor of 2
    if nw > budget:
        logger.warning(
            "num_workers=%d is high for %.1f GB of RAM (suggested <= %d). On Windows this "
            "can exhaust the commit limit mid-epoch and hang the run with "
            "'Couldn't open shared file mapping ... error code: 1455'. Lower data.num_workers "
            "if the run dies partway through epoch 0.", nw, total_gb, budget)


def train(cfg: ReVUAConfig, resume_from: str = None):
    # Fail loudly at startup, not silently mid-training (2026-09-08,
    # code-audit finding): `cfg.data.batch_size // cfg.optim.gradcache_
    # chunk_size` (used both for steps_per_epoch and chunks_per_step below)
    # is integer division -- a non-divisor chunk size silently drops the
    # remainder EVERY logical step (e.g. batch_size=192, chunk_size=7 ->
    # 27*7=189 samples actually used per step, 3 silently dropped every
    # time, with no error and no visible symptom besides a slightly-smaller
    # effective batch than the config claims). gradcache.py's own header
    # already documents this constraint; this makes it impossible to violate
    # by accident instead of relying on every config author remembering it.
    if cfg.optim.use_gradcache and cfg.data.batch_size % cfg.optim.gradcache_chunk_size != 0:
        raise ValueError(
            f"data.batch_size ({cfg.data.batch_size}) must be evenly divisible by "
            f"optim.gradcache_chunk_size ({cfg.optim.gradcache_chunk_size}) -- got a remainder of "
            f"{cfg.data.batch_size % cfg.optim.gradcache_chunk_size}. A non-divisor silently drops "
            f"that many samples from every logical training step.")
    set_seed(cfg.run.seed, cfg.run.deterministic)
    device = torch.device(cfg.run.device if torch.cuda.is_available() or cfg.run.device == "cpu" else "cpu")
    os.makedirs(cfg.run.output_dir, exist_ok=True)
    logger = get_logger("revua_cir.train", os.path.join(cfg.run.output_dir, "train.log"))
    logger.info("Config:\n%s", cfg.to_dict())

    train_transform = build_image_transform(cfg.data.image_size, train=True)
    eval_transform = build_image_transform(cfg.data.image_size, train=False)
    tokenizer = build_tokenizer(cfg.backbone)
    collate_fn = build_collate_fn(tokenizer)

    train_dataset = build_train_dataset(cfg, train_transform)
    val_dataset = build_val_dataset(cfg, eval_transform)
    logger.info("Train triplets: %d | Val triplets: %d", len(train_dataset), len(val_dataset))

    _warn_if_workers_exceed_memory(cfg, logger)

    if cfg.optim.use_gradcache:
        # The DataLoader's own batch_size is the gradcache CHUNK size, not the
        # logical training batch (data.batch_size, e.g. 1024). Collating the
        # full logical batch in one call -- even just to hand it to gradcache
        # for internal chunking -- required a single ~1.2 GB tensor per image
        # field, and pinning/prefetching that exhausted the Windows commit
        # charge ("Couldn't open shared file mapping ... error code: 1455")
        # regardless of num_workers. Loading at chunk granularity keeps every
        # buffer small; `_iter_gradcache_steps` below regroups chunks into
        # logical steps.
        loader_batch_size = cfg.optim.gradcache_chunk_size
        pin = False
    else:
        loader_batch_size = cfg.data.batch_size
        pin = True
    # A per-epoch seeded generator (not the ambient global RNG) makes each
    # epoch's shuffle order a pure function of (seed, epoch number). That is
    # what makes mid-epoch resume well-defined: rebuilding the loader for the
    # same epoch reproduces the same shuffle, so skipping the first
    # `step_in_epoch` chunks after a resume replays exactly the data already
    # trained on rather than a different random subset -- no duplicate-heavy or
    # skipped-heavy epoch from an interrupted run.
    def make_train_loader(epoch: int) -> DataLoader:
        g = torch.Generator().manual_seed(cfg.run.seed + epoch)
        return DataLoader(train_dataset, batch_size=loader_batch_size, shuffle=True, drop_last=True,
                          num_workers=cfg.data.num_workers, collate_fn=collate_fn, pin_memory=pin,
                          persistent_workers=False, generator=g)

    train_loader = make_train_loader(0)

    model = ReVUACIR(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = (len(train_loader) // (cfg.data.batch_size // cfg.optim.gradcache_chunk_size)
                       if cfg.optim.use_gradcache else len(train_loader))
    num_steps = steps_per_epoch * cfg.optim.max_epochs
    scheduler = build_scheduler(optimizer, cfg, num_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.optim.amp and device.type == "cuda")
    ema = EMA(model, cfg.optim.ema_decay) if cfg.optim.ema_decay > 0 else None

    def _early_stopper_state(es):
        if es is None:
            return {}
        return {"early_stopper": {"best": es.best, "best_epoch": es.best_epoch, "num_bad_epochs": es.num_bad_epochs}}

    early_stopper = EarlyStopper(cfg.early_stopping.mode, cfg.early_stopping.patience,
                                  cfg.early_stopping.min_delta, cfg.early_stopping.warmup_epochs) \
        if cfg.early_stopping.enabled else None
    ckpt_keeper = TopKCheckpointKeeper(cfg.run.save_top_k, cfg.early_stopping.mode)
    # Reconstruct top-K tracking from whatever epochNNN.pt files already
    # exist in output_dir (fixed 2026-09-09, code-audit finding #8): a fresh
    # keeper instance has no memory of a prior interrupted run's checkpoints,
    # so without this, resuming could accumulate more than save_top_k files
    # on disk (the keeper's own pruning only fires once len(_entries) >
    # self.k, but _entries starts empty every process start). A from-scratch
    # run has no epochNNN.pt files yet, so this is a no-op there.
    ckpt_keeper.rebuild_from_disk(cfg.run.output_dir)

    start_epoch = 0
    resume_step = 0
    if resume_from and os.path.exists(resume_from):
        from revua_cir.utils import load_checkpoint, restore_rng_state
        state = load_checkpoint(resume_from, model, optimizer, scheduler, map_location=device,
                                ema=ema, scaler=scaler)
        ckpt_epoch = state.get("epoch", 0)
        ckpt_step = state.get("step_in_epoch")
        has_ema_state = state.get("ema_shadow") is not None
        # Restore Python/NumPy/PyTorch-CPU/PyTorch-CUDA RNG state (fixed
        # 2026-09-09, code-audit finding #9): per-epoch DataLoader seeding
        # (make_train_loader below) already makes sample ORDER exactly
        # reproducible across resume on its own; this covers everything
        # else that draws from the same global RNG streams mid-training
        # (dropout masks, any other stochastic op) so a resumed run's
        # stochastic behavior from the very next step onward matches what
        # an uninterrupted run would have drawn, not just a differently-
        # seeded-but-still-valid alternative. Old checkpoints (no
        # "rng_state" key) silently no-op here -- optional reproducibility
        # polish, not required for ordinary training correctness.
        restore_rng_state(state.get("rng_state"))
        logger.info("Restored RNG state: %s", "yes" if state.get("rng_state") is not None else "no (old checkpoint)")
        # Restore EarlyStopper's tracked state (fixed 2026-09-09, code-audit
        # finding) so a resumed run's patience/best-so-far picks up exactly
        # where the interrupted run left off, instead of resetting `best` to
        # +-inf and `num_bad_epochs` to 0 -- which would silently let a
        # resumed run train `patience` epochs longer than an uninterrupted
        # one would have, past the point it should have stopped.
        es_state = (state.get("extra") or {}).get("early_stopper")
        if early_stopper is not None and es_state is not None:
            early_stopper.best = es_state["best"]
            early_stopper.best_epoch = es_state["best_epoch"]
            early_stopper.num_bad_epochs = es_state["num_bad_epochs"]
            logger.info("Restored early-stopper state: %s", es_state)
        if ckpt_step is not None:
            # Mid-epoch checkpoint: continue the SAME epoch from the next step,
            # not the next epoch. `global_step`/the scheduler must not be
            # rewound -- optimizer.step() already ran for steps < ckpt_step.
            start_epoch = ckpt_epoch
            resume_step = ckpt_step + 1
            logger.info("Resumed from %s mid-epoch: epoch %d, step %d (ema_shadow restored: %s)",
                        resume_from, start_epoch, resume_step,
                        has_ema_state if ema is not None else "n/a")
        else:
            start_epoch = ckpt_epoch + 1
            logger.info("Resumed from %s at epoch %d (ema_shadow restored: %s)",
                        resume_from, start_epoch, has_ema_state if ema is not None else "n/a")

    global_step = start_epoch * steps_per_epoch + resume_step
    chunks_per_step = cfg.data.batch_size // cfg.optim.gradcache_chunk_size if cfg.optim.use_gradcache else 1
    for epoch in range(start_epoch, cfg.optim.max_epochs):
        model.train()
        meters = {}
        t_epoch = time.time()
        # Rebuild with this epoch's seed (not the loader made before the loop) so
        # the shuffle order matches what a from-scratch run of this epoch would
        # see -- required for `skip_steps` below to skip the RIGHT samples.
        epoch_loader = train_loader if epoch == 0 and resume_step == 0 else make_train_loader(epoch)
        skip_steps = resume_step if epoch == start_epoch else 0
        step_source = (_iter_gradcache_steps(epoch_loader, chunks_per_step, skip_steps=skip_steps)
                      if cfg.optim.use_gradcache else
                      _iter_plain_steps(epoch_loader, skip_steps=skip_steps))
        for step, batch in step_source:
            # Under gradcache `batch` is a list of chunk-sized batches, kept on
            # the CPU; gradcache_step moves each chunk to the GPU as it is used,
            # so no single 1024-image tensor is ever assembled.
            if not cfg.optim.use_gradcache:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            if cfg.optim.use_gradcache:
                # Large-batch contrastive step at small-batch memory cost; the
                # optimizer step and clipping happen inside.
                step_losses = gradcache_step(model, batch, optimizer, cfg, scaler)
                step_skipped = step_losses.pop("_optimizer_step_skipped", False)
            else:
                with torch.autocast("cuda", enabled=cfg.optim.amp and device.type == "cuda"):
                    out = model(batch)
                    loss = out["losses"]["total"]

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip_norm)
                # Detect a skipped optimizer.step() (fixed 2026-09-10, real bug
                # found via external review of job 6300's real training log,
                # confirmed against the exact PyTorch warning it produced):
                # `scaler.step(optimizer)` silently skips the underlying
                # `optimizer.step()` on any inf/NaN unscaled gradient this
                # step -- calling `scheduler.step()` unconditionally right
                # after (as this code used to) advances the LR schedule one
                # step even though no parameter update happened, silently
                # desynchronizing "step count" from "actual optimizer steps
                # taken" for the rest of training. `scaler.get_scale()`
                # strictly decreases only on a skipped step (the adaptive
                # backoff after an overflow) -- comparing before/after
                # `update()` is GradScaler's own documented way to detect this.
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                step_skipped = scaler.get_scale() < scale_before
                # Detach before handing these to `meters`/logging. `out["losses"]`
                # values still carry the FULL forward+backward autograd graph
                # (every activation from this step) even though .backward() has
                # already run -- Python's float() reads the scalar but does not
                # release that graph, and `step_losses` outlives this iteration
                # (referenced until the next loop overwrites it), so the old
                # step's graph and the new step's graph were briefly resident at
                # once. On a memory-constrained GPU (a real Kaggle T4 session
                # with only 14.56 GiB, vs 15.6 GB reported by another session on
                # "the same" GPU model) this OOM'd on step 1, not step 0 --
                # exactly the signature of a graph retained one step too long.
                step_losses = {k: v.detach() for k, v in out["losses"].items()}
                del out

            # Only advance the LR schedule on a step that actually updated
            # parameters (fixed 2026-09-10, see step_skipped's own comment
            # above) -- an AMP-skipped step must not consume a schedule step.
            if not step_skipped:
                scheduler.step()
            else:
                logger.warning("epoch %d step %d: optimizer.step() skipped by GradScaler "
                                "(inf/NaN gradient detected) -- scheduler.step() also skipped "
                                "this iteration to keep the LR schedule in sync with real "
                                "optimizer steps.", epoch, step)
            if ema is not None:
                ema.update(model)

            for k, v in step_losses.items():
                meters.setdefault(k, AverageMeter()).update(float(v))
            global_step += 1

            if step % cfg.run.log_every == 0:
                lr = scheduler.get_last_lr()[-1]
                loss_str = " | ".join(f"{k}={m.avg:.4f}" for k, m in meters.items())
                logger.info("epoch %d step %d/%d lr=%.2e | %s", epoch, step, steps_per_epoch, lr, loss_str)

            # Mid-epoch checkpoint (Kaggle: 12h session cap, or a dropped
            # connection, can end a run at any point -- not just epoch
            # boundaries). Only the single latest is kept (fixed filename, no
            # ckpt_keeper/ranking involvement) so this does not multiply disk
            # usage; the validated top-k checkpoints below are unaffected.
            n = cfg.run.checkpoint_every_n_steps
            if n > 0 and (step + 1) % n == 0:
                # best_metric is informational only for a mid-epoch save (no
                # validation has run yet this epoch); fall back to 0.0 rather
                # than early_stopper's +-inf sentinel before any eval has
                # happened.
                best_so_far = early_stopper.best if early_stopper and abs(early_stopper.best) != float("inf") else 0.0
                latest_path = os.path.join(cfg.run.output_dir, "latest_mid_epoch.pt")
                save_checkpoint(latest_path, model, optimizer, scheduler, epoch,
                                best_metric=best_so_far,
                                config=cfg.to_dict(), ema=ema, live_model=model,
                                scaler=scaler, step_in_epoch=step,
                                extra=_early_stopper_state(early_stopper))
                logger.info("mid-epoch checkpoint: epoch %d step %d -> %s", epoch, step, latest_path)

        logger.info("epoch %d finished in %.1fs", epoch, time.time() - t_epoch)

        # -------- validation + early stopping -------- #
        if (epoch + 1) % cfg.run.eval_every_epochs == 0:
            eval_model = ema.shadow if ema is not None else model
            use_map = cfg.data.val_dataset == "circo"
            use_group_recall = cfg.data.val_dataset == "cirr"
            metrics = evaluate_dataset(eval_model, val_dataset, cfg, tokenizer, eval_transform, device,
                                        use_map=use_map, use_group_recall=use_group_recall,
                                        max_queries=min(len(val_dataset), 2000))
            logger.info("epoch %d val metrics: %s", epoch, metrics)

            monitor_key = cfg.early_stopping.monitor.replace("val_", "")
            # Was `metrics.get(monitor_key, metrics.get(f"{monitor_key}", 0.0))`
            # -- the inner .get() used the exact same key as the outer one
            # (an f-string of a plain string variable is that variable), so
            # it could never provide a different fallback; simplified
            # 2026-09-08, no behavior change.
            monitor_value = metrics.get(monitor_key, 0.0)

            # early_stopper.step() runs BEFORE save_checkpoint (fixed
            # 2026-09-09, code-audit finding #7): saving first and stepping
            # second meant the checkpoint's saved `extra["early_stopper"]`
            # reflected the PREVIOUS epoch's state, one validation event
            # behind the checkpoint's own monitor_value/epoch. Concretely: if
            # the live stopper was (best=36.7, num_bad_epochs=2) entering this
            # epoch and this epoch is another bad one, the old ordering saved
            # num_bad_epochs=2 into a checkpoint whose live stopper had
            # already advanced to num_bad_epochs=3 -- so a resume from that
            # checkpoint would silently get one extra epoch of patience it
            # shouldn't have. Stepping first means the saved state and the
            # checkpoint's own monitor_value always describe the same epoch.
            result = early_stopper.step(monitor_value, epoch) if early_stopper is not None else None
            if result is not None:
                logger.info("early-stopping: %s", result)

            ckpt_path = os.path.join(cfg.run.output_dir, f"epoch{epoch:03d}.pt")
            save_checkpoint(ckpt_path, eval_model, optimizer, scheduler, epoch, monitor_value, cfg.to_dict(),
                            ema=ema, live_model=model, scaler=scaler,
                            extra=_early_stopper_state(early_stopper))
            ckpt_keeper.offer(monitor_value, ckpt_path)
            # A validated end-of-epoch checkpoint supersedes any mid-epoch save
            # from the epoch that just finished -- resuming from the mid-epoch
            # file would otherwise incorrectly replay steps already covered by
            # this epoch's completed pass.
            stale_mid_epoch = os.path.join(cfg.run.output_dir, "latest_mid_epoch.pt")
            if os.path.exists(stale_mid_epoch):
                os.remove(stale_mid_epoch)

            if result is not None and result["should_stop"]:
                logger.info("Early stopping triggered at epoch %d (best=%.4f @ epoch %d)",
                            epoch, result["best"], result["best_epoch"])
                break

    logger.info("Training complete.")
    return model


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Newest usable checkpoint in `output_dir`: the mid-epoch save if one
    exists (most recent progress), else the highest-numbered validated
    `epochNNN.pt`. Used by `--resume auto` -- the point of that flag is that a
    Kaggle restart cell should not need to know or type an exact filename;
    the same `python -m revua_cir.train ... --resume auto` command is correct
    whether the previous session ended mid-epoch, at an epoch boundary, or
    never started.
    """
    if not os.path.isdir(output_dir):
        return None
    mid = os.path.join(output_dir, "latest_mid_epoch.pt")
    if os.path.exists(mid):
        return mid
    epochs = sorted(
        (f for f in os.listdir(output_dir) if f.startswith("epoch") and f.endswith(".pt")),
        key=lambda f: int(f[len("epoch"):-len(".pt")]),
    )
    return os.path.join(output_dir, epochs[-1]) if epochs else None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--module-ablation", type=str, default=None, help="e.g. 'fc' | 'fc+aep' | 'fc+aep+rv' | 'fc+aep+rv+ug'")
    p.add_argument("--resume", type=str, default=None,
                   help="checkpoint path, or 'auto' to resume from the newest checkpoint "
                        "(mid-epoch save if present, else the latest validated epoch) in "
                        "--output-dir. 'auto' is a no-op (fresh start) if output-dir is empty.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ReVUAConfig.from_yaml(args.config) if args.config else ReVUAConfig()
    if args.data_root:
        cfg.data.root = args.data_root
    if args.output_dir:
        cfg.run.output_dir = args.output_dir
    if args.module_ablation:
        cfg = cfg.apply_module_ablation(args.module_ablation)

    resume_from = args.resume
    if resume_from == "auto":
        resume_from = find_latest_checkpoint(cfg.run.output_dir)
        print(f"[--resume auto] {'resuming from ' + resume_from if resume_from else 'no checkpoint found, starting fresh'}")

    train(cfg, resume_from=resume_from)


if __name__ == "__main__":
    main()
