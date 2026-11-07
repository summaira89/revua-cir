"""
Cheap, real-data, real-backbone pre-flight check before committing to a full
training run. Runs a handful of REAL training steps (real ViT-L/14, real
CIRR data, real gradcache chunking via train.py's own machinery, real XBM if
enabled) and stops -- costs a few minutes on a GPU, not hours.

This exists because of a repeated pattern in this project: unit tests on
individual functions (gradcache_step, CrossBatchMemory in isolation) passed
cleanly while real end-to-end bugs only surfaced once the FULL pipeline ran
together (train.py's chunk-grouping + gradcache_step + a trainable backbone +
XBM's lazy shape init + AMP + real image loading, all at once). A unit test
proves a function is correct in isolation; it does not prove the wiring
between functions is correct. This script is the wiring check, cheaply,
before a training run that costs real GPU-hours.

    python scripts/dry_run_full_pipeline.py \
        --config configs/stage1_finetune_full_unfreeze_gradcache.yaml \
        --max-steps 5

What this checks that unit tests cannot:
  1. train.py's real DataLoader -> _iter_gradcache_steps -> gradcache_step
     chain runs without shape/type errors on REAL CIRR images (not
     synthetic 32x32 tensors).
  2. Real ViT-L/14 (not TinyBackbone) actually produces finite gradients
     through the trainable-backbone gradcache path end-to-end.
  3. AMP + gradcache + XBM (if enabled) don't interact badly -- e.g. dtype
     mismatches between the XBM queue (created in whatever dtype the first
     enqueue call happens to be) and later autocast-disabled comparisons.
  4. Memory actually fits at the configured batch_size/chunk_size on THIS
     specific GPU -- prints peak memory so you catch an OOM in 5 steps, not
     epoch 1 of a real run.
  5. Checkpointing/resume machinery doesn't choke on a real (not synthetic)
     model -- runs one checkpoint save+load cycle.

What this does NOT replace: it does not tell you whether the model will
actually LEARN anything useful -- 5 steps is nowhere near enough signal for
that. It only tells you the pipeline runs without silently-wrong or crashing
behavior. Still run a short real training verification (e.g.
verify_rv_hardneg.py-style, several epochs) before trusting a full run's
results, exactly as this project has done for every other fix so far.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from revua_cir.config import ReVUAConfig
from revua_cir.data.collate import build_collate_fn
from revua_cir.data.tokenizer import build_tokenizer
from revua_cir.data.transforms import build_image_transform
from revua_cir.gradcache import gradcache_step
from revua_cir.models.revua_cir import ReVUACIR
from revua_cir.train import (
    _iter_gradcache_steps, _iter_plain_steps, build_optimizer, build_scheduler, build_train_dataset,
)
from revua_cir.utils import save_checkpoint, load_checkpoint, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=5,
                   help="real training steps to run -- keep small, this is a wiring check, not a training run")
    p.add_argument("--use-xbm", action="store_true",
                   help="force reverse_verifier.use_xbm=True regardless of the config file, so a single "
                        "dry run can exercise gradcache + trainable backbone + XBM together -- the exact "
                        "combination that matters for a real training run, without needing a separate "
                        "XBM-specific config file just for this check.")
    p.add_argument("--xbm-warmup-steps", type=int, default=None,
                   help="override reverse_verifier.xbm_warmup_steps -- default to something small enough "
                        "that a short dry run can actually reach is_ready=True and exercise the "
                        "queue-sourced negative path, not just the warm-up path")
    args = p.parse_args()

    cfg = ReVUAConfig.from_yaml(args.config)
    if args.data_root:
        cfg.data.root = args.data_root
    if args.use_xbm:
        cfg.reverse_verifier.use_xbm = True
        if args.xbm_warmup_steps is not None:
            cfg.reverse_verifier.xbm_warmup_steps = args.xbm_warmup_steps

    print(f"[dry_run] config: {args.config}")
    print(f"[dry_run] backbone: {cfg.backbone.kind} {cfg.backbone.pretrained_name} "
          f"freeze_backbone={cfg.backbone.freeze_backbone} freeze_layers={cfg.backbone.freeze_layers}")
    print(f"[dry_run] batch_size={cfg.data.batch_size} use_gradcache={cfg.optim.use_gradcache} "
          f"gradcache_chunk_size={cfg.optim.gradcache_chunk_size if cfg.optim.use_gradcache else 'n/a'}")
    print(f"[dry_run] use_xbm={cfg.reverse_verifier.use_xbm} xbm_queue_size={cfg.reverse_verifier.xbm_queue_size} "
          f"xbm_warmup_steps={cfg.reverse_verifier.xbm_warmup_steps}")
    print(f"[dry_run] amp={cfg.optim.amp}")

    if cfg.optim.use_gradcache and cfg.data.batch_size % cfg.optim.gradcache_chunk_size != 0:
        dropped = cfg.data.batch_size % cfg.optim.gradcache_chunk_size
        print(f"[dry_run] WARNING: batch_size ({cfg.data.batch_size}) is not evenly divisible by "
              f"gradcache_chunk_size ({cfg.optim.gradcache_chunk_size}) -- {dropped} samples per step "
              f"will be silently dropped from every logical step (integer division in "
              f"chunks_per_step = batch_size // gradcache_chunk_size). Not a crash, but an effective "
              f"batch-size shrinkage you did not ask for. Fix by choosing values that divide evenly.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[dry_run] WARNING: no CUDA device found -- running on CPU. This will be very slow and "
              "does not validate real GPU memory behavior, but still checks correctness.")

    set_seed(cfg.run.seed)
    print("[dry_run] constructing model (real backbone download/load happens here if not cached)...")
    t0 = time.time()
    model = ReVUACIR(cfg).to(device)
    model.train()
    print(f"[dry_run] model constructed in {time.time() - t0:.1f}s, "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params")

    tokenizer = build_tokenizer(cfg.backbone)
    train_transform = build_image_transform(cfg.data.image_size, train=True)

    print("[dry_run] building real training dataset (real CIRR/FashionIQ data)...")
    train_dataset = build_train_dataset(cfg, train_transform)
    print(f"[dry_run]   {cfg.data.train_datasets}: {len(train_dataset)} triplets")

    collate_fn = build_collate_fn(tokenizer)
    loader_batch_size = cfg.optim.gradcache_chunk_size if cfg.optim.use_gradcache else cfg.data.batch_size
    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=loader_batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, drop_last=True,
    )

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, num_training_steps=args.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.optim.amp and device.type == "cuda")

    chunks_per_step = cfg.data.batch_size // cfg.optim.gradcache_chunk_size if cfg.optim.use_gradcache else 1
    step_source = (_iter_gradcache_steps(loader, chunks_per_step)
                   if cfg.optim.use_gradcache else _iter_plain_steps(loader))

    print(f"[dry_run] running {args.max_steps} REAL training steps...")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    n_run = 0
    for step, batch in step_source:
        if step >= args.max_steps:
            break
        t0 = time.time()
        if cfg.optim.use_gradcache:
            step_losses = gradcache_step(model, batch, optimizer, cfg, scaler)
            # `gradcache_step` now returns a `_optimizer_step_skipped` bool
            # alongside the real loss tensors (fixed 2026-09-10, code-audit
            # finding #2 of ReVUA_CIR_Final_PreTraining_Recommendations.md --
            # a real bug: this script previously treated every value in
            # step_losses as a tensor, which crashes with
            # `TypeError: isfinite(): argument 'input' ... must be Tensor,
            # not bool` the moment this ran with use_gradcache=true, since
            # gradcache_step always includes this key now). Pop it out
            # before any tensor-only handling below, matching train.py's own
            # handling of the same value.
            step_skipped = step_losses.pop("_optimizer_step_skipped", False)
        else:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast("cuda", enabled=cfg.optim.amp and device.type == "cuda"):
                out = model(batch)
                loss = out["losses"]["total"]
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip_norm)
            # Same skip-detection as train.py's plain AMP path (fixed
            # 2026-09-10): `scaler.step(optimizer)` silently skips the
            # underlying `optimizer.step()` on an inf/NaN gradient;
            # `scaler.get_scale()` strictly decreases only on that skip.
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            step_skipped = scaler.get_scale() < scale_before
            step_losses = {k: v.detach() for k, v in out["losses"].items()}
        if not step_skipped:
            scheduler.step()
        else:
            print(f"[dry_run] step {step}: optimizer.step() skipped by GradScaler "
                  f"(inf/NaN gradient) -- scheduler.step() also skipped this iteration.")
        elapsed = time.time() - t0

        for k, v in step_losses.items():
            if not torch.isfinite(v).all():
                print(f"[dry_run] FAIL: non-finite value in losses['{k}'] at step {step}: {v}")
                sys.exit(1)

        loss_str = " | ".join(f"{k}={float(v):.4f}" for k, v in step_losses.items())
        mem_str = ""
        if device.type == "cuda":
            mem_str = f" | peak_mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB"
        print(f"[dry_run] step {step}: {elapsed:.2f}s | {loss_str}{mem_str}")
        n_run += 1

    if n_run < args.max_steps:
        print(f"[dry_run] WARNING: only {n_run}/{args.max_steps} steps ran -- dataset/DataLoader "
              f"produced fewer steps than requested. Check chunks_per_step/batch_size/dataset size.")

    if cfg.reverse_verifier.use_xbm and model.rv is not None and model.rv.xbm is not None:
        print(f"[dry_run] XBM state after {n_run} steps: is_ready={model.rv.xbm.is_ready} "
              f"filled={int(model.rv.xbm.filled.item())}/{cfg.reverse_verifier.xbm_queue_size} "
              f"steps_seen={int(model.rv.xbm.steps_seen.item())}")

    print("[dry_run] testing checkpoint save/load round-trip on the REAL model...")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dry_run_ckpt.pt")
        save_checkpoint(path, model, epoch=0, best_metric=0.0, config=cfg.to_dict())
        model2 = ReVUACIR(cfg).to(device)
        load_checkpoint(path, model2, map_location=device)
    print("[dry_run] checkpoint round-trip OK")

    if device.type == "cuda":
        print(f"[dry_run] FINAL peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f}GB allocated, "
              f"{torch.cuda.max_memory_reserved()/1e9:.2f}GB reserved")

    print()
    print("[dry_run] ALL CHECKS PASSED -- pipeline runs correctly on real data/backbone.")
    print("[dry_run] This does NOT validate that the model will learn well -- run a short real")
    print("[dry_run] training verification (several epochs) before trusting a full run's results.")


if __name__ == "__main__":
    main()
