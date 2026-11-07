"""
Dedicated publication-quality latency benchmark (code-audit document finding
#6): `evaluate.py`'s `evaluate_dataset()` is a diagnostics-heavy function --
alongside the real base-retrieval and RV/UG-rerank passes it also runs
several diagnostic-only candidate-encode passes (top-1 pass, the
discrimination pair, the within-top-k spread pass) whose whole purpose is
producing metrics like `verification_auroc`, not measuring latency. Even
with CUDA synchronization (fixed 2026-09-09, code-audit finding #11) those
extra passes can warm CUDA kernels / OS file cache / image-decode paths
before the "real" timed region runs, contaminating a number meant to be
cited as ReVUA-CIR's inference latency.

This script measures ONLY the actual deployment inference path --
`ReVUACIR.retrieve_and_rerank()`, the same function a real serving
integration would call, with every diagnostic computation stripped out --
with a proper warmup, CUDA synchronization, and repeated measurement so
mean/p50/p95 are meaningful.

**Gallery methodology note** (added 2026-09-12, external-review finding):
this script builds its gallery with `evaluate.build_default_gallery()` --
the union of just this split's own reference+target images -- NOT
`scripts/evaluate_cirr.py`'s `build_cirr_full_gallery()` (the true full
official CIRR split, every image in `split.rc2.<split>.json`). This is
deliberate, not a bug or missing/invalid images: CIRR's official val
gallery has ~2297 images total, but not every one appears as a reference
or target in some triplet (some are pure distractors), so
`build_default_gallery` (~2265 images on val, confirmed job 6388) is
strictly smaller by design -- a fast, standard reduced gallery for
iterative benchmarking, matching the same tradeoff `evaluate_dataset()`
makes for its own default (non-`--full-gallery`) training-time validation
path. If you see a DIFFERENT gallery size reported here than in an
accuracy run against the same split (e.g. evaluate_cirr.py's ~2297), this
is why -- both are correct for their own purpose, but do not silently mix
them when writing up latency vs. accuracy numbers together: state which
gallery construction backs each number. The size difference has no
material effect on the LATENCY numbers themselves: RV/UG reranking only
touches the top-k/top-n candidates regardless of total gallery size, and
base-retrieval latency (a single query-vs-gallery matmul) differs by
<1.4% between the two gallery sizes -- well within this benchmark's own
measured min/max spread.

    python scripts/benchmark_latency.py \
        --config configs/stage1_finetune_full_unfreeze_gradcache.yaml \
        --checkpoint runs/ablation_partial_unfreeze_v2/fc_aep_rv_ug/epoch016.pt \
        --module-combo fc+aep+rv+ug \
        --eval-dataset cirr --n-queries 50 --n-repeats 30 --warmup 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F

from revua_cir.config import ReVUAConfig
from revua_cir.data.collate import build_collate_fn
from revua_cir.data.datasets import DATASET_REGISTRY
from revua_cir.data.tokenizer import build_tokenizer
from revua_cir.data.transforms import build_image_transform, safe_load_transform
from revua_cir.evaluate import build_default_gallery, encode_gallery_paths
from revua_cir.models.revua_cir import ReVUACIR
from revua_cir.utils import load_checkpoint


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values), p))


@torch.no_grad()
def benchmark(model: ReVUACIR, queries: list[dict], gallery_paths: list[str], gallery_pooled: torch.Tensor,
              transform, device: torch.device, top_n: int, n_repeats: int, warmup: int):
    """Times `retrieve_and_rerank` twice per query: once with `gallery_image_fn=
    None` (base retrieval only -- the function's own early-return branch,
    IDENTICAL code path a deployment would hit if RV/UG were disabled) and
    once with a real `gallery_image_fn` (the full pipeline, matching a real
    serving call with reranking enabled). No diagnostic-only computation
    runs at all -- this is the entire timed region.
    """
    def gallery_image_fn(indices: torch.Tensor) -> torch.Tensor:
        ids = indices.cpu().tolist()
        return torch.stack([safe_load_transform(gallery_paths[i], transform) for i in ids]).to(device)

    base_times, total_times = [], []

    def run_once(q, with_rerank: bool):
        ref = q["reference_pixel_values"].unsqueeze(0).to(device)
        ids = q["edit_input_ids"].unsqueeze(0).to(device)
        attn = q["edit_attention_mask"].unsqueeze(0).to(device)
        _sync(device)
        t0 = time.perf_counter()
        model.retrieve_and_rerank(ref, ids, attn, gallery_pooled,
                                  gallery_image_fn=gallery_image_fn if with_rerank else None,
                                  top_n=top_n)
        _sync(device)
        return time.perf_counter() - t0

    # Warmup: real forward passes through the exact same path, discarded --
    # lets CUDA kernels/cuDNN autotune/lazy-init finish before any timed
    # measurement, so the first "real" measurement isn't penalized by
    # one-time setup cost a deployed server wouldn't pay on every request.
    for i in range(warmup):
        q = queries[i % len(queries)]
        run_once(q, with_rerank=model.rv is not None)

    for i in range(n_repeats):
        q = queries[i % len(queries)]
        base_times.append(run_once(q, with_rerank=False))
        if model.rv is not None:
            total_times.append(run_once(q, with_rerank=True))

    return base_times, total_times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--module-combo", type=str, required=True)
    p.add_argument("--eval-dataset", type=str, default="cirr", choices=["cirr", "fashioniq", "circo"])
    p.add_argument("--eval-split", type=str, default="val")
    p.add_argument("--n-queries", type=int, default=50,
                   help="distinct queries to draw from (repeats cycle through them, not the same "
                        "single query every time, so image-decode cost isn't artificially cached "
                        "away for every measurement)")
    p.add_argument("--n-repeats", type=int, default=30, help="timed measurements per stage")
    p.add_argument("--warmup", type=int, default=5, help="untimed warmup iterations before measuring")
    p.add_argument("--top-n", type=int, default=50, help="matches ReVUACIR.retrieve_and_rerank's top_n")
    args = p.parse_args()

    cfg = ReVUAConfig.from_yaml(args.config)
    if args.data_root:
        cfg.data.root = args.data_root
    cfg = cfg.apply_module_ablation(args.module_combo)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_transform = build_image_transform(cfg.data.image_size, train=False)
    tokenizer = build_tokenizer(cfg.backbone)
    collate_fn = build_collate_fn(tokenizer)

    root = os.path.join(cfg.data.root, args.eval_dataset)
    dataset = DATASET_REGISTRY[args.eval_dataset](root=root, split=args.eval_split, transform=eval_transform)
    gallery_paths = build_default_gallery(dataset)
    print(f"[benchmark_latency] {args.eval_dataset}/{args.eval_split}: {len(dataset)} queries, "
          f"{len(gallery_paths)}-image gallery")

    model = ReVUACIR(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    print(f"[benchmark_latency] loaded {args.checkpoint} (module_combo={args.module_combo}, "
          f"device={device})")

    print("[benchmark_latency] encoding gallery (untimed, one-off)...")
    gallery_pooled = encode_gallery_paths(model, gallery_paths, eval_transform, device, cfg.data.eval_batch_size).to(device)

    n_q = min(args.n_queries, len(dataset))
    queries = [dataset[i] for i in range(n_q)]
    # collate_fn expects a list-of-dicts batch; here we need per-sample
    # tensors for retrieve_and_rerank's own (1, ...) batching in benchmark()
    # above, so run each single-sample dict through the tokenizer path
    # collate_fn already knows, then unbatch back to per-sample tensors --
    # keeps tokenization identical to the real training/eval code path
    # rather than re-deriving it here.
    collated = [collate_fn([q]) for q in queries]
    per_query = [{"reference_pixel_values": c["reference_pixel_values"][0],
                  "edit_input_ids": c["edit_input_ids"][0],
                  "edit_attention_mask": c["edit_attention_mask"][0]} for c in collated]

    base_times, total_times = benchmark(model, per_query, gallery_paths, gallery_pooled,
                                        eval_transform, device, args.top_n, args.n_repeats, args.warmup)

    def report(name: str, times: list[float]):
        if not times:
            print(f"[benchmark_latency] {name}: N/A (RV disabled)")
            return
        ms = [t * 1000 for t in times]
        print(f"[benchmark_latency] {name}: mean={np.mean(ms):.2f}ms  p50={_percentile(ms, 50):.2f}ms  "
              f"p95={_percentile(ms, 95):.2f}ms  min={min(ms):.2f}ms  max={max(ms):.2f}ms  (n={len(ms)})")

    print(f"[benchmark_latency] --- {args.n_repeats} measurements, {args.warmup} warmup, "
          f"{'CUDA-synchronized' if device.type == 'cuda' else 'CPU'} ---")
    report("base retrieval latency", base_times)
    if total_times:
        report("total online latency (base + RV/UG rerank)", total_times)
        rv_ug_ms = [t - b for t, b in zip(total_times, base_times[:len(total_times)])]
        report("RV+UG reranking overhead (total - base, paired)", rv_ug_ms)
        print("[benchmark_latency] NOTE: report total_online_latency, not base_retrieval alone, "
              "as 'ReVUA-CIR inference latency'.")


if __name__ == "__main__":
    main()
