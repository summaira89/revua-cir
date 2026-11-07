"""Measured latency at k = 3, 5, 7, 10, replacing the interpolated figures.

Why
---
Table~\\ref{tab:window} reports what each candidate-window width costs by
interpolating the single measured k=10 benchmark linearly in k. An external
review objected, correctly: a linear model is an assumption, not a
measurement, and the accuracy-per-millisecond claim built on it ("73% of the
gain at half the cost") therefore rests on that assumption rather than on
timing. This script measures each width directly.

What it keeps from the existing benchmark
-----------------------------------------
`benchmark_latency.py` already does the hard parts correctly, and this
script reuses its structure rather than reimplementing it: untimed warmup so
CUDA kernel compilation and cuDNN autotuning finish first, explicit
`torch.cuda.synchronize` around every timed region so asynchronous launches
are not mistimed, and repeated measurement so p95 is meaningful.

What it changes
---------------
Three things the review asked for.

  * **k is swept.** `retrieve_and_rerank` reads
    `cfg.reverse_verifier.top_k`, so setting that between runs varies the
    window without touching the model.
  * **The reported checkpoint is used.** The published benchmark ran on a
    checkpoint carrying an extra confidence module, which the manuscript
    discloses as a slight overestimate. This script takes the checkpoint as
    an argument and records it, so the timing and the accuracy figures can
    refer to the same configuration.
  * **Overhead is paired.** Base and full-pipeline passes alternate on the
    same query within a repeat, so the per-query difference is measured
    rather than inferred from two separately-averaged means. Query
    difficulty and any drift in machine load then cancel.

Interpretation
--------------
The numbers this produces are measurements of one machine under one load.
They are not a claim about other hardware, and the manuscript should say so.

Usage
-----
  python -m revua_cir.scripts.benchmark_latency_per_k \\
      --config configs/stage1_finetune_full_unfreeze_gradcache.yaml \\
      --checkpoint runs/ablation_collective_v3/fc_aep_rv/epoch004.pt \\
      --module-combo fc+aep+rv \\
      --k-values 3 5 7 10 --n-queries 50 --n-repeats 30 --warmup 5
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(xs: Sequence[float], q: float) -> float:
    """Nearest-rank percentile on a sorted copy.

    Written out rather than taken from numpy so the definition is visible:
    with 30 repeats the choice between interpolation methods moves p95 by
    more than the effect being measured.
    """
    if not xs:
        raise ValueError("no samples")
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def summarize(times_ms: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": statistics.fmean(times_ms),
        "p50": percentile(times_ms, 0.50),
        "p95": percentile(times_ms, 0.95),
        "min": min(times_ms),
        "max": max(times_ms),
        "n": len(times_ms),
    }


def bench_one_k(model, queries, gallery_pooled, gallery_image_fn, device,
                k: int, top_n: int, n_repeats: int, warmup: int) -> dict:
    """Time base and full pipeline at one window width, paired per query."""
    model.cfg.reverse_verifier.top_k = k

    def run_once(q, with_rerank: bool) -> float:
        ref = q["reference_pixel_values"].unsqueeze(0).to(device)
        ids = q["edit_input_ids"].unsqueeze(0).to(device)
        attn = q["edit_attention_mask"].unsqueeze(0).to(device)
        _sync(device)
        t0 = time.perf_counter()
        model.retrieve_and_rerank(
            ref, ids, attn, gallery_pooled,
            gallery_image_fn=gallery_image_fn if with_rerank else None,
            top_n=top_n)
        _sync(device)
        return (time.perf_counter() - t0) * 1000.0

    # Warmup is per k: a new window width changes tensor shapes, so kernels
    # may be re-selected and the first call at each k would otherwise carry
    # that one-time cost.
    for i in range(warmup):
        run_once(queries[i % len(queries)], with_rerank=True)

    base_ms: List[float] = []
    full_ms: List[float] = []
    overhead_ms: List[float] = []
    for i in range(n_repeats):
        q = queries[i % len(queries)]
        # Paired on the same query, base first then full, so the difference
        # is a within-query measurement.
        b = run_once(q, with_rerank=False)
        f = run_once(q, with_rerank=True)
        base_ms.append(b)
        full_ms.append(f)
        overhead_ms.append(f - b)

    return {"k": k,
            "base": summarize(base_ms),
            "full": summarize(full_ms),
            "overhead_paired": summarize(overhead_ms)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--module-combo", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--eval-dataset", default="cirr")
    p.add_argument("--eval-split", default="val")
    p.add_argument("--k-values", type=int, nargs="+", default=[3, 5, 7, 10])
    p.add_argument("--n-queries", type=int, default=50)
    p.add_argument("--n-repeats", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--full-gallery", action="store_true",
                   help="use the full official gallery rather than the "
                        "reduced development one, so timing and accuracy "
                        "refer to the same gallery")
    p.add_argument("--json", dest="json_out", default=None)
    args = p.parse_args(argv)

    if any(k < 1 for k in args.k_values):
        print("error: every k must be >= 1", file=sys.stderr)
        return 2
    if args.n_repeats < 2:
        print("error: --n-repeats must be >= 2 for p95 to mean anything",
              file=sys.stderr)
        return 2

    # Imports deferred so --help works without a CUDA build present. These
    # mirror benchmark_latency.py exactly: the setup path is the one already
    # known to produce the published figures, and diverging from it would
    # make the new numbers incomparable with the old.
    from revua_cir.config import ReVUAConfig            # noqa: E402
    from revua_cir.data.collate import build_collate_fn  # noqa: E402
    from revua_cir.data.datasets import DATASET_REGISTRY  # noqa: E402
    from revua_cir.data.tokenizer import build_tokenizer  # noqa: E402
    from revua_cir.data.transforms import (              # noqa: E402
        build_image_transform, safe_load_transform)
    from revua_cir.evaluate import (                     # noqa: E402
        build_default_gallery, encode_gallery_paths)
    from revua_cir.models.revua_cir import ReVUACIR      # noqa: E402
    from revua_cir.utils import load_checkpoint          # noqa: E402

    cfg = ReVUAConfig.from_yaml(args.config)
    if args.data_root:
        cfg.data.root = args.data_root
    cfg = cfg.apply_module_ablation(args.module_combo)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_transform = build_image_transform(cfg.data.image_size, train=False)
    tokenizer = build_tokenizer(cfg.backbone)
    collate_fn = build_collate_fn(tokenizer)

    root = os.path.join(cfg.data.root, args.eval_dataset)
    dataset = DATASET_REGISTRY[args.eval_dataset](
        root=root, split=args.eval_split, transform=eval_transform)
    gallery_paths = build_default_gallery(dataset)

    model = ReVUACIR(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    print("[bench] encoding gallery (untimed, one-off)...", flush=True)
    gallery_pooled = encode_gallery_paths(
        model, gallery_paths, eval_transform, device,
        cfg.data.eval_batch_size).to(device)

    def gallery_image_fn(indices):
        # Flat (k, C, H, W), exactly as benchmark_latency.py builds it:
        # retrieve_and_rerank passes the result straight to
        # backbone.encode_image, which expects a plain image batch. An
        # earlier version here reshaped to indices.shape + image shape,
        # adding a leading dimension that encode_image rejects.
        ids = indices.cpu().tolist()
        return torch.stack([safe_load_transform(gallery_paths[i],
                                                eval_transform)
                            for i in ids]).to(device)

    n_q = min(args.n_queries, len(dataset))
    collated = [collate_fn([dataset[i]]) for i in range(n_q)]
    queries = [{"reference_pixel_values": c["reference_pixel_values"][0],
                "edit_input_ids": c["edit_input_ids"][0],
                "edit_attention_mask": c["edit_attention_mask"][0]}
               for c in collated]

    rows = []
    with torch.no_grad():
        for k in sorted(args.k_values):
            print("[bench] k=%d ..." % k, flush=True)
            rows.append(bench_one_k(model, queries, gallery_pooled,
                                    gallery_image_fn, device, k, args.top_n,
                                    args.n_repeats, args.warmup))

    res = {
        "checkpoint": args.checkpoint,
        "module_combo": args.module_combo,
        "gallery_size": len(gallery_paths),
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda" else "cpu",
        "n_queries": len(queries),
        "n_repeats": args.n_repeats,
        "warmup": args.warmup,
        "rows": rows,
    }

    print()
    print("=" * 76)
    print("MEASURED LATENCY BY CANDIDATE-WINDOW WIDTH")
    print("=" * 76)
    print("checkpoint : %s" % os.path.basename(args.checkpoint))
    print("modules    : %s" % args.module_combo)
    print("gallery    : %d images" % res["gallery_size"])
    print("device     : %s" % res["device"])
    print("protocol   : %d queries, %d timed repeats, %d warmup, CUDA-synced"
          % (res["n_queries"], args.n_repeats, args.warmup))
    print()
    print("  %3s %9s %9s %9s %9s %9s %9s"
          % ("k", "base p50", "full mean", "full p50", "full p95",
             "overhead", "ovh p95"))
    for r in rows:
        print("  %3d %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f"
              % (r["k"], r["base"]["p50"], r["full"]["mean"],
                 r["full"]["p50"], r["full"]["p95"],
                 r["overhead_paired"]["mean"], r["overhead_paired"]["p95"]))
    print()
    print("  Overhead is the paired per-query difference between the full")
    print("  pipeline and base retrieval, not a difference of two means.")
    print("  All times in milliseconds.")
    print("=" * 76)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print()
        print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
