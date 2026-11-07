"""BBNV scoring-kernel cost estimate at real dimensions (v3.1 §7; source audit,
finding 8). CPU only; synthetic inputs of exactly the real shapes.

SCOPE: this times the per-query SCORING KERNEL of one forward/inverse head
pair with randomly initialised heads:
  input construction; forward head on every x in R_q; forward mixture over
  C_q (as one full table, and alternatively chunked over R_q); marginal,
  PMI_F and induced inverse; inverse head and inverse mixture over R_q;
  centred gate features and one integrated arm score.
It does NOT time a fitted gate or its decision, base-ranking fusion, real
feature/cache loading, or head training. The study's arms reuse heads: the
three cyclic seed pairs need 3 forward + 3 inverse heads, each scored once
per query, so the scoring cost is about 3 x the per-pair figure below.

Usage
-----
  python scripts/bbnv_profile.py --out-dir ../bbnv_runs/profile_<date> [--queries 20 --warmup 3 --torch-threads 1]

The output goes to results.json in a FRESH directory (refused if it holds files).
"""
from __future__ import annotations

import os
import sys

# see bbnv_oracle.py: keep numpy's MKL off OpenMP so torch's runtime is the only one
_NUMPY_PRELOADED = "numpy" in sys.modules
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import argparse  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
from typing import Dict, List, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bbnv_core as core  # noqa: E402
import bbnv_oracle as orc  # noqa: E402
import bbnv_provenance as prov  # noqa: E402

DIM, BANK, WINDOW, HIDDEN = 768, 1024, 10, 512
DATASET_QUERIES = {"cirr_val": 4181, "edir": 5000, "fashioniq_val_eval_halves": 3009}
STAGES = ("inputs", "forward_head", "forward_mixture", "forward_scores", "inverse_head",
          "inverse_mixture", "features_and_arm")


def _unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _one_query(f_head, i_head, bank_r, bank_c, rng, chunk: Optional[int]) -> Dict[str, float]:
    """Run and time one query's complete scoring kernel."""
    t = {}
    s = time.perf_counter()
    r, tvec = _unit(1, DIM, rng), _unit(1, DIM, rng)
    window = _unit(WINDOW, DIM, rng)
    r_q = np.concatenate([bank_r, r])
    c_q = np.concatenate([bank_c, window])
    c_ids = tuple(range(len(c_q)))
    win_idx = list(range(BANK, BANK + WINDOW))
    edit = rng.normal(size=WINDOW)
    t["inputs"] = time.perf_counter() - s
    s = time.perf_counter()
    hyp, lpi, kap = orc.head_outputs(f_head, r_q, np.repeat(tvec, len(r_q), axis=0))
    t["forward_head"] = time.perf_counter() - s
    log_rho = core.log_prior_uniform(len(r_q))
    s = time.perf_counter()
    if chunk is None:
        log_f = core.mixture_log_prob_batch(hyp, lpi, kap, c_q)
        t["forward_mixture"] = time.perf_counter() - s
        s = time.perf_counter()
        fq = core.forward_query_scores(log_f, log_rho, len(r_q) - 1, win_idx)
    else:
        def rows(a, b):
            return core.mixture_log_prob_batch(hyp[a:b], lpi[a:b], kap[a:b], c_q), c_ids
        log_marg = core.chunked_log_marginal(rows, log_rho, win_idx, chunk, c_ids)
        ref_row = core.mixture_log_prob_batch(hyp[-1:], lpi[-1:], kap[-1:], c_q)[0, win_idx]
        t["forward_mixture"] = time.perf_counter() - s
        s = time.perf_counter()
        pmi = ref_row - log_marg
        fq = {"log_marginal": log_marg, "pmi_f": pmi, "ie_induced": log_rho[-1] + pmi}
    t["forward_scores"] = time.perf_counter() - s
    s = time.perf_counter()
    hyp_i, lpi_i, kap_i = orc.head_outputs(i_head, window, np.repeat(tvec, WINDOW, axis=0))
    t["inverse_head"] = time.perf_counter() - s
    s = time.perf_counter()
    ie = core.inverse_query_scores(core.mixture_log_prob_batch(hyp_i, lpi_i, kap_i, r_q), len(r_q) - 1)
    t["inverse_mixture"] = time.perf_counter() - s
    s = time.perf_counter()
    core.gate_window_features(ie, fq["ie_induced"], fq["log_marginal"])
    core.integrate([fq["pmi_f"], ie], edit, gamma=1.0)
    t["features_and_arm"] = time.perf_counter() - s
    return t


def profile(n_queries: int, warmup: int, n_comp: int, seed: int, chunk: Optional[int]) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    f_head = orc.build_head(2 * DIM, DIM, HIDDEN, n_comp, seed)
    i_head = orc.build_head(2 * DIM, DIM, HIDDEN, n_comp, seed + 1)
    bank_r, bank_c = _unit(BANK, DIM, rng), _unit(BANK, DIM, rng)
    for _ in range(warmup):                                     # untimed warm-up
        _one_query(f_head, i_head, bank_r, bank_c, rng, chunk)
    rss_before = prov.process_rss_mb()
    tracemalloc.start()
    t_all = time.perf_counter()
    timings: List[Dict[str, float]] = [_one_query(f_head, i_head, bank_r, bank_c, rng, chunk)
                                       for _ in range(n_queries)]
    elapsed = time.perf_counter() - t_all
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    totals = np.array([sum(t.values()) for t in timings])
    per_stage = {st: {"median": float(np.median([t[st] for t in timings])),
                      "p90": float(np.percentile([t[st] for t in timings], 90))} for st in STAGES}
    median = float(np.median(totals))
    return {"n_components": n_comp, "chunk": chunk, "queries_timed": n_queries, "warmup_queries": warmup,
            "per_stage_seconds": per_stage,
            "per_query_seconds": {"median": median, "p90": float(np.percentile(totals, 90)),
                                  "mean": float(totals.mean())},
            "end_to_end_seconds": round(elapsed, 3),
            "peak_traced_python_allocations_mb": round(peak / 2 ** 20, 1),
            "process_rss_mb_before_after": [rss_before, prov.process_rss_mb()],
            "projected_minutes_per_head_pair_at_median": {
                k: round(median * q / 60, 1) for k, q in DATASET_QUERIES.items()},
            "params_per_head": orc.param_count(f_head)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True, help="fresh (new or empty) output directory")
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=128)
    args = ap.parse_args(argv)
    if args.queries < 1 or args.warmup < 0 or args.torch_threads < 1 or args.chunk < 1:
        ap.error("queries, threads and chunk must be positive; warmup non-negative")
    try:
        prov.prepare_fresh_dir(args.out_dir)
    except IOError as exc:
        ap.error(str(exc))
    import torch
    torch.set_num_threads(args.torch_threads)                 # set before any numerical work
    res = {"scope": "scoring-kernel estimate for one head pair; excludes gate fitting/decision, "
                    "base fusion, cache loading and training (see module docstring)",
           "config": {"dim": DIM, "bank": BANK, "window": WINDOW, "hidden": HIDDEN, "dtype": "float64"},
           "M1_full_table": profile(args.queries, args.warmup, 1, args.seed, None),
           "M1_chunked": profile(args.queries, args.warmup, 1, args.seed, args.chunk),
           "M8_full_table": profile(max(2, args.queries // 4), min(args.warmup, 1), 8, args.seed, None),
           "manifest": {"schema": prov.MANIFEST_SCHEMA, "script": "bbnv_profile.py", "args": vars(args),
                        "source_sha256": prov.source_hashes(),
                        "runtime": prov.runtime_info(_NUMPY_PRELOADED, args.torch_threads)}}
    prov.write_run(args.out_dir, res, {})
    prov.verify_run(args.out_dir)
    for key in ("M1_full_table", "M1_chunked", "M8_full_table"):
        r = res[key]
        print("[profile] %-14s median %.3fs/query (p90 %.3f) | traced peak %.0f MB | RSS %s MB | "
              "CIRR val %.1f min per head pair"
              % (key, r["per_query_seconds"]["median"], r["per_query_seconds"]["p90"],
                 r["peak_traced_python_allocations_mb"], r["process_rss_mb_before_after"],
                 r["projected_minutes_per_head_pair_at_median"]["cirr_val"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
