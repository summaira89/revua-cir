"""Numerical repeatability of the verifier vectors under a change of GPU batch
shape. CPU only; descriptive (nothing here is tuned or tested for significance).

Input: two probe dumps of the SAME first-n queries of one cache, made by
dump_verifier_vectors.py with the same checkpoint, config and module combo
and different --batch-size (so different numbers of candidates per GPU pass;
--batch-size also sets the query-encoder batch, so both change together.
The cache run encoded queries 128 at a time (no shuffle) and each query's k
candidates in their own pass; --batch-size 128 matches its query batching, not
its candidate pass, so the pilot (128 vs 32) measures the effect of a
realistic change of batch shape, not of the cache run's exact shape).
Reproducing each query's own cached cosines does not bound the error of the
bank penalty, which uses every candidate vector against OTHER queries'
targets, so this compares, between the two dumps:
  * every stored vector (max |difference|, max 1 - cosine);
  * the bank penalties of every term for K in {10, 5, 20, 50} and the full
    mean, with the bank = the probe's other queries (self excluded);
  * the CSLS-corrected calibrated scores (K = 10, coefficient 0.5);
  * verifier orderings, full bounded-fusion permutations, fused top-1 and
    R@1 hits, query by query (flips are counted, not assumed impossible);
  * within-window verifier AUROC of the corrected score;
and each dump's own residuals against the cache (float64).

Usage
-----
  python scripts/vector_repeatability.py CACHE PROBE_A.npz PROBE_B.npz --json OUT.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apply_frozen_policy import _order  # noqa: E402
from bank_normalized_verification import (K_PRIMARY, K_SENS, LAMBDA, SCHEMA, TERMS,  # noqa: E402
                                          ProtocolError, corrected_scores, hits_at,
                                          term_penalties, window_aucs)
from dump_verifier_vectors import reproduction_errors, sha256_file  # noqa: E402
from fusion_controls import load_records, relevant_set  # noqa: E402
from fusion_suite import r_bounded  # noqa: E402

VECTORS = ("e_inv", "t_hat", "q_comp", "rec", "cinv", "cedit", "c_pool")
SAME = ("schema", "dataset", "category", "split", "module_combo", "cache_sha256",
        "config_sha256", "checkpoint_sha256", "n_queries", "k", "dim",
        "mixture_w", "calibrator_scale", "calibrator_bias")


def load_probe(cache_path: str, npz_path: str, recs_all):
    """Manifest, vectors and the probe's cache records, cross-validated."""
    man_path = npz_path[:-4] + ".manifest.json"
    with open(man_path, encoding="utf-8") as fh:
        man = json.load(fh)
    if man.get("schema") != SCHEMA:
        raise ProtocolError("%s: schema %r" % (man_path, man.get("schema")))
    if man["npz_sha256"] != sha256_file(npz_path):
        raise ProtocolError("%s does not match its manifest" % npz_path)
    if man["cache_sha256"] != sha256_file(cache_path):
        raise ProtocolError("%s was not built from %s" % (npz_path, cache_path))
    n, k = man["n_queries"], man["k"]
    if not 0 < n <= len(recs_all):
        raise ProtocolError("%s: n_queries %r outside the cache" % (man_path, n))
    with np.load(npz_path) as z:
        vec = {name: z[name].astype(np.float32) for name in VECTORS}
        cand, qid = z["candidate_ids"], z["query_id"]
    if not np.array_equal(qid, np.arange(n)):
        raise ProtocolError("%s: query ids are not 0..n-1" % npz_path)
    for i in range(n):
        if list(cand[i]) != recs_all[i]["candidate_gallery_ids"] or len(cand[i]) != k:
            raise ProtocolError("%s: row %d does not match cache record %d" % (npz_path, i, i))
    return man, vec


def compare(cache_path: str, npz_a: str, npz_b: str) -> dict:
    recs_all, multi = load_records(cache_path)
    man_a, va = load_probe(cache_path, npz_a, recs_all)
    man_b, vb = load_probe(cache_path, npz_b, recs_all)
    differ = [key for key in SAME if man_a.get(key) != man_b.get(key)]
    if differ:
        raise ProtocolError("the probes are not the same run; they differ in %s" % differ)
    bs = (man_a["runtime"]["query_batch_size"], man_b["runtime"]["query_batch_size"])
    if bs[0] == bs[1]:
        raise ProtocolError("both probes use batch size %d; nothing is varied" % bs[0])
    n = man_a["n_queries"]
    if n - 1 < max(K_SENS):
        raise ProtocolError("%d probe queries leave a bank smaller than K=%d"
                            % (n, max(K_SENS)))
    recs = recs_all[:n]
    rels = [relevant_set(r, multi) for r in recs]
    idx = list(range(n))
    w, a, b = man_a["mixture_w"], man_a["calibrator_scale"], man_a["calibrator_bias"]

    out = {"n_queries": n, "k": man_a["k"], "query_batch_size": list(bs),
           "max_candidates_per_pass": [man_a["runtime"]["max_candidates_per_pass"],
                                       man_b["runtime"]["max_candidates_per_pass"]],
           "residual_vs_cache": [reproduction_errors(va, recs, w, a, b),
                                 reproduction_errors(vb, recs, w, a, b)],
           "vectors": {}, "penalty_max_abs": {}}
    for name in VECTORS:
        x, y = va[name].astype(np.float64), vb[name].astype(np.float64)
        out["vectors"][name] = {"max_abs": float(np.abs(x - y).max()),
                                "max_one_minus_cos": float((1.0 - (x * y).sum(-1)).max())}

    pens = []
    for vec in (va, vb):
        bank = {name: vec[name] for name in ("e_inv", "t_hat")}
        pens.append(term_penalties(vec, idx, bank, self_col=idx))
    for term in TERMS:
        out["penalty_max_abs"][term] = {
            str(key): float(np.abs(pens[0][term][key] - pens[1][term][key]).max())
            for key in pens[0][term]}

    man = {"mixture_w": w, "calibrator_scale": a, "calibrator_bias": b}
    sa, sb = (corrected_scores(recs, idx, p, man) for p in pens)
    out["corrected_score_max_abs"] = float(np.abs(np.array(sa) - np.array(sb)).max())

    fused = [[_order(r_bounded(recs[i]["base_score"], s[i], LAMBDA)) for i in idx]
             for s in (sa, sb)]
    out["flips"] = {
        "verifier_order": sum(_order(sa[i]) != _order(sb[i]) for i in idx),
        "fused_permutation": sum(fused[0][i] != fused[1][i] for i in idx),
        "fused_top1": sum(fused[0][i][0] != fused[1][i][0] for i in idx),
        "r1_hit": sum(x != y for x, y in zip(hits_at(recs, rels, idx, sa),
                                              hits_at(recs, rels, idx, sb)))}
    qa, auc_a = window_aucs(rels, idx, sa)
    qb, auc_b = window_aucs(rels, idx, sb)
    if qa != qb:
        raise ProtocolError("the probes cover different queries")
    out["auroc"] = {"n_covered": len(qa),
                    "mean": [float(np.mean(auc_a)), float(np.mean(auc_b))] if qa else None,
                    "max_abs_per_query": (float(np.abs(np.array(auc_a) - np.array(auc_b)).max())
                                          if qa else None)}
    out["provenance"] = {"script_sha256": sha256_file(os.path.abspath(__file__)),
                         "cache_sha256": man_a["cache_sha256"],
                         "npz_sha256": [man_a["npz_sha256"], man_b["npz_sha256"]],
                         "checkpoint_sha256": man_a["checkpoint_sha256"]}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache")
    ap.add_argument("probe_a")
    ap.add_argument("probe_b")
    ap.add_argument("--json", required=True)
    args = ap.parse_args(argv)
    for path in (args.probe_a, args.probe_b):
        if not path.endswith(".npz"):
            ap.error("%s must end in .npz" % path)
    try:
        res = compare(args.cache, args.probe_a, args.probe_b)
    except (OSError, ValueError, KeyError) as exc:
        ap.error("%s: %s" % (type(exc).__name__, exc))
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, allow_nan=False)
    print("[repeat] batch sizes %s, %d queries, k=%d" % (res["query_batch_size"],
                                                         res["n_queries"], res["k"]))
    print("[repeat] vectors max |diff|: %s"
          % {n: "%.2e" % v["max_abs"] for n, v in res["vectors"].items()})
    print("[repeat] K=%d penalty max |diff|: %s"
          % (K_PRIMARY, {t: "%.2e" % v[str(K_PRIMARY)] for t, v in res["penalty_max_abs"].items()}))
    print("[repeat] corrected score max |diff| %.2e | flips %s | AUROC %s"
          % (res["corrected_score_max_abs"], res["flips"], res["auroc"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
