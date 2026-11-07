"""Cross-fitted acceptance-policy comparison on CIRR (v3.1 §17). CPU, cached scores.

Every policy is fitted on one half of the frozen tune/report split and
decides the other half, in both directions, so each of the 4181 queries is
decided by a policy that never saw it. Identical-proposal policies gate the
deployed bounded fusion (lambda 1): R always reject, A always accept, BM
base-margin gate, LG original learned gate (gate_suite procedure), OF
original-features 1:2 gate and FW forward-only gate (bbnv_analyze §14
procedure). SW, a small fusion weight chosen on the fit half, is a different
proposal and reported apart. The strongest simple control (R, A, BM, SW) is
preselected per fold on fit-half utility; the primary exploratory contrast is
FW minus that control, with a bootstrap over reference-image clusters.
Matched-acceptance harm: BM's and LG's scores are re-thresholded on each
evaluation half's scores alone to accept as many changed queries as FW.

Usage
-----
  python scripts/bbnv_policy.py --windows ../bbnv_runs/windows_cirr_v1 \\
      --val-features ../bbnv_feats/v1/cirr_val --out-dir ../bbnv_runs/policy_cirr_v1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bbnv_analyze as ana  # noqa: E402
import bbnv_extract as ext  # noqa: E402
import bbnv_provenance as prov  # noqa: E402
import gate_suite as gs  # noqa: E402
from fusion_controls import LAMBDA_GRID  # noqa: E402
from fusion_suite import r_bounded  # noqa: E402

SIMPLE = ("R", "BM", "SW", "A")          # tie order: most conservative first
LEARNED = ("LG", "OF", "FW")
BOOT_SEED = 54321
FW_SEEDS = (0, 1, 2)


class PolicyError(ValueError):
    pass


# --------------------------------------------------------------- policies --
def outcome(view: Mapping[str, object], accept: bool) -> int:
    """Top-1 hit of the gated system for one query."""
    return int(view["hit_fused"]) if (view["changed"] and accept) else int(view["hit_base"])


def utility(hits: Sequence[int], base: Sequence[int]) -> float:
    return float(np.mean([(1.0 if h > b else -2.0 if h < b else 0.0) for h, b in zip(hits, base)]))


def fit_bm(fit: Sequence[Mapping[str, object]]) -> Tuple[Callable, float]:
    t, _ = gs.choose_threshold(list(fit), lambda v: v["feat"]["base_margin"], "le", 1.0)
    return (lambda vs: [bool(v["changed"] and v["feat"]["base_margin"] <= t) for v in vs]), t


def lg_scorer(fit: Sequence[Mapping[str, object]]) -> Callable:
    """gate_suite's learned model: ridge logistic on the fit half's changed,
    non-neutral queries (helped = 1, broken = 0)."""
    train = [v for v in fit if v["changed"] and v["hit_base"] != v["hit_fused"]]
    y = [int(v["hit_fused"]) for v in train]
    if len(set(y)) != 2:
        raise PolicyError("the learned gate needs both helped and broken queries")
    score, _ = gs.fit_logistic([[v["feat"][k] for k in gs.FEATURES] for v in train], y)
    return lambda v: score(v["feat"])


def fit_lg(fit: Sequence[Mapping[str, object]]) -> Tuple[Callable, float, Callable]:
    sc = lg_scorer(fit)
    t, _ = gs.choose_threshold(list(fit), sc, "ge", 1.0)
    return (lambda vs: [bool(v["changed"] and sc(v) >= t) for v in vs]), t, sc


def fit_bbnv(fit: Sequence[Mapping[str, object]], gate: str) -> Callable:
    """The §14 procedure: C by 5-fold CV utility on the fit half, then the
    1:2 weighted fit, accept iff p > 0.5."""
    names = ana.feature_names(gate)
    C, _ = ana.select_C(fit, names)
    accept, _ = ana._fit_gate(fit, names, C)
    return lambda vs: [bool(v["changed"] and a) for v, a in zip(vs, accept(vs))]


def sw_param(fit_rows: Sequence[Mapping[str, object]]) -> float:
    """Bounded lambda from LAMBDA_GRID maximising fit-half utility; ties -> smaller."""
    best = None
    base = [int(gs._top(r["base_score"]) == r["target_pos"]) for r in fit_rows]
    for lam in sorted(LAMBDA_GRID):
        hits = [int(gs._top(r_bounded(r["base_score"], r["verification_score"], lam)) == r["target_pos"])
                for r in fit_rows]
        u = utility(hits, base)
        if best is None or u > best[0]:
            best = (u, lam)
    return best[1]


def sw_hits(rows: Sequence[Mapping[str, object]], lam: float) -> List[int]:
    return [int(gs._top(r_bounded(r["base_score"], r["verification_score"], lam)) == r["target_pos"]) for r in rows]


def matched_acceptance(scores: Sequence[float], changed: Sequence[bool], k: int, higher_accepts: bool) -> List[bool]:
    """Accept exactly k changed queries: those with the most accepting scores
    (ties by index). Uses scores only, never labels."""
    idx = [i for i, c in enumerate(changed) if c]
    if not 0 <= k <= len(idx):
        raise PolicyError("cannot accept %d of %d changed queries" % (k, len(idx)))
    order = sorted(idx, key=lambda i: ((-scores[i]) if higher_accepts else scores[i], i))
    keep = set(order[:k])
    return [i in keep for i in range(len(scores))]


def clustered_bootstrap(diff: Sequence[float], cluster: Sequence[int], n_rep: int,
                        seed: int = BOOT_SEED) -> Dict[str, float]:
    """Mean of per-query differences, 95% percentile CI from resampling whole
    clusters (queries weighted equally), two-sided p from the sign of replicates."""
    d = np.asarray(diff, dtype=np.float64)
    cl = np.asarray(cluster)
    keys, inv = np.unique(cl, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    cnt = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(keys), size=(n_rep, len(keys)))
    reps = np.sort(sums[pick].sum(1) / cnt[pick].sum(1))
    obs = float(d.mean())
    tail = np.sum(reps <= 0) if obs >= 0 else np.sum(reps >= 0)
    return {"diff": obs, "ci_low": float(reps[int(0.025 * n_rep)]),
            "ci_high": float(reps[min(n_rep - 1, int(0.975 * n_rep))]),
            "p_two_sided": float(min(1.0, 2.0 * (tail + 1) / (n_rep + 1))), "n_clusters": int(len(keys))}


def threshold_record(t: float) -> object:
    """A gate_suite threshold for JSON: +/-inf (accept nothing / everything,
    depending on the direction) is recorded as a string, never as a number."""
    return t if np.isfinite(t) else ("+inf" if t > 0 else "-inf")


def summary(hits: Sequence[int], base: Sequence[int], accepted: Optional[Sequence[bool]] = None) -> Dict[str, object]:
    out = {"u": utility(hits, base), "r1": 100.0 * float(np.mean(hits)),
           "rescues": int(sum(1 for h, b in zip(hits, base) if h > b)),
           "breaks": int(sum(1 for h, b in zip(hits, base) if h < b))}
    if accepted is not None:
        out["accepted"] = int(sum(accepted))
    return out


# ------------------------------------------------------------------ folds --
def run_fold(fit_v, ev_v, fit_rows, ev_rows, fw_views_ev, fw_views_fit) -> Dict[str, object]:
    """Fit every policy on the fit half, decide the evaluation half."""
    base = [int(v["hit_base"]) for v in ev_v]
    fit_base = [int(v["hit_base"]) for v in fit_v]
    acc: Dict[str, List[bool]] = {"R": [False] * len(ev_v), "A": [bool(v["changed"]) for v in ev_v]}
    fit_acc: Dict[str, List[bool]] = {"R": [False] * len(fit_v), "A": [bool(v["changed"]) for v in fit_v]}
    bm, t_bm = fit_bm(fit_v)
    acc["BM"], fit_acc["BM"] = bm(ev_v), bm(fit_v)
    lg, t_lg, lg_sc = fit_lg(fit_v)
    acc["LG"] = lg(ev_v)
    acc["OF"] = fit_bbnv(fit_v, "base_only")(ev_v)
    acc["FW"] = fit_bbnv(fw_views_fit, "forward_only")(fw_views_ev)
    hits = {k: [outcome(v, a) for v, a in zip(ev_v, flags)] for k, flags in acc.items()}
    lam = sw_param(fit_rows)
    hits["SW"] = sw_hits(ev_rows, lam)
    fit_hits = {k: [outcome(v, a) for v, a in zip(fit_v, f)] for k, f in fit_acc.items()}
    fit_hits["SW"] = sw_hits(fit_rows, lam)
    fit_u = {k: utility(fit_hits[k], fit_base) for k in SIMPLE}
    best = max(fit_u.values())
    control = next(k for k in SIMPLE if fit_u[k] == best)
    # matched acceptance: BM and LG re-thresholded on THIS half's scores only
    changed = [bool(v["changed"]) for v in ev_v]
    k = int(sum(acc["FW"]))
    m_bm = matched_acceptance([v["feat"]["base_margin"] for v in ev_v], changed, k, higher_accepts=False)
    m_lg = matched_acceptance([lg_sc(v) for v in ev_v], changed, k, higher_accepts=True)
    matched = {"k": k, "BM": summary([outcome(v, a) for v, a in zip(ev_v, m_bm)], base, m_bm),
               "LG": summary([outcome(v, a) for v, a in zip(ev_v, m_lg)], base, m_lg),
               "FW": summary(hits["FW"], base, acc["FW"])}
    return {"hits": hits, "acc": acc, "base": base, "control": control, "fit_utility_simple": fit_u,
            "sw_lambda": lam, "bm_threshold": t_bm, "lg_threshold": t_lg, "matched": matched}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--windows", required=True)
    ap.add_argument("--val-features", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args(argv)
    if args.n_boot < 100:
        ap.error("--n-boot must be at least 100")
    try:
        prov.prepare_fresh_dir(args.out_dir)
    except IOError as exc:
        ap.error(str(exc))
    t0 = time.time()
    res, rows = ana.load_windows(args.windows)
    if res["target"] != "cirr_val":
        raise PolicyError("§17 is declared for CIRR validation")
    if res["inputs"]["val_features_results_sha256"] != prov.sha256_file(os.path.join(args.val_features, prov.RESULTS_NAME)):
        raise PolicyError("the window run was scored from other validation features")
    with np.load(os.path.join(args.val_features, ext.FEATURES_NAME), allow_pickle=False) as z:
        cluster = z["ref_row"].astype(np.int64)
    if len(cluster) != len(rows):
        raise PolicyError("features and windows disagree in query count")
    lam = float(res["lambda_star_frozen"])
    halves = {"tune": [i for i, r in enumerate(rows) if r["split"] == "tune"],
              "report": [i for i, r in enumerate(rows) if r["split"] == "report"]}
    views = {s: ana.gate_views(rows, lam, s) for s in FW_SEEDS}
    folds, pooled_hits = {}, {}
    for fit_name, ev_name in (("tune", "report"), ("report", "tune")):
        fi, ei = halves[fit_name], halves[ev_name]
        f = run_fold([views[0][i] for i in fi], [views[0][i] for i in ei], [rows[i] for i in fi],
                     [rows[i] for i in ei], [views[0][i] for i in ei], [views[0][i] for i in fi])
        for s in FW_SEEDS[1:]:
            f["hits"]["FW_seed%d" % s] = [outcome(v, a) for v, a in zip(
                [views[s][i] for i in ei], fit_bbnv([views[s][i] for i in fi], "forward_only")([views[s][i] for i in ei]))]
        folds["fit_%s" % fit_name] = f
        for k, h in f["hits"].items():
            pooled_hits.setdefault(k, {}).update(dict(zip(ei, h)))
        pooled_hits.setdefault("control", {}).update(dict(zip(ei, f["hits"][f["control"]])))
        pooled_hits.setdefault("base", {}).update(dict(zip(ei, f["base"])))
        print("[policy] fit on %s: control %s, SW lambda %g (%.0fs)" % (fit_name, f["control"], f["sw_lambda"],
                                                                     time.time() - t0), flush=True)
    order = list(range(len(rows)))
    H = {k: [v[i] for i in order] for k, v in pooled_hits.items()}
    base = H["base"]

    def per_query_u(hits):
        return [(1.0 if h > b else -2.0 if h < b else 0.0) for h, b in zip(hits, base)]
    pooled = {k: summary(H[k], base) for k in H if k != "base"}
    contrasts = {}
    for a, b in (("FW", "control"), ("LG", "control"), ("OF", "control"), ("FW", "LG"), ("FW", "OF"),
                 ("FW_seed1", "control"), ("FW_seed2", "control")):
        ua, ub = per_query_u(H[a]), per_query_u(H[b])
        contrasts["%s - %s" % (a, b)] = clustered_bootstrap([x - y for x, y in zip(ua, ub)], cluster.tolist(),
                                                            args.n_boot)
    per_fold = {}
    for name, f in folds.items():
        per_fold[name] = {"control": f["control"], "fit_utility_simple": f["fit_utility_simple"],
                          "sw_lambda": f["sw_lambda"], "bm_threshold": threshold_record(f["bm_threshold"]),
                          "lg_threshold": threshold_record(f["lg_threshold"]),
                          "policies": {k: summary(h, f["base"], f["acc"].get(k)) for k, h in f["hits"].items()},
                          "matched_acceptance": f["matched"]}
    results = {"schema": "bbnv-policy/1", "amendment": "v3.1 §17", "status": "exploratory, retrospective",
               "n_queries": len(rows), "primary": "FW - control", "pooled": pooled, "contrasts": contrasts,
               "per_fold": per_fold, "n_clusters": int(len(set(cluster.tolist()))),
               "inputs": {"windows_results_sha256": prov.sha256_file(os.path.join(args.windows, prov.RESULTS_NAME)),
                          "val_features_results_sha256": prov.sha256_file(
                              os.path.join(args.val_features, prov.RESULTS_NAME))},
               "manifest": {"schema": prov.MANIFEST_SCHEMA, "script": "bbnv_policy.py", "args": vars(args),
                            "source_sha256": prov.source_hashes(prov.BBNV_SOURCES + ("bbnv_policy.py",)),
                            "runtime": prov.runtime_info(None)},
               "wall_seconds": round(time.time() - t0, 1)}
    prov.write_run(args.out_dir, results, {})
    prov.verify_run(args.out_dir)
    print("[policy] wrote and verified %s in %.0fs" % (args.out_dir, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
