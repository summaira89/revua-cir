"""The registered EDIR analysis, Tracks A and C (v3.1 §15, §19). CPU. Run ONCE.

Written, tested and hashed before any EDIR score existed (§19.4). Reads the two
`bbnv_score.py` EDIR outputs (protocols `edir_annotated` and `edir_retrieval`)
and the frozen CIRR analysis; nothing is fitted or tuned on EDIR.

Shared (§15.1): queries weighted equally; uncertainty by resampling whole
source groups (the reference image; 4,784 groups), 10,000 draws, seed 54321,
the replicate procedure of bbnv_policy.clustered_bootstrap.

Track A (§2, §15.2): fixed sequence, two-sided alpha_A = 0.025.
  Q1a  F+I > F      on edir_annotated, macro AUROC (per query: pairwise
                    target-versus-distractor accuracy, ties one half)
  Q1b  F+I > F+F'   same
       statistic: mean over the three seed pairs of the paired difference,
       bootstrapped jointly (per query: the mean over pairs of the difference)
  Q2   frozen delta gate (seed 0) > frozen base_only (the strongest
       conservative control selected on CIRR), mean utility on edir_retrieval
  A step holds iff its estimate is positive AND its 97.5% CI excludes 0. A
  failed step makes every later step descriptive; a significant adverse
  result is reported as such and also stops the sequence.
  TOST for Q1a/Q1b at the frozen margin 0.01 AUROC (§3, §4.5), at alpha_A:
  two one-sided tests at 0.025 each, i.e. the 95% CI (1 - 2 alpha_A).
  Equivalence is established iff that whole CI lies strictly inside
  (-0.01, +0.01); otherwise "equivalence not established". The superiority/
  inferiority result of the step is kept beside it. The CI half-width is
  reported against 0.6 x margin as a precision diagnostic only, never as the
  decision (§19.4, revised before any EDIR outcome).
Track C (§15.3): frozen forward_only (seed 0) vs frozen base_only, mean
  utility on edir_retrieval, two-sided alpha_C = 0.025; holds iff
  delta_u > 0 and the 97.5% CI lies above 0; a significant negative delta_u
  is harm. delta = 0.01 is the smallest effect of interest (reported beside).
Descriptive only: arm AUROCs per seed, R@1 (base, ungated fusion, gates),
  rescues, breaks, acceptances, coverage; the other frozen seed-0 gates and the
  seed-1/2 forward_only gates against base_only (95% CIs, no multiplicity).

  python scripts/edir_analysis.py --annotated ../edir_work/scored_annotated --retrieval ../edir_work/scored_retrieval \\
      --features ../edir_work/feat_edir --analysis ../bbnv_runs/analysis_cirr_v1 --out-dir ../edir_work/analysis
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bbnv_analyze as ana  # noqa: E402
import bbnv_extract as ext  # noqa: E402
import bbnv_provenance as prov  # noqa: E402
import bbnv_train as trn  # noqa: E402
from bbnv_transfer import FROZEN_ANALYSIS_SHA256, frozen_gate  # noqa: E402

SCHEMA = "edir-analysis/1"
N_BOOT, BOOT_SEED = 10000, 54321
ALPHA = 0.025                       # alpha_A = alpha_C (§15.4)
EQ_MARGIN, PRECISION_RULE = 0.01, 0.6
SESOI_U = 0.01                      # Track C smallest effect of interest (§15.3)
EXPECTED_Q, EXPECTED_SOURCES = 5000, 4784
DESCRIPTIVE_GATES = ("delta", "stacker", "inverse_only", "unsigned", "same_direction", "forward_only")


class EdirAnalysisError(ValueError):
    pass


# ------------------------------------------------------------ statistics --
def cluster_replicates(d: Sequence[float], cluster: Sequence[int], n_rep: int = N_BOOT,
                       seed: int = BOOT_SEED) -> np.ndarray:
    """Sorted replicates of the mean of d, resampling whole clusters; exactly
    the procedure (and random stream) of bbnv_policy.clustered_bootstrap."""
    d = np.asarray(d, dtype=np.float64)
    keys, inv = np.unique(np.asarray(cluster), return_inverse=True)
    sums = np.bincount(inv, weights=d)
    cnt = np.bincount(inv).astype(np.float64)
    pick = np.random.default_rng(seed).integers(0, len(keys), size=(n_rep, len(keys)))
    return np.sort(sums[pick].sum(1) / cnt[pick].sum(1))


def interval(reps: np.ndarray, level: float) -> List[float]:
    """Percentile interval at `level`, indexed as clustered_bootstrap indexes its 95% one."""
    n, a = len(reps), (1.0 - level) / 2.0
    return [float(reps[int(a * n)]), float(reps[min(n - 1, int((1.0 - a) * n))])]


def significance(d: Sequence[float], cluster: Sequence[int], alpha: float = ALPHA) -> Dict[str, object]:
    reps = cluster_replicates(d, cluster)
    est = float(np.mean(d))
    lo, hi = interval(reps, 1.0 - alpha)
    tail = np.sum(reps <= 0) if est >= 0 else np.sum(reps >= 0)
    return {"estimate": est, "ci": [lo, hi], "ci_level": 1.0 - alpha,
            "p_two_sided": float(min(1.0, 2.0 * (tail + 1) / (len(reps) + 1))),
            "beneficial": bool(est > 0 and lo > 0), "adverse": bool(est < 0 and hi < 0),
            "ci_tost": interval(reps, 1.0 - 2.0 * alpha), "ci_tost_level": 1.0 - 2.0 * alpha,
            "n_clusters": int(len(np.unique(cluster)))}


def tost(res: Mapping[str, object], margin: float = EQ_MARGIN) -> Dict[str, object]:
    """TOST at the step's alpha: equivalence iff the (1 - 2 alpha) CI lies
    strictly inside (-margin, +margin). The half-width is a diagnostic only."""
    lo, hi = res["ci_tost"]
    half = (hi - lo) / 2.0
    eq = bool(-margin < lo and hi < margin)
    other = ("significant positive difference" if res["beneficial"] else
             "significant negative difference" if res["adverse"] else "no significant difference")
    return {"equivalent": eq, "status": "equivalence established" if eq else "equivalence not established",
            "ci": [lo, hi], "ci_level": res["ci_tost_level"], "margin": margin,
            "superiority_test": other,
            "precision_diagnostic": {"half_width": half, "threshold": PRECISION_RULE * margin,
                                     "adequate": bool(half <= PRECISION_RULE * margin)}}


def sequence(steps: Sequence[Mapping[str, object]]) -> List[str]:
    """Fixed-sequence status of each step: confirmed, or why the sequence stopped."""
    out, open_ = [], True
    for s in steps:
        if not open_:
            out.append("descriptive (sequence stopped earlier)")
        elif s["beneficial"]:
            out.append("CONFIRMED (beneficial, alpha_A = %.3f)" % ALPHA)
        else:
            out.append("adverse and significant; sequence stops" if s["adverse"]
                       else "not significant; sequence stops")
            open_ = False
    return out


# ------------------------------------------------------------------ inputs --
def load_inputs(annotated: str, retrieval: str, features: str, analysis: str) -> Dict[str, object]:
    if prov.sha256_file(os.path.join(analysis, prov.RESULTS_NAME)) != FROZEN_ANALYSIS_SHA256:
        raise EdirAnalysisError("%s is not the frozen CIRR analysis" % analysis)
    frozen = prov.verify_run(analysis)
    fres = prov.verify_run(features)
    if fres.get("target") != "edir_test":
        raise EdirAnalysisError("%s is not the edir_test feature run" % features)
    feat_sha = prov.sha256_file(os.path.join(features, prov.RESULTS_NAME))
    runs = {}
    for name, path in (("edir_annotated", annotated), ("edir_retrieval", retrieval)):
        res, rows = ana.load_windows(path)                     # verified, full (not smoke), in order
        if res.get("protocol") != name or res.get("target") != "edir_test":
            raise EdirAnalysisError("%s is not the %s run" % (path, name))
        if res["inputs"]["val_features_results_sha256"] != feat_sha:
            raise EdirAnalysisError("%s was scored on other features" % path)
        if float(res["lambda_star_frozen"]) != float(frozen["lambda_star"]):
            raise EdirAnalysisError("%s used another lambda*" % path)
        if len(rows) != EXPECTED_Q:
            raise EdirAnalysisError("%s has %d queries, expected %d" % (path, len(rows), EXPECTED_Q))
        runs[name] = {"results": res, "rows": rows,
                      "results_sha256": prov.sha256_file(os.path.join(path, prov.RESULTS_NAME))}
    if runs["edir_annotated"]["results"]["inputs"]["checkpoint_sha256"] != \
            runs["edir_retrieval"]["results"]["inputs"]["checkpoint_sha256"]:
        raise EdirAnalysisError("the two protocols were scored with different checkpoints")
    with np.load(os.path.join(features, ext.FEATURES_NAME), allow_pickle=False) as z:
        cluster = z["ref_index"].astype(np.int64)
    if len(cluster) != EXPECTED_Q or len(np.unique(cluster)) != EXPECTED_SOURCES:
        raise EdirAnalysisError("source groups: %d queries in %d groups, expected %d in %d"
                                % (len(cluster), len(np.unique(cluster)), EXPECTED_Q, EXPECTED_SOURCES))
    return {"frozen": frozen, "runs": runs, "cluster": cluster, "features_results_sha256": feat_sha}


# ---------------------------------------------------------------- Track A --
def q1_contrast(auroc: Mapping, a: str, b: str) -> np.ndarray:
    """Per query: mean over the three seed pairs of AUROC(a) - AUROC(b)."""
    d = [np.asarray(auroc[(a, s)], dtype=np.float64) - np.asarray(auroc[(b, s)], dtype=np.float64)
         for s in trn.SEEDS]
    return np.mean(d, axis=0)


def track_a(rows_ann, rows_ret, lam: float, gates: Mapping, cluster: np.ndarray) -> Dict[str, object]:
    outc = ana.arm_outcomes(rows_ann, lam)
    if any(v is None for vals in outc["auroc"].values() for v in vals):
        raise EdirAnalysisError("an annotated window lacks a positive or a negative")
    steps = {}
    for key, (a, b) in (("Q1a", ("F+I", "F")), ("Q1b", ("F+I", "F+F'"))):
        r = significance(q1_contrast(outc["auroc"], a, b), cluster)
        r["contrast"] = "%s - %s" % (a, b)
        r["by_seed_pair"] = {str(s): float(np.mean(np.asarray(outc["auroc"][(a, s)]) - np.asarray(outc["auroc"][(b, s)])))
                             for s in trn.SEEDS}
        r["tost"] = tost(r)
        steps[key] = r
    views = ana.gate_views(rows_ret, lam, 0)
    u = {g: ana.evaluate_gate(views, gates[g](views)) for g in ("delta", "base_only")}
    steps["Q2"] = significance(np.asarray(u["delta"]["per_query_u"]) - np.asarray(u["base_only"]["per_query_u"]), cluster)
    steps["Q2"]["contrast"] = "delta gate - base_only (frozen, seed 0)"
    steps["Q2"]["gates"] = {g: {k: v for k, v in u[g].items() if k != "per_query_u"} for g in u}
    for key, status in zip(("Q1a", "Q1b", "Q2"), sequence([steps["Q1a"], steps["Q1b"], steps["Q2"]])):
        steps[key]["status"] = status
    arms = {"%s|%d" % (arm, s): ana.core.macro_auroc(outc["auroc"][(arm, s)])[0]
            for arm in ana.ARMS for s in trn.SEEDS}
    return {"steps": steps, "macro_auroc_by_arm_seed": arms}


# ---------------------------------------------------------------- Track C --
def track_c(rows_ret, lam: float, gates_by_seed: Mapping, cluster: np.ndarray) -> Dict[str, object]:
    views = ana.gate_views(rows_ret, lam, 0)
    g0 = gates_by_seed[0]
    ev = {g: ana.evaluate_gate(views, g0[g](views)) for g in ("forward_only", "base_only")}
    primary = significance(np.asarray(ev["forward_only"]["per_query_u"]) - np.asarray(ev["base_only"]["per_query_u"]),
                          cluster)
    primary["status"] = ("CONFIRMED (beneficial, alpha_C = %.3f)" % ALPHA if primary["beneficial"]
                         else "HARM (significant negative)" if primary["adverse"] else "not significant")
    primary["smallest_effect_of_interest"] = SESOI_U
    hits_base = [v["hit_base"] for v in views]
    hits_fused = [v["hit_fused"] for v in views]
    changed = [v for v in views if v["changed"]]
    supporting = {
        "r1_base": 100.0 * float(np.mean(hits_base)), "r1_ungated_fusion": 100.0 * float(np.mean(hits_fused)),
        "coverage_target_in_window": float(np.mean([r["target_pos"] >= 0 for r in rows_ret])),
        "changed": len(changed),
        "rescues_available": sum(1 for v in changed if v["hit_fused"] > v["hit_base"]),
        "breaks_available": sum(1 for v in changed if v["hit_fused"] < v["hit_base"]),
        "u_ungated_fusion": float(np.mean([ana.query_utility(v, True) for v in views])),
        "gates": {g: {k: x for k, x in ev[g].items() if k != "per_query_u"} for g in ev}}
    for g in ev:
        supporting["gates"][g]["r1_ci95"] = [100.0 * x for x in interval(cluster_replicates(
            [v["hit_fused"] if (v["changed"] and a) else v["hit_base"] for v, a in zip(views, g0[g](views))],
            cluster), 0.95)]
    descriptive = {}
    for g in DESCRIPTIVE_GATES:
        if g == "forward_only":
            continue
        e = ana.evaluate_gate(views, g0[g](views))
        r = significance(np.asarray(e["per_query_u"]) - np.asarray(ev["base_only"]["per_query_u"]), cluster, 0.05)
        descriptive["seed0 %s - base_only" % g] = {"estimate": r["estimate"], "ci95": r["ci"]}
    for s in (1, 2):
        vs = ana.gate_views(rows_ret, lam, s)
        e_f = ana.evaluate_gate(vs, gates_by_seed[s]["forward_only"](vs))
        e_b = ana.evaluate_gate(vs, gates_by_seed[s]["base_only"](vs))
        r = significance(np.asarray(e_f["per_query_u"]) - np.asarray(e_b["per_query_u"]), cluster, 0.05)
        descriptive["seed%d forward_only - base_only" % s] = {"estimate": r["estimate"], "ci95": r["ci"]}
    descriptive["two forward heads averaged"] = (
        "not computed: its construction was not fixed in §15.3. §15.3 requires it before any directional "
        "attribution, so a directional attribution of Track C remains UNEVALUATED (not a null result).")
    return {"primary": primary, "supporting": supporting, "descriptive": descriptive}


def frozen_gates(frozen: Mapping) -> Dict[int, Dict[str, object]]:
    out = {}
    for s in trn.SEEDS:
        stored = frozen["gates"]["seed%d" % s]["gates"]
        out[s] = {g: frozen_gate(stored[g]["params"], ana.feature_names(g)) for g in stored if g in ana.GATES}
    if frozen["gates"]["seed0"]["strongest_control"] != "base_only":
        raise EdirAnalysisError("the frozen strongest control is not base_only")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--annotated", required=True, help="bbnv_score.py annotated output (edir_annotated)")
    ap.add_argument("--retrieval", required=True, help="bbnv_score.py windows output on EDIR (edir_retrieval)")
    ap.add_argument("--features", required=True, help="the edir_test feature run")
    ap.add_argument("--analysis", required=True, help="the frozen CIRR analysis (gates, lambda*)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    try:
        prov.prepare_fresh_dir(args.out_dir)
    except IOError as exc:
        ap.error(str(exc))
    t0 = time.time()
    inp = load_inputs(args.annotated, args.retrieval, args.features, args.analysis)
    lam = float(inp["frozen"]["lambda_star"])
    gates = frozen_gates(inp["frozen"])
    rows_ann, rows_ret = inp["runs"]["edir_annotated"]["rows"], inp["runs"]["edir_retrieval"]["rows"]
    results = {
        "schema": SCHEMA, "amendment": "v3.1 §15, §19", "alpha_each_track": ALPHA,
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "unit": "source group", "n_groups": int(len(np.unique(inp["cluster"])))},
        "track_A": track_a(rows_ann, rows_ret, lam, gates[0], inp["cluster"]),
        "track_C": track_c(rows_ret, lam, gates, inp["cluster"]),
        "inputs": {"annotated_results_sha256": inp["runs"]["edir_annotated"]["results_sha256"],
                   "retrieval_results_sha256": inp["runs"]["edir_retrieval"]["results_sha256"],
                   "features_results_sha256": inp["features_results_sha256"],
                   "frozen_analysis_sha256": FROZEN_ANALYSIS_SHA256},
        "manifest": {"schema": prov.MANIFEST_SCHEMA, "script": "edir_analysis.py", "args": vars(args),
                     "source_sha256": prov.source_hashes(prov.BBNV_SOURCES + ("bbnv_transfer.py", "edir_analysis.py")),
                     "runtime": prov.runtime_info(None)},
        "wall_seconds": round(time.time() - t0, 1)}
    prov.write_run(args.out_dir, results, {})
    prov.verify_run(args.out_dir)
    a, c = results["track_A"]["steps"], results["track_C"]["primary"]
    for k in ("Q1a", "Q1b", "Q2"):
        print("Track A %s %-28s %+.4f  97.5%% CI [%+.4f, %+.4f]  %s"
              % (k, a[k]["contrast"][:28], a[k]["estimate"], a[k]["ci"][0], a[k]["ci"][1], a[k]["status"]))
        if "tost" in a[k]:
            t = a[k]["tost"]
            print("        TOST +/-%.2f, %.0f%% CI [%+.4f, %+.4f]: %s"
                  % (t["margin"], 100 * t["ci_level"], t["ci"][0], t["ci"][1], t["status"]))
    print("Track C forward_only - base_only   %+.4f  97.5%% CI [%+.4f, %+.4f]  %s"
          % (c["estimate"], c["ci"][0], c["ci"][1], c["status"]))
    print("[edir-analysis] wrote and verified %s in %.0fs" % (args.out_dir, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
