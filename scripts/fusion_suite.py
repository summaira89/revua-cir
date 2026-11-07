"""Compact fusion suite with risk-sensitive evaluation (plan step 2).

What it answers
---------------
Whether bounded fusion's advantage survives fair simple alternatives,
evaluated the way risk-sensitive retrieval evaluates a system against a
baseline. All from the cached scores; no GPU.

Rules (each tuned on the tune half only, ties toward the smaller weight,
every grid containing 0 so ignoring the verifier is always available):
  base             first-stage order
  rank_wsum        the paper's -(2*base_rank + verifier_rank), parameter-free
  rrf60            reciprocal rank fusion, k=60, unweighted (Cormack et al.)
  wrrf             1/(60+r_b) + w/(60+r_v), w on the lambda grid
  linear           z(B) + lam*z(R)
  bounded          z(B) + lam*tanh(z(R)), tuned
  bounded_frozen   the published rule at lam = 1.0
  convex           (1-a)*mm(B) + a*mm(R), min-max per window (Bruch et al.)

Score representation: every score-based rule is run on the calibrated
verification score AND on the raw cosine it is computed from. Rank-based
rules must agree exactly across the two, because the Platt calibrator is
monotone with positive slope; the script checks that rather than assuming it.

Evaluation on the report half, each rule against base:
  R@1, delta, helped/broken, exact McNemar, paired bootstrap CI, top-1
  change rate, and URisk / TRisk (Dincer, Macdonald and Ounis, SIGIR 2014)
  at the prespecified alpha in {1, 5}. For a sample mean the jackknife
  standard error coincides with s/sqrt(n), so the paper's jackknife check
  is automatically met here.
Plus each rule against the strongest tuned rule, where "strongest" is chosen
on the TUNE half so the report half never selects anything.

Usage
-----
  python scripts/fusion_suite.py rv_score_cache_fc_aep_rv.jsonl --json ../fusion_suite.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fusion_controls import (  # noqa: E402
    LAMBDA_GRID, CacheError, _ranks_desc, _zscore, hit_at_1, load_records,
    mcnemar_exact, paired_bootstrap, relevant_set, split_indices,
)

CONVEX_GRID: Tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(11))
RRF_K = 60
ALPHAS: Tuple[float, ...] = (1.0, 5.0)
FROZEN_LAMBDA = 1.0


# ---------------------------------------------------------------- rules ----
def _minmax(xs: Sequence[float]) -> List[float]:
    lo, hi = min(xs), max(xs)
    if hi - lo <= 0.0:
        return [0.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


def r_base(b, r, p):
    return list(b)


def r_rank_wsum(b, r, p):
    rb, rr = _ranks_desc(b), _ranks_desc(r)
    return [-(2.0 * rb[i] + rr[i]) for i in range(len(rb))]


def r_rrf60(b, r, p):
    rb, rr = _ranks_desc(b), _ranks_desc(r)
    return [1.0 / (RRF_K + rb[i] + 1) + 1.0 / (RRF_K + rr[i] + 1) for i in range(len(rb))]


def r_wrrf(b, r, w):
    rb, rr = _ranks_desc(b), _ranks_desc(r)
    return [1.0 / (RRF_K + rb[i] + 1) + w / (RRF_K + rr[i] + 1) for i in range(len(rb))]


def r_linear(b, r, lam):
    zb, zr = _zscore(b), _zscore(r)
    return [zb[i] + lam * zr[i] for i in range(len(zb))]


def r_bounded(b, r, lam):
    zb, zr = _zscore(b), _zscore(r)
    return [zb[i] + lam * math.tanh(zr[i]) for i in range(len(zb))]


def r_convex(b, r, a):
    mb, mr = _minmax(b), _minmax(r)
    return [(1.0 - a) * mb[i] + a * mr[i] for i in range(len(mb))]


# name: (function, parameter grid or None for fixed, fixed value, rank-based?)
RULES: Dict[str, Tuple[Callable, Optional[Sequence[float]], float, bool]] = {
    "base": (r_base, None, 0.0, True),
    "rank_wsum": (r_rank_wsum, None, 0.0, True),
    "rrf60": (r_rrf60, None, 0.0, True),
    "wrrf": (r_wrrf, LAMBDA_GRID, 0.0, True),
    "linear": (r_linear, LAMBDA_GRID, 0.0, False),
    "bounded": (r_bounded, LAMBDA_GRID, 0.0, False),
    "bounded_frozen": (r_bounded, None, FROZEN_LAMBDA, False),
    "convex": (r_convex, CONVEX_GRID, 0.0, False),
}
REPRESENTATIONS = ("calibrated", "raw")
FIELD = {"calibrated": "verification_score", "raw": "cos_sim"}


def hits_for(recs, rel, fn, param, field, subset) -> List[int]:
    return [hit_at_1(fn(recs[i]["base_score"], recs[i][field], param), rel[i])
            for i in subset]


def top1_changed(recs, fn, param, field, subset) -> float:
    n = 0
    for i in subset:
        s = fn(recs[i]["base_score"], recs[i][field], param)
        top = max(range(len(s)), key=lambda j: (s[j], -j))
        n += 1 if top != 0 else 0
    return n / len(subset) if subset else 0.0


def tune(recs, rel, fn, grid, field, tune_idx) -> Tuple[float, int]:
    best_p, best_h = grid[0], -1
    for p in grid:
        h = sum(hits_for(recs, rel, fn, p, field, tune_idx))
        if h > best_h:
            best_p, best_h = p, h
    return best_p, best_h


# ----------------------------------------------------------------- risk ----
def urisk_trisk(rule_hits: Sequence[int], base_hits: Sequence[int],
                alpha: float) -> Dict[str, float]:
    """URisk and TRisk of a system against a baseline, per Dincer et al. 2014.

    x_i = delta_i if the query gains, (1 + alpha) * delta_i if it loses, 0
    otherwise; URisk = mean(x); TRisk = URisk / (s_x / sqrt(n)).
    """
    xs = []
    for r, b in zip(rule_hits, base_hits):
        d = r - b
        xs.append(d if d > 0 else (1.0 + alpha) * d)
    n = len(xs)
    u = sum(xs) / n
    sd = statistics.stdev(xs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    t = u / se if se > 0 else (0.0 if u == 0 else math.copysign(math.inf, u))
    return {"alpha": alpha, "urisk": u, "trisk": t}


# ------------------------------------------------------------- analysis ----
def analyse(path: str, n_rep: int = 10000) -> dict:
    recs, multipositive = load_records(path)
    has_raw = all("cos_sim" in r and len(r["cos_sim"]) == len(r["base_score"])
                  for r in recs)
    reps = REPRESENTATIONS if has_raw else ("calibrated",)
    rel = [relevant_set(r, multipositive) for r in recs]
    tune_idx, report_idx = split_indices(len(recs))

    base_report = hits_for(recs, rel, r_base, 0.0, "verification_score", report_idx)

    rows = {}
    for name, (fn, grid, fixed, rank_based) in RULES.items():
        for rep in reps:
            if name == "base" and rep != "calibrated":
                continue
            field = FIELD[rep]
            if grid is None:
                param, tune_hits = fixed, sum(hits_for(recs, rel, fn, fixed, field, tune_idx))
            else:
                param, tune_hits = tune(recs, rel, fn, grid, field, tune_idx)
            rep_hits = hits_for(recs, rel, fn, param, field, report_idx)
            key = name if name == "base" else "%s[%s]" % (name, rep)
            mc = mcnemar_exact(rep_hits, base_report)
            rows[key] = {
                "rule": name, "representation": rep, "param": param,
                "rank_based": rank_based,
                "tune_r1": 100.0 * tune_hits / len(tune_idx),
                "report_r1": 100.0 * sum(rep_hits) / len(report_idx),
                "helped": mc["fixed"], "broken": mc["broken"],
                "p_exact": mc["p_exact"],
                "vs_base": paired_bootstrap(rep_hits, base_report, n_rep),
                "top1_change_rate": top1_changed(recs, fn, param, field, report_idx),
                "risk": [urisk_trisk(rep_hits, base_report, a) for a in ALPHAS],
                "_hits": rep_hits,
            }

    # Rank-based rules must be invariant to the monotone calibrator.
    invariance = {}
    if has_raw:
        for name, (_, _, _, rank_based) in RULES.items():
            if rank_based and name != "base":
                a, b = rows["%s[calibrated]" % name], rows["%s[raw]" % name]
                invariance[name] = (a["_hits"] == b["_hits"] and a["param"] == b["param"])

    # Strongest tuned rule chosen on the TUNE half; every rule vs it on report.
    tuned = [k for k, v in rows.items() if RULES[v["rule"]][1] is not None]
    strongest = max(tuned, key=lambda k: (rows[k]["tune_r1"], k))
    vs_strongest = {k: paired_bootstrap(rows[k]["_hits"], rows[strongest]["_hits"], n_rep)
                    for k in rows if k != strongest}

    for v in rows.values():
        v.pop("_hits")
    return {"path": path, "multipositive": multipositive, "n": len(recs),
            "n_tune": len(tune_idx), "n_report": len(report_idx),
            "representations": list(reps), "alphas": list(ALPHAS),
            "rows": rows, "rank_rules_invariant_to_calibration": invariance,
            "strongest_on_tune": strongest, "vs_strongest": vs_strongest,
            "n_bootstrap": n_rep}


def report(res: dict) -> None:
    print("=" * 100)
    print("FUSION SUITE  %s  (n=%d: tune %d / report %d%s)"
          % (os.path.basename(res["path"]), res["n"], res["n_tune"], res["n_report"],
             ", multi-positive" if res["multipositive"] else ""))
    print("=" * 100)
    print("%-26s %6s %7s %7s %7s %5s %5s %9s %-17s %7s %7s"
          % ("rule[repr]", "param", "tuneR1", "repR1", "delta", "help", "brk",
             "p", "95% CI", "TRisk1", "TRisk5"))
    for k, v in res["rows"].items():
        d = v["vs_base"]
        print("%-26s %6s %7.2f %7.2f %+7.2f %5d %5d %9.3g [%+6.2f,%+6.2f] %7.2f %7.2f"
              % (k, ("%.2g" % v["param"]) if k != "base" else "-", v["tune_r1"],
                 v["report_r1"], d["delta_pp"], v["helped"], v["broken"], v["p_exact"],
                 d["ci_low"], d["ci_high"], v["risk"][0]["trisk"], v["risk"][1]["trisk"]))
    print()
    if res["rank_rules_invariant_to_calibration"]:
        ok = all(res["rank_rules_invariant_to_calibration"].values())
        print("rank-based rules identical on raw vs calibrated scores: %s %s"
              % (ok, res["rank_rules_invariant_to_calibration"]))
    print("strongest tuned rule (chosen on tune half): %s" % res["strongest_on_tune"])
    for k, d in res["vs_strongest"].items():
        print("  %-26s vs strongest: %+6.2f pp  CI [%+.2f, %+.2f]  p=%.3g"
              % (k, d["delta_pp"], d["ci_low"], d["ci_high"], d["p_two_sided"]))


# Published held-out values (Table 6, n_report = 2091) that the CIRR cache must
# reproduce before any new row is trusted: rule key -> (report R@1, parameter).
CIRR_ANCHORS = {"base": (36.63, None), "rank_wsum[calibrated]": (37.49, None),
                "bounded_frozen[calibrated]": (38.98, 1.0),
                "linear[calibrated]": (39.41, 0.75)}


def check_anchors(res: dict) -> List[str]:
    """Empty list if the run reproduces Table 6; else the mismatches."""
    if res["multipositive"] or res["n"] != 4181:
        return []
    bad = []
    for key, (r1, param) in CIRR_ANCHORS.items():
        row = res["rows"][key]
        if round(row["report_r1"], 2) != r1 or (param is not None and row["param"] != param):
            bad.append("%s: R@1 %.2f param %s (published %.2f, %s)"
                       % (key, row["report_r1"], row["param"], r1, param))
    return bad


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)
    try:
        res = analyse(args.cache, args.bootstrap)
    except CacheError as exc:
        print("CACHE ERROR: %s" % exc, file=sys.stderr)
        return 1
    report(res)
    bad = check_anchors(res)
    print()
    if bad:
        print("SELF-CHECK FAILED -- this run does not reproduce Table 6:")
        for b in bad:
            print("  " + b)
        return 1
    if not res["multipositive"] and res["n"] == 4181:
        print("Self-check passed: Table 6 reproduced (base, rank, bounded lam=1, linear lam=0.75).")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print("\nwrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
