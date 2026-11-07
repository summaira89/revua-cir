"""Post-verification acceptance gates (plan step 3). CPU only, from the cache.

Question
--------
Fusion breaks 10-15% of the queries base retrieval already answers correctly
(fusion_suite.py, report half). Can a label-free gate keep fusion's helped
queries while blocking some broken ones, better than simply trusting fusion
when base retrieval is unsure?

Design (fixed before results)
-----------------------------
Fusion rules gated: bounded at the frozen lambda=1 (most conservative) and
convex with its weight tuned on the tune half (most accurate).

For every query whose fused top-1 differs from the base top-1, a gate
accepts the fused ranking or keeps the base one. Gates:
  none      always accept (= the fusion rule itself)
  base      accept iff base margin <= t        (the bar to beat)
  vgap      accept iff verifier gap >= t
  learned   accept iff logistic score >= t; the logistic model is fitted
            on tune-half changed queries, helped=1 vs broken=0
Label-free features, all within-window z-scores: base margin (top-1 minus
runner-up), verifier gap (fused top minus base top), base gap (base top minus
fused top), base rank of the fused top, verifier's own margin.

Threshold: chosen on the TUNE half from the 0.0-1.0 decile grid of the gate
score over tune changed queries, plus accept-all and accept-none, to maximise
URisk at alpha=1 (primary; losses count double). The alpha=0 choice (pure
R@1) is reported as a sensitivity check only. Ties go to accepting fewer.

PRIMARY comparison, fixed before results: the learned gate against the
base-margin gate, on bounded fusion, under the alpha=1 objective, by the
paired difference in URisk(alpha=1) on the report half. Every other row is
exploratory.

Everything reported is on the REPORT half. This is post-verification gating:
it can limit damage but saves no verifier computation.

Usage
-----
  python scripts/gate_suite.py rv_score_cache_fc_aep_rv.jsonl --json ../gate_suite.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fusion_controls import (  # noqa: E402
    CacheError, _zscore, load_records, mcnemar_exact, relevant_set, split_indices,
)
from fusion_controls import LAMBDA_GRID  # noqa: E402
from fusion_suite import CONVEX_GRID, r_bounded, r_convex, tune, urisk_trisk  # noqa: E402

RISK_GRIDS = {"bounded_frozen": LAMBDA_GRID, "convex": CONVEX_GRID}

FEATURES = ("base_margin", "ver_gap", "base_gap", "fused_base_rank", "ver_margin")
ALPHA_PRIMARY = 1.0
ALPHAS_REPORTED = (1.0, 5.0)
DECILES = tuple(round(0.1 * i, 1) for i in range(11))
RIDGE = 1.0
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 54321


def _top(scores: Sequence[float]) -> int:
    return max(range(len(scores)), key=lambda i: (scores[i], -i))


def _second(scores: Sequence[float], first: int) -> int:
    rest = [i for i in range(len(scores)) if i != first]
    return max(rest, key=lambda i: (scores[i], -i))


def query_view(rec: dict, rel: set, fn, param) -> dict:
    """Outcome and label-free features for one query under one fusion rule."""
    b, v = rec["base_score"], rec["verification_score"]
    s = fn(b, v, param)
    tb, tf = _top(b), _top(s)
    zb, zv = _zscore(b), _zscore(v)
    base_order = sorted(range(len(b)), key=lambda i: (-b[i], i))
    if len(b) > 1:
        bm = zb[tb] - zb[_second(b, tb)]
        tv = _top(v)
        vm = zv[tv] - zv[_second(v, tv)]
    else:
        bm = vm = 0.0
    return {
        "changed": tf != tb,
        "hit_base": 1 if tb in rel else 0,
        "hit_fused": 1 if tf in rel else 0,
        "feat": {"base_margin": bm, "ver_gap": zv[tf] - zv[tb],
                 "base_gap": zb[tb] - zb[tf],
                 "fused_base_rank": float(base_order.index(tf)), "ver_margin": vm},
    }


# ------------------------------------------------------- logistic model ----
def fit_logistic(X: List[List[float]], y: List[int], ridge: float = RIDGE, iters: int = 100):
    """L2-regularised logistic regression by Newton/IRLS, features standardised
    on the training data. Returns a scorer f(features dict) -> probability."""
    import numpy as np
    A = np.asarray(X, dtype=float)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    Z = np.hstack([np.ones((len(A), 1)), (A - mu) / sd])
    t = np.asarray(y, dtype=float)
    w = np.zeros(Z.shape[1])
    R = ridge * np.eye(Z.shape[1])
    R[0, 0] = 0.0                                  # do not shrink the intercept
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        g = Z.T @ (p - t) + R @ w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break

    def score(f: Dict[str, float]) -> float:
        z = np.concatenate([[1.0], (np.array([f[k] for k in FEATURES]) - mu) / sd])
        return float(1.0 / (1.0 + np.exp(-z @ w)))
    return score, {"weights": w.tolist(), "mean": mu.tolist(), "sd": sd.tolist()}


# --------------------------------------------------------------- gating ----
def gated_hits(views: Sequence[dict], accept) -> List[int]:
    return [v["hit_fused"] if (v["changed"] and accept(v)) else v["hit_base"] for v in views]


def risk_value(hits: Sequence[int], base: Sequence[int], alpha: float) -> float:
    return urisk_trisk(hits, base, alpha)["urisk"]


def choose_threshold(views, score_fn, direction: str, alpha: float):
    """Decile thresholds of the gate score over tune changed queries, plus
    accept-all/none; maximise tune URisk(alpha), ties -> fewer accepted."""
    vals = sorted(score_fn(v) for v in views if v["changed"])
    if not vals:
        return math.inf, 0
    cands = [vals[min(len(vals) - 1, int(q * (len(vals) - 1)))] for q in DECILES]
    cands = sorted(set(cands + [-math.inf, math.inf]))
    base = [v["hit_base"] for v in views]
    best = None
    for t in cands:
        acc = (lambda v, t=t: score_fn(v) >= t) if direction == "ge" else \
              (lambda v, t=t: score_fn(v) <= t)
        hits = gated_hits(views, acc)
        n_acc = sum(1 for v in views if v["changed"] and acc(v))
        key = (risk_value(hits, base, alpha), -n_acc)
        if best is None or key > best[0]:
            best = (key, t, n_acc)
    return best[1], best[2]


def paired_mean_bootstrap(xa: Sequence[float], xb: Sequence[float],
                          n_rep: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
    """CI for mean(xa - xb) over queries (xa, xb are per-query values)."""
    n = len(xa)
    d = [a - b for a, b in zip(xa, xb)]
    obs = sum(d) / n
    rng = random.Random(seed)
    reps = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_rep))
    lo, hi = reps[int(0.025 * n_rep)], reps[min(n_rep - 1, int(0.975 * n_rep))]
    tail = sum(1 for r in reps if (r <= 0 if obs >= 0 else r >= 0))
    return {"diff": obs, "ci_low": lo, "ci_high": hi,
            "p_two_sided": min(1.0, 2.0 * (tail + 1) / (n_rep + 1))}


def risk_x(hits, base, alpha):
    out = []
    for h, b in zip(hits, base):
        d = h - b
        out.append(d if d > 0 else (1.0 + alpha) * d)
    return out


def summarise(name, hits, views, accept_flags, base):
    mc = mcnemar_exact(hits, base)
    changed = [i for i, v in enumerate(views) if v["changed"]]
    n_rev = sum(1 for i in changed if not accept_flags[i])
    return {"gate": name, "r1": 100.0 * sum(hits) / len(hits),
            "delta_pp": 100.0 * (sum(hits) - sum(base)) / len(hits),
            "helped": mc["fixed"], "broken": mc["broken"], "p_exact": mc["p_exact"],
            "reverted_share_of_changed": (n_rev / len(changed)) if changed else 0.0,
            "risk": [urisk_trisk(hits, base, a) for a in ALPHAS_REPORTED]}


def analyse(path: str, n_rep: int = N_BOOTSTRAP) -> dict:
    recs, multipositive = load_records(path)
    rel = [relevant_set(r, multipositive) for r in recs]
    tune_idx, report_idx = split_indices(len(recs))

    conv_a, _ = tune(recs, rel, r_convex, CONVEX_GRID, "verification_score", tune_idx)
    rules = {"bounded_frozen": (r_bounded, 1.0), "convex": (r_convex, conv_a)}

    out = {"path": path, "n": len(recs), "n_tune": len(tune_idx), "n_report": len(report_idx),
           "multipositive": multipositive, "convex_param": conv_a, "rules": {}}
    for rname, (fn, param) in rules.items():
        tv = [query_view(recs[i], rel[i], fn, param) for i in tune_idx]
        rv = [query_view(recs[i], rel[i], fn, param) for i in report_idx]
        base_r = [v["hit_base"] for v in rv]

        # learned gate, fitted on tune changed queries that are helped or broken
        train = [v for v in tv if v["changed"] and v["hit_base"] != v["hit_fused"]]
        X = [[v["feat"][k] for k in FEATURES] for v in train]
        y = [v["hit_fused"] for v in train]
        if len(set(y)) == 2:
            lr_score, lr_model = fit_logistic(X, y)
        else:
            lr_score, lr_model = (lambda f: 1.0), None

        scorers = {
            "base": (lambda v: v["feat"]["base_margin"], "le"),
            "vgap": (lambda v: v["feat"]["ver_gap"], "ge"),
            "learned": (lambda v: lr_score(v["feat"]), "ge"),
        }
        res = {"param": param, "n_train_gate": len(train), "logistic": lr_model, "gates": {}}
        rows = {}
        all_acc = [True] * len(rv)
        rows["none"] = (gated_hits(rv, lambda v: True), all_acc, None)
        for gname, (sf, direction) in scorers.items():
            for obj_alpha, tag in ((ALPHA_PRIMARY, ""), (0.0, "@a0")):
                t, _ = choose_threshold(tv, sf, direction, obj_alpha)
                acc = (lambda v, t=t, sf=sf: sf(v) >= t) if direction == "ge" else \
                      (lambda v, t=t, sf=sf: sf(v) <= t)
                flags = [acc(v) for v in rv]
                rows[gname + tag] = (gated_hits(rv, acc), flags, t)

        # The decisive control (added after the first CIRR run): the same fusion
        # rule with its weight re-tuned on the tune half for the SAME alpha=1
        # objective. A smaller weight also changes fewer rankings, so if this
        # matches the gate, the gate adds nothing beyond weakening the fusion.
        # The grid includes 0 (base only); ties go to the smaller weight.
        grid = RISK_GRIDS[rname]
        tune_base = [v["hit_base"] for v in tv]
        best_w = None
        for p in grid:
            tvw = [query_view(recs[i], rel[i], fn, p) for i in tune_idx]
            u = risk_value(gated_hits(tvw, lambda v: True), tune_base, ALPHA_PRIMARY)
            if best_w is None or u > best_w[0]:
                best_w = (u, p)
        rvw = [query_view(recs[i], rel[i], fn, best_w[1]) for i in report_idx]
        rows["weaker"] = (gated_hits(rvw, lambda v: True),
                          [w["changed"] for w in rvw], best_w[1])
        res["weaker_param"] = best_w[1]

        f_hits = rows["none"][0]
        f_broken = sum(1 for h, b in zip(f_hits, base_r) if b and not h)
        f_helped = sum(1 for h, b in zip(f_hits, base_r) if h and not b)
        for gname, (hits, flags, t) in rows.items():
            s = summarise(gname, hits, rv, flags, base_r)
            s["threshold"] = t
            s["helped_kept"] = (s["helped"] / f_helped) if f_helped else None
            s["broken_blocked"] = (1 - s["broken"] / f_broken) if f_broken else None
            xa = risk_x(hits, base_r, ALPHA_PRIMARY)
            s["vs_ungated_urisk1"] = paired_mean_bootstrap(
                xa, risk_x(f_hits, base_r, ALPHA_PRIMARY), n_rep)
            s["vs_base_gate_urisk1"] = paired_mean_bootstrap(
                xa, risk_x(rows["base"][0], base_r, ALPHA_PRIMARY), n_rep)
            s["vs_weaker_urisk1"] = paired_mean_bootstrap(
                xa, risk_x(rows["weaker"][0], base_r, ALPHA_PRIMARY), n_rep)
            s["vs_weaker_r1"] = paired_mean_bootstrap(hits, rows["weaker"][0], n_rep)
            res["gates"][gname] = s
        out["rules"][rname] = res
    return out


def report(res: dict) -> None:
    print("=" * 108)
    print("ACCEPTANCE GATES  %s  (report half n=%d; thresholds chosen on tune n=%d)"
          % (os.path.basename(res["path"]), res["n_report"], res["n_tune"]))
    print("=" * 108)
    for rname, r in res["rules"].items():
        print("\n%s (param %s; gate trained on %d tune changed queries)"
              % (rname, r["param"], r["n_train_gate"]))
        print("  %-12s %7s %7s %5s %5s %7s %7s %8s %8s %8s %-24s"
              % ("gate", "R@1", "delta", "help", "brk", "keptH", "blkB", "revert",
                 "TRisk1", "TRisk5", "URisk1 vs base-gate [CI]"))
        for g, s in r["gates"].items():
            d = s["vs_base_gate_urisk1"]
            print("  %-12s %7.2f %+7.2f %5d %5d %7s %7s %7.0f%% %8.2f %8.2f %+.4f [%+.4f,%+.4f]"
                  % (g, s["r1"], s["delta_pp"], s["helped"], s["broken"],
                     "%.0f%%" % (100 * s["helped_kept"]) if s["helped_kept"] is not None else "-",
                     "%.0f%%" % (100 * s["broken_blocked"]) if s["broken_blocked"] is not None else "-",
                     100 * s["reverted_share_of_changed"], s["risk"][0]["trisk"],
                     s["risk"][1]["trisk"], d["diff"], d["ci_low"], d["ci_high"]))
    print("\nkeptH = share of fusion's helped queries kept; blkB = share of its broken ones blocked;")
    print("revert = share of changed queries returned to the base ranking. '@a0' rows choose the")
    print("threshold for pure R@1 (sensitivity only); unmarked rows use the primary alpha=1 objective.")
    s = res["rules"]["bounded_frozen"]["gates"]["learned"]["vs_base_gate_urisk1"]
    print("\nPRIMARY (fixed in advance): learned vs base-margin gate, bounded fusion, URisk(1):")
    print("  diff %+.4f  95%% CI [%+.4f, %+.4f]  p=%.3g  -> %s" % (
        s["diff"], s["ci_low"], s["ci_high"], s["p_two_sided"],
        "learned gate better" if s["ci_low"] > 0 else
        ("base-margin gate better" if s["ci_high"] < 0 else "no detectable difference")))
    print("All other rows are exploratory.")
    print("\nDECISIVE CONTROL: each learned gate vs its fusion rule re-weighted for the same")
    print("alpha=1 objective ('weaker' row). Positive URisk diff = the gate adds something")
    print("beyond simply turning the fusion weight down.")
    for rname, r in res["rules"].items():
        s = r["gates"]["learned"]
        u, a = s["vs_weaker_urisk1"], s["vs_weaker_r1"]
        print("  %-15s weaker weight %-5s | URisk(1) diff %+.4f [%+.4f, %+.4f] p=%.3g"
              " | R@1 diff %+.2f pp [%+.2f, %+.2f]"
              % (rname, r["weaker_param"], u["diff"], u["ci_low"], u["ci_high"],
                 u["p_two_sided"], 100 * a["diff"], 100 * a["ci_low"], 100 * a["ci_high"]))
    print("\nLearned gate weights (standardised features; intercept first):")
    for rname, r in res["rules"].items():
        if r["logistic"]:
            w = r["logistic"]["weights"]
            print("  %-15s intercept %+.2f | %s" % (rname, w[0], "  ".join(
                "%s %+.2f" % (k, x) for k, x in zip(FEATURES, w[1:]))))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache")
    ap.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)
    try:
        res = analyse(args.cache, args.bootstrap)
    except CacheError as exc:
        print("CACHE ERROR: %s" % exc, file=sys.stderr)
        return 1
    report(res)
    # Self-check: ungated rules must reproduce fusion_suite's report-half rows.
    if not res["multipositive"] and res["n"] == 4181:
        got = (round(res["rules"]["bounded_frozen"]["gates"]["none"]["r1"], 2),
               round(res["rules"]["convex"]["gates"]["none"]["r1"], 2), res["convex_param"])
        if got != (38.98, 39.60, 0.5):
            print("\nSELF-CHECK FAILED: ungated bounded/convex/param = %s, expected (38.98, 39.6, 0.5)"
                  % (got,))
            return 1
        print("\nSelf-check passed: ungated bounded 38.98 and convex 39.60 (a=0.5) reproduced.")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
