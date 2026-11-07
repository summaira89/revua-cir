"""Does verification evidence improve integration beyond base-score conservatism? CPU only.

Question (fixed before any result of this script existed, 16 October 2026)
------------------------------------------------------------------------
The paper's integration claim rests on the learned acceptance gate. A gate
that reads only base-score features might do as well, in which case the gain
is conservative integration, not verifier evidence. This script tests that.

Feature sets (all label-free; the same z-scored quantities gate_suite.py uses)
  full       base_margin, ver_gap, base_gap, fused_base_rank, ver_margin
  base_only  base_margin, base_gap, fused_base_rank   (no verification score)
  ver_only   ver_gap, ver_margin                      (no base score)

Protocol
  Cross-fitting on CIRR validation: each gate (logistic model AND threshold)
  is fitted on one half of the frozen tune/report split and applied to the
  other half, in both directions, so every one of the queries receives an
  out-of-fold gate decision. Fitting is identical to gate_suite.py (ridge
  logistic on changed helped/broken queries; threshold maximising URisk at
  alpha=1 over deciles plus accept-all/none; ties to fewer acceptances).

PRIMARY comparison (fixed in advance)
  full vs base_only, bounded fusion at lambda=1, paired bootstrap difference
  in URisk(alpha=1) over all cross-fitted queries. "Verification evidence adds
  value" is claimed only if the 95% interval excludes zero in favour of full.
  Everything else is secondary or descriptive:
  - per-direction results and the convex rule;
  - ver_only;
  - a random gate at full's acceptance rate (expected value, analytic);
  - operating frontiers: maximum rescues at fixed break budgets, for every
    gate's threshold path and for the fusion-weight path, per evaluation half
    (descriptive: each frontier point is an oracle on the half it is read
    from, symmetrically for all families);
  - optional --transfer: tune-fitted gates applied unchanged to other caches.

All CIRR results are retrospective: the gate design and the checkpoint both
saw CIRR validation (manuscript Section 4).

Usage
  python scripts/evidence_ablation.py scripts/frozen_policy.json CIRR_CACHE --json ../ablation.json
  python scripts/evidence_ablation.py scripts/frozen_policy.json CIRR_CACHE --transfer FIQ1 FIQ2 FIQ3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apply_frozen_policy import (  # noqa: E402
    CIRR_REPORT_ANCHORS, FUSIONS, fisher_two_sided, logistic_scorer,
)
from fusion_controls import CacheError, load_records, relevant_set, split_indices  # noqa: E402
from fusion_suite import urisk_trisk  # noqa: E402
from gate_suite import (  # noqa: E402
    choose_threshold, fit_logistic, paired_mean_bootstrap, query_view, risk_x,
)

FEATURE_SETS = {
    "full": ("base_margin", "ver_gap", "base_gap", "fused_base_rank", "ver_margin"),
    "base_only": ("base_margin", "base_gap", "fused_base_rank"),
    "ver_only": ("ver_gap", "ver_margin"),
}
BREAK_BUDGETS = (5, 10, 15, 20, 30)
WEIGHT_STEPS = 100


def fit_gate(views_fit: Sequence[dict], feats: Sequence[str]):
    """Logistic gate + URisk(1) threshold, both from views_fit only."""
    train = [v for v in views_fit if v["changed"] and v["hit_base"] != v["hit_fused"]]
    y = [v["hit_fused"] for v in train]
    if len(set(y)) < 2:
        # same fallback as gate_suite.py: a constant score, and the threshold
        # rule then chooses between accepting all and accepting none
        def gate(v):
            return 1.0
    else:
        X = [[v["feat"][k] for k in feats] for v in train]
        _, model = fit_logistic(X, y)
        score = logistic_scorer(model, feats)

        def gate(v):
            return score(v["feat"])
    t, _ = choose_threshold(views_fit, gate, "ge", 1.0)
    return gate, t, len(train)


def decisions(views: Sequence[dict], score, t) -> List[bool]:
    return [v["changed"] and score(v) >= t for v in views]


def outcome(views: Sequence[dict], acc: Sequence[bool]) -> dict:
    hits = [v["hit_fused"] if a else v["hit_base"] for v, a in zip(views, acc)]
    base = [v["hit_base"] for v in views]
    H = sum(1 for v in views if v["changed"] and v["hit_fused"] and not v["hit_base"])
    B = sum(1 for v in views if v["changed"] and v["hit_base"] and not v["hit_fused"])
    kh = sum(1 for v, a in zip(views, acc) if a and v["hit_fused"] and not v["hit_base"])
    kb = sum(1 for v, a in zip(views, acc) if a and v["hit_base"] and not v["hit_fused"])
    return {"hits": hits, "base": base, "n": len(views),
            "r1": 100.0 * sum(hits) / len(hits) if hits else 0.0,
            "accepted": sum(1 for a in acc if a), "changed": sum(1 for v in views if v["changed"]),
            "helped": kh, "broken": kb, "H": H, "B": B,
            "odds_ratio": (kh * (B - kb)) / (kb * (H - kh)) if kb and (H - kh) else None,
            "fisher_p": fisher_two_sided(kh, kb, H - kh, B - kb),
            "risk1": urisk_trisk(hits, base, 1.0)}


SUMMARY_KEYS = ("r1", "accepted", "changed", "helped", "broken", "H", "B",
                "odds_ratio", "fisher_p", "risk1")


def summary(o: dict) -> dict:
    """The JSON-safe part of outcome() (drops the per-query lists)."""
    return {k: o[k] for k in SUMMARY_KEYS}


def random_gate_expectation(o: dict) -> dict:
    """A gate accepting each fused change independently at full's acceptance
    rate q has expected rescues qH and breaks qB (selectivity odds ratio 1)."""
    q = o["accepted"] / o["changed"] if o["changed"] else 0.0
    eh, eb = q * o["H"], q * o["B"]
    return {"rate": q, "exp_helped": eh, "exp_broken": eb,
            "exp_urisk1": (eh - 2.0 * eb) / o["n"]}


def threshold_frontier(views: Sequence[dict], score) -> List[tuple]:
    """(breaks, rescues) for every threshold on this half (accept score >= t)."""
    ch = sorted(((score(v), v) for v in views if v["changed"]), key=lambda x: -x[0])
    pts, h, b = [(0, 0)], 0, 0
    for j, (s, v) in enumerate(ch):
        h += 1 if (v["hit_fused"] and not v["hit_base"]) else 0
        b += 1 if (v["hit_base"] and not v["hit_fused"]) else 0
        if j + 1 == len(ch) or ch[j + 1][0] != s:   # a threshold cannot split tied scores
            pts.append((b, h))
    return pts


def weight_frontier(recs, rels, fn, param) -> List[tuple]:
    pts = []
    for i in range(WEIGHT_STEPS + 1):
        w = param * i / WEIGHT_STEPS
        h = b = 0
        for rec, rel in zip(recs, rels):
            v = query_view(rec, rel, fn, w)
            h += 1 if (v["changed"] and v["hit_fused"] and not v["hit_base"]) else 0
            b += 1 if (v["changed"] and v["hit_base"] and not v["hit_fused"]) else 0
        pts.append((b, h))
    return pts


def best_at_budgets(pts: Sequence[tuple]) -> Dict[int, int]:
    return {k: max(h for b, h in pts if b <= k) for k in BREAK_BUDGETS}


def analyse(policy_path: str, cirr_path: str, transfer: Sequence[str], n_rep: int) -> dict:
    policy = json.load(open(policy_path, encoding="utf-8"))
    recs, multi = load_records(cirr_path)
    rels = [relevant_set(r, multi) for r in recs]
    tune_idx, report_idx = split_indices(len(recs))
    halves = {"tune": tune_idx, "report": report_idx}
    t_recs, t_rels = [], []
    for p in transfer:
        r, m = load_records(p)
        t_recs.extend(r)
        t_rels.extend(relevant_set(x, m) for x in r)

    out = {"cirr": cirr_path, "n": len(recs), "transfer": list(transfer), "rules": {}}
    for rname, spec in policy["rules"].items():
        fn, param = FUSIONS[spec["fusion"]], spec["param"]
        views = {h: [query_view(recs[i], rels[i], fn, param) for i in idx]
                 for h, idx in halves.items()}
        res = {"param": param, "directions": {}, "crossfit": {}, "frontier": {}, "transfer": {}}
        cross_acc = {fs: [None] * len(recs) for fs in FEATURE_SETS}
        fitted_on_tune = {}
        for fit_h, ev_h in (("tune", "report"), ("report", "tune")):
            d = {}
            for fs, feats in FEATURE_SETS.items():
                score, t, n_train = fit_gate(views[fit_h], feats)
                acc = decisions(views[ev_h], score, t)
                for i, a in zip(halves[ev_h], acc):
                    cross_acc[fs][i] = a
                o = outcome(views[ev_h], acc)
                d[fs] = summary(o)
                d[fs].update(threshold=t, n_train=n_train,
                             frontier=best_at_budgets(threshold_frontier(views[ev_h], score)))
                if fit_h == "tune":
                    fitted_on_tune[fs] = (score, t)
            d["random_at_full_rate"] = random_gate_expectation(
                outcome(views[ev_h], [cross_acc["full"][i] for i in halves[ev_h]]))
            d["weights"] = best_at_budgets(weight_frontier(
                [recs[i] for i in halves[ev_h]], [rels[i] for i in halves[ev_h]], fn, param))
            res["directions"]["fit_%s_eval_%s" % (fit_h, ev_h)] = d

        # cross-fitted, all queries: every query's gate decision is out of fold
        allv = [None] * len(recs)
        for h, idx in halves.items():
            for i, v in zip(idx, views[h]):
                allv[i] = v
        outs = {fs: outcome(allv, cross_acc[fs]) for fs in FEATURE_SETS}
        for fs, o in outs.items():
            res["crossfit"][fs] = summary(o)
        xf = risk_x(outs["full"]["hits"], outs["full"]["base"], 1.0)
        for other in ("base_only", "ver_only"):
            res["crossfit"]["full_minus_" + other + "_urisk1"] = paired_mean_bootstrap(
                xf, risk_x(outs[other]["hits"], outs[other]["base"], 1.0), n_rep)
        res["crossfit"]["random_at_full_rate"] = random_gate_expectation(outs["full"])

        if t_recs:
            tv = [query_view(r, rel, fn, param) for r, rel in zip(t_recs, t_rels)]
            to = {fs: outcome(tv, decisions(tv, *fitted_on_tune[fs])) for fs in FEATURE_SETS}
            for fs, o in to.items():
                res["transfer"][fs] = summary(o)
            xt = risk_x(to["full"]["hits"], to["full"]["base"], 1.0)
            res["transfer"]["full_minus_base_only_urisk1"] = paired_mean_bootstrap(
                xt, risk_x(to["base_only"]["hits"], to["base_only"]["base"], 1.0), n_rep)
        out["rules"][rname] = res
    return out


def _row(name: str, e: dict) -> str:
    r = e["risk1"]
    orr = "%.2f" % e["odds_ratio"] if e.get("odds_ratio") is not None else "n/a"
    return "  %-11s acc %4d/%-4d  help %4d brk %4d  (of %d/%d)  OR %-5s p=%-7.3g URisk1 %+.4f TRisk1 %+.2f" % (
        name, e["accepted"], e["changed"], e["helped"], e["broken"], e["H"], e["B"],
        orr, e["fisher_p"], r["urisk"], r["trisk"])


def report(res: dict) -> None:
    print("=" * 112)
    print("EVIDENCE ABLATION  %s  n=%d  (retrospective on CIRR)" % (
        os.path.basename(res["cirr"]), res["n"]))
    print("=" * 112)
    for rname, r in res["rules"].items():
        print("\n### %s (param %s)" % (rname, r["param"]))
        for dname, d in r["directions"].items():
            print("\n %s" % dname)
            for fs in FEATURE_SETS:
                print(_row(fs, d[fs]))
            rg = d["random_at_full_rate"]
            print("  random@full rate %.3f: expected help %.1f brk %.1f URisk1 %+.4f" % (
                rg["rate"], rg["exp_helped"], rg["exp_broken"], rg["exp_urisk1"]))
            print("  frontier (max rescues at break budget %s):" % (BREAK_BUDGETS,))
            for fs in FEATURE_SETS:
                print("    %-10s %s" % (fs, [d[fs]["frontier"][k] for k in BREAK_BUDGETS]))
            print("    %-10s %s" % ("weights", [d["weights"][k] for k in BREAK_BUDGETS]))
        print("\n CROSS-FITTED (all queries, out-of-fold decisions)")
        for fs in FEATURE_SETS:
            print(_row(fs, r["crossfit"][fs]))
        for other in ("base_only", "ver_only"):
            c = r["crossfit"]["full_minus_" + other + "_urisk1"]
            print("  full - %-9s URisk1 %+.4f [%+.4f, %+.4f] p=%.4f" % (
                other, c["diff"], c["ci_low"], c["ci_high"], c["p_two_sided"]))
        if r["transfer"]:
            print("\n TRANSFER (tune-fitted gates, frozen thresholds)")
            for fs in FEATURE_SETS:
                print(_row(fs, r["transfer"][fs]))
            c = r["transfer"]["full_minus_base_only_urisk1"]
            print("  full - base_only URisk1 %+.4f [%+.4f, %+.4f] p=%.4f" % (
                c["diff"], c["ci_low"], c["ci_high"], c["p_two_sided"]))
    c = res["rules"]["bounded_frozen"]["crossfit"]["full_minus_base_only_urisk1"]
    print("\nPRIMARY (fixed in advance): full vs base_only, bounded, cross-fitted URisk(1):")
    print("  diff %+.4f  95%% CI [%+.4f, %+.4f]  p=%.4f  -> %s" % (
        c["diff"], c["ci_low"], c["ci_high"], c["p_two_sided"],
        "verification evidence adds value" if c["ci_low"] > 0 else
        ("base-only gate better" if c["ci_high"] < 0 else "no detectable difference")))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("policy")
    ap.add_argument("cirr_cache")
    ap.add_argument("--transfer", nargs="*", default=[])
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)
    try:
        res = analyse(args.policy, args.cirr_cache, args.transfer, args.bootstrap)
    except CacheError as exc:
        print("CACHE ERROR: %s" % exc, file=sys.stderr)
        return 1
    report(res)
    if res["n"] == 4181:                      # the CIRR validation cache
        bad = []
        for rname in res["rules"]:
            got = round(res["rules"][rname]["directions"]["fit_tune_eval_report"]["full"]["r1"], 2)
            want = CIRR_REPORT_ANCHORS[(rname, "gate_learned")]
            if got != want:
                bad.append("%s full gate R@1 %.2f, gate_suite %.2f" % (rname, got, want))
        print()
        if bad:
            print("SELF-CHECK FAILED -- the full gate does not reproduce gate_suite:")
            for b_ in bad:
                print("  " + b_)
            return 1
        print("Self-check passed: the full gate fitted on tune reproduces gate_suite's learned gate.")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
