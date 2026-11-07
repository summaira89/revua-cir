"""Apply the frozen CIRR policies to new caches, with no tuning (plan step 6).

Loads frozen_policy.json (fusion weights, learned-gate logistic model, gate
thresholds, all fitted on the CIRR tune half) and evaluates every policy on
the given cache(s). Nothing is fitted or selected here.

Policies per fusion rule: base, ungated fusion, the same fusion at its
frozen risk-tuned weight ("weaker"), and the base-margin, verifier-gap and
learned gates at their frozen thresholds.

Metrics: R@1 and R@5 (a top-10 rerank cannot change R@10/R@50), helped and
broken top-1 counts against base, URisk/TRisk at alpha 1 and 5, gate
selectivity (Fisher exact on helped vs broken among accepted changes), and
paired URisk(alpha=1) contrasts.

Self-check: --cirr-report-half restricts a CIRR cache to the frozen
report half and must reproduce gate_suite.py's report-half R@1 rows.

Usage
-----
  python scripts/apply_frozen_policy.py frozen_policy.json CACHE [CACHE ...] [--json OUT]
  python scripts/apply_frozen_policy.py frozen_policy.json rv_score_cache_fc_aep_rv.jsonl --cirr-report-half
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fusion_controls import CacheError, load_records, relevant_set, split_indices  # noqa: E402
from fusion_suite import r_bounded, r_convex, urisk_trisk  # noqa: E402
from gate_suite import paired_mean_bootstrap, query_view, risk_x  # noqa: E402

FUSIONS = {"bounded": r_bounded, "convex": r_convex}
ALPHAS = (1.0, 5.0)
# gate_suite.py report-half R@1 (CIRR), which the frozen application must reproduce.
CIRR_REPORT_ANCHORS = {
    ("bounded_frozen", "fusion"): 38.98, ("bounded_frozen", "weaker"): 37.69,
    ("bounded_frozen", "gate_base"): 36.82, ("bounded_frozen", "gate_vgap"): 38.21,
    ("bounded_frozen", "gate_learned"): 38.16,
    ("convex", "fusion"): 39.60, ("convex", "weaker"): 36.73,
    ("convex", "gate_base"): 36.63, ("convex", "gate_vgap"): 36.63,
    ("convex", "gate_learned"): 38.69,
}


def logistic_scorer(model: dict, features: Sequence[str]):
    w, mu, sd = model["weights"], model["mean"], model["sd"]

    def score(f: Dict[str, float]) -> float:
        z = w[0] + sum(w[i + 1] * (f[k] - mu[i]) / sd[i] for i, k in enumerate(features))
        return 1.0 / (1.0 + math.exp(-z))
    return score


def _order(scores: Sequence[float]) -> List[int]:
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def hit_at(order: Sequence[int], rel: set, k: int) -> int:
    return 1 if rel and any(i in rel for i in order[:k]) else 0


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    r1, c1, n = a + b, a + c, a + b + c + d
    if n == 0 or r1 in (0, n) or c1 in (0, n):
        return 1.0

    def p(x):
        return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    obs = p(a)
    lo, hi = max(0, r1 - (n - c1)), min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs * (1 + 1e-9)))


def evaluate_rule(recs, rels, spec: dict, features, n_rep: int) -> dict:
    fn = FUSIONS[spec["fusion"]]
    lr = logistic_scorer(spec["logistic"], features)
    th = spec["thresholds"]
    gates = {
        "gate_base": lambda v: v["feat"]["base_margin"] <= th["base"],
        "gate_vgap": lambda v: v["feat"]["ver_gap"] >= th["vgap"],
        "gate_learned": lambda v: lr(v["feat"]) >= th["learned"],
    }
    rows: Dict[str, Dict[str, List[int]]] = {k: {"h1": [], "h5": [], "acc": []}
                                             for k in ("base", "fusion", "weaker", *gates)}
    for rec, rel in zip(recs, rels):
        b, v = rec["base_score"], rec["verification_score"]
        ob = _order(b)
        of = _order(fn(b, v, spec["param"]))
        ow = _order(fn(b, v, spec["weaker_param"]))
        view = query_view(rec, rel, fn, spec["param"])
        rows["base"]["h1"].append(hit_at(ob, rel, 1))
        rows["base"]["h5"].append(hit_at(ob, rel, 5))
        rows["base"]["acc"].append(False)
        rows["fusion"]["h1"].append(hit_at(of, rel, 1))
        rows["fusion"]["h5"].append(hit_at(of, rel, 5))
        rows["fusion"]["acc"].append(view["changed"])
        rows["weaker"]["h1"].append(hit_at(ow, rel, 1))
        rows["weaker"]["h5"].append(hit_at(ow, rel, 5))
        rows["weaker"]["acc"].append(ow[0] != ob[0])
        for g, accept in gates.items():
            take = view["changed"] and accept(view)
            o = of if take else ob
            rows[g]["h1"].append(hit_at(o, rel, 1))
            rows[g]["h5"].append(hit_at(o, rel, 5))
            rows[g]["acc"].append(take)

    base1 = rows["base"]["h1"]
    fus1 = rows["fusion"]["h1"]
    n = len(base1)
    out = {}
    for k, r in rows.items():
        helped = sum(1 for h, b0 in zip(r["h1"], base1) if h and not b0)
        broken = sum(1 for h, b0 in zip(r["h1"], base1) if b0 and not h)
        entry = {"r1": 100.0 * sum(r["h1"]) / n, "r5": 100.0 * sum(r["h5"]) / n,
                 "helped": helped, "broken": broken,
                 "risk": [urisk_trisk(r["h1"], base1, a) for a in ALPHAS]}
        if k.startswith("gate_"):
            # selectivity among fusion's helped/broken changes
            hb = [(f, b0, a) for f, b0, a in zip(fus1, base1, r["acc"]) if f != b0]
            kh = sum(1 for f, b0, a in hb if a and f)
            kb = sum(1 for f, b0, a in hb if a and not f)
            H = sum(1 for f, b0, a in hb if f)
            B = len(hb) - H
            entry["selectivity"] = {"helped_kept": kh, "broken_kept": kb, "H": H, "B": B,
                                    "odds_ratio": (kh * (B - kb)) / (kb * (H - kh))
                                    if kb and (H - kh) else None,
                                    "fisher_p": fisher_two_sided(kh, kb, H - kh, B - kb)}
            xa = risk_x(r["h1"], base1, 1.0)
            entry["vs_fusion_urisk1"] = paired_mean_bootstrap(xa, risk_x(fus1, base1, 1.0), n_rep)
            entry["vs_weaker_urisk1"] = paired_mean_bootstrap(
                xa, risk_x(rows["weaker"]["h1"], base1, 1.0), n_rep)
        out[k] = entry
    return out


def analyse(policy_path: str, caches: Sequence[str], cirr_report_half: bool = False,
            n_rep: int = 10000) -> dict:
    policy = json.load(open(policy_path, encoding="utf-8"))
    recs, rels, sources = [], [], []
    for path in caches:
        r, multi = load_records(path)
        if cirr_report_half:
            _, report_idx = split_indices(len(r))
            r = [r[i] for i in report_idx]
        recs.extend(r)
        rels.extend(relevant_set(x, multi) for x in r)
        sources.append({"path": path, "n": len(r), "multipositive": multi})
    res = {"policy": policy_path, "sources": sources, "n": len(recs), "rules": {}}
    for rname, spec in policy["rules"].items():
        res["rules"][rname] = evaluate_rule(recs, rels, spec, policy["features"], n_rep)
    return res


def check_cirr_anchors(res: dict) -> List[str]:
    bad = []
    for (rname, k), want in CIRR_REPORT_ANCHORS.items():
        got = round(res["rules"][rname][k]["r1"], 2)
        if got != want:
            bad.append("%s/%s R@1 %.2f, gate_suite %.2f" % (rname, k, got, want))
    return bad


def report(res: dict) -> None:
    print("=" * 112)
    names = ", ".join(os.path.basename(s["path"]) for s in res["sources"])
    print("FROZEN POLICY  n=%d  from %s" % (res["n"], names))
    print("=" * 112)
    for rname, rows in res["rules"].items():
        print("\n%s" % rname)
        print("  %-13s %7s %7s %5s %5s %8s %8s  %-28s %-26s"
              % ("policy", "R@1", "R@5", "help", "brk", "TRisk1", "TRisk5",
                 "selectivity OR (Fisher p)", "URisk1 vs ungated [CI]"))
        for k, e in rows.items():
            sel = e.get("selectivity")
            sel_s = ("%s (%.2g)  %d/%d kept" % (
                "%.2f" % sel["odds_ratio"] if sel["odds_ratio"] is not None else "n/a",
                sel["fisher_p"], sel["helped_kept"] + sel["broken_kept"], sel["H"] + sel["B"])
                     if sel else "")
            vs = e.get("vs_fusion_urisk1")
            vs_s = ("%+.4f [%+.4f,%+.4f]" % (vs["diff"], vs["ci_low"], vs["ci_high"]) if vs else "")
            print("  %-13s %7.2f %7.2f %5d %5d %8.2f %8.2f  %-28s %-26s"
                  % (k, e["r1"], e["r5"], e["helped"], e["broken"], e["risk"][0]["trisk"],
                     e["risk"][1]["trisk"], sel_s, vs_s))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("policy")
    ap.add_argument("caches", nargs="+")
    ap.add_argument("--cirr-report-half", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)
    try:
        res = analyse(args.policy, args.caches, args.cirr_report_half, args.bootstrap)
    except CacheError as exc:
        print("CACHE ERROR: %s" % exc, file=sys.stderr)
        return 1
    report(res)
    if args.cirr_report_half:
        bad = check_cirr_anchors(res)
        print()
        if bad:
            print("SELF-CHECK FAILED -- frozen application does not reproduce gate_suite:")
            for b in bad:
                print("  " + b)
            return 1
        print("Self-check passed: all 10 gate_suite report-half R@1 rows reproduced from the frozen file.")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
