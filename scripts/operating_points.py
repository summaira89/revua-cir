"""Matched operating points and acceptance accounting for the frozen gates. CPU only.

Two questions a reader of the gate results can fairly ask.

1. Does the learned gate do more than lower the fusion weight? The gate is
   compared with the same fusion rule at weights matched to the gate on the
   TUNE half, where the gate itself was fitted:
     matched-damage  tune-half broken <= the gate's, maximising tune helped
     matched-gain    tune-half helped >= the gate's, minimising tune broken
   ties go to the smaller weight. The weight grid is 101 evenly spaced values
   from 0 to the rule's frozen weight, which contains the frozen re-weighted
   control. Weights are then frozen: evaluated on the CIRR report half, or
   applied unchanged to other caches with --weights.

2. How often does each policy intervene? For every policy: queries whose
   returned top-1 differs from the base top-1 ("accepted"), split into helped,
   broken and neutral (wrong to wrong; right to right is possible only with
   multiple positives).

Contrasts are against the learned gate, by paired bootstrap over queries:
URisk at alpha=1 and R@1.

Usage
-----
  python scripts/operating_points.py scripts/frozen_policy.json CIRR_CACHE --cirr --json ../op_cirr.json
  python scripts/operating_points.py scripts/frozen_policy.json FIQ_CACHE [FIQ_CACHE ...] --weights ../op_cirr.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apply_frozen_policy import (  # noqa: E402
    CIRR_REPORT_ANCHORS, FUSIONS, _order, hit_at, logistic_scorer,
)
from fusion_controls import CacheError, load_records, relevant_set, split_indices  # noqa: E402
from fusion_suite import urisk_trisk  # noqa: E402
from gate_suite import paired_mean_bootstrap, query_view, risk_x  # noqa: E402

GRID_STEPS = 100
ALPHAS = (1.0, 5.0)
GATES = ("gate_base", "gate_vgap", "gate_learned")


def prepare(recs, rels, spec: dict, features) -> List[dict]:
    """Per-query quantities that do not depend on the policy."""
    fn = FUSIONS[spec["fusion"]]
    lr = logistic_scorer(spec["logistic"], features)
    out = []
    for rec, rel in zip(recs, rels):
        b, v = rec["base_score"], rec["verification_score"]
        view = query_view(rec, rel, fn, spec["param"])
        out.append({"b": b, "v": v, "rel": rel, "ob": _order(b),
                    "of": _order(fn(b, v, spec["param"])), "view": view,
                    "lr": lr(view["feat"])})
    return out


def weight_policy(prep: Sequence[dict], fn, w: float):
    """Fusion at weight w: per-query hit@1 and whether the top-1 moved."""
    hits, moved = [], []
    for q in prep:
        o = _order(fn(q["b"], q["v"], w))
        hits.append(hit_at(o, q["rel"], 1))
        moved.append(o[0] != q["ob"][0])
    return hits, moved


def gate_policy(prep: Sequence[dict], spec: dict, gate: str):
    th = spec["thresholds"]
    accept = {
        "gate_base": lambda q: q["view"]["feat"]["base_margin"] <= th["base"],
        "gate_vgap": lambda q: q["view"]["feat"]["ver_gap"] >= th["vgap"],
        "gate_learned": lambda q: q["lr"] >= th["learned"],
    }[gate]
    hits, moved = [], []
    for q in prep:
        take = q["view"]["changed"] and accept(q)
        o = q["of"] if take else q["ob"]
        hits.append(hit_at(o, q["rel"], 1))
        moved.append(o[0] != q["ob"][0])
    return hits, moved


def counts(hits: Sequence[int], base: Sequence[int], moved: Sequence[bool]) -> dict:
    helped = sum(1 for h, b in zip(hits, base) if h and not b)
    broken = sum(1 for h, b in zip(hits, base) if b and not h)
    accepted = sum(1 for m in moved if m)
    if any((h != b) and not m for h, b, m in zip(hits, base, moved)):
        raise AssertionError("an outcome changed without the top-1 moving")
    return {"helped": helped, "broken": broken, "accepted": accepted,
            "neutral": accepted - helped - broken}


def grid_for(param: float) -> List[float]:
    return [round(param * i / GRID_STEPS, 10) for i in range(GRID_STEPS + 1)]


def select_matched(prep_tune, fn, grid, target_h: int, target_b: int) -> dict:
    """Weights matched to the gate's tune-half damage and gain."""
    base = [hit_at(q["ob"], q["rel"], 1) for q in prep_tune]
    rows = []
    for w in grid:
        hits, moved = weight_policy(prep_tune, fn, w)
        c = counts(hits, base, moved)
        rows.append((w, c["helped"], c["broken"]))
    dmg = [r for r in rows if r[2] <= target_b]
    gain = [r for r in rows if r[1] >= target_h]
    md = max(dmg, key=lambda r: (r[1], -r[0])) if dmg else None
    mg = min(gain, key=lambda r: (r[2], r[0])) if gain else None
    return {"matched_damage": md[0] if md else None,
            "matched_gain": mg[0] if mg else None,
            "tune_target": {"helped": target_h, "broken": target_b},
            "tune_matched_damage": {"helped": md[1], "broken": md[2]} if md else None,
            "tune_matched_gain": {"helped": mg[1], "broken": mg[2]} if mg else None}


def evaluate_policies(prep, spec: dict, weights: dict, n_rep: int) -> Dict[str, dict]:
    fn = FUSIONS[spec["fusion"]]
    base = [hit_at(q["ob"], q["rel"], 1) for q in prep]
    pol = {"base": (base, [False] * len(base), None)}
    for name, w in (("fusion", spec["param"]), ("weaker", spec["weaker_param"]),
                    ("matched_damage", weights.get("matched_damage")),
                    ("matched_gain", weights.get("matched_gain"))):
        if w is not None:
            h, m = weight_policy(prep, fn, w)
            pol[name] = (h, m, w)
    for g in GATES:
        h, m = gate_policy(prep, spec, g)
        pol[g] = (h, m, None)

    n = len(base)
    ref_h = pol["gate_learned"][0]
    ref_x = risk_x(ref_h, base, 1.0)
    out = {}
    for name, (h, m, w) in pol.items():
        e = dict(counts(h, base, m), weight=w, r1=100.0 * sum(h) / n,
                 accept_rate=100.0 * sum(1 for x in m if x) / n,
                 risk=[urisk_trisk(h, base, a) for a in ALPHAS])
        if name != "gate_learned":
            e["learned_minus_this_urisk1"] = paired_mean_bootstrap(
                ref_x, risk_x(h, base, 1.0), n_rep)
            r1 = paired_mean_bootstrap(ref_h, h, n_rep)
            e["learned_minus_this_r1_pp"] = {k: (100.0 * v if k != "p_two_sided" else v)
                                             for k, v in r1.items()}
        out[name] = e
    return out


def analyse(policy_path: str, caches: Sequence[str], cirr: bool,
            weights_path: Optional[str], n_rep: int) -> dict:
    policy = json.load(open(policy_path, encoding="utf-8"))
    recs, rels, sources = [], [], []
    for path in caches:
        r, multi = load_records(path)
        recs.extend(r)
        rels.extend(relevant_set(x, multi) for x in r)
        sources.append({"path": path, "n": len(r), "multipositive": multi})
    if cirr and len(caches) != 1:
        raise CacheError("--cirr takes exactly one CIRR cache")
    if cirr:
        tune_idx, eval_idx = split_indices(len(recs))
    else:
        tune_idx, eval_idx = [], list(range(len(recs)))
        if weights_path is None:
            raise CacheError("non-CIRR caches need --weights from a --cirr run")
        frozen_w = json.load(open(weights_path, encoding="utf-8"))["matched_weights"]

    res = {"policy": policy_path, "sources": sources, "n_eval": len(eval_idx),
           "mode": "cirr_report_half" if cirr else "frozen_weights",
           "matched_weights": {}, "rules": {}}
    for rname, spec in policy["rules"].items():
        fn = FUSIONS[spec["fusion"]]
        if cirr:
            prep_t = prepare([recs[i] for i in tune_idx], [rels[i] for i in tune_idx],
                             spec, policy["features"])
            base_t = [hit_at(q["ob"], q["rel"], 1) for q in prep_t]
            gh, gm = gate_policy(prep_t, spec, "gate_learned")
            gc = counts(gh, base_t, gm)
            w = select_matched(prep_t, fn, grid_for(spec["param"]), gc["helped"], gc["broken"])
        else:
            w = frozen_w[rname]
        res["matched_weights"][rname] = w
        prep_e = prepare([recs[i] for i in eval_idx], [rels[i] for i in eval_idx],
                         spec, policy["features"])
        res["rules"][rname] = evaluate_policies(prep_e, spec, w, n_rep)
    return res


def check_cirr_anchors(res: dict) -> List[str]:
    bad = []
    for (rname, k), want in CIRR_REPORT_ANCHORS.items():
        got = round(res["rules"][rname][k]["r1"], 2)
        if got != want:
            bad.append("%s/%s R@1 %.2f, expected %.2f" % (rname, k, got, want))
    return bad


def report(res: dict) -> None:
    print("=" * 128)
    print("OPERATING POINTS  %s  n=%d  (%s)" % (
        ", ".join(os.path.basename(s["path"]) for s in res["sources"]),
        res["n_eval"], res["mode"]))
    print("=" * 128)
    for rname, rows in res["rules"].items():
        w = res["matched_weights"][rname]
        print("\n%s  matched weights: damage %s, gain %s  (tune target %s)" % (
            rname, w["matched_damage"], w["matched_gain"], w.get("tune_target")))
        print("  %-15s %7s %7s %5s %5s %6s %5s %7s %7s %7s  %-30s %-26s" % (
            "policy", "weight", "R@1", "help", "brk", "accept", "neut", "acc%",
            "TRisk1", "TRisk5", "learned-this URisk1 [CI] p", "learned-this R@1pp [CI]"))
        for name, e in rows.items():
            u = e.get("learned_minus_this_urisk1")
            r = e.get("learned_minus_this_r1_pp")
            us = "%+.4f [%+.4f,%+.4f] %.4f" % (u["diff"], u["ci_low"], u["ci_high"],
                                               u["p_two_sided"]) if u else ""
            rs = "%+.2f [%+.2f,%+.2f]" % (r["diff"], r["ci_low"], r["ci_high"]) if r else ""
            print("  %-15s %7s %7.3f %5d %5d %6d %5d %7.2f %7.2f %7.2f  %-30s %-26s" % (
                name, "-" if e["weight"] is None else "%.4g" % e["weight"], e["r1"],
                e["helped"], e["broken"], e["accepted"], e["neutral"], e["accept_rate"],
                e["risk"][0]["trisk"], e["risk"][1]["trisk"], us, rs))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("policy")
    ap.add_argument("caches", nargs="+")
    ap.add_argument("--cirr", action="store_true",
                    help="select matched weights on the tune half, report on the report half")
    ap.add_argument("--weights", default=None, help="JSON from a --cirr run")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)
    try:
        res = analyse(args.policy, args.caches, args.cirr, args.weights, args.bootstrap)
    except CacheError as exc:
        print("CACHE ERROR: %s" % exc, file=sys.stderr)
        return 1
    report(res)
    if args.cirr:
        bad = check_cirr_anchors(res)
        print()
        if bad:
            print("SELF-CHECK FAILED -- does not reproduce gate_suite's report-half rows:")
            for b in bad:
                print("  " + b)
            return 1
        print("Self-check passed: all 10 gate_suite report-half R@1 rows reproduced.")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
