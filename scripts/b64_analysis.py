"""B6.4 analysis: the two pre-registered paired tests (lumiere/B64_PREREGISTRATION.md).

Confirmatory family (Holm, alpha 0.05):
  1. fc+aep vs fc            -- base-retrieval hit@1, per query
  2. noinv vs fc+aep+rv      -- bounded-fusion hit@1 at the frozen lambda
Exploratory (no multiplicity claim):
  * noinv vs fc+aep+rv on base-retrieval hit@1
  * noinv's learned edit weight w, recovered from its cache

Inputs are all produced elsewhere and only paired here, on CPU:
  fc hits      fc_per_query_hits.json      (job 7246)
  fc+aep hits  fc_aep_per_query_hits.json  (scripts/per_query_hits.py)
  caches       rv_score_cache_fc_aep_rv.jsonl (published) and the noinv
               cache (scripts/evaluate_cirr.py --rerank --cache-path)

Every pairing is proved per query by identical target gallery indices, and
the published counts (FC 1505, fc+aep+rv base 1550, bounded 1646) must be
reproduced, before any statistic is computed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from typing import List, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fusion_controls import CacheError, load_records  # noqa: E402
from fc_vs_bounded_paired import PairingError, cache_outcomes, check_alignment  # noqa: E402
from ablation_significance import compare  # noqa: E402

N_QUERIES = 4181
FC_HITS, PUB_BASE_HITS, PUB_BOUNDED_HITS = 1505, 1550, 1646
PUBLISHED_W = 0.488847


def holm_adjusted(pvals: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def load_hits(path: str):
    d = json.load(open(path, encoding="utf-8"))
    hits = [int(x) for x in d["hit_at_1"]]
    targets = [int(t) for t in d["targets"]]
    valid = [int(v) for v in d.get("valid", [1 if t >= 0 else 0 for t in targets])]
    if not (len(hits) == len(targets) == len(valid)):
        raise PairingError("%s: field lengths differ" % path)
    return hits, targets, valid, d.get("query_ids")


def recover_noinv_w(recs: Sequence[dict]):
    """w from cos = (a*recon + w*edit)/(a + w), a = 0.5(1 - w), solved per row:
    w = (recon - cos) / (cos + recon - 2*edit). Returns (median, sd, n, max
    residual of the no-invariant formula at the median w)."""
    ws = []
    for r in recs:
        for c, rc, e in zip(r["cos_sim"], r["recon_cos"], r["edit_cos"]):
            den = c + rc - 2.0 * e
            if abs(den) > 1e-4:
                ws.append((rc - c) / den)
    if not ws:
        return None, None, 0, None
    w = statistics.median(ws)
    sd = statistics.pstdev(ws)
    a = 0.5 * (1.0 - w)
    resid = max(abs(c - (a * rc + w * e) / (a + w))
                for r in recs
                for c, rc, e in zip(r["cos_sim"], r["recon_cos"], r["edit_cos"]))
    return w, sd, len(ws), resid


def full_formula_residual(recs: Sequence[dict], w: float) -> float:
    """Max misfit of the reported three-term score at weight w."""
    return max(abs(c - ((1 - w) * (0.5 * rc + 0.5 * i) + w * e))
               for r in recs
               for c, rc, i, e in zip(r["cos_sim"], r["recon_cos"],
                                      r["invariant_cos"], r["edit_cos"]))


def analyse(fc_hits_path, fcaep_hits_path, pub_cache, noinv_cache, lam):
    fc, fc_t, fc_v, fc_q = load_hits(fc_hits_path)
    fa, fa_t, fa_v, fa_q = load_hits(fcaep_hits_path)
    check_alignment(fc_t, fa_t)
    if fc_q and fa_q and fc_q != fa_q:
        raise PairingError("fc and fc+aep query_ids differ")
    if len(fc) != N_QUERIES or sum(fc) != FC_HITS:
        raise PairingError("FC hits %d/%d, published %d/%d"
                           % (sum(fc), len(fc), FC_HITS, N_QUERIES))

    pub, pub_multi = load_records(pub_cache)
    nov, nov_multi = load_records(noinv_cache)
    if pub_multi or nov_multi or len(pub) != N_QUERIES or len(nov) != N_QUERIES:
        raise PairingError("both caches must be single-target CIRR with %d records"
                           % N_QUERIES)
    pub_base, pub_bnd, pub_t = cache_outcomes(pub, lam)
    nov_base, nov_bnd, nov_t = cache_outcomes(nov, lam)
    check_alignment(pub_t, nov_t)
    check_alignment(fc_t, pub_t)
    if sum(pub_base) != PUB_BASE_HITS or sum(pub_bnd) != PUB_BOUNDED_HITS:
        raise PairingError("published cache gives base %d / bounded %d, expected %d / %d"
                           % (sum(pub_base), sum(pub_bnd), PUB_BASE_HITS, PUB_BOUNDED_HITS))

    t1 = compare(fc, fa, fc_v, fa_v, "FC", "FC+AEP")
    all_valid = [1] * N_QUERIES
    t2 = compare(pub_bnd, nov_bnd, all_valid, all_valid,
                 "FC+AEP+RV bounded", "noinv bounded")
    adj = holm_adjusted([t1["mcnemar_exact_p"], t2["mcnemar_exact_p"]])
    t1["holm_p"], t2["holm_p"] = adj
    t1["holm_reject_0.05"], t2["holm_reject_0.05"] = adj[0] <= 0.05, adj[1] <= 0.05

    expl = compare(pub_base, nov_base, all_valid, all_valid,
                   "FC+AEP+RV base", "noinv base")
    w, w_sd, w_n, w_resid = recover_noinv_w(nov)
    return {"confirmatory": {"fcaep_vs_fc": t1, "noinv_vs_rv_bounded": t2},
            "exploratory": {
                "noinv_vs_rv_base": expl,
                "noinv_learned_w": w, "noinv_w_sd": w_sd, "noinv_w_rows": w_n,
                "noinv_formula_max_residual": w_resid,
                "full_formula_max_residual_on_noinv_cache":
                    full_formula_residual(nov, w) if w is not None else None,
                "published_w": PUBLISHED_W},
            "lam": lam}


def _line(t):
    return ("%-18s %7.3f -> %7.3f  %+.3f pp  fixed %d broken %d  p=%.3g  "
            "CI [%+.3f, %+.3f]" % (t["label_a"] + " vs", t["r1_a_pct"], t["r1_b_pct"],
                                   t["delta_pp"], t["help"], t["harm"],
                                   t["mcnemar_exact_p"], t["ci95_low_pp"], t["ci95_high_pp"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fc-hits", required=True)
    ap.add_argument("--fcaep-hits", required=True)
    ap.add_argument("--published-cache", required=True)
    ap.add_argument("--noinv-cache", required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--json-out", required=True)
    args = ap.parse_args(argv)
    try:
        res = analyse(args.fc_hits, args.fcaep_hits, args.published_cache,
                      args.noinv_cache, args.lam)
    except (PairingError, CacheError) as exc:
        print("ABORT: %s" % exc, file=sys.stderr)
        return 1

    c, e = res["confirmatory"], res["exploratory"]
    print("pairing verified on all %d queries; published counts reproduced" % N_QUERIES)
    print("\nCONFIRMATORY (Holm over 2)")
    for t in (c["fcaep_vs_fc"], c["noinv_vs_rv_bounded"]):
        print("  " + _line(t))
        print("  %18s Holm p = %.3g  -> %s" % ("", t["holm_p"],
              "significant" if t["holm_reject_0.05"] else "not significant"))
    print("\nEXPLORATORY")
    print("  " + _line(e["noinv_vs_rv_base"]))
    if e["noinv_learned_w"] is not None:
        print("  noinv learned w = %.6f (sd %.2g over %d rows; published %.6f)"
              % (e["noinv_learned_w"], e["noinv_w_sd"], e["noinv_w_rows"], e["published_w"]))
        print("  no-invariant formula max residual %.2g | full formula %.2g"
              % (e["noinv_formula_max_residual"],
                 e["full_formula_max_residual_on_noinv_cache"]))
    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print("\nwrote %s" % args.json_out)
    return 0 if not math.isnan(c["fcaep_vs_fc"]["delta_pp"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
