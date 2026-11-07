"""Multi-positive-aware cache analysis: CIRCO definitional check, conditional
usefulness, and error taxonomy.

**Why this file exists separately from `rv_cache_analysis.py`.**
`rv_cache_analysis.py` keys every code path off the cache's single
`target_pos` field. On CIRR that is correct -- CIRR has exactly one target
per query. On CIRCO it is NOT: CIRCO queries carry a mean of 4.16 relevant
targets (max 14), recorded in `relevant_gallery_indices` /
`relevant_positions_in_candidates`, which `rv_cache_analysis.py` never
reads. Measured consequences on the 220-query CIRCO cache:

* 78/220 queries (35%) have MORE THAN ONE relevant item inside the top-10
  candidate window, so "did we get the target" is ambiguous under the
  single-target definition;
* 42/220 queries have `target_pos == -1` (the designated target missed the
  window) while a DIFFERENT relevant item sits inside it -- the
  single-target definition scores these as total misses when they are hits;
* net effect: single-target base hit@1 reads 23.18% where the correct
  multi-positive figure is 31.82%, an 8.64pp understatement.

`target_pos != min(relevant_positions_in_candidates)` on 64/220 records.
That is not a cache bug -- `target_pos` faithfully tracks the *designated*
target while `relevant_positions_in_candidates` tracks *all* positives --
but it means single-target code silently answers a different question than
the one CIRCO asks.

This module therefore treats a query as correct at rank k if ANY relevant
item appears at rank k, matching the official CIRCO evaluator's notion of
relevance.

**Scope limit, stated honestly.** The analyses here operate on the cached
top-k candidate window, not the full 123k gallery. They reproduce and
explain the *reranking delta*, which is confined to that window; they are
NOT a reimplementation of the official mAP evaluator and must not be quoted
as headline mAP. Headline mAP comes from `scripts/evaluate_circo.py`, which
uses the official definition over the full gallery.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from revua_cir.significance import (  # noqa: E402
    mcnemar_exact,
    mcnemar_midp,
    paired_bootstrap_ci,
)

# Bootstrap replicate count for reported CIs. 1000 is adequate as a
# diagnostic, but measured seed-to-seed drift at 1000 is ~0.024pp --
# exactly one query on a 4181-query set (100/4181 = 0.0239pp), so the
# endpoints can wobble by a full quantum of the metric. At 10000 the
# CI is stable across seeds and costs ~25s on CPU, which is free
# relative to a GPU eval pass. Paper numbers should not visibly depend
# on Monte-Carlo noise.
N_BOOTSTRAP = 10000



# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_cache(path):
    """Read a JSONL score cache, rejecting malformed/ragged records loudly."""
    recs = []
    required = ("base_score", "verification_score", "candidate_gallery_ids")
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("%s:%d: malformed JSON: %s" % (path, lineno, exc)) from exc
            for key in required:
                if key not in rec:
                    raise ValueError("%s:%d: missing required field %r" % (path, lineno, key))
            n = len(rec["candidate_gallery_ids"])
            if len(rec["base_score"]) != n or len(rec["verification_score"]) != n:
                raise ValueError(
                    "%s:%d: ragged record -- %d candidates but %d base / %d rv scores"
                    % (path, lineno, n, len(rec["base_score"]),
                       len(rec["verification_score"]))
                )
            _validate_record(rec, path, lineno, n)
            recs.append(rec)
    if not recs:
        raise ValueError("%s: no records" % path)
    return recs


def _validate_record(rec, path, lineno, n):
    """Structural checks beyond field presence and length.

    Demonstrated necessity: a `relevant_positions_in_candidates` entry of 7
    in a 3-candidate window loads without complaint and then scores that
    query as a permanent MISS, because no position ever matches. Nothing
    raises; the statistics just come out quietly wrong. Since this analysis
    is used to VALIDATE previous cluster results, silent acceptance of a
    malformed cache is the worst possible failure mode.
    """
    where = "%s:%d" % (path, lineno)

    ids = rec["candidate_gallery_ids"]
    if len(set(ids)) != len(ids):
        raise ValueError("%s: duplicate candidate_gallery_ids -- a candidate "
                         "appears twice in one window" % where)

    tp = rec.get("target_pos")
    if tp is not None and not (tp == -1 or 0 <= tp < n):
        raise ValueError("%s: target_pos %r outside [-1, %d)" % (where, tp, n))

    rp = rec.get("relevant_positions_in_candidates")
    if rp is not None:
        if not all(isinstance(p, int) and not isinstance(p, bool) for p in rp):
            raise ValueError("%s: relevant_positions_in_candidates must be "
                             "integers, got %r" % (where, rp))
        if len(set(rp)) != len(rp):
            raise ValueError("%s: duplicate relevant_positions_in_candidates "
                             "%r" % (where, rp))
        bad = [p for p in rp if not 0 <= p < n]
        if bad:
            raise ValueError("%s: relevant_positions_in_candidates %r outside "
                             "[0, %d) -- these would silently score as misses"
                             % (where, bad, n))
        rel_ids = rec.get("relevant_gallery_indices")
        if rel_ids is not None:
            if len(rp) > len(rel_ids):
                raise ValueError("%s: %d relevant positions but only %d "
                                 "relevant_gallery_indices"
                                 % (where, len(rp), len(rel_ids)))
            rel_set = set(rel_ids)
            mism = [p for p in rp if ids[p] not in rel_set]
            if mism:
                raise ValueError(
                    "%s: candidate(s) at relevant position(s) %r are not in "
                    "relevant_gallery_indices -- the two fields disagree"
                    % (where, mism))


def is_multipositive(recs):
    """True when the cache carries multi-positive ground truth (CIRCO)."""
    return any("relevant_positions_in_candidates" in r for r in recs)


def relevant_positions(rec):
    """All candidate-window positions holding a relevant item.

    Falls back to the single-target field for CIRR-style caches, so one
    code path serves both datasets.
    """
    if "relevant_positions_in_candidates" in rec:
        return set(rec["relevant_positions_in_candidates"])
    tp = rec.get("target_pos", -1)
    return {tp} if tp is not None and tp >= 0 else set()


# --------------------------------------------------------------------------
# orderings
# --------------------------------------------------------------------------
def base_order(rec):
    """Candidate positions by base score, descending.

    Ties break by original position (stable), matching the production
    path's `stable=True` argsort so CPU and CUDA cannot disagree.
    """
    n = len(rec["base_score"])
    return sorted(range(n), key=lambda i: (-rec["base_score"][i], i))


def rank_fuse_order(rec):
    """Production `_rank_fuse`: 2*base_rank + adj_rank, lower is better."""
    n = len(rec["base_score"])
    b_sorted = sorted(range(n), key=lambda i: (-rec["base_score"][i], i))
    r_sorted = sorted(range(n), key=lambda i: (-rec["verification_score"][i], i))
    b_rank = {p: r for r, p in enumerate(b_sorted)}
    r_rank = {p: r for r, p in enumerate(r_sorted)}
    return sorted(range(n), key=lambda i: (2.0 * b_rank[i] + r_rank[i], i))


def _zscore(xs):
    """Population z-score; constant input maps to zeros, never NaN."""
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    sd = math.sqrt(var)
    return [(x - m) / (sd + 1e-8) for x in xs]


def bounded_fusion_order(rec, lam):
    """S_i = z(B_i) + lambda * tanh(z(R_i)), descending."""
    zb = _zscore(rec["base_score"])
    zr = _zscore(rec["verification_score"])
    s = [zb[i] + lam * math.tanh(zr[i]) for i in range(len(zb))]
    return sorted(range(len(s)), key=lambda i: (-s[i], i))


def rv_only_order(rec):
    n = len(rec["verification_score"])
    return sorted(range(n), key=lambda i: (-rec["verification_score"][i], i))


def hit_at_1(order, rel):
    """Multi-positive hit@1: is ANY relevant item ranked first."""
    return 1 if order and order[0] in rel else 0


# --------------------------------------------------------------------------
# section 16 -- multi-positive definitional check
# --------------------------------------------------------------------------
def definitional_check(recs):
    """Quantify how wrong the single-target definition is on this cache.

    This is the audit evidence for why CIRCO needs its own analysis path.
    It does NOT recompute official mAP (that needs the full gallery); it
    measures the definitional error inside the cached window.
    """
    n = len(recs)
    n_rel = [len(r.get("relevant_gallery_indices", [])) for r in recs]
    multi_in_window = sum(1 for r in recs if len(relevant_positions(r)) > 1)
    missed_but_present = sum(
        1 for r in recs
        if r.get("target_pos", -1) < 0 and len(relevant_positions(r)) > 0
    )
    mismatch = 0
    for r in recs:
        rp = relevant_positions(r)
        tp = r.get("target_pos", -1)
        if rp and tp != min(rp):
            mismatch += 1
        elif not rp and tp >= 0:
            mismatch += 1

    st_hit1 = sum(1 for r in recs if r.get("target_pos", -1) == 0)
    mp_hit1 = sum(1 for r in recs if 0 in relevant_positions(r))

    return {
        "n_queries": n,
        "mean_relevant_per_query": (sum(n_rel) / n) if n_rel else 0.0,
        "max_relevant_per_query": max(n_rel) if n_rel else 0,
        "queries_with_multiple_relevant_in_window": multi_in_window,
        "queries_designated_missed_but_relevant_present": missed_but_present,
        "records_target_pos_disagrees_with_relevant_positions": mismatch,
        "single_target_hit_at_1_pct": 100.0 * st_hit1 / n,
        "multipositive_hit_at_1_pct": 100.0 * mp_hit1 / n,
        "understatement_pp": 100.0 * (mp_hit1 - st_hit1) / n,
    }


# --------------------------------------------------------------------------
# section 17 -- conditional usefulness
# --------------------------------------------------------------------------
def _paired_summary(base_hits, new_hits):
    n = len(base_hits)
    if n == 0:
        return {"n": 0, "base_hit_at_1_pct": 0.0, "new_hit_at_1_pct": 0.0,
                "delta_pp": 0.0, "help": 0, "harm": 0, "discordant": 0,
                "mcnemar_exact_p": 1.0, "mcnemar_midp_p": 1.0,
                "ci95_low_pp": float("nan"), "ci95_high_pp": float("nan"),
                "ci_straddles_zero": True}
    help_ = sum(1 for b, x in zip(base_hits, new_hits) if not b and x)
    harm = sum(1 for b, x in zip(base_hits, new_hits) if b and not x)
    base_r1 = 100.0 * sum(base_hits) / n
    new_r1 = 100.0 * sum(new_hits) / n
    boot = paired_bootstrap_ci(base_hits, new_hits, n_replicates=N_BOOTSTRAP, seed=0)
    lo, hi = boot["ci_low"], boot["ci_high"]
    return {
        "n": n,
        "base_hit_at_1_pct": base_r1,
        "new_hit_at_1_pct": new_r1,
        "delta_pp": new_r1 - base_r1,
        "help": help_,
        "harm": harm,
        "discordant": help_ + harm,
        "mcnemar_exact_p": mcnemar_exact(help_, harm),
        "mcnemar_midp_p": mcnemar_midp(help_, harm),
        "ci95_low_pp": lo,
        "ci95_high_pp": hi,
        "ci_straddles_zero": boot["straddles_zero"],
    }


def _auroc_within_window(recs):
    """Within-query POOLED pairwise concordance of verification_score
    separating relevant from non-relevant candidates.

    Restricting comparisons to within a query is deliberate: verification
    scores are not calibrated across queries, so a global pool would measure
    query difficulty rather than verifier discrimination. Ties get explicit
    0.5 credit (Mann-Whitney U convention).

    NOT the mean of per-query AUROCs. This pools every within-query
    positive-negative pair into one concordance ratio, sum(C_q)/sum(N_q), so
    queries contributing more pairs carry more weight. On the CIRCO cache
    the per-query pair count ranges 9..25 (a 3x spread) and the two
    statistics differ measurably: pooled 0.5124 vs macro-average 0.5056.
    Both are reported by `auroc_summary` so neither can be quoted as the
    other.
    """
    conc = 0.0
    total = 0
    for r in recs:
        rel = relevant_positions(r)
        if not rel:
            continue
        scores = r["verification_score"]
        pos = [scores[i] for i in rel]
        neg = [scores[i] for i in range(len(scores)) if i not in rel]
        if not pos or not neg:
            continue
        for p in pos:
            for q in neg:
                total += 1
                if p > q:
                    conc += 1.0
                elif p == q:
                    conc += 0.5
    return (conc / total) if total else None


def auroc_summary(recs):
    """Both weightings of the within-query verifier statistic, plus n.

    Reported together because they answer different questions: the pooled
    figure weights by evidence volume, the macro-average weights every
    query equally. Quoting one under the other's name is the failure this
    guards against.
    """
    pooled = _auroc_within_window(recs)
    per_query = []
    for r in recs:
        rel = relevant_positions(r)
        scores = r["verification_score"]
        pos = [scores[i] for i in rel]
        neg = [scores[i] for i in range(len(scores)) if i not in rel]
        if not pos or not neg:
            continue
        conc = 0.0
        for p in pos:
            for q in neg:
                conc += 1.0 if p > q else (0.5 if p == q else 0.0)
        per_query.append(conc / (len(pos) * len(neg)))
    macro = (sum(per_query) / len(per_query)) if per_query else None
    return {
        "within_query_pooled_pairwise_auroc": pooled,
        "macro_average_per_query_auroc": macro,
        "n_evaluable_queries": len(per_query),
    }


def conditional_usefulness(recs, lam=None):
    """Does verification help WHEN the retriever already found a target?

    Reports candidate coverage C_K, verifier AUROC over relevant vs
    non-relevant candidates, and each fusion delta restricted to covered
    queries -- the subset where reranking can possibly do anything.
    """
    n = len(recs)
    covered = [r for r in recs if len(relevant_positions(r)) > 0]

    orders = {"rank_fusion": rank_fuse_order, "rv_only": rv_only_order}
    if lam is not None:
        orders["bounded_fusion"] = lambda r: bounded_fusion_order(r, lam)

    out = {
        "n_queries": n,
        "candidate_coverage_C_K": len(covered) / n,
        "n_covered": len(covered),
        "verifier_auroc": auroc_summary(recs),
        "variants": {},
    }

    for name, fn in orders.items():
        b_all = [hit_at_1(base_order(r), relevant_positions(r)) for r in recs]
        n_all = [hit_at_1(fn(r), relevant_positions(r)) for r in recs]
        b_cov = [hit_at_1(base_order(r), relevant_positions(r)) for r in covered]
        n_cov = [hit_at_1(fn(r), relevant_positions(r)) for r in covered]
        out["variants"][name] = {
            "overall": _paired_summary(b_all, n_all),
            "covered_only": _paired_summary(b_cov, n_cov),
        }
    return out


# --------------------------------------------------------------------------
# section 18 -- error taxonomy
# --------------------------------------------------------------------------
def _profile_rows(rows):
    """Mean covariates for a taxonomy bucket, for 'when does it help'."""
    if not rows:
        return {"mean_base_margin": None, "mean_rv_margin": None,
                "mean_n_relevant": None, "mean_top1_base_score": None,
                "mean_rv_spread": None, "mean_target_base_margin": None,
                "mean_target_rv_margin": None, "mean_target_base_rank": None,
                "mean_target_rv_rank": None, "n_with_target_margin": 0}

    def _margin(vals):
        s = sorted(vals, reverse=True)
        return (s[0] - s[1]) if len(s) > 1 else 0.0

    def _target_margin(r, key):
        """Best-relevant minus best-non-relevant, on `key`'s scale.

        This is the quantity that actually explains a rescue or a break.
        The generic top1-vs-top2 margin does NOT: on 57/156 (36.5%) of
        covered CIRCO queries both top-1 and top-2 are non-relevant, so
        that margin compares two wrong answers to each other. Returns None
        when the window holds no relevant item (nothing to rescue) or only
        relevant items (nothing to displace it).
        """
        rel = relevant_positions(r)
        vals = r[key]
        pos = [vals[i] for i in rel if 0 <= i < len(vals)]
        neg = [vals[i] for i in range(len(vals)) if i not in rel]
        if not pos or not neg:
            return None
        return max(pos) - max(neg)

    def _target_rank(r, key):
        """Rank of the best-scoring relevant candidate under `key` (0-based)."""
        rel = relevant_positions(r)
        if not rel:
            return None
        order = sorted(range(len(r[key])), key=lambda i: (-r[key][i], i))
        for pos, cand in enumerate(order):
            if cand in rel:
                return pos
        return None

    bm = [_margin(r["base_score"]) for r in rows]
    rm = [_margin(r["verification_score"]) for r in rows]
    tbm = [v for v in (_target_margin(r, "base_score") for r in rows) if v is not None]
    trm = [v for v in (_target_margin(r, "verification_score") for r in rows) if v is not None]
    tbr = [v for v in (_target_rank(r, "base_score") for r in rows) if v is not None]
    trr = [v for v in (_target_rank(r, "verification_score") for r in rows) if v is not None]
    nr = [len(r.get("relevant_gallery_indices", [])) or len(relevant_positions(r))
          for r in rows]
    t1 = [max(r["base_score"]) for r in rows]
    sp = [max(r["verification_score"]) - min(r["verification_score"]) for r in rows]

    def mean(v):
        return sum(v) / len(v)

    return {
        "mean_base_margin": mean(bm),
        "mean_rv_margin": mean(rm),
        "mean_n_relevant": mean(nr),
        "mean_top1_base_score": mean(t1),
        "mean_rv_spread": mean(sp),
        # target-relative: best-relevant minus best-non-relevant
        "mean_target_base_margin": mean(tbm) if tbm else None,
        "mean_target_rv_margin": mean(trm) if trm else None,
        "mean_target_base_rank": mean(tbr) if tbr else None,
        "mean_target_rv_rank": mean(trr) if trr else None,
        "n_with_target_margin": len(tbm),
    }


def error_taxonomy(recs, variant="rank_fusion", lam=None):
    """Classify every query into Type A/B/C/D and profile each type.

    A: base wrong -> fused right  (verification rescued it)
    B: base right -> fused wrong  (verification broke it)
    C: both wrong
    D: both right
    """
    if variant == "rank_fusion":
        fn = rank_fuse_order
    elif variant == "rv_only":
        fn = rv_only_order
    elif variant == "bounded_fusion":
        if lam is None:
            raise ValueError("bounded_fusion requires lam")

        def fn(r):
            return bounded_fusion_order(r, lam)
    else:
        raise ValueError("unknown variant %r" % variant)

    buckets = {"A": [], "B": [], "C": [], "D": []}
    for r in recs:
        rel = relevant_positions(r)
        b = hit_at_1(base_order(r), rel)
        x = hit_at_1(fn(r), rel)
        if b and x:
            key = "D"
        elif not b and x:
            key = "A"
        elif b and not x:
            key = "B"
        else:
            key = "C"
        buckets[key].append(r)

    profile = {}
    for key, rows in buckets.items():
        entry = {"count": len(rows), "pct": 100.0 * len(rows) / len(recs)}
        entry.update(_profile_rows(rows))
        profile[key] = entry
    return {"variant": variant, "lam": lam, "n_queries": len(recs), "types": profile}


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Multi-positive-aware cache analysis (CIRCO-safe).")
    ap.add_argument("--cache", required=True, help="path to a JSONL score cache")
    ap.add_argument("--lam", type=float, default=None,
                    help="lambda for bounded fusion; omit to skip that variant")
    ap.add_argument("--json-out", default=None, help="write full results as JSON")
    args = ap.parse_args(argv)

    recs = load_cache(args.cache)
    mp = is_multipositive(recs)

    print("=" * 76)
    print("Multi-positive cache analysis: %s" % args.cache)
    print("records: %d   multi-positive ground truth: %s" % (len(recs), mp))
    print("=" * 76)

    results = {"cache": str(args.cache), "multipositive": mp}

    if mp:
        print("\n--- 16. Multi-positive definitional check ---")
        chk = definitional_check(recs)
        results["definitional_check"] = chk
        for k, v in chk.items():
            if isinstance(v, float):
                print("  %-55s %.4f" % (k, v))
            else:
                print("  %-55s %s" % (k, v))
        print("\n  NOTE: headline mAP comes from scripts/evaluate_circo.py")
        print("        (official evaluator, full gallery). The figures above")
        print("        describe the cached top-k window only.")

    print("\n--- 17. Conditional usefulness ---")
    cu = conditional_usefulness(recs, lam=args.lam)
    results["conditional_usefulness"] = cu
    print("  candidate coverage C_K            : %.4f (%d/%d)"
          % (cu["candidate_coverage_C_K"], cu["n_covered"], cu["n_queries"]))
    au = cu["verifier_auroc"]
    if au["within_query_pooled_pairwise_auroc"] is None:
        print("  verifier statistic                : n/a (no evaluable query)")
    else:
        print("  within-query pooled AUROC (all neg): %.4f  (weights by pair count)"
              % au["within_query_pooled_pairwise_auroc"])
        print("    NOTE: this scores the relevant candidate against ALL non-relevant")
        print("    candidates in the window. evaluate.py's logged `verification_auroc`")
        print("    instead uses ONE deliberately hard negative (the highest-ranked")
        print("    incorrect candidate), so it reads LOWER -- 0.5735 vs 0.7277 on")
        print("    CIRR. Both are valid; they answer different questions. Do not")
        print("    quote one under the other's name.")
        print("  macro-average per-query AUROC     : %.4f  (weights queries equally)"
              % au["macro_average_per_query_auroc"])
        print("  evaluable queries                 : %d"
              % au["n_evaluable_queries"])
    for name, d in cu["variants"].items():
        print("\n  [%s]" % name)
        for scope in ("overall", "covered_only"):
            s = d[scope]
            print("    %-13s n=%5d  base %6.2f -> %6.2f  delta %+6.2f pp  "
                  "help %4d harm %4d  p=%.4g  CI[%+.2f,%+.2f]"
                  % (scope, s["n"], s["base_hit_at_1_pct"], s["new_hit_at_1_pct"],
                     s["delta_pp"], s["help"], s["harm"], s["mcnemar_exact_p"],
                     s["ci95_low_pp"], s["ci95_high_pp"]))

    print("\n--- 18. Error taxonomy ---")
    variants = ["rank_fusion"] + (["bounded_fusion"] if args.lam is not None else [])
    results["error_taxonomy"] = {}
    for v in variants:
        tax = error_taxonomy(recs, variant=v, lam=args.lam)
        results["error_taxonomy"][v] = tax
        suffix = (" lam=%s" % args.lam) if v == "bounded_fusion" else ""
        print("\n  [%s]%s" % (v, suffix))
        print("    %-6s %6s %7s %10s %9s %7s %11s %10s %9s"
              % ("type", "count", "pct", "base_marg", "rv_marg", "n_rel",
                 "tgt_b_marg", "tgt_r_marg", "tgt_rank"))
        for key in ("A", "B", "C", "D"):
            t = tax["types"][key]
            if t["count"] == 0:
                print("    %-6s %6d %6.2f%%        --" % (key, t["count"], t["pct"]))
                continue

            def _f(v, w=10, p=4):
                return ("%*.*f" % (w, p, v)) if v is not None else ("%*s" % (w, "n/a"))

            print("    %-6s %6d %6.2f%% %10.4f %9.4f %7.2f %11s %10s %9s"
                  % (key, t["count"], t["pct"], t["mean_base_margin"],
                     t["mean_rv_margin"], t["mean_n_relevant"],
                     _f(t["mean_target_base_margin"], 11),
                     _f(t["mean_target_rv_margin"], 10),
                     _f(t["mean_target_base_rank"], 9, 2)))
        print("    A=rescued  B=broken  C=both wrong  D=both right")
        print("    tgt_* = best-RELEVANT minus best-non-relevant (the quantity that")
        print("            explains a rescue/break); base_marg/rv_marg are the")
        print("            generic top1-vs-top2 margins, which can compare two")
        print("            NON-relevant candidates (measured on CIRCO: 36.5%% of")
        print("            covered queries; CIRR has one target so it differs).")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print("\nwrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
