"""Fusion-rule controls on cached scores: is the gain the signal or the rule?

The question
------------
The manuscript reports that bounded fusion gains significantly where rank
fusion does not, and concludes that the integration rule matters. An external
review raised two objections to that conclusion as stated:

  (a) it compares one significant result against one non-significant one,
      which does not establish that the two rules differ from each other;
  (b) no control rules out the possibility that per-query normalisation and
      the fusion arithmetic produce apparent gains on their own.

This script answers both from the cached scores, with no GPU.

The rules compared
------------------
All operate on the same cached base scores B and verification scores R, and
all are given the same tuning budget: every rule with a free parameter has it
selected on the tune half and evaluated on the report half.

  base            B alone. The floor every rule must beat.
  bounded         z(B) + lam*tanh(z(R)).            The deployed rule.
  linear          z(B) + lam*z(R).                  Bounded, without tanh --
                  isolates what the saturation contributes.
  rank            -(2*base_rank + adj_rank).        The rank-based rule the
                  project first deployed; parameter-free, so its "tuning" is
                  a no-op, which is itself worth stating.

The controls
------------
  shuffled        R permuted WITHIN each query's own window. This is the
                  control the review asks for. Permuting within the window
                  preserves that query's R distribution exactly -- the same
                  values, the same z-scores, the same tanh outputs -- and
                  destroys only which candidate each score belongs to. Any
                  gain that survives is produced by the arithmetic rather
                  than by the signal.
  noise           R replaced by uniform noise. A coarser check: it changes
                  the distribution as well as the correspondence, so it
                  cannot separate the two, and is reported only because a
                  reader may ask for it.

Permuting ACROSS queries was considered and rejected: it changes each
window's score distribution at the same time as the correspondence, so a null
result would not say which of the two mattered.

What this script does not do
----------------------------
It does not re-tune the checkpoint, and it inherits whatever selection
history the cache carries. The tune/report split (seed 12345) isolates
lambda selection only; it does not make the report half untouched by
checkpoint selection, and the manuscript says so.

Usage
-----
  python -m revua_cir.scripts.fusion_controls CACHE.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

SPLIT_SEED = 12345        # the project's frozen tune/report split
SHUFFLE_SEED = 20261005   # distinct stream for the control permutations
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 54321

# The same grid the lambda sweep used, so no rule gets a finer search than
# another. Rank fusion has no free parameter and simply ignores it.
LAMBDA_GRID: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5,
                                  2.0, 3.0, 4.0, 6.0, 8.0)


class CacheError(ValueError):
    pass


def _zscore(xs: Sequence[float]) -> List[float]:
    """Population z-score with the project's 1e-8 floor.

    Identical to multipositive_analysis._zscore and
    window_width_analysis._zscore; the three must not drift apart.
    """
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    sd = math.sqrt(var)
    return [(x - m) / (sd + 1e-8) for x in xs]


def _ranks_desc(xs: Sequence[float]) -> List[int]:
    """0-based rank of each element under descending sort, ties by index.

    Rank 0 is the highest-scoring candidate. The project's `_rank_fuse` uses
    this convention; stating it explicitly here because an off-by-one would
    silently invert the rule.
    """
    order = sorted(range(len(xs)), key=lambda i: (-xs[i], i))
    out = [0] * len(xs)
    for pos, i in enumerate(order):
        out[i] = pos
    return out


def _is_index(v) -> bool:
    """A genuine integer (bool is an int subclass in Python, so exclude it)."""
    return isinstance(v, int) and not isinstance(v, bool)


def _validate(rec: dict, path: str, lineno: int) -> None:
    where = "%s:%d" % (os.path.basename(path), lineno)
    if not isinstance(rec, dict):
        raise CacheError("%s: record is not a JSON object" % where)
    for f in ("candidate_gallery_ids", "base_score", "verification_score"):
        if not isinstance(rec.get(f), list):
            raise CacheError("%s: missing or non-list field %r" % (where, f))
    n = len(rec["candidate_gallery_ids"])
    if n == 0:
        raise CacheError("%s: empty candidate window" % where)
    if not all(_is_index(c) for c in rec["candidate_gallery_ids"]):
        raise CacheError("%s: candidate_gallery_ids must be integers" % where)
    if len(set(rec["candidate_gallery_ids"])) != n:
        raise CacheError("%s: duplicate candidate_gallery_ids" % where)
    for f in ("base_score", "verification_score"):
        if len(rec[f]) != n:
            raise CacheError("%s: %s has %d entries for %d candidates"
                             % (where, f, len(rec[f]), n))
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) for v in rec[f]):
            raise CacheError("%s: %s has a non-finite value" % (where, f))
    tp = rec.get("target_pos")
    if tp is not None and not (_is_index(tp) and (tp == -1 or 0 <= tp < n)):
        raise CacheError("%s: target_pos %r is not an integer in [-1, %d)" % (where, tp, n))
    if "relevant_positions_in_candidates" in rec:
        # Presence of the key marks a multi-positive cache, so its value must
        # be a list (possibly empty: no relevant candidate in this window).
        rp = rec["relevant_positions_in_candidates"]
        if not isinstance(rp, list) or not all(_is_index(p) and 0 <= p < n for p in rp):
            raise CacheError("%s: relevant positions must be integers in [0, %d)" % (where, n))
        if len(set(rp)) != len(rp):
            raise CacheError("%s: duplicate relevant positions" % where)


def load_records(path: str) -> Tuple[List[dict], bool]:
    recs: List[dict] = []
    flags: List[bool] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CacheError("%s:%d: malformed JSON (%s)"
                                 % (os.path.basename(path), lineno, exc))
            _validate(rec, path, lineno)
            recs.append(rec)
            flags.append("relevant_positions_in_candidates" in rec)
    if not recs:
        raise CacheError("%s: no records" % path)
    if len(set(flags)) != 1:
        raise CacheError("%s: cache mixes single-target and multi-positive "
                         "records" % path)
    return recs, flags[0]


def relevant_set(rec: dict, multipositive: bool) -> set:
    if multipositive:
        return set(rec["relevant_positions_in_candidates"])
    tp = rec.get("target_pos", -1)
    return {tp} if tp is not None and tp >= 0 else set()


# ---------------------------------------------------------------------------
# the rules
# ---------------------------------------------------------------------------

def rule_base(b: Sequence[float], r: Sequence[float], lam: float) -> List[float]:
    return list(b)


def rule_bounded(b: Sequence[float], r: Sequence[float],
                 lam: float) -> List[float]:
    zb, zr = _zscore(b), _zscore(r)
    return [zb[i] + lam * math.tanh(zr[i]) for i in range(len(zb))]


def rule_linear(b: Sequence[float], r: Sequence[float],
                lam: float) -> List[float]:
    """Bounded fusion without the saturation, to isolate what tanh adds."""
    zb, zr = _zscore(b), _zscore(r)
    return [zb[i] + lam * zr[i] for i in range(len(zb))]


def rule_rank(b: Sequence[float], r: Sequence[float],
              lam: float) -> List[float]:
    """The project's `_rank_fuse`: -(2*base_rank + adj_rank), descending.

    Parameter-free, so `lam` is ignored. Negated because the caller sorts
    descending and low ranks are better.
    """
    rb, rr = _ranks_desc(b), _ranks_desc(r)
    return [-(2.0 * rb[i] + rr[i]) for i in range(len(rb))]


RULES: Dict[str, Tuple[Callable, bool]] = {
    # name: (function, has_free_parameter)
    "base": (rule_base, False),
    "bounded": (rule_bounded, True),
    "linear": (rule_linear, True),
    "rank": (rule_rank, False),
}


def hit_at_1(scores: Sequence[float], rel: set) -> int:
    if not rel:
        return 0
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return 1 if order and order[0] in rel else 0


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

N_SHUFFLE_REPEATS = 25   # one permutation is not a control; see analyse()


def shuffled_within_window(recs: Sequence[dict], seed: int) -> List[dict]:
    """R permuted inside each query's own window.

    Preserves the multiset of verification scores per query exactly, so the
    z-scores and tanh outputs are unchanged as a set; only the pairing with
    candidates is destroyed. A rule that still gains under this control is
    gaining from the arithmetic, not the signal.
    """
    rng = random.Random(seed)
    out = []
    for rec in recs:
        new = dict(rec)
        v = list(rec["verification_score"])
        rng.shuffle(v)
        new["verification_score"] = v
        out.append(new)
    return out


def noise_verifier(recs: Sequence[dict], seed: int) -> List[dict]:
    """R replaced by uniform noise on the observed global range.

    Coarser than the within-window shuffle: it changes the distribution as
    well as the correspondence, so a null here cannot say which mattered.
    Reported as a secondary check only.
    """
    rng = random.Random(seed)
    allv = [s for rec in recs for s in rec["verification_score"]]
    lo, hi = min(allv), max(allv)
    out = []
    for rec in recs:
        new = dict(rec)
        new["verification_score"] = [rng.uniform(lo, hi)
                                     for _ in rec["verification_score"]]
        out.append(new)
    return out


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def split_indices(n: int, seed: int = SPLIT_SEED) -> Tuple[List[int], List[int]]:
    """The project's frozen tune/report split: a seeded shuffle, halved."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    half = n // 2
    return sorted(idx[:half]), sorted(idx[half:])


def evaluate(recs: Sequence[dict], rel: Sequence[set], rule: Callable,
             lam: float, subset: Sequence[int]) -> Tuple[int, List[int]]:
    hits = 0
    per_query = []
    for i in subset:
        rec = recs[i]
        s = rule(rec["base_score"], rec["verification_score"], lam)
        h = hit_at_1(s, rel[i])
        hits += h
        per_query.append(h)
    return hits, per_query


def tune_lambda(recs: Sequence[dict], rel: Sequence[set], rule: Callable,
                has_param: bool, tune: Sequence[int]) -> float:
    """Pick lambda on the tune half. Parameter-free rules return 0.0.

    Ties are broken toward the SMALLER lambda, which is the conservative
    choice: it leans on the verifier less.
    """
    if not has_param:
        return 0.0
    best_lam, best_hits = LAMBDA_GRID[0], -1
    for lam in LAMBDA_GRID:
        hits, _ = evaluate(recs, rel, rule, lam, tune)
        if hits > best_hits:
            best_lam, best_hits = lam, hits
    return best_lam


def paired_bootstrap(a: Sequence[int], b: Sequence[int],
                     n_rep: int = N_BOOTSTRAP,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    """CI for mean(a) - mean(b), resampling QUERIES with replacement.

    Paired: the same resampled indices are used for both arms, so the shared
    query-difficulty component cancels. This is the comparison the review
    asks for -- rule against rule directly, rather than each against base.
    """
    n = len(a)
    if n != len(b):
        raise ValueError("arms differ in length (%d vs %d)" % (n, len(b)))
    obs = (sum(a) - sum(b)) / n
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_rep):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(a[i] - b[i] for i in idx) / n)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]
    tail = (sum(1 for d in deltas if d <= 0.0) if obs >= 0
            else sum(1 for d in deltas if d >= 0.0))
    p = min(1.0, 2.0 * (tail + 1) / (len(deltas) + 1))
    return {"delta_pp": 100.0 * obs, "ci_low": 100.0 * lo,
            "ci_high": 100.0 * hi, "p_two_sided": p}


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> dict:
    """Exact binomial McNemar with mid-p, matching the project's protocol."""
    if len(a) != len(b):
        raise ValueError("arms differ in length (%d vs %d)" % (len(a), len(b)))
    if any(x not in (0, 1) for x in a) or any(y not in (0, 1) for y in b):
        raise ValueError("McNemar needs binary per-query outcomes")
    n01 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n10 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n = n01 + n10
    if n == 0:
        return {"fixed": 0, "broken": 0, "p_exact": 1.0, "p_midp": 1.0}

    def C(n, k):
        return math.comb(n, k)
    k = min(n01, n10)
    tail = sum(C(n, i) for i in range(0, k + 1)) / (2 ** n)
    p_exact = min(1.0, 2.0 * tail)
    at_k = C(n, k) / (2 ** n)
    p_midp = min(1.0, 2.0 * (tail - 0.5 * at_k))
    return {"fixed": n01, "broken": n10, "p_exact": p_exact,
            "p_midp": p_midp}


def analyse(path: str, n_rep: int = N_BOOTSTRAP) -> dict:
    recs, multipositive = load_records(path)
    rel = [relevant_set(r, multipositive) for r in recs]
    tune, report = split_indices(len(recs))

    results: Dict[str, Dict[str, dict]] = {"real": {}}
    per_query: Dict[str, Dict[str, List[int]]] = {"real": {}}

    for rname, (fn, has_param) in RULES.items():
        lam = tune_lambda(recs, rel, fn, has_param, tune)
        hits, pq = evaluate(recs, rel, fn, lam, report)
        results["real"][rname] = {"lambda": lam,
                                  "hit1_pct": 100.0 * hits / len(report),
                                  "n_hits": hits}
        per_query["real"][rname] = pq

    # The controls run over MANY permutations, not one. A single shuffle is
    # itself a random draw: on the CIRCO cache one seed put the shuffled
    # rank rule at 35.45% against the real signal's 30.00%, while the median
    # over 25 seeds was 28.18%. Reporting a single permutation would have
    # manufactured a result, in either direction.
    for vname, maker in (("shuffled", shuffled_within_window),
                         ("noise", noise_verifier)):
        results[vname] = {}
        per_query[vname] = {}
        for rname, (fn, has_param) in RULES.items():
            draws = []   # (R@1 %, repeat index, per-query hits)
            for rep_i in range(N_SHUFFLE_REPEATS):
                vrecs = maker(recs, SHUFFLE_SEED + 97 * rep_i)
                lam = tune_lambda(vrecs, rel, fn, has_param, tune)
                hits, pq = evaluate(vrecs, rel, fn, lam, report)
                draws.append((100.0 * hits / len(report), rep_i, pq))
            draws.sort(key=lambda d: (d[0], d[1]))
            median = draws[len(draws) // 2]
            base_pct = results["real"]["base"]["hit1_pct"]
            results[vname][rname] = {
                "hit1_pct": median[0],
                "hit1_min": draws[0][0],
                "hit1_max": draws[-1][0],
                # Across-permutation summary of the change against base: the
                # descriptive statement the control supports. The paired test
                # below is on the median draw, which was selected using report
                # outcomes, so its interval is conditional on that selection.
                "delta_median": median[0] - base_pct,
                "delta_min": draws[0][0] - base_pct,
                "delta_max": draws[-1][0] - base_pct,
                "n_repeats": N_SHUFFLE_REPEATS,
                "median_repeat": median[1],
            }
            # The paired contrast must use the SAME permutation whose R@1 is
            # reported (the median one); using an arbitrary draw made the
            # contrast and the R@1 column disagree.
            per_query[vname][rname] = median[2]

    # Direct rule-vs-rule comparisons on the report half, which is the
    # comparison the review asks for.
    base_pq = per_query["real"]["base"]
    contrasts = {}
    for rname in ("bounded", "linear", "rank"):
        pq = per_query["real"][rname]
        contrasts["%s vs base" % rname] = {
            **paired_bootstrap(pq, base_pq, n_rep),
            **mcnemar_exact(pq, base_pq),
        }
    contrasts["bounded vs rank"] = {
        **paired_bootstrap(per_query["real"]["bounded"],
                           per_query["real"]["rank"], n_rep),
        **mcnemar_exact(per_query["real"]["bounded"],
                        per_query["real"]["rank"]),
    }
    contrasts["bounded vs linear"] = {
        **paired_bootstrap(per_query["real"]["bounded"],
                           per_query["real"]["linear"], n_rep),
        **mcnemar_exact(per_query["real"]["bounded"],
                        per_query["real"]["linear"]),
    }

    # Each control rule against the same base, to show the gain vanishes.
    control_contrasts = {}
    for vname in ("shuffled", "noise"):
        for rname in ("bounded", "rank"):
            control_contrasts["%s / %s vs base" % (vname, rname)] = \
                paired_bootstrap(per_query[vname][rname], base_pq, n_rep)

    return {
        "path": path,
        "n_queries": len(recs),
        "n_tune": len(tune),
        "n_report": len(report),
        "multipositive": multipositive,
        "lambda_grid": list(LAMBDA_GRID),
        "results": results,
        "contrasts": contrasts,
        "control_contrasts": control_contrasts,
        "n_bootstrap": n_rep,
    }


def report(res: dict) -> None:
    print("=" * 78)
    print("FUSION-RULE CONTROLS")
    print("=" * 78)
    print("cache      : %s" % os.path.basename(res["path"]))
    print("queries    : %d  (tune %d / report %d, seed %d)"
          % (res["n_queries"], res["n_tune"], res["n_report"], SPLIT_SEED))
    print("relevance  : %s" % ("multi-positive" if res["multipositive"]
                               else "single-target"))
    print("lambda grid: %s" % ", ".join("%g" % x for x in res["lambda_grid"]))
    print()
    print("Every rule with a free parameter is tuned on the tune half and")
    print("evaluated on the report half. Rank fusion has no parameter.")
    print()

    print("REPORT-HALF \\rat{1} BY RULE AND VERIFIER VARIANT".replace(
        "\\rat{1}", "R@1"))
    print("  %-17s %8s %10s %10s   %s"
          % ("rule", "real", "shuf med", "noise med", "[shuffled range]"))
    for rname in RULES:
        row = []
        for v in ("real", "shuffled", "noise"):
            d = res["results"][v][rname]
            row.append("%.2f%%" % d["hit1_pct"])
        lam = res["results"]["real"][rname]["lambda"]
        tag = rname if rname == "base" or rname == "rank" else \
            "%s (lam=%g)" % (rname, lam)
        sh = res["results"]["shuffled"][rname]
        print("  %-17s %8s %10s %10s   [%.2f-%.2f]"
              % (tag, row[0], row[1], row[2],
                 sh["hit1_min"], sh["hit1_max"]))
    print()
    print("  'shuffled' permutes R inside each query's own window, so the")
    print("  score distribution is unchanged and only the candidate pairing")
    print("  is destroyed. A rule that still gains there is gaining from the")
    print("  arithmetic rather than the verification signal.")
    print()
    print("  Control change vs base, pp (median and range over %d permutations):"
          % N_SHUFFLE_REPEATS)
    for vname in ("shuffled", "noise"):
        for rname in ("bounded", "rank"):
            d = res["results"][vname][rname]
            print("    %-8s %-8s %+.2f  [%+.2f, %+.2f]" % (
                vname, rname, d["delta_median"], d["delta_min"], d["delta_max"]))

    print()
    print("DIRECT PAIRED CONTRASTS ON THE REPORT HALF")
    print("  %-20s %9s %-20s %9s %7s %7s"
          % ("contrast", "delta", "95% CI", "p(boot)", "fixed", "broken"))
    for name, d in res["contrasts"].items():
        print("  %-20s %+8.2f  [%+6.2f, %+6.2f] %9.4f %7d %7d"
              % (name, d["delta_pp"], d["ci_low"], d["ci_high"],
                 d["p_two_sided"], d["fixed"], d["broken"]))

    print()
    print("CONTROL CONTRASTS (should be indistinguishable from zero)")
    print("  %-26s %9s %-20s %9s" % ("contrast", "delta", "95% CI", "p"))
    for name, d in res["control_contrasts"].items():
        print("  %-26s %+8.2f  [%+6.2f, %+6.2f] %9.4f"
              % (name, d["delta_pp"], d["ci_low"], d["ci_high"],
                 d["p_two_sided"]))
    print("=" * 78)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache")
    ap.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)

    if args.bootstrap < 1:
        print("error: --bootstrap must be >= 1", file=sys.stderr)
        return 2
    try:
        res = analyse(args.cache, n_rep=args.bootstrap)
    except CacheError as exc:
        print("CACHE ERROR: %s" % exc, file=sys.stderr)
        return 1

    report(res)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print()
        print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
