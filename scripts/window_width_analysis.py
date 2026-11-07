"""Does the fusion gain survive a narrower candidate window?

Why this exists
---------------
Verification costs ~150 ms per query against ~21 ms for base retrieval, and
that cost is dominated by encoding and scoring the top-$k$ candidates. If most
of the fusion gain is recovered at $k = 5$, the operating point is materially
cheaper for roughly the same accuracy. If it is not, the trade-off should be
reported rather than left for a reviewer to ask about.

The cached scores answer this for any $k$ up to the cached width without a GPU.

The subtlety that makes this non-trivial
----------------------------------------
Bounded fusion is `S_i = z(B_i) + lambda * tanh(z(R_i))`, where both
z-scores are computed **per query over the candidate window**. Truncating the
window to $k$ therefore changes the normalisation, not just the candidate set:
the mean and standard deviation are recomputed over $k$ items. Scoring the
first $k$ candidates with statistics taken from the full 10 would measure
something that no deployed system could compute, since a system operating at
$k = 5$ never sees candidates 6--10.

This script recomputes the z-scores inside each truncated window, which is
what a system actually running at that width would do. `truncate_rescores`
below is the function that does it, and
`test_truncation_recomputes_zscores` pins the distinction.

Latency model
-------------
Reported latency is a linear interpolation, not a measurement:

    total(k) = base + (k / k_ref) * overhead_ref

with `base = 20.61` ms and `overhead_ref = 157.42` ms at `k_ref = 10`, from
the paper's own benchmark table. The linear form assumes per-candidate
verification cost dominates, which holds because the verifier runs once per
candidate; it ignores fixed per-query overhead inside the verification stage,
so small-$k$ figures are **optimistic**. Labelled as an estimate wherever it
is printed, and excluded from the JSON's measured fields.

Usage
-----
  python -m revua_cir.scripts.window_width_analysis CACHE.jsonl --lam 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import List, Optional, Sequence, Tuple

WIDTHS: Tuple[int, ...] = (1, 2, 3, 5, 7, 10)

# From Table "Latency in milliseconds for a single query" in the manuscript.
BASE_LATENCY_MS = 20.61
OVERHEAD_AT_10_MS = 157.42
REF_WIDTH = 10


class CacheError(ValueError):
    pass


def _zscore(xs: Sequence[float]) -> List[float]:
    """Population z-score; constant input maps to zeros, never NaN.

    Identical to `multipositive_analysis._zscore`, including the 1e-8 floor,
    so the two scripts cannot disagree about the fusion rule.
    """
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    sd = math.sqrt(var)
    return [(x - m) / (sd + 1e-8) for x in xs]


def _validate(rec: dict, path: str, lineno: int) -> None:
    where = "%s:%d" % (os.path.basename(path), lineno)
    for f in ("candidate_gallery_ids", "base_score", "verification_score"):
        if f not in rec:
            raise CacheError("%s: missing field %r" % (where, f))
    n = len(rec["candidate_gallery_ids"])
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
    if tp is not None and not (tp == -1 or 0 <= tp < n):
        raise CacheError("%s: target_pos %r outside [-1, %d)" % (where, tp, n))
    rp = rec.get("relevant_positions_in_candidates")
    if rp is not None and any(not 0 <= p < n for p in rp):
        raise CacheError("%s: relevant position outside [0, %d)" % (where, n))


def load_records(path: str) -> Tuple[List[dict], bool, int]:
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
    widths = {len(r["candidate_gallery_ids"]) for r in recs}
    if len(widths) != 1:
        raise CacheError("%s: records have differing window widths %r"
                         % (path, sorted(widths)))
    return recs, flags[0], widths.pop()


def relevant_set(rec: dict, multipositive: bool) -> set:
    if multipositive:
        return set(rec["relevant_positions_in_candidates"])
    tp = rec.get("target_pos", -1)
    return {tp} if tp is not None and tp >= 0 else set()


def truncate_rescores(rec: dict, k: int) -> Tuple[List[float], List[float]]:
    """The first k candidates' scores, z-scored WITHIN that truncated window.

    This is the whole point of the script: a system operating at width k
    never sees candidates k+1..n, so its per-query normalisation is taken
    over k items. Reusing the full-window statistics would attribute to
    width k an advantage it could not have.
    """
    if k < 1:
        raise ValueError("k must be >= 1, got %d" % k)
    b = rec["base_score"][:k]
    r = rec["verification_score"][:k]
    return _zscore(b), _zscore(r)


def base_order(rec: dict, k: int) -> List[int]:
    b = rec["base_score"][:k]
    return sorted(range(len(b)), key=lambda i: (-b[i], i))


def fused_order(rec: dict, k: int, lam: float) -> List[int]:
    zb, zr = truncate_rescores(rec, k)
    s = [zb[i] + lam * math.tanh(zr[i]) for i in range(len(zb))]
    return sorted(range(len(s)), key=lambda i: (-s[i], i))


def estimated_latency_ms(k: int) -> float:
    """Linear interpolation of the measured benchmark. An ESTIMATE."""
    return BASE_LATENCY_MS + (k / REF_WIDTH) * OVERHEAD_AT_10_MS


def analyse(path: str, lam: float,
            widths: Sequence[int] = WIDTHS) -> dict:
    recs, multipositive, cached_width = load_records(path)

    usable = [k for k in widths if k <= cached_width]
    skipped = [k for k in widths if k > cached_width]

    rel = [relevant_set(r, multipositive) for r in recs]
    n = len(recs)

    rows = []
    for k in usable:
        base_hits = fused_hits = 0
        helped = harmed = 0
        for rec, rset in zip(recs, rel):
            bo = base_order(rec, k)
            fo = fused_order(rec, k, lam)
            b = 1 if bo and bo[0] in rset else 0
            f = 1 if fo and fo[0] in rset else 0
            base_hits += b
            fused_hits += f
            if f and not b:
                helped += 1
            elif b and not f:
                harmed += 1
        rows.append({
            "k": k,
            "base_hit1_pct": 100.0 * base_hits / n,
            "fused_hit1_pct": 100.0 * fused_hits / n,
            "delta_pp": 100.0 * (fused_hits - base_hits) / n,
            "helped": helped,
            "harmed": harmed,
            "net": helped - harmed,
            "estimated_latency_ms": estimated_latency_ms(k),
        })

    ref = next((r for r in rows if r["k"] == cached_width), rows[-1])
    for r in rows:
        r["delta_retained_pct"] = (
            None if abs(ref["delta_pp"]) < 1e-12
            else 100.0 * r["delta_pp"] / ref["delta_pp"])

    return {
        "path": path,
        "lam": lam,
        "n_queries": n,
        "multipositive": multipositive,
        "cached_width": cached_width,
        "widths_skipped": skipped,
        "reference_k": ref["k"],
        "rows": rows,
    }


def report(res: dict) -> None:
    print("=" * 78)
    print("CANDIDATE-WINDOW WIDTH VERSUS FUSION GAIN")
    print("=" * 78)
    print("cache        : %s" % os.path.basename(res["path"]))
    print("queries      : %d" % res["n_queries"])
    print("relevance    : %s" % ("multi-positive" if res["multipositive"]
                                 else "single-target"))
    print("lambda       : %s" % res["lam"])
    print("cached width : %d" % res["cached_width"])
    if res["widths_skipped"]:
        print("skipped      : %s (wider than the cache; needs a GPU pass)"
              % ", ".join(str(k) for k in res["widths_skipped"]))
    print()
    print("  z-scores are recomputed within each truncated window, which is")
    print("  what a system operating at that width would do.")
    print()
    print("  %3s %10s %10s %9s %7s %7s %9s %12s"
          % ("k", "base@1", "fused@1", "delta", "help", "harm", "retained",
             "est. latency"))
    for r in res["rows"]:
        ret = r["delta_retained_pct"]
        print("  %3d %9.2f%% %9.2f%% %+8.2f %7d %7d %8s %10.1f ms"
              % (r["k"], r["base_hit1_pct"], r["fused_hit1_pct"],
                 r["delta_pp"], r["helped"], r["harmed"],
                 "--" if ret is None else "%.0f%%" % ret,
                 r["estimated_latency_ms"]))
    print()
    print("  'retained' is this width's delta as a percentage of the delta at")
    print("  k=%d. 'est. latency' interpolates the measured benchmark"
          % res["reference_k"])
    print("  linearly in k and is an ESTIMATE, optimistic at small k because")
    print("  it charges no fixed per-query verification overhead.")
    print("=" * 78)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache")
    ap.add_argument("--lam", type=float, required=True,
                    help="bounded-fusion weight; must be the frozen value")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)

    if args.lam < 0:
        print("error: --lam must be >= 0", file=sys.stderr)
        return 2
    try:
        res = analyse(args.cache, args.lam)
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
