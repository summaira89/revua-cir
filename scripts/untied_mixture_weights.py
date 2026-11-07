"""Untied verification-score mixture weights, selected on the tune half.

What this answers
-----------------
B6.4 item 4: the trained verifier ties the two non-edit weights, so
reconstruction and preservation each carry $(1-w)/2 \\approx 0.256$ while
edit relevance carries $w \\approx 0.489$. The review asks whether that
allocation is a good one -- the two branches need not deserve equal weight,
and Section 5.5 already shows they do not behave alike: reconstruction
helps and preservation hurts.

No GPU is needed. The cache stores `recon_cos`, `invariant_cos` and
`edit_cos` separately for every candidate, so any convex combination of the
three can be scored offline.

Why the split matters here
--------------------------
The existing component ablation evaluates FIXED reweightings (drop a term,
keep the rest), so reporting it on all 4181 queries is legitimate. This
script SELECTS weights, which is a different statistical act: choosing the
best of many candidates on the same data that reports the result would
report the selection noise as an effect. Weights are therefore chosen on
the tune half (seed 12345, 2090 queries) and the chosen weights alone are
evaluated on the report half (2091 queries), matching the protocol
Section 4 uses for lambda.

The comparison is against the TRAINED tied weights evaluated on the same
report half, so both arms are scored on identical queries and the
difference is paired.

What a negative result would mean
---------------------------------
If untying does not beat the tied allocation on held-out queries, that is
informative: it says the trained weights are not leaving much on the table,
and it removes a reviewer's open question rather than adding a claim. The
script is written to report that outcome as clearly as a positive one.

Usage
-----
  python -m revua_cir.scripts.untied_mixture_weights CACHE.jsonl \\
      --lam 1.0 --grid 21
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fusion_controls import (  # noqa: E402
    CacheError, SPLIT_SEED, _zscore, split_indices,
)
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 54321

# The trained model's own allocation: w = sigmoid(edit_weight) = 0.488847,
# with the two non-edit terms tied at (1 - w) / 2.
W_EDIT_TRAINED = 0.488847


def trained_weights() -> Tuple[float, float, float]:
    """(recon, invariant, edit) as the trained model ties them."""
    tied = (1.0 - W_EDIT_TRAINED) / 2.0
    return (tied, tied, W_EDIT_TRAINED)


def load_records(path: str) -> List[dict]:
    """Load a CIRR cache, requiring the three component cosines.

    A CIRCO cache would load fine here but must not be used: its
    multi-positive annotation makes `target_pos` ill-defined as a single
    rank, and the paper's component results are CIRR-only. The caller is
    warned by the explicit `relevant_gallery_indices` check below.
    """
    recs: List[dict] = []
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
            for f in ("candidate_gallery_ids", "base_score", "recon_cos",
                      "invariant_cos", "edit_cos", "target_pos"):
                if f not in rec:
                    raise CacheError(
                        "%s:%d: missing %r -- this script needs a cache with "
                        "the three component cosines"
                        % (os.path.basename(path), lineno, f))
            n = len(rec["candidate_gallery_ids"])
            for f in ("base_score", "recon_cos", "invariant_cos", "edit_cos"):
                if len(rec[f]) != n:
                    raise CacheError("%s:%d: %s has %d entries for %d "
                                     "candidates"
                                     % (os.path.basename(path), lineno, f,
                                        len(rec[f]), n))
            recs.append(rec)
    if not recs:
        raise CacheError("%s: no records" % path)
    if any(len(r.get("relevant_gallery_indices") or []) > 1 for r in recs):
        raise CacheError(
            "this cache has queries with multiple ground truths, which means "
            "it is CIRCO; the component analysis is CIRR-only (see the "
            "multi-positive note in Section 5.6)")
    return recs


def covered(rec: dict) -> bool:
    """True when the true target is inside the cached window.

    Queries whose target never entered the window cannot be rescued or
    broken by any reweighting, so they contribute an identical constant to
    both arms and are excluded from the paired comparison.
    """
    tp = rec.get("target_pos")
    return tp is not None and tp >= 0


def fused_rank1(rec: dict, w: Tuple[float, float, float],
                lam: float) -> bool:
    """Is the true target ranked first under this weighting?

    Mirrors the deployed rule exactly: the three cosines are mixed, the
    mixture is z-scored within the window, squashed by tanh, and added to
    the z-scored base score with weight lam.
    """
    recon = rec["recon_cos"]
    inv = rec["invariant_cos"]
    edit = rec["edit_cos"]
    k = len(recon)
    mixed = [w[0] * recon[i] + w[1] * inv[i] + w[2] * edit[i]
             for i in range(k)]
    zb = _zscore(rec["base_score"])
    zr = _zscore(mixed)
    fused = [zb[i] + lam * math.tanh(zr[i]) for i in range(k)]
    # Ties broken by original order, as elsewhere in the project.
    best = max(range(k), key=lambda i: (fused[i], -i))
    return best == rec["target_pos"]


def rate(recs: Sequence[dict], w: Tuple[float, float, float],
         lam: float) -> float:
    if not recs:
        return 0.0
    return 100.0 * sum(1 for r in recs if fused_rank1(r, w, lam)) / len(recs)


def weight_grid(steps: int) -> List[Tuple[float, float, float]]:
    """Convex weights on a simplex grid, excluding the all-zero corner.

    A convex constraint keeps the mixture comparable to the trained one,
    which is also convex; unconstrained weights would change the score's
    scale as well as its direction and would not isolate the allocation.
    """
    grid: List[Tuple[float, float, float]] = []
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            k = steps - i - j
            grid.append((i / steps, j / steps, k / steps))
    return grid


def split(recs: Sequence[dict], seed: int = SPLIT_SEED
          ) -> Tuple[List[dict], List[dict]]:
    """The project's tune/report partition.

    Delegates to `fusion_controls.split_indices` rather than reimplementing
    the shuffle, so this script cannot drift away from the partition the
    rest of the paper uses. Verified identical at n = 10, 220, 2000 and
    4181 before the delegation was introduced.
    """
    tune_idx, report_idx = split_indices(len(recs), seed)
    return [recs[i] for i in tune_idx], [recs[i] for i in report_idx]


def paired_bootstrap(a: Sequence[bool], b: Sequence[bool],
                     n_rep: int = N_BOOTSTRAP,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    """CI for the paired rank-1 rate difference, resampling queries."""
    n = len(a)
    if n != len(b):
        raise ValueError("arms differ in length")
    obs = 100.0 * (sum(a) - sum(b)) / n
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_rep):
        tot = 0
        for _ in range(n):
            i = rng.randrange(n)
            tot += (1 if a[i] else 0) - (1 if b[i] else 0)
        deltas.append(100.0 * tot / n)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]
    tail = (sum(1 for d in deltas if d <= 0.0) if obs >= 0
            else sum(1 for d in deltas if d >= 0.0))
    p = min(1.0, 2.0 * (tail + 1) / (len(deltas) + 1))
    return {"delta": obs, "ci_low": lo, "ci_high": hi, "p_two_sided": p}


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """Exact binomial McNemar on the discordant pairs."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    lo = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2.0 ** n)
    return {"n01": n01, "n10": n10, "p": min(1.0, 2.0 * tail)}


def analyse(path: str, lam: float, steps: int,
            n_rep: int = N_BOOTSTRAP) -> dict:
    recs = [r for r in load_records(path) if covered(r)]
    tune, report = split(recs)

    tied = trained_weights()
    grid = weight_grid(steps)

    # Select on the tune half only.
    scored = [(rate(tune, w, lam), w) for w in grid]
    best_tune, best_w = max(scored, key=lambda t: (t[0], -sum(t[1])))
    tied_tune = rate(tune, tied, lam)

    # Evaluate the single chosen weighting on the untouched report half.
    a = [fused_rank1(r, best_w, lam) for r in report]
    b = [fused_rank1(r, tied, lam) for r in report]

    return {
        "path": path, "lam": lam, "grid_steps": steps,
        "n_covered": len(recs), "n_tune": len(tune), "n_report": len(report),
        "tied_weights": tied, "selected_weights": best_w,
        "tune": {"tied": tied_tune, "selected": best_tune,
                 "delta": best_tune - tied_tune},
        "report": {"tied": 100.0 * sum(b) / len(b),
                   "selected": 100.0 * sum(a) / len(a)},
        "contrast": paired_bootstrap(a, b, n_rep),
        "mcnemar": mcnemar_exact(a, b),
        "n_grid": len(grid),
    }


def report_out(res: dict) -> None:
    tw, sw = res["tied_weights"], res["selected_weights"]
    print("=" * 74)
    print("UNTIED MIXTURE WEIGHTS  (selected on tune half, scored on report)")
    print("=" * 74)
    print("cache    : %s" % os.path.basename(res["path"]))
    print("covered  : %d queries -> tune %d / report %d"
          % (res["n_covered"], res["n_tune"], res["n_report"]))
    print("grid     : %d convex weightings (%d steps per axis)"
          % (res["n_grid"], res["grid_steps"]))
    print("lambda   : %s (frozen)" % res["lam"])
    print()
    print("  weighting            recon  invar   edit")
    print("  trained (tied)      %6.3f %6.3f %6.3f" % tw)
    print("  selected on tune    %6.3f %6.3f %6.3f" % sw)
    print()
    print("TUNE HALF (selection -- these numbers are optimistic by")
    print("construction and are NOT the result)")
    print("  tied      %6.2f" % res["tune"]["tied"])
    print("  selected  %6.2f   (%+.2f)"
          % (res["tune"]["selected"], res["tune"]["delta"]))
    print()
    print("REPORT HALF (the result)")
    print("  tied      %6.2f" % res["report"]["tied"])
    print("  selected  %6.2f" % res["report"]["selected"])
    c = res["contrast"]
    print("  delta     %+6.2f   95%% CI [%+.2f, %+.2f], bootstrap p=%.4f"
          % (c["delta"], c["ci_low"], c["ci_high"], c["p_two_sided"]))
    m = res["mcnemar"]
    print("  McNemar   %d/%d discordant, exact p=%.4f"
          % (m["n01"], m["n10"], m["p"]))
    print()
    crosses_zero = c["ci_low"] <= 0.0 <= c["ci_high"]
    if crosses_zero:
        print("  READ: the interval includes zero, so untying the two non-edit")
        print("  weights does not measurably beat the trained tied allocation")
        print("  on held-out queries. The trained weights are not leaving a")
        print("  detectable amount on the table.")
    else:
        print("  READ: the interval excludes zero. Note this is a single")
        print("  selected weighting evaluated once on held-out queries, not a")
        print("  retrained model -- the allocation is post-hoc, as in")
        print("  Section 5.5.")
    print("=" * 74)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache")
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--grid", type=int, default=21,
                    help="steps per simplex axis; 21 gives 253 weightings")
    ap.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)

    if args.lam < 0:
        print("error: --lam must be >= 0", file=sys.stderr)
        return 2
    if args.grid < 2:
        print("error: --grid must be >= 2", file=sys.stderr)
        return 2

    try:
        res = analyse(args.cache, args.lam, args.grid, args.bootstrap)
    except CacheError as exc:
        print("CACHE ERROR: %s" % exc, file=sys.stderr)
        return 1

    report_out(res)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print()
        print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
