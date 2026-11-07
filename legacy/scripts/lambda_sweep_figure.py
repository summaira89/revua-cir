"""Emit the full lambda sweep as CSV, for the manuscript's lambda figure.

**Why this exists.** The manuscript specification forbids cherry-picking
lambda values, and only 4 of the 12 grid points were recorded in earlier
write-ups. Rather than plot a partial curve, this recomputes every point
from the score cache so the figure shows the complete swept range.

Reports BOTH halves of the tune/report split at every lambda:

* the TUNE half is what lambda was actually selected on;
* the REPORT half is the held-out estimate.

Plotting both makes the selection protocol visible, and shows whether the
selected optimum transfers. Also emits the full-set curve at the frozen
lambda for reference.

CPU only, no GPU allocation, runs in seconds on a login node.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from multipositive_analysis import (  # noqa: E402
    base_order,
    bounded_fusion_order,
    hit_at_1,
    load_cache,
    relevant_positions,
)
from revua_cir.significance import mcnemar_exact, split_indices  # noqa: E402

# The grid actually swept in the reported experiments.
LAMBDA_GRID = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70,
               1.00, 1.50, 2.00, 3.00, 5.00, 8.00)


def r_at_1(recs, lam=None):
    """R@1 over `recs`; lam=None gives the base ordering."""
    if not recs:
        return 0.0, [], []
    base, new = [], []
    for r in recs:
        rel = relevant_positions(r)
        base.append(hit_at_1(base_order(r), rel))
        order = base_order(r) if lam is None else bounded_fusion_order(r, lam)
        new.append(hit_at_1(order, rel))
    return 100.0 * sum(new) / len(new), base, new


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tune-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--csv-out", default="lambda_sweep.csv")
    args = ap.parse_args(argv)

    recs = load_cache(args.cache)
    n = len(recs)
    tune_idx, report_idx = split_indices(n, tune_frac=args.tune_frac,
                                         seed=args.seed)
    tune = [recs[i] for i in tune_idx]
    report = [recs[i] for i in report_idx]

    print("=" * 74)
    print("Lambda sweep: %s" % args.cache)
    print("n = %d   tune = %d   report = %d   (seed %d)"
          % (n, len(tune), len(report), args.seed))
    print("=" * 74)

    base_tune, _, _ = r_at_1(tune)
    base_report, _, _ = r_at_1(report)
    base_full, _, _ = r_at_1(recs)
    print("base R@1   tune %.4f   report %.4f   full %.4f"
          % (base_tune, base_report, base_full))
    print()
    print("%8s %10s %10s %10s %12s" %
          ("lambda", "tune R@1", "report R@1", "full R@1", "full p"))

    rows = []
    for lam in LAMBDA_GRID:
        t, _, _ = r_at_1(tune, lam)
        rp, _, _ = r_at_1(report, lam)
        f, fb, fn = r_at_1(recs, lam)
        help_ = sum(1 for b, x in zip(fb, fn) if not b and x)
        harm = sum(1 for b, x in zip(fb, fn) if b and not x)
        p = mcnemar_exact(help_, harm)
        rows.append({
            "lambda": lam, "tune_r1": t, "report_r1": rp, "full_r1": f,
            "full_delta_pp": f - base_full, "help": help_, "harm": harm,
            "full_p": p,
        })
        print("%8.2f %10.4f %10.4f %10.4f %12.4g" % (lam, t, rp, f, p))

    best = max(rows, key=lambda r: (r["tune_r1"], -r["lambda"]))
    print()
    print("selected on TUNE half: lambda = %.2f (tune R@1 %.4f)"
          % (best["lambda"], best["tune_r1"]))
    print("  -> held-out report R@1 at that lambda: %.4f (base %.4f, %+.2f pp)"
          % (best["report_r1"], base_report, best["report_r1"] - base_report))

    with open(args.csv_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
        # baselines recorded so the figure can draw them without guessing
        fh.write("# base_tune,%.6f\n" % base_tune)
        fh.write("# base_report,%.6f\n" % base_report)
        fh.write("# base_full,%.6f\n" % base_full)
    print("\nwrote %s" % args.csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
