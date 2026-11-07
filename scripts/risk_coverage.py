"""Risk--coverage curves and AURC for the confidence proxies.

What the review asked for
-------------------------
Section 5.7 reports AUROC and a coverage table. An external review asked for
four additions:

  (1) risk--coverage curves and the area under them, with paired
      uncertainty, this being the standard selective-prediction framing;
  (2) a comparison of "base entropy + verification" against base entropy
      alone -- the manuscript currently combines verification with
      base_margin, which is not the strongest single parent;
  (3) a note on how across-query rank averaging would work at deployment;
  (4) a statement of what abstention actually does.

This script covers (1) and (2). Items (3) and (4) are manuscript text, not
measurements, and are handled there.

AURC and E-AURC
---------------
With queries sorted by decreasing confidence, the selective risk at coverage
k/N is the error rate among the k most confident:

    r(k/N) = (1/k) * |{i <= k : query i is wrong}|
    AURC   = (1/N) * sum_{k=1..N} r(k/N)

Lower is better. Raw AURC is not comparable between proxies whose base
accuracy differs, because a more accurate model has a lower floor. Geifman
et al. (ICLR 2019) therefore subtract the oracle:

    E-AURC = AURC - AURC_oracle

where the oracle ranking places every correct query before every incorrect
one. E-AURC isolates ranking quality from accuracy, which is what we want
when comparing proxies measured on the same predictions.

Verified numerically before use: a random ranking gives AURC close to the
base error rate (1 - accuracy), the oracle gives much less, and an
adversarial ranking much more. `test_aurc_ordering_oracle_random_worst`
pins those three cases.

Uncertainty
-----------
Differences between proxies are paired by query and bootstrapped over
queries, so the shared difficulty component cancels. Reporting an interval
for each proxy separately and eyeballing the overlap would be the weaker
analysis the review objects to elsewhere.

Usage
-----
  python -m revua_cir.scripts.risk_coverage CACHE.jsonl [-k 1]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from selective_retrieval_analysis import (  # noqa: E402
    CacheError, PROXIES, UG_PROXY, combine_by_rank, confidence, is_correct,
    load_records,
)

N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 54321

# Coverage levels printed in the curve table. The AURC integral uses every
# k from 1 to N regardless; these are only for display.
DISPLAY_COVERAGES: Tuple[float, ...] = (1.00, 0.90, 0.75, 0.50, 0.25, 0.10)


def _sorted_outcomes(conf: Sequence[float],
                     correct: Sequence[bool]) -> List[bool]:
    """Outcomes ordered by decreasing confidence, ties by original index.

    The stable tie-break matters: a proxy with many equal values would
    otherwise give a different curve on each run.
    """
    order = sorted(range(len(conf)), key=lambda i: (-conf[i], i))
    return [bool(correct[i]) for i in order]


def aurc(conf: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the risk--coverage curve; lower is better."""
    outcomes = _sorted_outcomes(conf, correct)
    n = len(outcomes)
    if n == 0:
        raise ValueError("no queries")
    total = 0.0
    errors = 0
    for k in range(1, n + 1):
        if not outcomes[k - 1]:
            errors += 1
        total += errors / k
    return total / n


def oracle_aurc(correct: Sequence[bool]) -> float:
    """AURC of a perfect ranking: every correct query before every wrong one.

    Depends only on how many are correct, not on any confidence signal, so
    it is the floor that E-AURC subtracts.
    """
    n_correct = sum(1 for c in correct if c)
    ideal = [True] * n_correct + [False] * (len(correct) - n_correct)
    return aurc(list(range(len(ideal), 0, -1)), ideal)


def e_aurc(conf: Sequence[float], correct: Sequence[bool]) -> float:
    """Excess AURC: AURC minus the oracle's. Zero is perfect ranking."""
    return aurc(conf, correct) - oracle_aurc(correct)


def risk_at_coverages(conf: Sequence[float], correct: Sequence[bool],
                      coverages: Sequence[float] = DISPLAY_COVERAGES
                      ) -> List[dict]:
    outcomes = _sorted_outcomes(conf, correct)
    n = len(outcomes)
    rows = []
    for cov in coverages:
        k = max(1, min(n, int(round(cov * n))))
        wrong = sum(1 for o in outcomes[:k] if not o)
        rows.append({"coverage": cov, "n_kept": k,
                     "risk": wrong / k, "accuracy": 1.0 - wrong / k})
    return rows


def paired_bootstrap_aurc_delta(conf_a: Sequence[float],
                                conf_b: Sequence[float],
                                correct: Sequence[bool],
                                n_rep: int = N_BOOTSTRAP,
                                seed: int = BOOTSTRAP_SEED) -> dict:
    """CI for AURC(a) - AURC(b), resampling queries with replacement.

    Both proxies are re-scored on the SAME resample, so the component of
    variance driven by which queries are hard cancels. A negative delta
    means proxy a ranks better, since lower AURC is better.
    """
    n = len(correct)
    if not (len(conf_a) == len(conf_b) == n):
        raise ValueError("inputs differ in length (%d, %d, %d)"
                         % (len(conf_a), len(conf_b), n))
    observed = aurc(conf_a, correct) - aurc(conf_b, correct)
    rng = random.Random(seed)
    deltas: List[float] = []
    for _ in range(n_rep):
        idx = [rng.randrange(n) for _ in range(n)]
        cr = [correct[i] for i in idx]
        if all(cr) or not any(cr):
            continue
        a = aurc([conf_a[i] for i in idx], cr)
        b = aurc([conf_b[i] for i in idx], cr)
        deltas.append(a - b)
    if not deltas:
        return {"observed_delta": observed, "ci_low": None, "ci_high": None,
                "p_two_sided": None, "n_valid": 0}
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]
    tail = (sum(1 for d in deltas if d >= 0.0) if observed < 0
            else sum(1 for d in deltas if d <= 0.0))
    p = min(1.0, 2.0 * (tail + 1) / (len(deltas) + 1))
    return {"observed_delta": observed, "ci_low": lo, "ci_high": hi,
            "p_two_sided": p, "n_valid": len(deltas)}


def analyse(path: str, k: int = 1, n_rep: int = N_BOOTSTRAP) -> dict:
    records, multipositive, has_ug = load_records(path)
    correct = [is_correct(rec, k, multipositive) for rec in records]
    if not any(correct) or all(correct):
        raise CacheError("every query has the same outcome at k=%d; AURC is "
                         "undefined" % k)

    names = list(PROXIES) + ([UG_PROXY] if has_ug else [])
    conf: Dict[str, List[float]] = {
        nm: [confidence(rec, nm) for rec in records] for nm in names
    }

    # The combinations the review asks about. base_entropy is the strongest
    # single parent on CIRR, so pairing verification with it is the test the
    # manuscript's base_margin pairing does not make.
    combos = {
        "base_entropy + rv_margin": ("base_entropy", "rv_margin"),
        "base_margin + rv_margin": ("base_margin", "rv_margin"),
        "base_top1 + rv_margin": ("base_top1", "rv_margin"),
    }
    for cname, (a, b) in combos.items():
        if a in conf and b in conf:
            conf[cname] = combine_by_rank(conf[a], conf[b])

    orc = oracle_aurc(correct)
    metrics = {nm: {"aurc": aurc(c, correct),
                    "e_aurc": aurc(c, correct) - orc,
                    "curve": risk_at_coverages(c, correct)}
               for nm, c in conf.items()}

    # Every combination against its own strongest parent, paired.
    contrasts = {}
    for cname, (a, b) in combos.items():
        if cname not in conf:
            continue
        parent = a if metrics[a]["aurc"] <= metrics[b]["aurc"] else b
        contrasts["%s vs %s" % (cname, parent)] = {
            "parent": parent,
            **paired_bootstrap_aurc_delta(conf[cname], conf[parent],
                                          correct, n_rep),
        }

    # And the single proxies against the best single proxy, so the table is
    # not only self-referential.
    singles = [nm for nm in names]
    best_single = min(singles, key=lambda nm: metrics[nm]["aurc"])

    # Every combination also against that same best single proxy (added
    # 2026-10-13). The manuscript's risk-coverage table reports one common
    # reference column; a combination whose own stronger parent is not the
    # best single proxy (base_margin + rv_margin, whose parent is base_margin)
    # otherwise has no entry in that column.
    for cname, (a, b) in combos.items():
        if cname not in conf:
            continue
        key = "%s vs %s" % (cname, best_single)
        if key in contrasts:
            continue
        contrasts[key] = {
            "parent": best_single,
            **paired_bootstrap_aurc_delta(conf[cname], conf[best_single],
                                          correct, n_rep),
        }
    for nm in singles:
        if nm == best_single:
            continue
        contrasts["%s vs %s" % (nm, best_single)] = {
            "parent": best_single,
            **paired_bootstrap_aurc_delta(conf[nm], conf[best_single],
                                          correct, n_rep),
        }

    return {
        "path": path, "k": k, "n_queries": len(records),
        "multipositive": multipositive, "has_ug_confidence": has_ug,
        "base_accuracy": sum(correct) / len(correct),
        "oracle_aurc": orc,
        "best_single": best_single,
        "metrics": metrics,
        "contrasts": contrasts,
        "n_bootstrap": n_rep,
    }


def report(res: dict) -> None:
    print("=" * 76)
    print("RISK--COVERAGE AND AURC")
    print("=" * 76)
    print("cache        : %s" % os.path.basename(res["path"]))
    print("queries      : %d   correct at top-%d: %.2f%%"
          % (res["n_queries"], res["k"], 100.0 * res["base_accuracy"]))
    print("oracle AURC  : %.4f   (the floor E-AURC subtracts)"
          % res["oracle_aurc"])
    print()
    print("Lower AURC is better. E-AURC removes the accuracy floor, so it is")
    print("the figure to compare between proxies.")
    print()
    print("  %-26s %8s %8s" % ("proxy", "AURC", "E-AURC"))
    for nm, m in sorted(res["metrics"].items(), key=lambda kv: kv[1]["aurc"]):
        star = "  <- best single" if nm == res["best_single"] else ""
        print("  %-26s %8.4f %8.4f%s" % (nm, m["aurc"], m["e_aurc"], star))

    print()
    print("RISK AT SELECTED COVERAGES (risk = error rate among those kept)")
    hdr = "  %-26s" % "proxy"
    for c in DISPLAY_COVERAGES:
        hdr += "%8.0f%%" % (100 * c)
    print(hdr)
    for nm, m in sorted(res["metrics"].items(), key=lambda kv: kv[1]["aurc"]):
        row = "  %-26s" % nm
        for r in m["curve"]:
            row += "%9.3f" % r["risk"]
        print(row)

    print()
    print("PAIRED AURC CONTRASTS (negative favours the first proxy)")
    print("  %-38s %9s %-20s %8s" % ("contrast", "delta", "95% CI", "p"))
    for name, d in res["contrasts"].items():
        if d["ci_low"] is None:
            print("  %-38s %9s" % (name, "undefined"))
            continue
        print("  %-38s %+9.4f [%+7.4f, %+7.4f] %8.4f"
              % (name, d["observed_delta"], d["ci_low"], d["ci_high"],
                 d["p_two_sided"]))
    print("=" * 76)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache")
    ap.add_argument("-k", type=int, default=1)
    ap.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)

    if args.k < 1:
        print("error: -k must be >= 1", file=sys.stderr)
        return 2
    if args.bootstrap < 1:
        print("error: --bootstrap must be >= 1", file=sys.stderr)
        return 2
    try:
        res = analyse(args.cache, k=args.k, n_rep=args.bootstrap)
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
