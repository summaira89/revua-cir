"""How many usable hard negatives does the retriever's top-k actually contain?

Why this exists
---------------
The manuscript's Future Work proposes training the verifier against the
retriever's own top-$k$ rather than against in-batch targets, on the grounds
that the former is what the verifier faces at inference. Before spending
~6 GPU-hours to find out, this measures the precondition: after excluding
likely false negatives, does the top-$k$ contain enough separable negatives
to train on?

That precondition is not rhetorical. RocketQA (Qu et al., NAACL 2021)
measured ~70% of a converged retriever's hardest mined negatives to be
unannotated true positives. CIRR makes the risk structural rather than
incidental: every query sits inside a curated subset of six visually similar
images, so the top-10 is built to contain near-misses.

What the model already does
---------------------------
`revua_cir.py:405--450` mines semi-hard negatives in-batch under the FaceNet
rule: the hardest candidate whose similarity is still below the positive's
(`sim < pos_sim`), excluding self. That is already a positive-aware threshold
in the NV-Retriever sense, with the threshold at 100% of the positive. The
difference this script probes is the negative POOL -- other queries' targets
versus this query's own retrieved gallery candidates -- not the presence of
mining.

Thresholds compared
-------------------
  none      every non-target candidate in the window. The naive rule, and
            the one the recommendation document proposes. Included as the
            control that shows what the guard is worth.
  pct-100   admit a negative only if score <= 1.00 * target_score. Our
            current in-batch rule, transplanted to this pool.
  pct-95    admit only if score <= 0.95 * target_score. NV-Retriever's
            TopK-PercPos, whose ablation found 95% optimal.
  pct-90    a stricter variant, to show the trend rather than one point.
  qure      QuRe's (ICML 2025) rule: of the candidates ranked below the
            target, find the two largest adjacent score drops and keep what
            lies between them. Reimplemented from the paper's description,
            not from its code; see `qure_window` for the exact reading and
            its ambiguities.

A query contributes nothing when the target is absent from the window, since
there is then no positive to anchor a threshold to. Those queries are counted
and excluded rather than silently scored as zero-negative.

Usage
-----
  python -m revua_cir.scripts.hardneg_mining_analysis CACHE.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# Thresholds as fractions of the target's own base score. 1.00 reproduces the
# rule already in the model; 0.95 is NV-Retriever's reported optimum.
PCT_THRESHOLDS: Tuple[float, ...] = (1.00, 0.95, 0.90)


class CacheError(ValueError):
    """Raised for any structural problem in the cache file."""


def _validate(rec: dict, path: str, lineno: int) -> None:
    where = "%s:%d" % (os.path.basename(path), lineno)
    for field in ("candidate_gallery_ids", "base_score"):
        if field not in rec:
            raise CacheError("%s: missing required field %r" % (where, field))

    ids = rec["candidate_gallery_ids"]
    n = len(ids)
    if len(set(ids)) != len(ids):
        raise CacheError("%s: duplicate candidate_gallery_ids" % where)
    if len(rec["base_score"]) != n:
        raise CacheError("%s: base_score has %d entries for %d candidates"
                         % (where, len(rec["base_score"]), n))
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) for v in rec["base_score"]):
        raise CacheError("%s: base_score has a non-finite value" % where)

    tp = rec.get("target_pos")
    if tp is not None and not (tp == -1 or 0 <= tp < n):
        raise CacheError("%s: target_pos %r outside [-1, %d)" % (where, tp, n))

    rp = rec.get("relevant_positions_in_candidates")
    if rp is not None:
        bad = [p for p in rp if not 0 <= p < n]
        if bad:
            raise CacheError("%s: relevant positions %r outside [0, %d)"
                             % (where, bad, n))


def load_records(path: str) -> Tuple[List[dict], bool]:
    """Returns (records, multipositive).

    The multi-positive determination is per FILE and a mixture aborts, for
    the reason `multipositive_analysis.py` documents: scoring a CIRCO cache
    under CIRR's single-target rule silently mislabels every query with more
    than one valid target -- which here would mean counting true positives as
    minable negatives, the exact error this script exists to quantify.
    """
    records: List[dict] = []
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
            records.append(rec)
            flags.append("relevant_positions_in_candidates" in rec)

    if not records:
        raise CacheError("%s: no records" % path)
    if len(set(flags)) != 1:
        raise CacheError(
            "%s: %d of %d records carry relevant_positions_in_candidates; a "
            "cache must be wholly single-target or wholly multi-positive"
            % (path, sum(flags), len(records)))
    return records, flags[0]


def positive_positions(rec: dict, multipositive: bool) -> List[int]:
    """Every position in the window that must NOT be used as a negative.

    Under multi-positive relevance this is the full relevant set, not just
    one target. Treating the others as negatives would be the false-negative
    error by construction rather than by accident.
    """
    if multipositive:
        return sorted(rec["relevant_positions_in_candidates"])
    tp = rec.get("target_pos", -1)
    return [tp] if tp is not None and tp >= 0 else []


def anchor_score(rec: dict, positives: Sequence[int]) -> Optional[float]:
    """The score a threshold is measured against.

    With several positives the LOWEST is used, which is the conservative
    choice: it admits fewer negatives than anchoring on the best positive
    would, and a negative scoring above the weakest positive is exactly the
    case a threshold should be suspicious of.
    """
    if not positives:
        return None
    return min(rec["base_score"][p] for p in positives)


def mine_threshold(rec: dict, positives: Sequence[int],
                   pct: Optional[float]) -> List[int]:
    """Candidate positions admitted as negatives under a positive-aware
    threshold. `pct=None` applies no threshold (the naive rule).

    Note the sign convention: base scores are cosine similarities and can in
    principle be negative, in which case `pct * anchor` INCREASES the bound
    rather than tightening it. Guarded explicitly rather than left to
    arithmetic, since a silently inverted threshold would admit more
    negatives while appearing stricter.
    """
    scores = rec["base_score"]
    pos_set = set(positives)
    cand = [i for i in range(len(scores)) if i not in pos_set]
    if pct is None:
        return cand

    anchor = anchor_score(rec, positives)
    if anchor is None:
        return []
    if anchor <= 0.0:
        # A non-positive anchor makes a multiplicative threshold meaningless.
        # Report the query as unusable rather than produce a bound that
        # tightens or loosens depending on the sign.
        return []
    bound = pct * anchor
    return [i for i in cand if scores[i] <= bound]


def qure_window(rec: dict, positives: Sequence[int]) -> List[int]:
    """QuRe's (ICML 2025) selection, reimplemented from its description.

    The paper describes finding "two positions with the largest adjacent
    score differences" among the candidates ranked below the target, and
    keeping the samples between those steep drops.

    Two ambiguities are resolved here explicitly, because the paper's text
    alone does not fix them and a silent choice would misattribute whatever
    this produces to QuRe:

      * "below the target" is read as strictly lower-scoring than the anchor
        positive, using the same anchor as the threshold rules.
      * the region kept is between the two largest drops in rank order, i.e.
        bounded by the earlier and later of the two drop positions, not the
        top-two drops in magnitude order.

    This is therefore a reading of QuRe, not QuRe. It is reported alongside
    the threshold rules for comparison, not as a reproduction.
    """
    scores = rec["base_score"]
    anchor = anchor_score(rec, positives)
    if anchor is None:
        return []
    pos_set = set(positives)

    below = [i for i in range(len(scores))
             if i not in pos_set and scores[i] < anchor]
    below.sort(key=lambda i: -scores[i])
    if len(below) < 3:
        # Fewer than three candidates cannot exhibit two interior drops.
        return below

    drops = []
    for j in range(len(below) - 1):
        drops.append((scores[below[j]] - scores[below[j + 1]], j))
    drops.sort(key=lambda t: -t[0])
    j1, j2 = sorted((drops[0][1], drops[1][1]))
    # Keep what lies between the two drops: positions after the first drop
    # up to and including the second drop's upper side.
    return below[j1 + 1: j2 + 1]


def summarize(records: Sequence[dict], multipositive: bool) -> dict:
    n = len(records)
    window = len(records[0]["candidate_gallery_ids"])

    covered, uncovered = [], 0
    for rec in records:
        if positive_positions(rec, multipositive):
            covered.append(rec)
        else:
            uncovered += 1

    rules: Dict[str, dict] = {}

    def add(name: str, miner) -> None:
        counts, margins, usable = [], [], 0
        for rec in covered:
            pos = positive_positions(rec, multipositive)
            negs = miner(rec, pos)
            counts.append(len(negs))
            if negs:
                usable += 1
                anchor = anchor_score(rec, pos)
                top_neg = max(rec["base_score"][i] for i in negs)
                margins.append(anchor - top_neg)
        rules[name] = {
            "mean_negatives": statistics.fmean(counts) if counts else 0.0,
            "median_negatives": statistics.median(counts) if counts else 0.0,
            "queries_with_at_least_one": usable,
            "queries_with_none": len(covered) - usable,
            "mean_margin_to_hardest": statistics.fmean(margins) if margins else None,
            "median_margin_to_hardest": statistics.median(margins) if margins else None,
        }

    add("none (naive)", lambda r, p: mine_threshold(r, p, None))
    for pct in PCT_THRESHOLDS:
        add("pct-%d" % round(pct * 100), lambda r, p, q=pct: mine_threshold(r, p, q))
    add("qure (our reading)", qure_window)

    # How many non-target candidates outscore the anchor positive? Under the
    # naive rule every one of these becomes a negative the model is told is
    # wrong while the retriever ranks it above the right answer.
    above = []
    for rec in covered:
        pos = positive_positions(rec, multipositive)
        anchor = anchor_score(rec, pos)
        pos_set = set(pos)
        above.append(sum(1 for i, s in enumerate(rec["base_score"])
                         if i not in pos_set and s > anchor))

    return {
        "n_queries": n,
        "window": window,
        "multipositive": multipositive,
        "n_covered": len(covered),
        "n_uncovered": uncovered,
        "mean_candidates_above_anchor": statistics.fmean(above) if above else 0.0,
        "queries_with_any_above_anchor": sum(1 for a in above if a > 0),
        "rules": rules,
    }


def report(res: dict) -> None:
    print("=" * 76)
    print("HARD-NEGATIVE AVAILABILITY IN THE RETRIEVER'S TOP-%d" % res["window"])
    print("=" * 76)
    print("queries            : %d" % res["n_queries"])
    print("relevance          : %s" % ("multi-positive" if res["multipositive"]
                                       else "single-target"))
    print("target in window   : %d  (these can anchor a threshold)"
          % res["n_covered"])
    print("target absent      : %d  (no anchor; unusable for mining)"
          % res["n_uncovered"])
    print()
    print("CANDIDATES OUTSCORING THE TARGET")
    print("  mean per covered query : %.2f"
          % res["mean_candidates_above_anchor"])
    print("  queries with >= 1      : %d of %d (%.1f%%)"
          % (res["queries_with_any_above_anchor"], res["n_covered"],
             100.0 * res["queries_with_any_above_anchor"]
             / max(1, res["n_covered"])))
    print("  Under the naive rule every one of these is used as a negative")
    print("  although the retriever ranks it above the true target.")
    print()
    print("NEGATIVES ADMITTED PER COVERED QUERY")
    print("  %-20s %8s %8s %10s %12s" % ("rule", "mean", "median",
                                         "no negs", "margin"))
    for name, r in res["rules"].items():
        m = r["mean_margin_to_hardest"]
        print("  %-20s %8.2f %8.1f %10d %12s"
              % (name, r["mean_negatives"], r["median_negatives"],
                 r["queries_with_none"],
                 "--" if m is None else "%.4f" % m))
    print()
    print("  'margin' is the mean gap between the target's score and the")
    print("  hardest admitted negative -- what the training hinge acts on.")
    print("  A margin near zero means the negatives are target-adjacent and")
    print("  a hinge would push against near-misses.")
    print("=" * 76)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache", help="rv_score_cache_*.jsonl")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)

    try:
        records, multipositive = load_records(args.cache)
    except CacheError as exc:
        print("CACHE ERROR: %s" % exc, file=sys.stderr)
        return 1

    res = summarize(records, multipositive)
    res["path"] = args.cache
    report(res)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print()
        print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
