"""Selective retrieval and confidence analysis from a cached score file.

The question this answers
------------------------
ReVUA-CIR's uncertainty gate does not change rankings: the logged
`ug_changes_final_order_rate` is ~0%, so as a reranking signal it is inert.
That leaves an obvious reviewer question -- what is the "U" for? -- which the
manuscript currently cannot answer beyond "it changed no rankings".

Selective retrieval is a different use for a confidence signal, and one the
cached scores can settle with no GPU. Instead of reordering candidates, rank
QUERIES by confidence and withhold the least confident ones. If a confidence
proxy is informative, accuracy on the retained queries rises monotonically as
coverage falls. That is the standard risk--coverage framing (El-Yaniv and
Wiener, JMLR 2010; Geifman and El-Yaniv, NeurIPS 2017).

What this script does NOT do
---------------------------
It does not recompute ECE. ECE as logged by `evaluate.py` is
`expected_calibration_error(ug_confidence, base_top1_correct)` -- the gate's
confidence against whether the BASE retriever's top-1 was right. That needs
`ug_confidence`, which `evaluate.py` writes to the cache only when the gate is
enabled (`if ug_b is not None`). The headline checkpoint is `fc+aep+rv`, with
no gate, so its cache has no `ug_confidence` field and no ECE can be derived
from it at any level of effort. The ECE numbers reported in the manuscript
come from the separate `fc+aep+rv+ug` training log and are labelled as such.
If this script is pointed at a gate-enabled cache it will additionally report
the `ug_confidence`-based proxy, and says so; otherwise it omits it rather
than substituting a different quantity under the same name.

Confidence proxies
------------------
All are computed from the BASE retrieval scores or the cached RV terms, i.e.
they are available at query time without knowing the answer:

  base_margin      base_score[0] - base_score[1], the top-1/top-2 gap. The
                   standard uncertainty proxy for a ranker: a query whose
                   best two candidates score almost identically is one the
                   model cannot separate.
  base_top1        base_score[0] alone. A control -- absolute similarity
                   conflates "confident" with "this image matches lots of
                   things", so if base_margin does no better than this, the
                   margin framing is adding nothing.
  base_entropy     negative Shannon entropy of softmax(base_score/T) over the
                   window. Uses the whole score distribution rather than just
                   its first two elements.
  rv_margin        verification_score[0] - verification_score[1] under the
                   base ordering. Tests whether the VERIFIER supplies
                   selective signal the base scores do not -- the question
                   that makes this analysis about ReVUA-CIR rather than about
                   CLIP.
  ug_confidence    the gate's own confidence, reduced by its top-1 entry.
                   Only when present in the cache.

Correctness
-----------
Single-target (CIRR): a query is correct at k if `target_pos` is in [0, k).
Multi-positive (CIRCO): correct at k if ANY entry of
`relevant_positions_in_candidates` is < k. Mixing the two definitions is the
error `multipositive_analysis.py` was written to prevent, so the definition is
chosen per file from which fields exist, and a file mixing both aborts.

Significance
------------
A selective-retrieval curve that rises can rise by chance, especially at low
coverage where few queries remain. Each coverage level therefore gets a
permutation test: under the null that confidence carries no information, ask
how often a randomly chosen subset of the same size is at least this accurate.
This is the right null -- it holds the outcome distribution and the subset size
fixed and varies only whether confidence is informative.

That null count is hypergeometric, so it is sampled directly (`N_PERMUTE`
draws via `_hypergeom_rvs`) rather than by shuffling the whole outcome vector.
The two are equivalent, and `test_permutation_matches_an_explicit_shuffle`
pins them against each other; the direct version is O(1) per replicate instead
of O(n), which matters at 4181 queries. A closed form exists for this p-value
and `test_permutation_p_matches_the_exact_hypergeometric_tail` checks against
it; Monte Carlo is retained so the reported floor, 1/(N_PERMUTE+1), states
plainly how much resolution the estimate has.

Usage
-----
  python -m revua_cir.scripts.selective_retrieval_analysis CACHE.jsonl [-k 1]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# Coverage levels reported. 1.00 is included because it is the anchor: every
# other row is only interpretable as a change from full coverage.
COVERAGES: Tuple[float, ...] = (1.00, 0.90, 0.80, 0.70, 0.60,
                                0.50, 0.40, 0.30, 0.20, 0.10)

N_PERMUTE = 10000
PERMUTE_SEED = 12345  # same seed as the tune/report split, for traceability

# Paired bootstrap over queries, for the combined-vs-parent AUROC comparison.
# 10,000 replicates matches the fusion analysis's existing protocol so the
# two are directly comparable; the seed is distinct from PERMUTE_SEED so the
# two resampling procedures cannot accidentally share a random stream.
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 54321

# Softmax temperature for base_entropy. The base scores are cosine
# similarities in roughly [0, 1], so the raw spread across a top-k window is
# small (~0.05); without sharpening, softmax is near-uniform for every query
# and the entropy proxy would be near-constant by construction rather than by
# finding. 0.01 puts the resulting distributions on a usable scale. This is a
# fixed presentation choice, not a tuned parameter -- see
# `sensitivity_to_temperature` below, which reports the proxy's AUROC across a
# temperature range so the choice can be checked.
ENTROPY_TEMPERATURE = 0.01

PROXIES: Tuple[str, ...] = ("base_margin", "base_top1", "base_entropy",
                            "rv_margin")
UG_PROXY = "ug_confidence"

# Combined proxies, each averaging the ACROSS-QUERY ranks of two single
# proxies. Rank averaging, not score averaging: base_margin is a cosine
# difference of order 0.01 and rv_margin is a difference of sigmoid outputs
# on an unrelated scale, so adding the raw values would let whichever happens
# to have the larger spread dominate, and the "combination" would report that
# arbitrary scale choice rather than any joint signal. Ranks are scale-free,
# which is the same reasoning behind the production `_rank_fuse`.
#
# These exist to answer one question: the verifier's margin is nearly
# uncorrelated with every base proxy (Spearman ~0.15 on CIRR), so does it
# carry selective information the base scores do not already have? A
# combination that beats both parents says yes.
COMBINED_PROXIES: Dict[str, Tuple[str, str]] = {
    "base_margin+rv_margin": ("base_margin", "rv_margin"),
    "base_top1+rv_margin": ("base_top1", "rv_margin"),
}


class CacheError(ValueError):
    """Raised for any structural problem in the cache file."""


def _softmax(xs: Sequence[float], temperature: float) -> List[float]:
    m = max(xs)
    exps = [math.exp((x - m) / temperature) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def _neg_entropy(scores: Sequence[float], temperature: float) -> float:
    """Negative Shannon entropy, so that HIGHER means MORE confident.

    Returning entropy directly would invert this proxy relative to every
    other one and make the shared sort order wrong -- a sign error that a
    rising curve would disguise as a finding.
    """
    p = _softmax(scores, temperature)
    h = -sum(pi * math.log(pi) for pi in p if pi > 0.0)
    return -h


def _hypergeom_rvs(rng: random.Random, n: int, n_success: int,
                   n_draw: int) -> int:
    """One hypergeometric draw: successes among `n_draw` taken from `n`.

    Equivalent to shuffling a vector of `n_success` hits and `n - n_success`
    misses and counting hits in the first `n_draw` positions, which is exactly
    the permutation null a selective-retrieval curve needs. Drawing the count
    directly is O(n_draw) rather than O(n) per replicate and allocates
    nothing.

    Sampling without replacement, updating the remaining population as it
    goes -- the textbook sequential construction, not an approximation.
    """
    # Draw from whichever side is smaller; the count is recovered exactly
    # either way, and it halves the work when coverage is above 50%.
    if n_draw > n - n_draw:
        complement = _hypergeom_rvs(rng, n, n_success, n - n_draw)
        return n_success - complement

    remaining_pop = n
    remaining_success = n_success
    hits = 0
    for _ in range(n_draw):
        if rng.random() < remaining_success / remaining_pop:
            hits += 1
            remaining_success -= 1
        remaining_pop -= 1
    return hits


def _validate_record(rec: dict, path: str, lineno: int) -> None:
    where = "%s:%d" % (os.path.basename(path), lineno)

    for field in ("candidate_gallery_ids", "base_score", "verification_score"):
        if field not in rec:
            raise CacheError("%s: missing required field %r" % (where, field))

    ids = rec["candidate_gallery_ids"]
    n = len(ids)
    if n < 2:
        raise CacheError("%s: window of %d candidate(s); base_margin and "
                         "rv_margin both need at least 2" % (where, n))
    if len(set(ids)) != len(ids):
        raise CacheError("%s: duplicate candidate_gallery_ids -- a candidate "
                         "appears twice in one window" % where)

    for field in ("base_score", "verification_score"):
        if len(rec[field]) != n:
            raise CacheError("%s: %s has %d entries for %d candidates"
                             % (where, field, len(rec[field]), n))
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) for v in rec[field]):
            raise CacheError("%s: %s contains a non-finite or non-numeric "
                             "value" % (where, field))

    tp = rec.get("target_pos")
    if tp is not None and not (tp == -1 or 0 <= tp < n):
        raise CacheError("%s: target_pos %r outside [-1, %d)" % (where, tp, n))

    rp = rec.get("relevant_positions_in_candidates")
    if rp is not None:
        if not all(isinstance(p, int) and not isinstance(p, bool) for p in rp):
            raise CacheError("%s: relevant_positions_in_candidates must be "
                             "integers, got %r" % (where, rp))
        if len(set(rp)) != len(rp):
            raise CacheError("%s: duplicate relevant_positions_in_candidates "
                             "%r" % (where, rp))
        bad = [p for p in rp if not 0 <= p < n]
        if bad:
            raise CacheError("%s: relevant_positions_in_candidates %r outside "
                             "[0, %d) -- these would silently score as misses"
                             % (where, bad, n))

    ug = rec.get("ug_confidence")
    if ug is not None:
        if len(ug) != n:
            raise CacheError("%s: ug_confidence has %d entries for %d "
                             "candidates" % (where, len(ug), n))
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) for v in ug):
            raise CacheError("%s: ug_confidence contains a non-finite or "
                             "non-numeric value" % where)


def load_records(path: str) -> Tuple[List[dict], bool, bool]:
    """Returns (records, multipositive, has_ug).

    `multipositive` is decided once for the whole file, not per record, and a
    file that mixes the two conventions aborts. A CIRCO cache scored under
    CIRR's single-target rule under-reports accuracy silently -- exactly the
    failure mode that prompted the multi-positive rewrite -- so the
    determination has to be unambiguous.
    """
    records: List[dict] = []
    multi_flags: List[bool] = []
    ug_flags: List[bool] = []

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
            _validate_record(rec, path, lineno)
            records.append(rec)
            multi_flags.append("relevant_positions_in_candidates" in rec)
            ug_flags.append("ug_confidence" in rec)

    if not records:
        raise CacheError("%s: no records" % path)

    if len(set(multi_flags)) != 1:
        n_multi = sum(multi_flags)
        raise CacheError(
            "%s: %d of %d records carry relevant_positions_in_candidates. A "
            "cache must be entirely single-target or entirely multi-positive; "
            "scoring a mixture applies the wrong correctness rule to part of "
            "the file." % (path, n_multi, len(records)))

    if len(set(ug_flags)) != 1:
        n_ug = sum(ug_flags)
        raise CacheError(
            "%s: %d of %d records carry ug_confidence. A partially "
            "gate-enabled cache cannot be analysed as either."
            % (path, n_ug, len(records)))

    ids = [rec.get("query_id") for rec in records]
    if all(i is not None for i in ids) and len(set(ids)) != len(ids):
        raise CacheError("%s: duplicate query_id values -- the same query "
                         "appears more than once, so every statistic below "
                         "would be weighted wrongly" % path)

    return records, multi_flags[0], ug_flags[0]


def is_correct(rec: dict, k: int, multipositive: bool) -> bool:
    """Correct at k under the file's own relevance convention."""
    if multipositive:
        return any(p < k for p in rec["relevant_positions_in_candidates"])
    tp = rec.get("target_pos", -1)
    return tp is not None and 0 <= tp < k


def confidence(rec: dict, proxy: str) -> float:
    base = rec["base_score"]
    if proxy == "base_margin":
        return base[0] - base[1]
    if proxy == "base_top1":
        return base[0]
    if proxy == "base_entropy":
        return _neg_entropy(base, ENTROPY_TEMPERATURE)
    if proxy == "rv_margin":
        rv = rec["verification_score"]
        return rv[0] - rv[1]
    if proxy == UG_PROXY:
        return rec["ug_confidence"][0]
    raise KeyError("unknown proxy %r" % proxy)


def selective_curve(
    confidences: Sequence[float],
    correct: Sequence[bool],
    coverages: Sequence[float] = COVERAGES,
    n_permute: int = N_PERMUTE,
    seed: int = PERMUTE_SEED,
) -> List[dict]:
    """Accuracy on the most-confident fraction of queries, at each coverage.

    Ties are broken by original index (a stable sort), so a proxy that is
    constant across many queries degrades to "keep the first m" rather than to
    an arbitrary ordering that would vary between runs.
    """
    n = len(confidences)
    if n != len(correct):
        raise ValueError("confidences and correct differ in length (%d vs %d)"
                         % (n, len(correct)))
    if n == 0:
        raise ValueError("no queries")

    order = sorted(range(n), key=lambda i: (-confidences[i], i))
    outcomes = [bool(correct[i]) for i in order]
    total_correct = sum(outcomes)

    rng = random.Random(seed)
    rows: List[dict] = []

    for cov in coverages:
        m = int(round(cov * n))
        m = max(1, min(n, m))
        kept = outcomes[:m]
        n_kept_correct = sum(kept)
        acc = n_kept_correct / m

        if m == n:
            # At full coverage the "subset" is everything, so a permutation
            # test is vacuous: every shuffle yields the same accuracy. Report
            # the p-value as undefined rather than as 1.0, which would read as
            # a genuine null result.
            p = None
        else:
            # Under the null, the number of correct answers among m queries
            # drawn without replacement from n (of which total_correct are
            # correct) is hypergeometric. Sampling that count directly is
            # equivalent to shuffling the whole outcome vector and summing
            # its first m entries, but costs O(1) per replicate instead of
            # O(n) -- at 4181 queries the shuffle version takes minutes per
            # k, which on a shared login node is worth avoiding.
            # `test_permutation_matches_an_explicit_shuffle` pins the two
            # against each other.
            hits = 0
            need = n_kept_correct
            for _ in range(n_permute):
                if _hypergeom_rvs(rng, n, total_correct, m) >= need:
                    hits += 1
            # Add-one smoothing: with n_permute resamples the smallest
            # observable p is 1/(n_permute+1), and reporting 0.0 would
            # overstate what this many permutations can resolve.
            p = (hits + 1) / (n_permute + 1)

        rows.append({
            "coverage": cov,
            "n_kept": m,
            "n_correct": n_kept_correct,
            "accuracy": acc,
            "p_permutation": p,
        })

    return rows


def auroc(confidences: Sequence[float],
          correct: Sequence[bool]) -> Optional[float]:
    """Rank AUROC of confidence as a predictor of correctness, ties at 0.5.

    A single scalar summary of the whole curve: the probability that a
    randomly chosen correct query is ranked above a randomly chosen incorrect
    one. Returns None when one class is empty, where AUROC is undefined.
    """
    pos = [c for c, ok in zip(confidences, correct) if ok]
    neg = [c for c, ok in zip(confidences, correct) if not ok]
    if not pos or not neg:
        return None

    # Rank-sum formulation with explicit mid-ranks for ties -- an O(n log n)
    # equivalent of counting concordant pairs, which matters at 4181 queries
    # where the pairwise loop is millions of comparisons per proxy.
    merged = sorted([(c, 1) for c in pos] + [(c, 0) for c in neg])
    ranks: List[float] = [0.0] * len(merged)
    i = 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[t] = mid
        i = j + 1

    rank_sum_pos = sum(r for r, (_, lab) in zip(ranks, merged) if lab == 1)
    n_p, n_n = len(pos), len(neg)
    return (rank_sum_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n)


def _midranks(xs: Sequence[float]) -> List[float]:
    """Ascending ranks, ties sharing their mean rank.

    Shared by `spearman` and `combine_by_rank` so the two cannot drift apart
    in how they treat ties.
    """
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            out[order[t]] = mid
        i = j + 1
    return out


def combine_by_rank(a: Sequence[float], b: Sequence[float]) -> List[float]:
    """Mean of two proxies' across-query ranks, higher still meaning more
    confident.

    Both inputs already follow the "higher is more confident" convention that
    `confidence` enforces, so ascending ranks preserve it and no sign flip is
    needed. Equal weight is deliberate: any tuned weighting would need its own
    held-out split to be honest, and the question here is only whether the two
    signals are complementary at all, not what the best mixture is.
    """
    if len(a) != len(b):
        raise ValueError("inputs differ in length (%d vs %d)" % (len(a), len(b)))
    ra, rb = _midranks(a), _midranks(b)
    return [(x + y) / 2.0 for x, y in zip(ra, rb)]


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation with mid-ranks for ties.

    Reported between proxies because two proxies that rank queries almost
    identically are one finding, not two: `base_entropy` reduces to a
    rescaled margin in some limits, so its separate AUROC only means
    something if its ordering is actually different.
    """
    n = len(a)
    if n != len(b):
        raise ValueError("inputs differ in length (%d vs %d)" % (n, len(b)))
    if n < 2:
        return None

    ra, rb = _midranks(a), _midranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if da == 0.0 or db == 0.0:
        # One proxy is constant, so it induces no ordering to correlate with.
        return None
    return num / (da * db)


def paired_bootstrap_auroc_delta(
    conf_a: Sequence[float],
    conf_b: Sequence[float],
    correct: Sequence[bool],
    n_replicates: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Bootstrap CI for AUROC(a) - AUROC(b), resampling QUERIES.

    Needed because "combined beats both parents" is a comparison of two
    numbers estimated from the same queries, and a raw difference says
    nothing about whether it would survive a different sample. The first
    CIRR run made this concrete: at k=5 the combined proxy beat its best
    parent by +0.0025, which a boolean verdict reported as a win and which
    is plainly inside sampling noise.

    Queries are resampled with replacement and BOTH proxies are re-scored on
    the same resample, so the pairing is preserved and the shared component
    of their variance cancels -- the same reason the fusion analysis uses a
    paired bootstrap rather than two independent ones.

    Replicates where either AUROC is undefined (a resample drawing only one
    class) are skipped and counted, rather than silently treated as zero.
    """
    n = len(correct)
    if not (len(conf_a) == len(conf_b) == n):
        raise ValueError("inputs differ in length (%d, %d, %d)"
                         % (len(conf_a), len(conf_b), n))
    if n == 0:
        raise ValueError("no queries")

    observed_a = auroc(conf_a, correct)
    observed_b = auroc(conf_b, correct)
    if observed_a is None or observed_b is None:
        return {"observed_delta": None, "ci_low": None, "ci_high": None,
                "p_two_sided": None, "n_valid": 0, "n_replicates": n_replicates}

    observed = observed_a - observed_b
    rng = random.Random(seed)
    deltas: List[float] = []

    for _ in range(n_replicates):
        idx = [rng.randrange(n) for _ in range(n)]
        cr = [correct[i] for i in idx]
        da = auroc([conf_a[i] for i in idx], cr)
        db = auroc([conf_b[i] for i in idx], cr)
        if da is None or db is None:
            continue
        deltas.append(da - db)

    if not deltas:
        return {"observed_delta": observed, "ci_low": None, "ci_high": None,
                "p_two_sided": None, "n_valid": 0,
                "n_replicates": n_replicates}

    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]

    # Two-sided p by the sign-reversal convention: how often the resampled
    # difference lands on the other side of zero from the observed one.
    if observed >= 0:
        tail = sum(1 for d in deltas if d <= 0.0)
    else:
        tail = sum(1 for d in deltas if d >= 0.0)
    p = min(1.0, 2.0 * (tail + 1) / (len(deltas) + 1))

    return {
        "observed_delta": observed,
        "ci_low": lo,
        "ci_high": hi,
        "p_two_sided": p,
        "n_valid": len(deltas),
        "n_replicates": n_replicates,
    }


def sensitivity_to_temperature(
    records: Sequence[dict],
    correct: Sequence[bool],
    temperatures: Sequence[float] = (0.003, 0.01, 0.03, 0.1),
) -> List[dict]:
    """AUROC of base_entropy across temperatures.

    ENTROPY_TEMPERATURE is a presentation choice, and a reviewer is entitled
    to ask whether the proxy's behaviour is an artefact of it. Reporting the
    sweep is cheaper than defending the single value.
    """
    out = []
    for t in temperatures:
        conf = [_neg_entropy(r["base_score"], t) for r in records]
        out.append({"temperature": t, "auroc": auroc(conf, correct)})
    return out


def analyse(path: str, k: int = 1, n_permute: int = N_PERMUTE,
            n_bootstrap: int = N_BOOTSTRAP) -> dict:
    records, multipositive, has_ug = load_records(path)
    correct = [is_correct(rec, k, multipositive) for rec in records]

    proxies = list(PROXIES) + ([UG_PROXY] if has_ug else [])

    curves: Dict[str, List[dict]] = {}
    aurocs: Dict[str, Optional[float]] = {}
    distinct: Dict[str, int] = {}
    conf_by_proxy: Dict[str, List[float]] = {}
    for proxy in proxies:
        conf = [confidence(rec, proxy) for rec in records]
        conf_by_proxy[proxy] = conf
        distinct[proxy] = len(set(conf))
        curves[proxy] = selective_curve(conf, correct, n_permute=n_permute)
        aurocs[proxy] = auroc(conf, correct)

    cross: Dict[str, Optional[float]] = {}
    for i, p in enumerate(proxies):
        for q in proxies[i + 1:]:
            cross["%s|%s" % (p, q)] = spearman(conf_by_proxy[p],
                                               conf_by_proxy[q])

    # Combined proxies are computed and reported separately from the single
    # ones, and each records its parents' AUROCs alongside its own, so the
    # only claim they can support -- "the combination beats both parts" --
    # can be read off without cross-referencing another table.
    combined: Dict[str, dict] = {}
    for name, (pa, pb) in COMBINED_PROXIES.items():
        if pa not in conf_by_proxy or pb not in conf_by_proxy:
            continue
        conf = combine_by_rank(conf_by_proxy[pa], conf_by_proxy[pb])
        a_comb = auroc(conf, correct)
        a_pa, a_pb = aurocs[pa], aurocs[pb]
        best_parent = None
        best_parent_name = None
        if a_pa is not None and a_pb is not None:
            best_parent_name = pa if a_pa >= a_pb else pb
            best_parent = max(a_pa, a_pb)

        # Compare against the BETTER parent, with a paired bootstrap. A bare
        # inequality is not enough: on the first CIRR run the combination
        # beat its best parent by +0.0025 at k=5, which is inside sampling
        # noise but which a boolean test called a win.
        boot = None
        if best_parent_name is not None:
            boot = paired_bootstrap_auroc_delta(
                conf, conf_by_proxy[best_parent_name], correct,
                n_replicates=n_bootstrap)

        beats = False
        if boot is not None and boot["ci_low"] is not None:
            # Significant improvement: the whole 95% CI for the difference
            # sits above zero.
            beats = boot["ci_low"] > 0.0

        combined[name] = {
            "parents": [pa, pb],
            "auroc": a_comb,
            "parent_auroc": {pa: a_pa, pb: a_pb},
            "best_parent": best_parent_name,
            "best_parent_auroc": best_parent,
            "delta_vs_best_parent": (None if (a_comb is None
                                              or best_parent is None)
                                     else a_comb - best_parent),
            "bootstrap_vs_best_parent": boot,
            "beats_best_parent_significantly": beats,
            "parent_spearman": spearman(conf_by_proxy[pa], conf_by_proxy[pb]),
            "curve": selective_curve(conf, correct, n_permute=n_permute),
            "n_distinct_values": len(set(conf)),
        }

    return {
        "path": path,
        "n_queries": len(records),
        "k": k,
        "multipositive": multipositive,
        "has_ug_confidence": has_ug,
        "n_correct_full": sum(correct),
        "base_accuracy": sum(correct) / len(correct),
        "proxies": proxies,
        "curves": curves,
        "auroc": aurocs,
        "n_distinct_values": distinct,
        "proxy_spearman": cross,
        "combined": combined,
        "n_permute": n_permute,
        "permute_seed": PERMUTE_SEED,
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "entropy_temperature": ENTROPY_TEMPERATURE,
        "entropy_temperature_sweep": sensitivity_to_temperature(records,
                                                               correct),
    }


def _fmt_p(p: Optional[float], n_permute: int) -> str:
    if p is None:
        return "     --"
    floor = 1.0 / (n_permute + 1)
    if p <= floor:
        return "<%.0e" % floor
    return "%7.4f" % p


def report(res: dict) -> None:
    n_permute = res["n_permute"]
    print("=" * 74)
    print("SELECTIVE RETRIEVAL AND CONFIDENCE ANALYSIS")
    print("=" * 74)
    print("cache            : %s" % os.path.basename(res["path"]))
    print("queries          : %d" % res["n_queries"])
    print("relevance        : %s" % ("multi-positive (CIRCO)"
                                     if res["multipositive"]
                                     else "single-target (CIRR)"))
    print("correct at       : top-%d" % res["k"])
    print("full-coverage acc: %.2f%% (%d/%d)"
          % (100.0 * res["base_accuracy"], res["n_correct_full"],
             res["n_queries"]))
    print("ug_confidence    : %s" % ("present" if res["has_ug_confidence"]
                                     else "ABSENT (gate disabled in this run)"))
    print("permutations     : %d, seed %d" % (n_permute, res["permute_seed"]))

    print()
    print("AUROC OF CONFIDENCE AS A PREDICTOR OF TOP-%d CORRECTNESS" % res["k"])
    print("  (0.5 = uninformative; a selective curve can only rise if this "
          "> 0.5)")
    for proxy in res["proxies"]:
        a = res["auroc"][proxy]
        print("    %-16s %s" % (proxy, "undefined (one class empty)"
                                if a is None else "%.4f" % a))

    print()
    print("SELECTIVE RETRIEVAL CURVES")
    for proxy in res["proxies"]:
        rows = res["curves"][proxy]
        print()
        print("  %s (%d distinct values across %d queries)"
              % (proxy, res["n_distinct_values"][proxy], res["n_queries"]))
        print("    %-9s %7s %9s %9s %9s" % ("coverage", "kept", "accuracy",
                                            "delta", "perm p"))
        base_acc = rows[0]["accuracy"]
        for r in rows:
            print("    %7.0f%%  %7d  %8.2f%%  %+8.2f  %s"
                  % (100.0 * r["coverage"], r["n_kept"],
                     100.0 * r["accuracy"],
                     100.0 * (r["accuracy"] - base_acc),
                     _fmt_p(r["p_permutation"], n_permute)))

    if res.get("combined"):
        print()
        print("COMBINED PROXIES (mean of the two parents' across-query ranks)")
        print("  Tests whether the verifier's margin carries selective signal")
        print("  the base scores do not already have.")
        for name, c in res["combined"].items():
            pa, pb = c["parents"]
            print()
            print("  %s" % name)
            print("    AUROC combined  %s" % ("undefined" if c["auroc"] is None
                                              else "%.4f" % c["auroc"]))
            for p in (pa, pb):
                a = c["parent_auroc"][p]
                print("    AUROC %-14s %s" % (p, "undefined" if a is None
                                              else "%.4f" % a))
            rho = c["parent_spearman"]
            print("    parents' Spearman  %s"
                  % ("undefined" if rho is None else "%+.4f" % rho))
            d = c["delta_vs_best_parent"]
            print("    delta vs best parent (%s)  %s"
                  % (c["best_parent"],
                     "undefined" if d is None else "%+.4f" % d))
            boot = c["bootstrap_vs_best_parent"]
            if boot and boot["ci_low"] is not None:
                print("    paired bootstrap 95%% CI  [%+.4f, %+.4f]  p=%.4f"
                      % (boot["ci_low"], boot["ci_high"],
                         boot["p_two_sided"]))
                print("      (%d/%d replicates valid)"
                      % (boot["n_valid"], boot["n_replicates"]))
            if c["beats_best_parent_significantly"]:
                verdict = "YES"
            elif boot and boot["ci_high"] is not None and boot["ci_high"] < 0.0:
                # The combination is significantly WORSE, which is a distinct
                # finding from "no detectable difference" and must not be
                # reported as though the CI merely spanned zero.
                verdict = "no -- significantly WORSE than the best parent"
            elif boot and boot["ci_low"] is not None:
                verdict = "no -- CI includes zero"
            else:
                verdict = "no"
            print("    beats best parent significantly: %s" % verdict)
            rows = c["curve"]
            base_acc = rows[0]["accuracy"]
            print("    %-9s %7s %9s %9s %9s" % ("coverage", "kept", "accuracy",
                                                "delta", "perm p"))
            for r in rows:
                print("    %7.0f%%  %7d  %8.2f%%  %+8.2f  %s"
                      % (100.0 * r["coverage"], r["n_kept"],
                         100.0 * r["accuracy"],
                         100.0 * (r["accuracy"] - base_acc),
                         _fmt_p(r["p_permutation"], n_permute)))

    print()
    print("PROXY RANK AGREEMENT (Spearman)")
    print("  (near 1.0 = the same ordering, so not an independent finding)")
    for key, rho in res["proxy_spearman"].items():
        p, q = key.split("|")
        print("    %-16s %-16s %s" % (p, q, "undefined (constant proxy)"
                                      if rho is None else "%+.4f" % rho))

    print()
    print("ENTROPY TEMPERATURE SENSITIVITY (base_entropy AUROC)")
    for row in res["entropy_temperature_sweep"]:
        a = row["auroc"]
        mark = ("  <- reported"
                if abs(row["temperature"] - res["entropy_temperature"]) < 1e-12
                else "")
        print("    T=%-7.3f %s%s" % (row["temperature"],
                                     "undefined" if a is None else "%.4f" % a,
                                     mark))

    print()
    print("NOTE ON ECE")
    print("  Not computed here. evaluate.py's ECE compares ug_confidence")
    print("  against base top-1 correctness and therefore requires a")
    print("  gate-enabled run. See the module docstring.")
    print("=" * 74)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache", help="rv_score_cache_*.jsonl")
    ap.add_argument("-k", type=int, default=1,
                    help="correctness threshold: correct if a relevant "
                         "candidate is in the top k (default 1)")
    ap.add_argument("--permutations", type=int, default=N_PERMUTE)
    ap.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP,
                    help="paired-bootstrap replicates for the "
                         "combined-vs-parent AUROC comparison")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="also write the full result as JSON here")
    args = ap.parse_args(argv)

    if args.k < 1:
        print("error: -k must be >= 1", file=sys.stderr)
        return 2
    if args.permutations < 1:
        print("error: --permutations must be >= 1", file=sys.stderr)
        return 2
    if args.bootstrap < 1:
        print("error: --bootstrap must be >= 1", file=sys.stderr)
        return 2

    try:
        res = analyse(args.cache, k=args.k, n_permute=args.permutations,
                      n_bootstrap=args.bootstrap)
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
