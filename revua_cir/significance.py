"""
Significance testing for paired binary retrieval outcomes (R@1 hit/miss on
the same query set under two systems).

**Why these specific tests** -- chosen after a dedicated literature pass,
not by default:

* **McNemar's exact binomial** is the correct test for paired binary
  outcomes: every query is scored by both systems, and the test conditions
  on exactly the discordant pairs (base-right/new-wrong vs
  base-wrong/new-right). Note that IR's own methodological convention
  (Smucker, Allan & Carterette, "A Comparison of Statistical Significance
  Tests for Information Retrieval Evaluation", CIKM 2007) recommends
  Fisher's randomization test and argues AGAINST the sign test because it
  discards magnitude information. McNemar is the sign test's binary
  sibling -- but for a strictly binary metric like R@1 there IS no
  magnitude to discard, so that objection does not apply here. Reporting
  the paired bootstrap CI alongside (below) covers the field-convention
  expectation.

* **Exact binomial, NOT the continuity-corrected chi-square.** Fagerland,
  Lydersen & Laake ("The McNemar test for binary matched-pairs data:
  mid-p and asymptotic are better than exact conditional", BMC Medical
  Research Methodology, 2013) find the continuity-corrected asymptotic
  test (Edwards' correction) over-corrects and performs poorly. At the
  discordant-pair counts this project actually sees (b+c in the hundreds)
  exact, mid-p and uncorrected asymptotic all agree to several decimals,
  so this is a formality -- but the exact/mid-p pair is the defensible
  choice and costs nothing. `mcnemar_midp` is provided for completeness.

* **Paired bootstrap** gives what McNemar cannot: a confidence interval on
  the effect size itself (the R@1 difference in percentage points), which
  is more informative than a p-value and is the increasingly common format
  in neural-retrieval papers. It resamples QUERIES with their (base, new)
  outcome pair kept together -- resampling outcomes independently would
  destroy the pairing and inflate the interval.

Deliberately dependency-free (pure stdlib `math`/`random`): `scipy.stats`
is NOT a dependency of this project, because importing it alongside torch +
scikit-learn triggered a real, fatal `OMP: Error #15` duplicate-OpenMP
crash on this project's dev environment (see revua_cir/evaluate.py's
`_spearman_corr` docstring for that incident). The exact-binomial
implementation here was validated against `scipy.stats.binomtest` to
machine precision on seven cases including the degenerate b=c=0 and b=c=1
before being committed.
"""
from __future__ import annotations

import random
from math import comb
from typing import Sequence


def mcnemar_exact(n_help: int, n_harm: int) -> float:
    """Two-sided exact-binomial McNemar p-value.

    `n_help`: queries the new system fixed (base wrong -> new right).
    `n_harm`: queries the new system broke (base right -> new wrong).
    Concordant pairs (both right / both wrong) are correctly ignored --
    that conditioning is the whole point of the test.

    Returns 1.0 when there are no discordant pairs at all (nothing to
    distinguish the systems), rather than dividing by zero.

    **On tail-doubling vs "the general exact two-sided binomial test"**
    (external review, 2026-09-12, asked whether doubling the smaller tail
    is only an approximation that could diverge on asymmetric counts like
    help=8/harm=3): under McNemar's null the success probability is
    ALWAYS exactly 0.5, which makes the binomial pmf symmetric about n/2.
    scipy's general two-sided rule -- sum the probabilities of all
    outcomes at least as extreme as (i.e. no more probable than) the
    observed one -- therefore selects exactly the set {x <= lo} union
    {x >= n-lo}, whose mass is precisely 2*P(X <= lo). Verified two ways
    before keeping this implementation: (a) the two rejection REGIONS were
    shown to be identical sets for every n <= 40 and every possible
    observation, and (b) the p-values matched `scipy.stats.binomtest` to
    <= 1.7e-15 across all 3721 (help, harm) pairs in 0..60 x 0..60 plus
    extreme-imbalance cases up to 1000:0 and 0:1000. A control at p=0.3
    confirmed the two regions DO diverge when the null is asymmetric --
    i.e. the equality is a genuine consequence of p=0.5, not a numerical
    coincidence. So this is the exact two-sided test, not an approximation
    to it, and the name `mcnemar_exact` is accurate.
    """
    if n_help < 0 or n_harm < 0:
        raise ValueError(f"counts must be non-negative, got help={n_help} harm={n_harm}")
    n = n_help + n_harm
    if n == 0:
        return 1.0
    lo = min(n_help, n_harm)
    # P(X <= lo) under Binomial(n, 0.5), doubled for a two-sided test.
    p = sum(comb(n, i) for i in range(lo + 1)) / (2 ** n) * 2
    return min(p, 1.0)


def mcnemar_midp(n_help: int, n_harm: int) -> float:
    """Mid-p variant (Fagerland et al. 2013): the exact p-value minus half
    the point mass at the observed value. Slightly less conservative than
    the exact conditional test while avoiding Edwards' over-correction.
    Reported alongside the exact value, never instead of it.
    """
    if n_help < 0 or n_harm < 0:
        raise ValueError(f"counts must be non-negative, got help={n_help} harm={n_harm}")
    n = n_help + n_harm
    if n == 0:
        return 1.0
    lo = min(n_help, n_harm)
    p_exact = sum(comb(n, i) for i in range(lo + 1)) / (2 ** n) * 2
    p_point = comb(n, lo) / (2 ** n)
    return min(max(p_exact - p_point, 0.0), 1.0)


def paired_bootstrap_ci(base_hits: Sequence[int], new_hits: Sequence[int],
                        n_replicates: int = 1000, seed: int = 0,
                        alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI on the R@1 difference (new - base), in
    percentage points.

    `base_hits`/`new_hits`: parallel 0/1 sequences, one entry per query,
    ALIGNED BY QUERY. Resampling draws query INDICES, so each query's
    (base, new) pair moves together -- preserving the pairing that makes
    this a paired test. Resampling the two sequences independently would
    be a different (and wrong) analysis.
    """
    if len(base_hits) != len(new_hits):
        raise ValueError(f"base_hits ({len(base_hits)}) and new_hits ({len(new_hits)}) "
                          f"must be the same length -- they are paired per query")
    n = len(base_hits)
    if n == 0:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_replicates": 0, "straddles_zero": True}
    if n_replicates < 1:
        raise ValueError(f"n_replicates must be >= 1, got {n_replicates}")

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_replicates):
        idx = [rng.randrange(n) for _ in range(n)]
        b = sum(base_hits[i] for i in idx)
        v = sum(new_hits[i] for i in idx)
        diffs.append(100.0 * (v - b) / n)
    diffs.sort()

    lo_i = int((alpha / 2) * n_replicates)
    hi_i = int((1 - alpha / 2) * n_replicates) - 1
    lo_i = max(0, min(lo_i, n_replicates - 1))
    hi_i = max(0, min(hi_i, n_replicates - 1))
    ci_low, ci_high = diffs[lo_i], diffs[hi_i]

    observed = 100.0 * (sum(new_hits) - sum(base_hits)) / n
    return {
        "observed_diff": observed,
        "mean_diff": sum(diffs) / len(diffs),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_replicates": n_replicates,
        "straddles_zero": bool(ci_low <= 0.0 <= ci_high),
    }


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down correction for a family of `pvalues`
    (e.g. one per lambda in a sweep). Returns a parallel list of booleans:
    True where the hypothesis is rejected at family-wise alpha.

    Holm rather than plain Bonferroni: uniformly more powerful, same
    family-wise error control, no extra assumptions. Used as a SUPPLEMENT
    to a proper tune/report split, never as a substitute for it -- a
    correction bounds the false-positive rate across a sweep but does not
    undo the optimistic bias of selecting the argmax on the reported data
    (Jensen & Cohen, "Multiple Comparisons in Induction Algorithms",
    Machine Learning 38(3), 2000).
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    rejected = [False] * m
    for rank, i in enumerate(order):
        threshold = alpha / (m - rank)
        if pvalues[i] <= threshold:
            rejected[i] = True
        else:
            break  # step-down: once one fails, all larger p-values fail too
    return rejected


def split_indices(n: int, tune_frac: float = 0.5, seed: int = 12345) -> tuple[list[int], list[int]]:
    """Deterministic random split of `n` queries into (tune, report) index
    lists. Selection of any hyperparameter (lambda, margin, ...) must
    happen on TUNE only; the reported number and its significance test
    come from REPORT only.

    Fixed default seed so the split is reproducible across runs and across
    machines -- a split that moved between invocations would let a caller
    (even unintentionally) reroll it until the reported number improved,
    which is the same selection bias this split exists to prevent.
    """
    if not 0.0 < tune_frac < 1.0:
        raise ValueError(f"tune_frac must be strictly between 0 and 1, got {tune_frac}")
    if n < 2:
        raise ValueError(f"need at least 2 queries to split, got {n}")
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = int(round(n * tune_frac))
    cut = max(1, min(cut, n - 1))  # never produce an empty side
    return sorted(idx[:cut]), sorted(idx[cut:])
