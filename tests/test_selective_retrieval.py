"""Regression tests for selective_retrieval_analysis.

The failure mode this analysis is most exposed to is a curve that rises for a
reason other than the one claimed: a sign error in a proxy, a correctness rule
applied under the wrong relevance convention, a tie-ordering that happens to
correlate with the answer, or a p-value floor mistaken for significance. Each
test below targets one of those specifically.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

try:
    import pytest
except ImportError:  # pragma: no cover - cluster venv without pytest
    # Minimal stand-ins so the standalone runner at the bottom of this file
    # works on the login node, where the venv has no pytest. Only the three
    # pytest features these tests actually use are provided; anything else
    # would fail loudly rather than silently pass.
    class _Approx:
        def __init__(self, expected, abs=None, rel=None):
            self.expected, self.abs, self.rel = expected, abs, rel

        def __eq__(self, other):
            tol = self.abs if self.abs is not None else max(
                1e-12, (self.rel if self.rel is not None else 1e-6)
                * abs(self.expected))
            return abs(other - self.expected) <= tol

        def __repr__(self):
            return "approx(%r)" % (self.expected,)

    class _Skipped(Exception):
        pass

    class _RaisesCtx:
        def __init__(self, exc, match=None):
            self.exc, self.match = exc, match

        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            if et is None:
                raise AssertionError("DID NOT RAISE %s" % (self.exc,))
            if not issubclass(et, self.exc):
                return False
            if self.match is not None:
                import re as _re
                if not _re.search(self.match, str(ev)):
                    raise AssertionError(
                        "pattern %r not found in %r" % (self.match, str(ev)))
            return True

    class _PytestShim:
        Skipped = _Skipped

        @staticmethod
        def approx(expected, abs=None, rel=None):
            return _Approx(expected, abs=abs, rel=rel)

        @staticmethod
        def raises(exc, match=None):
            return _RaisesCtx(exc, match=match)

        @staticmethod
        def skip(reason=""):
            raise _Skipped(reason)

    pytest = _PytestShim()  # type: ignore[assignment]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import selective_retrieval_analysis as sra  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_record(query_id, base, rv=None, target_pos=None, relevant=None,
                ug=None):
    n = len(base)
    rec = {
        "query_id": query_id,
        "candidate_gallery_ids": list(range(query_id * 100,
                                            query_id * 100 + n)),
        "base_score": list(base),
        "verification_score": list(rv) if rv is not None else [0.5] * n,
    }
    if target_pos is not None:
        rec["target_pos"] = target_pos
    if relevant is not None:
        rec["relevant_positions_in_candidates"] = list(relevant)
    if ug is not None:
        rec["ug_confidence"] = list(ug)
    return rec


def write_cache(records):
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8")
    for rec in records:
        fh.write(json.dumps(rec) + "\n")
    fh.close()
    return fh.name


# ---------------------------------------------------------------------------
# proxy definitions
# ---------------------------------------------------------------------------

def test_neg_entropy_is_higher_for_a_peaked_distribution():
    """The sign convention: HIGHER must mean MORE confident.

    If this returned entropy rather than its negation, every proxy would sort
    the opposite way from the others and a decreasing curve would be reported
    as increasing. The mistake is invisible in the output, so it is tested at
    the source.
    """
    peaked = sra._neg_entropy([1.0, 0.0, 0.0, 0.0], 0.1)
    flat = sra._neg_entropy([0.25, 0.25, 0.25, 0.25], 0.1)
    assert peaked > flat


def test_softmax_sums_to_one_and_is_shift_invariant():
    p = sra._softmax([0.61, 0.60, 0.59], 0.01)
    assert abs(sum(p) - 1.0) < 1e-12
    q = sra._softmax([1.61, 1.60, 1.59], 0.01)
    for a, b in zip(p, q):
        assert abs(a - b) < 1e-12


def test_base_margin_and_rv_margin_read_the_right_fields():
    rec = make_record(0, [0.9, 0.4, 0.1], rv=[0.2, 0.15, 0.1], target_pos=0)
    assert sra.confidence(rec, "base_margin") == pytest.approx(0.5)
    assert sra.confidence(rec, "rv_margin") == pytest.approx(0.05)
    assert sra.confidence(rec, "base_top1") == pytest.approx(0.9)


def test_unknown_proxy_raises():
    rec = make_record(0, [0.9, 0.4], target_pos=0)
    with pytest.raises(KeyError):
        sra.confidence(rec, "no_such_proxy")


# ---------------------------------------------------------------------------
# correctness conventions
# ---------------------------------------------------------------------------

def test_single_target_correctness_respects_k():
    rec = make_record(0, [0.9, 0.5, 0.2], target_pos=2)
    assert not sra.is_correct(rec, 1, multipositive=False)
    assert not sra.is_correct(rec, 2, multipositive=False)
    assert sra.is_correct(rec, 3, multipositive=False)


def test_target_pos_minus_one_is_never_correct():
    """target_pos == -1 means the target missed the window entirely."""
    rec = make_record(0, [0.9, 0.5], target_pos=-1)
    for k in (1, 2, 10):
        assert not sra.is_correct(rec, k, multipositive=False)


def test_multipositive_correctness_uses_any_relevant_position():
    rec = make_record(0, [0.9, 0.5, 0.2], relevant=[1, 2])
    assert not sra.is_correct(rec, 1, multipositive=True)
    assert sra.is_correct(rec, 2, multipositive=True)


def test_multipositive_cache_scored_as_single_target_would_understate():
    """Why the convention is chosen per file rather than assumed.

    This record's only relevant candidate sits at position 0, but it carries
    no target_pos -- scoring it under the single-target rule would call it a
    miss. That is the silent under-reporting the per-file determination
    prevents.
    """
    rec = make_record(0, [0.9, 0.5], relevant=[0])
    assert sra.is_correct(rec, 1, multipositive=True)
    assert not sra.is_correct(rec, 1, multipositive=False)


def test_mixed_relevance_conventions_abort():
    path = write_cache([
        make_record(0, [0.9, 0.5], target_pos=0),
        make_record(1, [0.8, 0.5], relevant=[0]),
    ])
    try:
        with pytest.raises(sra.CacheError, match="entirely single-target"):
            sra.load_records(path)
    finally:
        os.unlink(path)


def test_mixed_ug_presence_aborts():
    path = write_cache([
        make_record(0, [0.9, 0.5], target_pos=0, ug=[0.7, 0.6]),
        make_record(1, [0.8, 0.5], target_pos=0),
    ])
    try:
        with pytest.raises(sra.CacheError, match="ug_confidence"):
            sra.load_records(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# structural validation
# ---------------------------------------------------------------------------

def test_out_of_range_target_pos_is_rejected():
    path = write_cache([make_record(0, [0.9, 0.5], target_pos=7)])
    try:
        with pytest.raises(sra.CacheError, match="outside"):
            sra.load_records(path)
    finally:
        os.unlink(path)


def test_out_of_range_relevant_position_is_rejected():
    path = write_cache([make_record(0, [0.9, 0.5], relevant=[5])])
    try:
        with pytest.raises(sra.CacheError, match="silently score as misses"):
            sra.load_records(path)
    finally:
        os.unlink(path)


def test_duplicate_candidate_ids_are_rejected():
    rec = make_record(0, [0.9, 0.5], target_pos=0)
    rec["candidate_gallery_ids"] = [4, 4]
    path = write_cache([rec])
    try:
        with pytest.raises(sra.CacheError, match="duplicate candidate"):
            sra.load_records(path)
    finally:
        os.unlink(path)


def test_duplicate_query_ids_are_rejected():
    path = write_cache([
        make_record(0, [0.9, 0.5], target_pos=0),
        make_record(0, [0.8, 0.5], target_pos=1),
    ])
    try:
        with pytest.raises(sra.CacheError, match="duplicate query_id"):
            sra.load_records(path)
    finally:
        os.unlink(path)


def test_length_mismatch_is_rejected():
    rec = make_record(0, [0.9, 0.5, 0.2], target_pos=0)
    rec["verification_score"] = [0.5, 0.4]
    path = write_cache([rec])
    try:
        with pytest.raises(sra.CacheError, match="verification_score has 2"):
            sra.load_records(path)
    finally:
        os.unlink(path)


def test_nonfinite_score_is_rejected():
    rec = make_record(0, [0.9, 0.5], target_pos=0)
    rec["base_score"] = [float("nan"), 0.5]
    path = write_cache([rec])
    try:
        with pytest.raises(sra.CacheError, match="non-finite"):
            sra.load_records(path)
    finally:
        os.unlink(path)


def test_single_candidate_window_is_rejected():
    """A margin proxy needs two candidates; a 1-wide window would IndexError
    deep inside the curve rather than at load."""
    path = write_cache([make_record(0, [0.9], target_pos=0)])
    try:
        with pytest.raises(sra.CacheError, match="at least 2"):
            sra.load_records(path)
    finally:
        os.unlink(path)


def test_empty_cache_is_rejected():
    path = write_cache([])
    try:
        with pytest.raises(sra.CacheError, match="no records"):
            sra.load_records(path)
    finally:
        os.unlink(path)


def test_malformed_json_names_the_line():
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8")
    fh.write(json.dumps(make_record(0, [0.9, 0.5], target_pos=0)) + "\n")
    fh.write("{not json\n")
    fh.close()
    try:
        with pytest.raises(sra.CacheError, match=":2:"):
            sra.load_records(fh.name)
    finally:
        os.unlink(fh.name)


def test_blank_lines_are_skipped_without_shifting_line_numbers():
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8")
    fh.write(json.dumps(make_record(0, [0.9, 0.5], target_pos=0)) + "\n")
    fh.write("\n")
    fh.write(json.dumps(make_record(1, [0.8, 0.5], target_pos=1)) + "\n")
    fh.close()
    try:
        records, multi, has_ug = sra.load_records(fh.name)
        assert len(records) == 2
        assert multi is False and has_ug is False
    finally:
        os.unlink(fh.name)


# ---------------------------------------------------------------------------
# the curve itself
# ---------------------------------------------------------------------------

def test_perfect_confidence_gives_a_rising_curve():
    """Confidence ordered exactly with correctness: the curve must rise to
    100% once all the failures have been withheld."""
    conf = [float(10 - i) for i in range(10)]
    correct = [True] * 5 + [False] * 5
    rows = sra.selective_curve(conf, correct, coverages=(1.0, 0.5),
                               n_permute=200)
    assert rows[0]["accuracy"] == pytest.approx(0.5)
    assert rows[1]["accuracy"] == pytest.approx(1.0)


def test_anticorrelated_confidence_gives_a_falling_curve():
    """The counterpart. Without this, a sign error that inverted every proxy
    would still pass the rising-curve test above on some other input."""
    conf = [float(i) for i in range(10)]
    correct = [True] * 5 + [False] * 5
    rows = sra.selective_curve(conf, correct, coverages=(1.0, 0.5),
                               n_permute=200)
    assert rows[1]["accuracy"] == pytest.approx(0.0)


def test_constant_confidence_leaves_accuracy_at_the_base_rate():
    """A proxy with no information must not appear to work.

    With every confidence equal, the stable tie-break keeps the first m
    queries. Interleaving outcomes makes that subset representative, so the
    accuracy should stay at the base rate -- if it moved, the tie-break
    itself would be carrying signal.
    """
    conf = [0.5] * 10
    correct = [i % 2 == 0 for i in range(10)]
    rows = sra.selective_curve(conf, correct, coverages=(1.0, 0.8, 0.6, 0.4),
                               n_permute=200)
    for r in rows:
        assert r["accuracy"] == pytest.approx(0.5)


def test_full_coverage_p_value_is_none_not_one():
    rows = sra.selective_curve([1.0, 2.0, 3.0], [True, False, True],
                               coverages=(1.0,), n_permute=100)
    assert rows[0]["p_permutation"] is None


def test_permutation_p_is_floored_not_zero():
    conf = [float(100 - i) for i in range(100)]
    correct = [True] * 50 + [False] * 50
    rows = sra.selective_curve(conf, correct, coverages=(0.5,), n_permute=100)
    p = rows[0]["p_permutation"]
    assert p == pytest.approx(1.0 / 101.0)
    assert p > 0.0


def test_uninformative_confidence_gives_a_large_p_value():
    """The null actually behaves like a null. A random confidence ordering
    should not produce a significant p at any coverage."""
    import random as _random
    rnd = _random.Random(7)
    n = 400
    correct = [rnd.random() < 0.4 for _ in range(n)]
    conf = [rnd.random() for _ in range(n)]
    rows = sra.selective_curve(conf, correct, coverages=(0.8, 0.5),
                               n_permute=2000)
    for r in rows:
        assert r["p_permutation"] > 0.01


def test_permutation_p_matches_the_exact_hypergeometric_tail():
    """Pins the permutation p against a closed form.

    This test exists because of a real defect. The first version of
    `selective_curve` compared `sum(pool[:m]) / m >= acc` -- two float
    divisions of the same rational -- and dropped borderline replicates where
    they failed to compare equal. On the real CIRCO cache at 90% coverage
    that reported p = 0.107 where the exact answer is 0.0102, a tenfold
    overstatement in the conservative direction. Comparing integer counts
    fixed it; this checks the fix against the closed form rather than against
    another sampler.
    """
    from math import comb

    n, n_success, m = 220, 70, 198
    need = 68

    # Construct outcomes whose most-confident m contain exactly `need` hits.
    correct = [True] * need + [False] * (m - need)
    correct += [True] * (n_success - need) + [False] * (n - m - (n_success - need))
    assert len(correct) == n and sum(correct) == n_success
    conf = [float(n - i) for i in range(n)]  # already in confidence order

    rows = sra.selective_curve(conf, correct, coverages=(m / n,),
                               n_permute=40000)
    assert rows[0]["n_kept"] == m
    assert rows[0]["n_correct"] == need

    exact = sum(comb(n_success, x) * comb(n - n_success, m - x)
                for x in range(need, min(n_success, m) + 1)) / comb(n, m)
    assert exact == pytest.approx(0.010216, abs=1e-5)

    got = rows[0]["p_permutation"]
    se = math.sqrt(exact * (1 - exact) / 40000)
    assert abs(got - exact) < max(4 * se, 0.002), (got, exact)


def test_hypergeom_respects_its_bounds():
    """A draw cannot exceed the successes available or the draws taken."""
    import random as _random
    rnd = _random.Random(5)
    for n, k, m in [(100, 40, 30), (100, 40, 70), (10, 10, 3), (10, 0, 3),
                    (50, 25, 50), (7, 3, 1)]:
        for _ in range(200):
            v = sra._hypergeom_rvs(rnd, n, k, m)
            assert 0 <= v <= min(k, m)
            assert v >= max(0, m - (n - k))


def test_hypergeom_degenerate_cases_are_exact():
    import random as _random
    rnd = _random.Random(5)
    # No successes in the population: never draw one.
    assert sra._hypergeom_rvs(rnd, 20, 0, 5) == 0
    # Every member is a success: every draw is one.
    assert sra._hypergeom_rvs(rnd, 20, 20, 5) == 5
    # Drawing the whole population recovers the success count exactly.
    assert sra._hypergeom_rvs(rnd, 20, 7, 20) == 7


def test_hypergeom_mean_matches_the_analytic_value():
    """E[X] = m * K / N. A sequential sampler that failed to decrement the
    population would be biased, and this is where that shows."""
    import random as _random
    rnd = _random.Random(17)
    n, k, m, trials = 200, 80, 50, 20000
    total = sum(sra._hypergeom_rvs(rnd, n, k, m) for _ in range(trials))
    mean = total / trials
    expected = m * k / n
    # Analytic sd of the hypergeometric, so the tolerance is principled
    # rather than guessed.
    var = m * (k / n) * (1 - k / n) * (n - m) / (n - 1)
    se = math.sqrt(var / trials)
    assert abs(mean - expected) < 4 * se, (mean, expected, se)


def test_permutation_matches_an_explicit_shuffle():
    """The optimisation this pins down.

    `selective_curve` samples the null count hypergeometrically instead of
    shuffling the outcome vector, because the shuffle costs minutes at 4181
    queries. The two must agree, so this recomputes a p-value the slow,
    obviously-correct way and compares. A discrepancy here would mean every
    reported p-value is wrong.
    """
    import random as _random
    rnd = _random.Random(23)
    n = 300
    correct = [rnd.random() < 0.35 for _ in range(n)]
    # Confidence mildly informative, so the p-value lands away from 0 and 1
    # where a disagreement would actually be visible.
    conf = [(0.6 if ok else 0.4) + rnd.random() for ok in correct]

    cov = 0.5
    rows = sra.selective_curve(conf, correct, coverages=(cov,),
                               n_permute=20000)
    fast_p = rows[0]["p_permutation"]

    order = sorted(range(n), key=lambda i: (-conf[i], i))
    outcomes = [correct[i] for i in order]
    m = int(round(cov * n))
    need = sum(outcomes[:m])
    total_correct = sum(outcomes)

    slow_rng = _random.Random(sra.PERMUTE_SEED)
    pool = [True] * total_correct + [False] * (n - total_correct)
    hits = 0
    reps = 20000
    for _ in range(reps):
        slow_rng.shuffle(pool)
        if sum(pool[:m]) >= need:
            hits += 1
    slow_p = (hits + 1) / (reps + 1)

    # Both are Monte Carlo estimates of the same quantity, so they agree to
    # sampling error, not exactly. 4 sigma on a binomial proportion.
    se = math.sqrt(max(slow_p * (1 - slow_p), 1e-9) / reps)
    assert abs(fast_p - slow_p) < max(4 * se, 0.005), (fast_p, slow_p, se)


def test_curve_is_deterministic_across_calls():
    conf = [float(50 - i) for i in range(50)]
    correct = [i % 3 != 0 for i in range(50)]
    a = sra.selective_curve(conf, correct, n_permute=500)
    b = sra.selective_curve(conf, correct, n_permute=500)
    assert a == b


def test_coverage_never_yields_an_empty_subset():
    """At small n, a low coverage rounds toward zero; m must clamp to 1 rather
    than divide by zero."""
    rows = sra.selective_curve([1.0, 2.0], [True, False],
                               coverages=(0.1,), n_permute=50)
    assert rows[0]["n_kept"] == 1


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="differ in length"):
        sra.selective_curve([1.0, 2.0], [True], n_permute=10)


def test_n_correct_matches_accuracy_times_n_kept():
    """Guards the denominator bug class: a numerator counted over a filtered
    subset and a denominator taken from the unfiltered total."""
    conf = [float(20 - i) for i in range(20)]
    correct = [i % 2 == 0 for i in range(20)]
    for r in sra.selective_curve(conf, correct, n_permute=100):
        assert r["n_correct"] == pytest.approx(r["accuracy"] * r["n_kept"])


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------

def test_auroc_is_one_for_perfect_separation():
    assert sra.auroc([3.0, 2.0, 1.0, 0.0],
                     [True, True, False, False]) == pytest.approx(1.0)


def test_auroc_is_zero_for_perfect_inversion():
    assert sra.auroc([0.0, 1.0, 2.0, 3.0],
                     [True, True, False, False]) == pytest.approx(0.0)


def test_auroc_is_half_for_all_ties():
    assert sra.auroc([1.0] * 6,
                     [True, True, True, False, False, False]) == pytest.approx(0.5)


def test_auroc_undefined_when_one_class_is_empty():
    assert sra.auroc([1.0, 2.0], [True, True]) is None
    assert sra.auroc([1.0, 2.0], [False, False]) is None


def test_auroc_matches_a_brute_force_pair_count_with_ties():
    """The rank-sum shortcut must agree with the definition, including the
    half-credit convention for ties."""
    import random as _random
    rnd = _random.Random(11)
    conf = [round(rnd.random(), 2) for _ in range(120)]  # forces ties
    correct = [rnd.random() < 0.45 for _ in range(120)]

    pos = [c for c, ok in zip(conf, correct) if ok]
    neg = [c for c, ok in zip(conf, correct) if not ok]
    wins = sum((1.0 if p > q else 0.5 if p == q else 0.0)
               for p in pos for q in neg)
    expected = wins / (len(pos) * len(neg))

    assert sra.auroc(conf, correct) == pytest.approx(expected)


def test_auroc_above_half_is_necessary_for_a_rising_curve():
    """Ties the two summaries together: an AUROC below 0.5 cannot coexist
    with a curve that rises at every coverage."""
    conf = [float(i) for i in range(40)]
    correct = [i < 20 for i in range(40)]  # confidence anti-correlated
    a = sra.auroc(conf, correct)
    rows = sra.selective_curve(conf, correct, coverages=(1.0, 0.5),
                               n_permute=100)
    assert a < 0.5
    assert rows[1]["accuracy"] < rows[0]["accuracy"]


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_analyse_end_to_end_single_target():
    records = []
    for i in range(40):
        hit = i < 16
        # Confident queries are the correct ones, by construction.
        gap = 0.20 if hit else 0.01
        records.append(make_record(
            i, [0.70, 0.70 - gap, 0.70 - gap - 0.01],
            rv=[0.60, 0.59, 0.58],
            target_pos=0 if hit else -1))
    path = write_cache(records)
    try:
        res = sra.analyse(path, k=1, n_permute=500)
        assert res["n_queries"] == 40
        assert res["multipositive"] is False
        assert res["has_ug_confidence"] is False
        assert sra.UG_PROXY not in res["proxies"]
        assert res["base_accuracy"] == pytest.approx(16 / 40)
        assert res["auroc"]["base_margin"] == pytest.approx(1.0)
        curve = res["curves"]["base_margin"]
        assert curve[0]["accuracy"] < curve[-1]["accuracy"]
        # rv_margin is constant here, so it must show no selective effect.
        assert res["auroc"]["rv_margin"] == pytest.approx(0.5)
        report_ok = res["curves"]["rv_margin"][0]["accuracy"]
        assert report_ok == pytest.approx(16 / 40)
    finally:
        os.unlink(path)


def test_analyse_includes_ug_proxy_when_present():
    records = [make_record(i, [0.7, 0.6], rv=[0.5, 0.4],
                           target_pos=0 if i < 5 else -1,
                           ug=[0.9 if i < 5 else 0.1, 0.5])
               for i in range(10)]
    path = write_cache(records)
    try:
        res = sra.analyse(path, k=1, n_permute=200)
        assert res["has_ug_confidence"] is True
        assert sra.UG_PROXY in res["proxies"]
        assert res["auroc"][sra.UG_PROXY] == pytest.approx(1.0)
    finally:
        os.unlink(path)


def test_analyse_on_multipositive_cache_uses_the_multipositive_rule():
    records = [make_record(i, [0.7, 0.6, 0.5], rv=[0.5, 0.4, 0.3],
                           relevant=[0] if i < 6 else [2])
               for i in range(12)]
    path = write_cache(records)
    try:
        res = sra.analyse(path, k=1, n_permute=200)
        assert res["multipositive"] is True
        assert res["base_accuracy"] == pytest.approx(0.5)
    finally:
        os.unlink(path)


def test_report_runs_without_error(capsys):
    records = [make_record(i, [0.7, 0.65, 0.6], rv=[0.5, 0.45, 0.4],
                           target_pos=0 if i % 2 == 0 else -1)
               for i in range(20)]
    path = write_cache(records)
    try:
        res = sra.analyse(path, k=1, n_permute=100)
        sra.report(res)
        out = capsys.readouterr().out
        assert "SELECTIVE RETRIEVAL CURVES" in out
        assert "NOTE ON ECE" in out
        assert "ABSENT" in out  # no ug_confidence in this fixture
    finally:
        os.unlink(path)


def test_main_returns_nonzero_on_a_bad_cache(capsys):
    path = write_cache([make_record(0, [0.9, 0.5], target_pos=99)])
    try:
        assert sra.main([path]) == 1
        assert "CACHE ERROR" in capsys.readouterr().err
    finally:
        os.unlink(path)


def test_main_rejects_bad_arguments(capsys):
    path = write_cache([make_record(0, [0.9, 0.5], target_pos=0),
                        make_record(1, [0.8, 0.5], target_pos=-1)])
    try:
        assert sra.main([path, "-k", "0"]) == 2
        assert sra.main([path, "--permutations", "0"]) == 2
    finally:
        os.unlink(path)


def test_main_writes_json_that_round_trips():
    records = [make_record(i, [0.7, 0.6], rv=[0.5, 0.4],
                           target_pos=0 if i < 4 else -1)
               for i in range(10)]
    path = write_cache(records)
    out = path + ".json"
    try:
        assert sra.main([path, "--permutations", "100", "--json", out]) == 0
        with open(out, encoding="utf-8") as fh:
            loaded = json.load(fh)
        assert loaded["n_queries"] == 10
        assert "base_margin" in loaded["curves"]
    finally:
        os.unlink(path)
        if os.path.exists(out):
            os.unlink(out)


def test_temperature_sweep_reports_the_configured_value():
    records = [make_record(i, [0.7, 0.6, 0.5], target_pos=0 if i < 5 else -1)
               for i in range(10)]
    path = write_cache(records)
    try:
        res = sra.analyse(path, k=1, n_permute=50)
        temps = [r["temperature"] for r in res["entropy_temperature_sweep"]]
        assert sra.ENTROPY_TEMPERATURE in temps
        for r in res["entropy_temperature_sweep"]:
            assert r["auroc"] is None or 0.0 <= r["auroc"] <= 1.0
    finally:
        os.unlink(path)


def test_midranks_assigns_mean_rank_to_ties():
    assert sra._midranks([10.0, 20.0, 30.0]) == [1.0, 2.0, 3.0]
    assert sra._midranks([10.0, 10.0, 30.0]) == [1.5, 1.5, 3.0]
    assert sra._midranks([5.0, 5.0, 5.0]) == [2.0, 2.0, 2.0]


def test_combine_by_rank_is_invariant_to_each_input_scale():
    """The property the whole combined-proxy result rests on.

    base_margin is a cosine difference of order 0.01; rv_margin is a
    difference of sigmoid outputs on an unrelated scale. If the combination
    were sensitive to those scales it would be reporting an arbitrary unit
    choice rather than a joint signal. Multiplying one input by 1000 must not
    change the combined ordering at all.
    """
    a = [0.001, 0.004, 0.002, 0.009, 0.005]
    b = [0.9, 0.1, 0.5, 0.3, 0.7]

    base = sra.combine_by_rank(a, b)
    scaled = sra.combine_by_rank([x * 1000.0 for x in a], b)
    shifted = sra.combine_by_rank([x + 50.0 for x in a], b)

    order = lambda xs: sorted(range(len(xs)), key=lambda i: (-xs[i], i))  # noqa: E731
    assert order(base) == order(scaled)
    assert order(base) == order(shifted)


def test_combine_by_rank_preserves_the_higher_is_confident_convention():
    """Both parents agree perfectly, so the combination must agree too --
    an accidental sign flip inside the averaging would show up here."""
    a = [1.0, 2.0, 3.0, 4.0]
    b = [10.0, 20.0, 30.0, 40.0]
    combined = sra.combine_by_rank(a, b)
    assert combined == sorted(combined)
    assert combined[-1] > combined[0]


def test_combine_by_rank_of_identical_inputs_reproduces_the_ranks():
    a = [3.0, 1.0, 2.0]
    assert sra.combine_by_rank(a, a) == sra._midranks(a)


def test_combine_by_rank_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="differ in length"):
        sra.combine_by_rank([1.0, 2.0], [1.0])


def test_combined_proxy_beats_parents_when_signals_are_complementary():
    """A constructed case where each parent alone is weak but together they
    separate the classes. Confirms `beats_both_parents` can fire, so a `no`
    on real data is informative rather than structural."""
    # Each parent identifies half the positives and is uninformative on the
    # other half; the average of ranks recovers more than either.
    correct, a, b = [], [], []
    for i in range(100):
        ok = i < 50
        correct.append(ok)
        if ok:
            a.append(1.0 if i < 25 else 0.0)
            b.append(0.0 if i < 25 else 1.0)
        else:
            a.append(0.0)
            b.append(0.0)
    auroc_a = sra.auroc(a, correct)
    auroc_b = sra.auroc(b, correct)
    auroc_c = sra.auroc(sra.combine_by_rank(a, b), correct)
    assert auroc_c > auroc_a and auroc_c > auroc_b


def test_combined_proxy_does_not_beat_parents_when_one_is_noise():
    """The counterpart: averaging a good signal with pure noise should not
    improve on the good signal, so `beats_both_parents` is not trivially
    true."""
    import random as _random
    rnd = _random.Random(31)
    n = 400
    correct = [i < 200 for i in range(n)]
    good = [(1.0 if ok else 0.0) + 0.01 * rnd.random() for ok in correct]
    noise = [rnd.random() for _ in range(n)]
    auroc_good = sra.auroc(good, correct)
    auroc_comb = sra.auroc(sra.combine_by_rank(good, noise), correct)
    assert auroc_comb < auroc_good


def test_bootstrap_ci_excludes_zero_for_a_real_difference():
    import random as _random
    rnd = _random.Random(41)
    n = 600
    correct = [i < 300 for i in range(n)]
    strong = [(1.0 if ok else 0.0) + 0.6 * rnd.random() for ok in correct]
    weak = [rnd.random() for _ in range(n)]
    res = sra.paired_bootstrap_auroc_delta(strong, weak, correct,
                                           n_replicates=400)
    assert res["observed_delta"] > 0
    assert res["ci_low"] > 0.0
    assert res["p_two_sided"] < 0.05


def test_bootstrap_ci_includes_zero_for_no_real_difference():
    """The case that matters: the k=5 result showed +0.0025, and a boolean
    test called that a win. A CI must decline to."""
    import random as _random
    rnd = _random.Random(43)
    n = 600
    correct = [i < 300 for i in range(n)]
    a = [(1.0 if ok else 0.0) + 0.8 * rnd.random() for ok in correct]
    b = list(a)
    # Perturb b negligibly, so any AUROC difference is pure noise.
    b[0] += 1e-9
    res = sra.paired_bootstrap_auroc_delta(a, b, correct, n_replicates=400)
    assert res["ci_low"] <= 0.0 <= res["ci_high"]
    assert res["p_two_sided"] > 0.05


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    import random as _random
    rnd = _random.Random(47)
    n = 200
    correct = [rnd.random() < 0.4 for _ in range(n)]
    a = [rnd.random() for _ in range(n)]
    b = [rnd.random() for _ in range(n)]
    r1 = sra.paired_bootstrap_auroc_delta(a, b, correct, n_replicates=200)
    r2 = sra.paired_bootstrap_auroc_delta(a, b, correct, n_replicates=200)
    assert r1 == r2


def test_bootstrap_is_paired_not_independent():
    """Two identical proxies must show a delta of exactly zero with a CI
    that collapses onto zero. Independent resampling would produce spurious
    spread here, so this pins the pairing."""
    import random as _random
    rnd = _random.Random(53)
    n = 300
    correct = [rnd.random() < 0.5 for _ in range(n)]
    a = [rnd.random() for _ in range(n)]
    res = sra.paired_bootstrap_auroc_delta(a, list(a), correct,
                                           n_replicates=200)
    assert res["observed_delta"] == pytest.approx(0.0)
    assert res["ci_low"] == pytest.approx(0.0)
    assert res["ci_high"] == pytest.approx(0.0)


def test_report_distinguishes_worse_from_indistinguishable(capsys):
    """A CI entirely below zero means the combination is significantly WORSE,
    not that the CI 'includes zero'. Both print 'no', so only the explanation
    catches the difference -- and on the real CIRCO cache the first version
    printed the wrong one.
    """
    records = []
    for i in range(60):
        hit = i < 24
        gap = 0.20 if hit else 0.01
        # rv deliberately anti-correlated with correctness, as on CIRCO.
        rv_gap = 0.01 if hit else 0.15
        records.append(make_record(
            i, [0.70, 0.70 - gap, 0.68 - gap],
            rv=[0.60, 0.60 - rv_gap, 0.55],
            target_pos=0 if hit else -1))
    path = write_cache(records)
    try:
        res = sra.analyse(path, k=1, n_permute=100, n_bootstrap=500)
        sra.report(res)
        out = capsys.readouterr().out
        worse = [c for c in res["combined"].values()
                 if c["bootstrap_vs_best_parent"]["ci_high"] is not None
                 and c["bootstrap_vs_best_parent"]["ci_high"] < 0.0]
        if worse:
            assert "significantly WORSE" in out
    finally:
        os.unlink(path)


def test_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="differ in length"):
        sra.paired_bootstrap_auroc_delta([1.0, 2.0], [1.0], [True, False])


def test_bootstrap_handles_a_single_class_without_crashing():
    res = sra.paired_bootstrap_auroc_delta([1.0, 2.0], [2.0, 1.0],
                                           [True, True], n_replicates=50)
    assert res["observed_delta"] is None
    assert res["n_valid"] == 0


def test_combined_verdict_requires_significance_not_just_a_larger_number():
    """Ties the verdict to the CI rather than to a bare inequality."""
    records = []
    for i in range(80):
        hit = i < 32
        gap = 0.20 if hit else 0.01
        records.append(make_record(
            i, [0.70, 0.70 - gap, 0.68 - gap],
            rv=[0.60, 0.60 - (0.15 if hit else 0.01), 0.55],
            target_pos=0 if hit else -1))
    path = write_cache(records)
    try:
        res = sra.analyse(path, k=1, n_permute=100, n_bootstrap=300)
        for name, c in res["combined"].items():
            boot = c["bootstrap_vs_best_parent"]
            assert boot is not None
            # The verdict must agree with the CI, never with the raw delta.
            if c["beats_best_parent_significantly"]:
                assert boot["ci_low"] > 0.0, name
            else:
                assert boot["ci_low"] is None or boot["ci_low"] <= 0.0, name
    finally:
        os.unlink(path)


def test_analyse_reports_combined_proxies_with_their_parents():
    records = []
    for i in range(60):
        hit = i < 24
        gap = 0.20 if hit else 0.01
        rv_gap = 0.15 if (i % 2 == 0) else 0.01
        records.append(make_record(
            i, [0.70, 0.70 - gap, 0.68 - gap],
            rv=[0.60, 0.60 - rv_gap, 0.58 - rv_gap],
            target_pos=0 if hit else -1))
    path = write_cache(records)
    try:
        res = sra.analyse(path, k=1, n_permute=200)
        assert "combined" in res
        assert "base_margin+rv_margin" in res["combined"]
        c = res["combined"]["base_margin+rv_margin"]
        assert c["parents"] == ["base_margin", "rv_margin"]
        # The parent AUROCs quoted inside must match the top-level ones,
        # or the "beats both parents" verdict compares the wrong numbers.
        assert c["parent_auroc"]["base_margin"] == res["auroc"]["base_margin"]
        assert c["parent_auroc"]["rv_margin"] == res["auroc"]["rv_margin"]
        assert c["best_parent_auroc"] == max(res["auroc"]["base_margin"],
                                             res["auroc"]["rv_margin"])
        assert c["delta_vs_best_parent"] == pytest.approx(
            c["auroc"] - c["best_parent_auroc"])
        assert isinstance(c["beats_best_parent_significantly"], bool)
        assert len(c["curve"]) == len(sra.COVERAGES)
    finally:
        os.unlink(path)


def test_spearman_is_one_for_an_identical_ordering():
    assert sra.spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)


def test_spearman_is_minus_one_for_a_reversed_ordering():
    assert sra.spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_spearman_is_monotone_invariant():
    """Rank correlation must ignore a monotone rescaling; if it did not, the
    number would be reporting the proxies' units rather than their ordering."""
    a = [0.1, 0.4, 0.2, 0.9, 0.5]
    b = [x ** 3 for x in a]
    assert sra.spearman(a, b) == pytest.approx(1.0)


def test_spearman_undefined_for_a_constant_proxy():
    assert sra.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None


def test_spearman_handles_ties_with_midranks():
    rho = sra.spearman([1.0, 1.0, 2.0, 3.0], [5.0, 5.0, 6.0, 7.0])
    assert rho == pytest.approx(1.0)


def test_spearman_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="differ in length"):
        sra.spearman([1.0, 2.0], [1.0])


def test_spearman_matches_pearson_on_ranks_for_a_random_case():
    import random as _random
    rnd = _random.Random(3)
    a = [rnd.random() for _ in range(80)]
    b = [rnd.random() for _ in range(80)]
    rho = sra.spearman(a, b)
    assert -1.0 <= rho <= 1.0
    # Untied random data: Spearman equals Pearson on the integer ranks.
    ra = sorted(range(80), key=lambda i: a[i])
    rb = sorted(range(80), key=lambda i: b[i])
    rank_a = [0] * 80
    rank_b = [0] * 80
    for pos, i in enumerate(ra):
        rank_a[i] = pos + 1
    for pos, i in enumerate(rb):
        rank_b[i] = pos + 1
    n = 80
    ma, mb = sum(rank_a) / n, sum(rank_b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(rank_a, rank_b))
    den = (math.sqrt(sum((x - ma) ** 2 for x in rank_a))
           * math.sqrt(sum((y - mb) ** 2 for y in rank_b)))
    assert rho == pytest.approx(num / den)


def test_analyse_reports_pairwise_spearman_for_every_proxy_pair():
    records = [make_record(i, [0.7, 0.6, 0.5], rv=[0.5, 0.4, 0.3],
                           target_pos=0 if i < 5 else -1)
               for i in range(12)]
    path = write_cache(records)
    try:
        res = sra.analyse(path, k=1, n_permute=50)
        n = len(res["proxies"])
        assert len(res["proxy_spearman"]) == n * (n - 1) // 2
        for key in res["proxy_spearman"]:
            p, q = key.split("|")
            assert p in res["proxies"] and q in res["proxies"]
    finally:
        os.unlink(path)


def test_entropy_does_not_overflow_at_the_configured_temperature():
    """The base scores are cosines near 1.0 and T is 0.01, so exp() runs on
    arguments up to ~100 before the max-subtraction. Confirms the shift keeps
    it finite on realistic input."""
    scores = [0.9401, 0.6038, 0.0, -0.4]
    val = sra._neg_entropy(scores, sra.ENTROPY_TEMPERATURE)
    assert math.isfinite(val)


# ---------------------------------------------------------------------------
# real cache
# ---------------------------------------------------------------------------

CIRCO_CACHE_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "rv_score_cache_circo_fc_aep_rv.jsonl"),
    os.path.join(os.path.dirname(__file__), "..",
                 "rv_score_cache_circo_fc_aep_rv.jsonl"),
]


def _find_circo_cache():
    for p in CIRCO_CACHE_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def test_real_circo_cache_pins_the_reported_numbers():
    """Pins the published CIRCO figures against the real cache.

    31.818% agrees with test_multipositive.py's independently hand-checked
    base hit rate on the same file, which is the cross-check that matters:
    an earlier ad-hoc pass reported 32.27% because it applied the
    single-target rule to a multi-positive cache. Pinning the number here
    stops that regressing silently.
    """
    path = _find_circo_cache()
    if path is None:
        import pytest as _pytest
        _pytest.skip("CIRCO cache not present")

    res = sra.analyse(path, k=1, n_permute=2000)

    assert res["n_queries"] == 220
    assert res["multipositive"] is True
    assert res["has_ug_confidence"] is False
    assert res["n_correct_full"] == 70
    assert 100.0 * res["base_accuracy"] == pytest.approx(31.818, abs=0.01)

    # base_entropy is the strongest proxy and rv_margin is not informative.
    assert res["auroc"]["base_entropy"] == pytest.approx(0.7282, abs=0.002)
    assert res["auroc"]["base_margin"] == pytest.approx(0.6679, abs=0.002)
    assert res["auroc"]["base_top1"] == pytest.approx(0.5777, abs=0.002)
    assert res["auroc"]["rv_margin"] == pytest.approx(0.4470, abs=0.002)

    # The headline claim: withholding the least confident half raises
    # accuracy materially under base_entropy.
    rows = {r["coverage"]: r for r in res["curves"]["base_entropy"]}
    assert 100.0 * rows[0.50]["accuracy"] == pytest.approx(46.36, abs=0.01)

    # rv_margin's curve must FALL -- recorded as a negative result, not
    # quietly dropped.
    rv_rows = {r["coverage"]: r for r in res["curves"]["rv_margin"]}
    assert rv_rows[0.50]["accuracy"] < rv_rows[1.00]["accuracy"]

    # Entropy is distinct from the margin rather than a rescaling of it.
    rho = res["proxy_spearman"]["base_margin|base_entropy"]
    assert rho == pytest.approx(0.8124, abs=0.002)
    assert rho < 0.95


def test_real_circo_cache_entropy_advantage_survives_temperature():
    """The reported temperature must not be the only one that works."""
    path = _find_circo_cache()
    if path is None:
        import pytest as _pytest
        _pytest.skip("CIRCO cache not present")

    res = sra.analyse(path, k=1, n_permute=1)
    margin_auroc = res["auroc"]["base_margin"]
    for row in res["entropy_temperature_sweep"]:
        assert row["auroc"] > margin_auroc, (
            "base_entropy at T=%g (%.4f) does not beat base_margin (%.4f); "
            "the entropy proxy's advantage would then be a temperature "
            "artefact" % (row["temperature"], row["auroc"], margin_auroc))


if __name__ == "__main__":
    # Standalone runner, for the cluster login node where pytest may not be
    # installed in the venv. Mirrors test_multipositive.py's convention.
    import traceback

    _fns = [(name, obj) for name, obj in sorted(globals().items())
            if name.startswith("test_") and callable(obj)]
    _failed = 0
    for _name, _fn in _fns:
        try:
            if _fn.__code__.co_argcount:
                print("  [skip] %s (needs a pytest fixture)" % _name)
                continue
            _fn()
            print("  [ok] %s" % _name)
        except Exception as exc:  # noqa: BLE001
            if exc.__class__.__name__ == "Skipped":
                print("  [skip] %s (%s)" % (_name, exc))
                continue
            _failed += 1
            print("  [FAIL] %s: %s" % (_name, exc))
            traceback.print_exc()
    print()
    if _failed:
        print("%d TEST(S) FAILED." % _failed)
        raise SystemExit(1)
    print("ALL SELECTIVE-RETRIEVAL TESTS PASSED.")
