"""Regression tests for risk_coverage.

AURC is an easy metric to implement subtly wrong: an off-by-one in the
coverage loop, a sort in the wrong direction, or an oracle that is not
actually optimal all produce plausible-looking numbers. Each is pinned
below, and `test_aurc_ordering_oracle_random_worst` checks the three
reference cases against each other rather than against hard-coded values.
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile

try:
    import pytest
except ImportError:  # pragma: no cover
    class _Approx:
        def __init__(self, expected, abs=None, rel=None):
            self.expected, self.abs, self.rel = expected, abs, rel

        def __eq__(self, other):
            tol = self.abs if self.abs is not None else max(
                1e-12, (self.rel if self.rel is not None else 1e-6)
                * abs(self.expected))
            return abs(other - self.expected) <= tol

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
                    raise AssertionError("pattern %r not in %r"
                                         % (self.match, str(ev)))
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

import risk_coverage as rc  # noqa: E402


def make_record(qid, base, rv, target_pos=None, relevant=None):
    n = len(base)
    rec = {"query_id": qid,
           "candidate_gallery_ids": list(range(qid * 100, qid * 100 + n)),
           "base_score": list(base),
           "verification_score": list(rv)}
    if target_pos is not None:
        rec["target_pos"] = target_pos
    if relevant is not None:
        rec["relevant_positions_in_candidates"] = list(relevant)
    return rec


def write_cache(records):
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8")
    for r in records:
        fh.write(json.dumps(r) + "\n")
    fh.close()
    return fh.name


# ---------------------------------------------------------------------------
# AURC itself
# ---------------------------------------------------------------------------

def test_aurc_matches_a_hand_computed_case():
    """Confidences [3,2,1] over outcomes [T,F,T] sort to [T,F,T]:
       k=1 risk 0/1 = 0
       k=2 risk 1/2 = 0.5
       k=3 risk 1/3
       AURC = (0 + 0.5 + 1/3)/3 = 0.277778
    """
    got = rc.aurc([3.0, 2.0, 1.0], [True, False, True])
    assert got == pytest.approx((0 + 0.5 + 1 / 3) / 3)


def test_aurc_perfect_ranking_is_the_oracle():
    correct = [True] * 6 + [False] * 4
    conf = list(range(10, 0, -1))        # already in the ideal order
    assert rc.aurc(conf, correct) == pytest.approx(rc.oracle_aurc(correct))


def test_aurc_is_zero_when_everything_is_correct():
    assert rc.aurc([3.0, 2.0, 1.0], [True, True, True]) == pytest.approx(0.0)


def test_aurc_is_one_when_everything_is_wrong():
    assert rc.aurc([3.0, 2.0, 1.0], [False, False, False]) == \
        pytest.approx(1.0)


def test_aurc_ordering_oracle_random_worst():
    """The three reference cases must order correctly, and a random ranking
    should land near the base error rate."""
    n, ncorr = 200, 120
    correct = [True] * ncorr + [False] * (n - ncorr)
    oracle = rc.aurc(list(range(n, 0, -1)), correct)
    worst = rc.aurc(list(range(n)), correct)       # ascending = worst
    rnd = random.Random(3)
    conf = [rnd.random() for _ in range(n)]
    mid = rc.aurc(conf, correct)
    assert oracle < mid < worst
    assert mid == pytest.approx(1 - ncorr / n, abs=0.08)


def test_aurc_sorts_by_decreasing_confidence():
    """If the sort direction were reversed, a good proxy would look bad."""
    correct = [True, True, False, False]
    good = [4.0, 3.0, 2.0, 1.0]      # confidence aligned with correctness
    bad = [1.0, 2.0, 3.0, 4.0]       # anti-aligned
    assert rc.aurc(good, correct) < rc.aurc(bad, correct)


def test_aurc_ties_break_stably():
    correct = [True, False, True, False]
    a = rc.aurc([1.0] * 4, correct)
    b = rc.aurc([1.0] * 4, correct)
    assert a == b


def test_aurc_rejects_empty_input():
    with pytest.raises(ValueError, match="no queries"):
        rc.aurc([], [])


# ---------------------------------------------------------------------------
# the oracle and E-AURC
# ---------------------------------------------------------------------------

def test_oracle_depends_only_on_the_number_correct():
    a = [True, False, True, False, True]
    b = [True, True, True, False, False]
    assert rc.oracle_aurc(a) == pytest.approx(rc.oracle_aurc(b))


def test_oracle_lower_bounds_every_ranking():
    correct = [True] * 7 + [False] * 5
    orc = rc.oracle_aurc(correct)
    rnd = random.Random(11)
    for _ in range(25):
        conf = [rnd.random() for _ in correct]
        assert rc.aurc(conf, correct) >= orc - 1e-12


def test_e_aurc_is_zero_for_a_perfect_ranking():
    correct = [True] * 5 + [False] * 5
    assert rc.e_aurc(list(range(10, 0, -1)), correct) == \
        pytest.approx(0.0, abs=1e-12)


def test_e_aurc_is_never_negative():
    correct = [True] * 6 + [False] * 6
    rnd = random.Random(5)
    for _ in range(20):
        conf = [rnd.random() for _ in correct]
        assert rc.e_aurc(conf, correct) >= -1e-12


def test_e_aurc_removes_the_accuracy_floor():
    """Two proxies with identical ranking quality but different accuracy
    should differ in AURC and agree in E-AURC."""
    hi = [True] * 8 + [False] * 2
    lo = [True] * 4 + [False] * 6
    conf_hi = list(range(10, 0, -1))
    conf_lo = list(range(10, 0, -1))
    assert rc.aurc(conf_hi, hi) != pytest.approx(rc.aurc(conf_lo, lo))
    assert rc.e_aurc(conf_hi, hi) == pytest.approx(rc.e_aurc(conf_lo, lo),
                                                   abs=1e-12)


# ---------------------------------------------------------------------------
# the curve
# ---------------------------------------------------------------------------

def test_risk_at_full_coverage_is_the_base_error_rate():
    correct = [True] * 7 + [False] * 3
    rows = rc.risk_at_coverages(list(range(10, 0, -1)), correct, (1.0,))
    assert rows[0]["risk"] == pytest.approx(0.3)


def test_risk_falls_as_coverage_falls_for_a_good_proxy():
    correct = [True] * 10 + [False] * 10
    conf = list(range(20, 0, -1))
    rows = rc.risk_at_coverages(conf, correct, (1.0, 0.5, 0.25))
    risks = [r["risk"] for r in rows]
    assert risks[0] > risks[1] >= risks[2]


def test_risk_and_accuracy_sum_to_one():
    correct = [True, False, True, True]
    for r in rc.risk_at_coverages([4.0, 3.0, 2.0, 1.0], correct):
        assert r["risk"] + r["accuracy"] == pytest.approx(1.0)


def test_coverage_never_yields_zero_kept():
    rows = rc.risk_at_coverages([2.0, 1.0], [True, False], (0.01,))
    assert rows[0]["n_kept"] == 1


# ---------------------------------------------------------------------------
# paired bootstrap
# ---------------------------------------------------------------------------

def test_paired_bootstrap_identical_proxies_gives_zero():
    correct = [True] * 30 + [False] * 30
    conf = list(range(60, 0, -1))
    r = rc.paired_bootstrap_aurc_delta(conf, list(conf), correct, n_rep=200)
    assert r["observed_delta"] == pytest.approx(0.0)
    assert r["ci_low"] == pytest.approx(0.0)
    assert r["ci_high"] == pytest.approx(0.0)


def test_paired_bootstrap_detects_a_real_difference():
    n = 200
    correct = [True] * 100 + [False] * 100
    good = list(range(n, 0, -1))
    rnd = random.Random(7)
    bad = [rnd.random() for _ in range(n)]
    r = rc.paired_bootstrap_aurc_delta(good, bad, correct, n_rep=400)
    assert r["observed_delta"] < 0          # lower AURC = better
    assert r["ci_high"] < 0
    assert r["p_two_sided"] < 0.05


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="differ in length"):
        rc.paired_bootstrap_aurc_delta([1.0, 2.0], [1.0], [True, False],
                                       n_rep=10)


def test_paired_bootstrap_is_deterministic():
    correct = [True] * 20 + [False] * 20
    rnd = random.Random(2)
    a = [rnd.random() for _ in range(40)]
    b = [rnd.random() for _ in range(40)]
    r1 = rc.paired_bootstrap_aurc_delta(a, b, correct, n_rep=150)
    r2 = rc.paired_bootstrap_aurc_delta(a, b, correct, n_rep=150)
    assert r1 == r2


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_analyse_includes_the_entropy_combination():
    """The combination the review asks for must be present, since pairing
    verification with base_margin alone was the objection."""
    recs = [make_record(i, [0.70, 0.60, 0.50], [0.6, 0.5, 0.4],
                        target_pos=0 if i % 3 else 1) for i in range(60)]
    path = write_cache(recs)
    try:
        res = rc.analyse(path, k=1, n_rep=100)
        assert "base_entropy + rv_margin" in res["metrics"]
        assert any("base_entropy + rv_margin" in c for c in res["contrasts"])
    finally:
        os.unlink(path)


def test_every_combination_is_contrasted_with_the_best_single_proxy():
    """The manuscript table has one common reference column (the best single
    proxy); every combination must have an entry against it, including one
    whose own stronger parent is a different proxy."""
    rng = __import__("random").Random(3)
    recs = [make_record(i, [rng.random() for _ in range(4)],
                        [rng.random() for _ in range(4)],
                        target_pos=rng.choice([0, 0, 1, 2, -1]))
            for i in range(120)]
    path = write_cache(recs)
    try:
        res = rc.analyse(path, k=1, n_rep=100)
        best = res["best_single"]
        for combo in ("base_entropy + rv_margin", "base_margin + rv_margin",
                      "base_top1 + rv_margin"):
            key = "%s vs %s" % (combo, best)
            assert key in res["contrasts"], key
            d = res["contrasts"][key]
            want = res["metrics"][combo]["aurc"] - res["metrics"][best]["aurc"]
            assert abs(d["observed_delta"] - want) < 1e-12
    finally:
        os.unlink(path)


def test_analyse_rejects_a_degenerate_outcome_set():
    """All-correct or all-wrong makes AURC meaningless; fail loudly."""
    recs = [make_record(i, [0.9, 0.5], [0.1, 0.2], target_pos=0)
            for i in range(20)]
    path = write_cache(recs)
    try:
        with pytest.raises(rc.CacheError, match="same outcome"):
            rc.analyse(path, k=1, n_rep=50)
    finally:
        os.unlink(path)


def test_report_runs(capsys):
    recs = [make_record(i, [0.70, 0.60, 0.50], [0.6, 0.5, 0.4],
                        target_pos=0 if i % 3 else 1) for i in range(40)]
    path = write_cache(recs)
    try:
        res = rc.analyse(path, k=1, n_rep=100)
        rc.report(res)
        out = capsys.readouterr().out
        assert "E-AURC" in out and "RISK AT SELECTED COVERAGES" in out
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# real cache
# ---------------------------------------------------------------------------

def _find_circo():
    for p in [os.path.join(os.path.dirname(__file__), "..", "..",
                           "rv_score_cache_circo_fc_aep_rv.jsonl"),
              os.path.join(os.path.dirname(__file__), "..",
                           "rv_score_cache_circo_fc_aep_rv.jsonl")]:
        if os.path.exists(p):
            return p
    return None


def test_real_cache_oracle_matches_its_accuracy():
    """The oracle depends only on the number correct, so it must agree with
    one recomputed from the known 70/220 base accuracy."""
    path = _find_circo()
    if path is None:
        pytest.skip("CIRCO cache not present")
    res = rc.analyse(path, k=1, n_rep=50)
    assert res["n_queries"] == 220
    assert res["base_accuracy"] == pytest.approx(70 / 220, abs=1e-6)
    expected = rc.oracle_aurc([True] * 70 + [False] * 150)
    assert res["oracle_aurc"] == pytest.approx(expected)


def test_real_cache_every_e_aurc_is_nonnegative():
    path = _find_circo()
    if path is None:
        pytest.skip("CIRCO cache not present")
    res = rc.analyse(path, k=1, n_rep=50)
    for nm, m in res["metrics"].items():
        assert m["e_aurc"] >= -1e-12, nm


if __name__ == "__main__":
    import traceback
    fns = [(n, o) for n, o in sorted(globals().items())
           if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in fns:
        try:
            if fn.__code__.co_argcount:
                print("  [skip] %s (needs a pytest fixture)" % name)
                continue
            fn()
            print("  [ok] %s" % name)
        except Exception as exc:  # noqa: BLE001
            if exc.__class__.__name__ == "Skipped":
                print("  [skip] %s (%s)" % (name, exc))
                continue
            failed += 1
            print("  [FAIL] %s: %s" % (name, exc))
            traceback.print_exc()
    print()
    if failed:
        print("%d TEST(S) FAILED." % failed)
        raise SystemExit(1)
    print("ALL RISK-COVERAGE TESTS PASSED.")
