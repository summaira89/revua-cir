"""Regression tests for hardneg_mining_analysis.

The failure this analysis is most exposed to is counting a true positive as a
minable negative, which is the very quantity it is supposed to measure. Under
multi-positive relevance that happens silently if only one target is excluded.
Several tests below target that case specifically, plus the threshold sign
convention, which would invert on a negative anchor.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

try:
    import pytest
except ImportError:  # pragma: no cover - cluster venv without pytest
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

import hardneg_mining_analysis as hm  # noqa: E402


def make_record(qid, base, target_pos=None, relevant=None):
    n = len(base)
    rec = {
        "query_id": qid,
        "candidate_gallery_ids": list(range(qid * 100, qid * 100 + n)),
        "base_score": list(base),
    }
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
# positives must never be minable
# ---------------------------------------------------------------------------

def test_single_target_is_excluded_from_negatives():
    rec = make_record(0, [0.9, 0.8, 0.7], target_pos=1)
    pos = hm.positive_positions(rec, multipositive=False)
    assert pos == [1]
    assert 1 not in hm.mine_threshold(rec, pos, None)


def test_every_relevant_position_is_excluded_under_multipositive():
    """The error this whole script measures, committed by the script itself
    if it excluded only one target."""
    rec = make_record(0, [0.9, 0.8, 0.7, 0.6], relevant=[0, 2])
    pos = hm.positive_positions(rec, multipositive=True)
    assert pos == [0, 2]
    negs = hm.mine_threshold(rec, pos, None)
    assert 0 not in negs and 2 not in negs
    assert sorted(negs) == [1, 3]


def test_missing_target_yields_no_positives_and_no_negatives():
    rec = make_record(0, [0.9, 0.8], target_pos=-1)
    pos = hm.positive_positions(rec, multipositive=False)
    assert pos == []
    assert hm.mine_threshold(rec, pos, 0.95) == []
    assert hm.anchor_score(rec, pos) is None


# ---------------------------------------------------------------------------
# the anchor
# ---------------------------------------------------------------------------

def test_anchor_is_the_weakest_positive():
    """Anchoring on the strongest positive would admit more negatives; the
    conservative choice is the weakest."""
    rec = make_record(0, [0.9, 0.5, 0.7], relevant=[0, 1])
    assert hm.anchor_score(rec, [0, 1]) == pytest.approx(0.5)


def test_threshold_is_measured_against_the_anchor_not_the_top_score():
    rec = make_record(0, [0.90, 0.50, 0.60], relevant=[0, 1])
    # anchor = 0.50, so pct-100 admits only scores <= 0.50; 0.60 is excluded.
    assert hm.mine_threshold(rec, [0, 1], 1.00) == []


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------

def test_naive_rule_admits_candidates_outscoring_the_target():
    """The control. Without a threshold, a candidate the retriever ranks
    ABOVE the true target still becomes a negative."""
    rec = make_record(0, [0.95, 0.60, 0.40], target_pos=1)
    negs = hm.mine_threshold(rec, [1], None)
    assert 0 in negs, "naive rule must admit the higher-scoring candidate"


def test_pct100_excludes_candidates_outscoring_the_target():
    rec = make_record(0, [0.95, 0.60, 0.40], target_pos=1)
    negs = hm.mine_threshold(rec, [1], 1.00)
    assert 0 not in negs
    assert 2 in negs


def test_pct95_is_stricter_than_pct100():
    # target 0.60; 0.58 passes at 100% (<=0.60) but fails at 95% (<=0.57).
    rec = make_record(0, [0.58, 0.60, 0.20], target_pos=1)
    assert 0 in hm.mine_threshold(rec, [1], 1.00)
    assert 0 not in hm.mine_threshold(rec, [1], 0.95)


def test_thresholds_are_monotone_in_strictness():
    """Tightening the threshold can only remove negatives, never add them.
    A sign slip in `pct * anchor` would break this."""
    rec = make_record(0, [0.95, 0.70, 0.66, 0.60, 0.30], target_pos=1)
    sets = [set(hm.mine_threshold(rec, [1], p))
            for p in (None, 1.00, 0.95, 0.90)]
    for a, b in zip(sets, sets[1:]):
        assert b <= a, "stricter threshold admitted a new negative"


def test_nonpositive_anchor_is_reported_unusable_not_inverted():
    """With a negative anchor, `0.95 * anchor` is GREATER than the anchor, so
    a multiplicative threshold silently loosens. Returning [] instead."""
    rec = make_record(0, [-0.10, -0.50, -0.90], target_pos=1)
    assert hm.anchor_score(rec, [1]) == pytest.approx(-0.50)
    assert hm.mine_threshold(rec, [1], 0.95) == []
    # the naive rule is unaffected, since it does not scale anything
    assert len(hm.mine_threshold(rec, [1], None)) == 2


def test_zero_anchor_is_unusable():
    rec = make_record(0, [0.5, 0.0, -0.2], target_pos=1)
    assert hm.mine_threshold(rec, [1], 0.95) == []


# ---------------------------------------------------------------------------
# QuRe reading
# ---------------------------------------------------------------------------

def test_qure_only_considers_candidates_below_the_anchor():
    rec = make_record(0, [0.99, 0.50, 0.40, 0.39, 0.10, 0.09],
                      target_pos=1)
    got = hm.qure_window(rec, [1])
    assert 0 not in got, "a candidate above the anchor must not be selected"
    assert 1 not in got, "the target must not be selected"


def test_qure_returns_everything_below_when_too_few_candidates():
    """Fewer than three candidates cannot show two interior drops; the
    function must degrade rather than index out of range."""
    rec = make_record(0, [0.9, 0.5, 0.4], target_pos=1)
    got = hm.qure_window(rec, [1])
    assert got == [2]


def test_qure_picks_the_region_between_the_two_largest_drops():
    """Hand-traced. Below-anchor scores, descending:
         0.40, 0.39, 0.38, 0.10, 0.09
       adjacent drops:  .01, .01, .28, .01
       The two largest are at j=2 (0.28) and then one of the 0.01s; with
       ties broken by sort order the second is j=0. Sorted: j1=0, j2=2, so
       the kept slice is below[1:3] -> the 0.39 and 0.38 candidates.
    """
    rec = make_record(0, [0.99, 0.50, 0.40, 0.39, 0.38, 0.10, 0.09],
                      target_pos=1)
    got = hm.qure_window(rec, [1])
    scores = [rec["base_score"][i] for i in got]
    assert scores == [pytest.approx(0.39), pytest.approx(0.38)], scores


def test_qure_never_returns_a_positive():
    rec = make_record(0, [0.9, 0.8, 0.5, 0.49, 0.2, 0.19], relevant=[0, 1])
    got = hm.qure_window(rec, [0, 1])
    assert 0 not in got and 1 not in got


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_mixed_relevance_conventions_abort():
    path = write_cache([make_record(0, [0.9, 0.5], target_pos=0),
                        make_record(1, [0.8, 0.5], relevant=[0])])
    try:
        with pytest.raises(hm.CacheError, match="wholly single-target"):
            hm.load_records(path)
    finally:
        os.unlink(path)


def test_out_of_range_target_rejected():
    path = write_cache([make_record(0, [0.9, 0.5], target_pos=9)])
    try:
        with pytest.raises(hm.CacheError, match="outside"):
            hm.load_records(path)
    finally:
        os.unlink(path)


def test_duplicate_candidates_rejected():
    rec = make_record(0, [0.9, 0.5], target_pos=0)
    rec["candidate_gallery_ids"] = [7, 7]
    path = write_cache([rec])
    try:
        with pytest.raises(hm.CacheError, match="duplicate"):
            hm.load_records(path)
    finally:
        os.unlink(path)


def test_ragged_scores_rejected():
    rec = make_record(0, [0.9, 0.5, 0.2], target_pos=0)
    rec["base_score"] = [0.9, 0.5]
    path = write_cache([rec])
    try:
        with pytest.raises(hm.CacheError, match="base_score has 2"):
            hm.load_records(path)
    finally:
        os.unlink(path)


def test_nonfinite_score_rejected():
    rec = make_record(0, [0.9, 0.5], target_pos=0)
    rec["base_score"] = [float("inf"), 0.5]
    path = write_cache([rec])
    try:
        with pytest.raises(hm.CacheError, match="non-finite"):
            hm.load_records(path)
    finally:
        os.unlink(path)


def test_empty_cache_rejected():
    path = write_cache([])
    try:
        with pytest.raises(hm.CacheError, match="no records"):
            hm.load_records(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def test_uncovered_queries_are_counted_not_scored():
    """A query whose target missed the window has no anchor. It must be
    excluded from the per-rule means, not counted as zero negatives, which
    would drag every mean toward zero."""
    recs = [make_record(0, [0.9, 0.5, 0.2], target_pos=1),
            make_record(1, [0.9, 0.5, 0.2], target_pos=-1)]
    res = hm.summarize(recs, multipositive=False)
    assert res["n_queries"] == 2
    assert res["n_covered"] == 1
    assert res["n_uncovered"] == 1
    # only the covered query contributes: 2 non-target candidates
    assert res["rules"]["none (naive)"]["mean_negatives"] == pytest.approx(2.0)


def test_above_anchor_count_is_reported():
    recs = [make_record(0, [0.95, 0.60, 0.40], target_pos=1)]
    res = hm.summarize(recs, multipositive=False)
    assert res["mean_candidates_above_anchor"] == pytest.approx(1.0)
    assert res["queries_with_any_above_anchor"] == 1


def test_margin_is_target_minus_hardest_admitted_negative():
    recs = [make_record(0, [0.30, 0.60, 0.20], target_pos=1)]
    res = hm.summarize(recs, multipositive=False)
    # pct-100 admits 0.30 and 0.20; hardest is 0.30, so margin = 0.60-0.30
    assert res["rules"]["pct-100"]["mean_margin_to_hardest"] == pytest.approx(0.30)


def test_all_rules_present_in_summary():
    recs = [make_record(i, [0.9, 0.7, 0.5, 0.3, 0.1], target_pos=1)
            for i in range(5)]
    res = hm.summarize(recs, multipositive=False)
    for name in ("none (naive)", "pct-100", "pct-95", "pct-90",
                 "qure (our reading)"):
        assert name in res["rules"], name


def test_report_runs(capsys):
    recs = [make_record(i, [0.9, 0.7, 0.5, 0.3, 0.1],
                        target_pos=1 if i % 2 == 0 else -1)
            for i in range(8)]
    res = hm.summarize(recs, multipositive=False)
    res["path"] = "fixture.jsonl"
    hm.report(res)
    out = capsys.readouterr().out
    assert "HARD-NEGATIVE AVAILABILITY" in out
    assert "CANDIDATES OUTSCORING THE TARGET" in out


def test_main_returns_nonzero_on_bad_cache(capsys):
    path = write_cache([make_record(0, [0.9, 0.5], target_pos=42)])
    try:
        assert hm.main([path]) == 1
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


def test_real_circo_cache_is_handled_as_multipositive():
    path = _find_circo()
    if path is None:
        pytest.skip("CIRCO cache not present")
    records, multi = hm.load_records(path)
    assert multi is True
    assert len(records) == 220
    res = hm.summarize(records, multi)
    # Every rule must admit strictly fewer negatives than the naive one, or
    # equal when no candidate outscores the anchor.
    naive = res["rules"]["none (naive)"]["mean_negatives"]
    for name in ("pct-100", "pct-95", "pct-90"):
        assert res["rules"][name]["mean_negatives"] <= naive + 1e-9, name
    assert res["n_covered"] + res["n_uncovered"] == 220


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
    print("ALL HARD-NEGATIVE MINING TESTS PASSED.")
