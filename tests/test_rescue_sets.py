"""Regression tests for rescue_set_analysis.

The claim this script exists to check is a set claim, so the dangerous bug is
one that makes two different sets look identical. `test_equal_counts_can_have
_disjoint_sets` constructs exactly that case. The other risk is the
both-wrong split silently counting an unreachable query as a reranking
failure, which `test_absent_and_present_are_counted_separately` pins.
"""
from __future__ import annotations

import json
import os
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

import rescue_set_analysis as rs  # noqa: E402


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
# rank convention
# ---------------------------------------------------------------------------

def test_rank_zero_is_the_top_candidate():
    assert rs.ranked_order([0.9, 0.5, 0.7])[0] == 0
    assert rs.ranked_order([0.5, 0.9, 0.7])[0] == 1


def test_best_relevant_rank_is_zero_based():
    assert rs.best_relevant_rank([0.9, 0.5, 0.7], {0}) == 0
    assert rs.best_relevant_rank([0.9, 0.5, 0.7], {2}) == 1   # 0.7 is 2nd
    assert rs.best_relevant_rank([0.9, 0.5, 0.7], {1}) == 2


def test_best_relevant_rank_takes_the_best_of_several():
    """Multi-positive: the best-placed relevant candidate, not the first by
    index."""
    assert rs.best_relevant_rank([0.9, 0.5, 0.7], {1, 2}) == 1


def test_best_relevant_rank_is_none_when_nothing_relevant():
    assert rs.best_relevant_rank([0.9, 0.5], set()) is None


def test_ties_break_by_index_deterministically():
    assert rs.ranked_order([0.5, 0.5, 0.5]) == [0, 1, 2]


# ---------------------------------------------------------------------------
# the central claim: counts are not sets
# ---------------------------------------------------------------------------

def test_equal_counts_can_have_disjoint_sets():
    """The bug this script exists to prevent.

    Two rules each rescue exactly one query, but different ones. A
    count-based summary would call them identical; the Jaccard must be 0.
    """
    a = {3}
    b = {7}
    assert len(a) == len(b)
    assert rs.jaccard(a, b) == pytest.approx(0.0)


def test_jaccard_identical_sets_is_one():
    assert rs.jaccard({1, 2, 3}, {1, 2, 3}) == pytest.approx(1.0)


def test_jaccard_two_empty_sets_is_one():
    """Two rules that rescue nothing agree completely; returning 0/0 or
    crashing would both be wrong."""
    assert rs.jaccard(set(), set()) == pytest.approx(1.0)


def test_jaccard_partial_overlap():
    assert rs.jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(2 / 4)


def test_rescue_and_break_are_disjoint_by_construction():
    """A query cannot be both rescued and broken by the same rule."""
    recs = [make_record(i, [0.90, 0.85, 0.50], [0.1, 0.9, 0.2],
                        target_pos=i % 3) for i in range(30)]
    rel = [{r["target_pos"]} for r in recs]
    sets = rs.outcome_sets(recs, rel, lam=2.0)
    for name in ("bounded", "rank"):
        assert not (sets[name]["rescued"] & sets[name]["broken"])


def test_rescue_requires_base_wrong_and_rule_right():
    """Hand-built: base puts the target second, a large lambda promotes it."""
    recs = [make_record(0, [0.90, 0.89, 0.50], [0.1, 0.9, 0.2],
                        target_pos=1)]
    rel = [{1}]
    sets = rs.outcome_sets(recs, rel, lam=5.0)
    assert 0 in sets["bounded"]["rescued"]
    assert 0 not in sets["bounded"]["broken"]


def test_break_requires_base_right_and_rule_wrong():
    recs = [make_record(0, [0.90, 0.89, 0.50], [0.1, 0.9, 0.2],
                        target_pos=0)]
    rel = [{0}]
    sets = rs.outcome_sets(recs, rel, lam=5.0)
    assert 0 in sets["bounded"]["broken"]
    assert 0 not in sets["bounded"]["rescued"]


# ---------------------------------------------------------------------------
# the both-wrong split
# ---------------------------------------------------------------------------

def test_absent_and_present_are_counted_separately():
    """The distinction the review asks for: a query whose target never
    entered the window is a first-stage failure, not a reranking one."""
    recs = [
        # target absent: no relevant position at all
        make_record(0, [0.9, 0.5, 0.2], [0.3, 0.3, 0.3], target_pos=-1),
        # target present at rank 2 but nothing promotes it
        make_record(1, [0.9, 0.5, 0.2], [0.9, 0.5, 0.1], target_pos=2),
    ]
    rel = [rs.relevant_set(r, False) for r in recs]
    sets = rs.outcome_sets(recs, rel, lam=0.1)
    bw = rs.both_wrong_breakdown(recs, rel, sets)
    assert bw["target_absent_from_window"] == 1
    assert bw["target_present_not_promoted"] == 1
    assert bw["both_wrong_total"] == 2


def test_both_wrong_excludes_queries_any_rule_gets_right():
    recs = [make_record(0, [0.9, 0.5], [0.1, 0.2], target_pos=0)]
    rel = [{0}]
    sets = rs.outcome_sets(recs, rel, lam=0.1)
    bw = rs.both_wrong_breakdown(recs, rel, sets)
    assert bw["both_wrong_total"] == 0


def test_both_wrong_reports_both_rank_conventions():
    recs = [make_record(i, [0.9, 0.5, 0.2], [0.9, 0.5, 0.1], target_pos=2)
            for i in range(5)]
    rel = [{2} for _ in recs]
    sets = rs.outcome_sets(recs, rel, lam=0.1)
    bw = rs.both_wrong_breakdown(recs, rel, sets)
    assert bw["median_base_rank_0based"] == 2
    assert bw["median_base_rank_1based"] == 3


def test_median_rank_is_none_when_no_present_failures():
    recs = [make_record(0, [0.9, 0.5], [0.1, 0.2], target_pos=-1)]
    rel = [set()]
    sets = rs.outcome_sets(recs, rel, lam=0.1)
    bw = rs.both_wrong_breakdown(recs, rel, sets)
    assert bw["median_base_rank_0based"] is None


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_analyse_reports_overlap_fields():
    recs = [make_record(i, [0.90, 0.85, 0.50], [0.1, 0.9, 0.2],
                        target_pos=i % 3) for i in range(40)]
    path = write_cache(recs)
    try:
        res = rs.analyse(path, lam=2.0)
        for k in ("both", "bounded_only", "rank_only", "jaccard"):
            assert k in res["rescued"] and k in res["broken"]
        # counts must be internally consistent
        d = res["rescued"]
        assert d["bounded"] == d["both"] + d["bounded_only"]
        assert d["rank"] == d["both"] + d["rank_only"]
    finally:
        os.unlink(path)


def test_coverage_counts_queries_with_a_relevant_candidate():
    recs = [make_record(0, [0.9, 0.5], [0.1, 0.2], target_pos=0),
            make_record(1, [0.9, 0.5], [0.1, 0.2], target_pos=-1)]
    path = write_cache(recs)
    try:
        res = rs.analyse(path, lam=1.0)
        assert res["coverage"] == 1
    finally:
        os.unlink(path)


def test_report_runs(capsys):
    recs = [make_record(i, [0.9, 0.85, 0.5], [0.1, 0.9, 0.2],
                        target_pos=i % 3) for i in range(20)]
    path = write_cache(recs)
    try:
        res = rs.analyse(path, lam=2.0)
        rs.report(res)
        out = capsys.readouterr().out
        assert "0-BASED" in out
        assert "Jaccard" in out
        assert "no reranker could help" in out
    finally:
        os.unlink(path)


def test_main_rejects_negative_lambda():
    recs = [make_record(0, [0.9, 0.5], [0.1, 0.2], target_pos=0)]
    path = write_cache(recs)
    try:
        assert rs.main([path, "--lam", "-1"]) == 2
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


def test_real_cache_coverage_matches_the_known_figure():
    """CIRCO: 156 of 220 queries have a relevant candidate in the window,
    the figure the multi-positive analysis independently reports."""
    path = _find_circo()
    if path is None:
        pytest.skip("CIRCO cache not present")
    res = rs.analyse(path, lam=1.0)
    assert res["n_queries"] == 220
    assert res["coverage"] == 156


def test_real_cache_both_wrong_split_sums_correctly():
    path = _find_circo()
    if path is None:
        pytest.skip("CIRCO cache not present")
    res = rs.analyse(path, lam=1.0)
    bw = res["both_wrong"]
    assert (bw["target_absent_from_window"]
            + bw["target_present_not_promoted"]) == bw["both_wrong_total"]


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
    print("ALL RESCUE-SET TESTS PASSED.")
