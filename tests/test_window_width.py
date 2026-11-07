"""Regression tests for window_width_analysis.

The error this analysis is most exposed to is scoring a truncated window with
statistics taken from the full one. That would credit a narrow window with
information it could not have, and the output would look entirely reasonable.
`test_truncation_recomputes_zscores` and
`test_narrow_window_cannot_use_discarded_candidates` target it directly.
"""
from __future__ import annotations

import json
import math
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

import window_width_analysis as ww  # noqa: E402


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
# the central correctness property
# ---------------------------------------------------------------------------

def test_truncation_recomputes_zscores():
    """The whole point. Truncating must renormalise over k items, not reuse
    the full-window mean and sd."""
    rec = make_record(0, [0.9, 0.8, 0.1, 0.0], [0.5, 0.4, 0.3, 0.2])
    zb3, _ = ww.truncate_rescores(rec, 3)
    # Recompute independently from the first three base scores only.
    sub = [0.9, 0.8, 0.1]
    m = sum(sub) / 3
    sd = math.sqrt(sum((x - m) ** 2 for x in sub) / 3)
    expected = [(x - m) / (sd + 1e-8) for x in sub]
    assert zb3 == [pytest.approx(e) for e in expected]

    # And it must DIFFER from the full-window z-scores of the same items.
    zb4, _ = ww.truncate_rescores(rec, 4)
    assert zb3[0] != pytest.approx(zb4[0]), \
        "truncation reused the full-window normalisation"


def test_narrow_window_cannot_use_discarded_candidates():
    """Changing a candidate outside the window must not change the result
    inside it."""
    a = make_record(0, [0.9, 0.8, 0.7, 0.6], [0.1, 0.2, 0.3, 0.4])
    b = make_record(0, [0.9, 0.8, 0.7, 0.6], [0.1, 0.2, 0.9, 0.9])
    assert ww.fused_order(a, 2, 1.0) == ww.fused_order(b, 2, 1.0)


def test_zscore_matches_multipositive_analysis():
    """Both scripts must implement one fusion rule, including the 1e-8 floor."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                    "scripts"))
    import multipositive_analysis as mp
    xs = [0.63, 0.60, 0.59, 0.55]
    assert ww._zscore(xs) == [pytest.approx(v) for v in mp._zscore(xs)]


def test_zscore_constant_input_is_zeros_not_nan():
    for v in ww._zscore([0.5, 0.5, 0.5]):
        assert math.isfinite(v)
        assert v == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# fusion behaviour
# ---------------------------------------------------------------------------

def test_lambda_zero_fusion_equals_base_order():
    rec = make_record(0, [0.9, 0.5, 0.7], [0.1, 0.9, 0.2])
    assert ww.fused_order(rec, 3, 0.0) == ww.base_order(rec, 3)


def test_large_lambda_follows_verification():
    rec = make_record(0, [0.90, 0.89, 0.88], [0.1, 0.9, 0.2])
    order = ww.fused_order(rec, 3, 50.0)
    assert order[0] == 1, "huge lambda should put the top RV candidate first"


def test_fused_order_is_a_permutation_of_the_window():
    rec = make_record(0, [0.9, 0.5, 0.7, 0.2], [0.1, 0.9, 0.2, 0.3])
    for k in (1, 2, 3, 4):
        assert sorted(ww.fused_order(rec, k, 1.0)) == list(range(k))


def test_ties_break_by_index_deterministically():
    rec = make_record(0, [0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
    assert ww.fused_order(rec, 3, 1.0) == [0, 1, 2]


def test_truncate_rejects_k_below_one():
    rec = make_record(0, [0.9, 0.5], [0.1, 0.2])
    with pytest.raises(ValueError, match="k must be >= 1"):
        ww.truncate_rescores(rec, 0)


# ---------------------------------------------------------------------------
# k=1 is a degenerate but meaningful case
# ---------------------------------------------------------------------------

def test_k1_fusion_cannot_change_anything():
    """With one candidate there is nothing to reorder, so base and fused
    hit rates must be identical at every lambda."""
    recs = [make_record(i, [0.9, 0.5, 0.2], [0.1, 0.9, 0.2],
                        target_pos=i % 3) for i in range(9)]
    path = write_cache(recs)
    try:
        res = ww.analyse(path, lam=1.0, widths=(1,))
        row = res["rows"][0]
        assert row["delta_pp"] == pytest.approx(0.0)
        assert row["helped"] == 0 and row["harmed"] == 0
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# help/harm accounting
# ---------------------------------------------------------------------------

def test_help_and_harm_are_counted_separately():
    recs = [
        # fusion promotes the target: helped
        make_record(0, [0.90, 0.89], [0.1, 0.9], target_pos=1),
        # fusion demotes the target: harmed
        make_record(1, [0.90, 0.89], [0.1, 0.9], target_pos=0),
    ]
    path = write_cache(recs)
    try:
        res = ww.analyse(path, lam=5.0, widths=(2,))
        row = res["rows"][0]
        assert row["helped"] == 1 and row["harmed"] == 1
        assert row["net"] == 0
        assert row["delta_pp"] == pytest.approx(0.0)
    finally:
        os.unlink(path)


def test_delta_equals_fused_minus_base():
    recs = [make_record(i, [0.9, 0.5, 0.2], [0.2, 0.9, 0.1],
                        target_pos=1 if i < 5 else 0) for i in range(10)]
    path = write_cache(recs)
    try:
        res = ww.analyse(path, lam=2.0)
        for r in res["rows"]:
            assert r["delta_pp"] == pytest.approx(
                r["fused_hit1_pct"] - r["base_hit1_pct"])
            assert r["net"] == r["helped"] - r["harmed"]
    finally:
        os.unlink(path)


def test_retained_is_relative_to_the_cached_width():
    """Every query's target sits at position 1 with the top verification
    score, so a large lambda promotes it and the delta is strictly positive
    at the reference width -- which is what makes the ratio defined."""
    recs = [make_record(i, [0.90, 0.89, 0.50, 0.20, 0.10],
                        [0.10, 0.90, 0.20, 0.30, 0.40],
                        target_pos=1)
            for i in range(20)]
    path = write_cache(recs)
    try:
        res = ww.analyse(path, lam=5.0, widths=(1, 3, 5))
        ref = [r for r in res["rows"] if r["k"] == 5][0]
        assert ref["delta_pp"] > 0.0, "fixture must produce a non-zero delta"
        assert ref["delta_retained_pct"] == pytest.approx(100.0)
    finally:
        os.unlink(path)


def test_retained_is_none_when_the_reference_delta_is_zero():
    """Guards the division. A zero reference delta makes the ratio undefined,
    and reporting 0% or 100% would both be misleading."""
    recs = [make_record(i, [0.9, 0.5, 0.2], [0.2, 0.1, 0.05], target_pos=0)
            for i in range(6)]
    path = write_cache(recs)
    try:
        res = ww.analyse(path, lam=1.0, widths=(1, 3))
        ref = [r for r in res["rows"] if r["k"] == 3][0]
        assert ref["delta_pp"] == pytest.approx(0.0)
        assert ref["delta_retained_pct"] is None
    finally:
        os.unlink(path)


def test_widths_wider_than_the_cache_are_skipped_not_crashed():
    recs = [make_record(i, [0.9, 0.5], [0.2, 0.1], target_pos=0)
            for i in range(4)]
    path = write_cache(recs)
    try:
        res = ww.analyse(path, lam=1.0, widths=(1, 2, 5, 20))
        assert [r["k"] for r in res["rows"]] == [1, 2]
        assert res["widths_skipped"] == [5, 20]
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# relevance conventions
# ---------------------------------------------------------------------------

def test_multipositive_hit_uses_any_relevant():
    recs = [make_record(0, [0.9, 0.5, 0.2], [0.1, 0.2, 0.3],
                        relevant=[0, 2])]
    path = write_cache(recs)
    try:
        res = ww.analyse(path, lam=0.0, widths=(3,))
        assert res["multipositive"] is True
        assert res["rows"][0]["base_hit1_pct"] == pytest.approx(100.0)
    finally:
        os.unlink(path)


def test_missing_target_never_counts_as_a_hit():
    recs = [make_record(0, [0.9, 0.5], [0.1, 0.2], target_pos=-1)]
    path = write_cache(recs)
    try:
        res = ww.analyse(path, lam=1.0, widths=(2,))
        assert res["rows"][0]["base_hit1_pct"] == pytest.approx(0.0)
        assert res["rows"][0]["fused_hit1_pct"] == pytest.approx(0.0)
    finally:
        os.unlink(path)


def test_mixed_conventions_abort():
    path = write_cache([make_record(0, [0.9, 0.5], [0.1, 0.2], target_pos=0),
                        make_record(1, [0.8, 0.5], [0.1, 0.2], relevant=[0])])
    try:
        with pytest.raises(ww.CacheError, match="mixes"):
            ww.load_records(path)
    finally:
        os.unlink(path)


def test_ragged_widths_abort():
    path = write_cache([make_record(0, [0.9, 0.5], [0.1, 0.2], target_pos=0),
                        make_record(1, [0.8, 0.5, 0.3], [0.1, 0.2, 0.3],
                                    target_pos=0)])
    try:
        with pytest.raises(ww.CacheError, match="differing window widths"):
            ww.load_records(path)
    finally:
        os.unlink(path)


def test_nonfinite_rejected():
    rec = make_record(0, [0.9, 0.5], [0.1, 0.2], target_pos=0)
    rec["verification_score"] = [float("nan"), 0.2]
    path = write_cache([rec])
    try:
        with pytest.raises(ww.CacheError, match="non-finite"):
            ww.load_records(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# latency estimate
# ---------------------------------------------------------------------------

def test_latency_estimate_reproduces_the_measured_point():
    """At k=10 the estimate must equal base + measured overhead exactly."""
    assert ww.estimated_latency_ms(10) == pytest.approx(
        ww.BASE_LATENCY_MS + ww.OVERHEAD_AT_10_MS)


def test_latency_estimate_is_monotone_and_above_base():
    prev = 0.0
    for k in (1, 2, 5, 10):
        v = ww.estimated_latency_ms(k)
        assert v > ww.BASE_LATENCY_MS
        assert v > prev
        prev = v


def test_latency_is_labelled_an_estimate_in_the_report(capsys):
    recs = [make_record(i, [0.9, 0.5, 0.2], [0.2, 0.9, 0.1], target_pos=1)
            for i in range(6)]
    path = write_cache(recs)
    try:
        res = ww.analyse(path, lam=1.0)
        ww.report(res)
        out = capsys.readouterr().out
        assert "ESTIMATE" in out, "latency must be marked as an estimate"
        assert "optimistic" in out
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def test_main_requires_lambda():
    recs = [make_record(0, [0.9, 0.5], [0.1, 0.2], target_pos=0)]
    path = write_cache(recs)
    try:
        raised = False
        try:
            ww.main([path])
        except SystemExit:
            raised = True
        assert raised, "--lam must be required"
    finally:
        os.unlink(path)


def test_main_rejects_negative_lambda():
    recs = [make_record(0, [0.9, 0.5], [0.1, 0.2], target_pos=0)]
    path = write_cache(recs)
    try:
        assert ww.main([path, "--lam", "-1"]) == 2
    finally:
        os.unlink(path)


def test_main_writes_json():
    recs = [make_record(i, [0.9, 0.5, 0.2], [0.2, 0.9, 0.1], target_pos=1)
            for i in range(6)]
    path = write_cache(recs)
    out = path + ".json"
    try:
        assert ww.main([path, "--lam", "1.0", "--json", out]) == 0
        with open(out, encoding="utf-8") as fh:
            d = json.load(fh)
        assert d["lam"] == 1.0 and d["n_queries"] == 6
    finally:
        os.unlink(path)
        if os.path.exists(out):
            os.unlink(out)


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


def test_real_circo_k10_reproduces_the_published_base_rate():
    """At k=10 the base hit rate must match the figure the multi-positive
    analysis independently reports (31.818%), which anchors this script to
    the rest of the project."""
    path = _find_circo()
    if path is None:
        pytest.skip("CIRCO cache not present")
    res = ww.analyse(path, lam=1.0, widths=(10,))
    assert res["n_queries"] == 220
    assert res["rows"][0]["base_hit1_pct"] == pytest.approx(31.818, abs=0.01)


def test_real_circo_base_rate_is_monotone_in_k():
    """A wider window can only contain more relevant candidates, so the base
    hit rate cannot fall as k grows."""
    path = _find_circo()
    if path is None:
        pytest.skip("CIRCO cache not present")
    res = ww.analyse(path, lam=1.0)
    rates = [r["base_hit1_pct"] for r in res["rows"]]
    for a, b in zip(rates, rates[1:]):
        assert b >= a - 1e-9, rates


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
    print("ALL WINDOW-WIDTH TESTS PASSED.")
