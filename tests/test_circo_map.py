"""Regression tests for circo_bounded_map.

The dangerous bug here is an AP formula that is close but not the official
one: a denominator of n_gt instead of min(n_gt, K), or precision counted at
non-relevant positions, both produce plausible numbers that disagree with
the published evaluator. Both are pinned against hand-computed cases, and
`test_real_cache_base_map5_matches_published` checks the whole pipeline
against the 24.49 already in the paper.
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

import circo_bounded_map as cm  # noqa: E402


def make_record(qid, base, rv, gt, top50=None):
    n = len(base)
    cand = list(range(qid * 1000, qid * 1000 + n))
    if top50 is None:
        top50 = cand + list(range(qid * 1000 + n, qid * 1000 + 50))
    return {"query_id": qid,
            "candidate_gallery_ids": cand,
            "base_score": list(base),
            "verification_score": list(rv),
            "relevant_gallery_indices": list(gt),
            "base_top50_gallery_ids": list(top50)}


def write_cache(records):
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8")
    for r in records:
        fh.write(json.dumps(r) + "\n")
    fh.close()
    return fh.name


# ---------------------------------------------------------------------------
# the AP formula
# ---------------------------------------------------------------------------

def test_ap_matches_the_official_formula():
    """Hand-traced against
       precisions = cumsum(labels)*labels / arange(1,n+1)
       AP@K       = sum(precisions[:K]) / min(n_gt, K)

    ranked [a,b,c,d], gt {a,c}, K=4:
       pos 1 a relevant -> 1/1
       pos 2 b no       -> 0
       pos 3 c relevant -> 2/3
       pos 4 d no       -> 0
       AP = (1 + 2/3) / min(2,4) = 1.6667/2 = 0.8333
    """
    got = cm.average_precision_at_k(["a", "b", "c", "d"], ["a", "c"], 4)
    assert got == pytest.approx((1.0 + 2 / 3) / 2)


def test_ap_denominator_is_min_ngt_k_not_ngt():
    """The distinction that matters: with 4 ground truths and K=2, a query
    with both top slots relevant must score 1.0. A denominator of n_gt would
    give 0.5."""
    got = cm.average_precision_at_k(["a", "b", "x", "y"],
                                    ["a", "b", "c", "d"], 2)
    assert got == pytest.approx(1.0)


def test_ap_zeroes_non_relevant_positions():
    """Only ranks holding a ground truth contribute. If precision were
    accumulated at every rank the value would be larger."""
    got = cm.average_precision_at_k(["x", "a"], ["a"], 2)
    assert got == pytest.approx(0.5)      # 1 hit at rank 2 -> 1/2, /min(1,2)


def test_ap_perfect_ranking_is_one():
    assert cm.average_precision_at_k(["a", "b", "c"], ["a", "b"], 3) == \
        pytest.approx(1.0)


def test_ap_is_zero_when_nothing_relevant_is_retrieved():
    assert cm.average_precision_at_k(["x", "y"], ["a"], 2) == \
        pytest.approx(0.0)


def test_ap_respects_the_cutoff():
    """A relevant image beyond K must not count."""
    assert cm.average_precision_at_k(["x", "y", "a"], ["a"], 2) == \
        pytest.approx(0.0)
    assert cm.average_precision_at_k(["x", "y", "a"], ["a"], 3) > 0


def test_ap_with_no_ground_truth_is_zero():
    assert cm.average_precision_at_k(["a", "b"], [], 2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# the splice
# ---------------------------------------------------------------------------

def test_rerank_preserves_the_tail_order():
    rec = make_record(0, [0.9, 0.5, 0.2], [0.1, 0.9, 0.2], gt=[1])
    out = cm.bounded_rerank_top50(rec, lam=5.0)
    assert out[3:] == rec["base_top50_gallery_ids"][3:]


def test_rerank_keeps_every_id_exactly_once():
    rec = make_record(0, [0.9, 0.5, 0.2], [0.1, 0.9, 0.2], gt=[1])
    out = cm.bounded_rerank_top50(rec, lam=2.0)
    assert sorted(out) == sorted(rec["base_top50_gallery_ids"])
    assert len(out) == len(set(out))


def test_rerank_at_lambda_zero_leaves_the_order_unchanged():
    rec = make_record(0, [0.9, 0.5, 0.2], [0.1, 0.9, 0.2], gt=[1])
    assert cm.bounded_rerank_top50(rec, lam=0.0) == \
        rec["base_top50_gallery_ids"]


def test_large_lambda_promotes_the_top_verification_candidate():
    rec = make_record(0, [0.90, 0.89, 0.88], [0.1, 0.9, 0.2], gt=[1])
    out = cm.bounded_rerank_top50(rec, lam=50.0)
    assert out[0] == rec["candidate_gallery_ids"][1]


def test_splice_rejects_a_mismatched_window():
    """If the cached window is not the head of the top-50 the splice would
    duplicate or drop ids. Fail loudly instead."""
    rec = make_record(0, [0.9, 0.5], [0.1, 0.2], gt=[0])
    rec["base_top50_gallery_ids"] = [999, 998] + \
        rec["base_top50_gallery_ids"][2:]
    with pytest.raises(cm.CacheError, match="not the head"):
        cm.bounded_rerank_top50(rec, lam=1.0)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def test_loader_rejects_a_cirr_style_cache():
    """A CIRR cache has no top-50 and no multi-positive annotation; scoring
    it here would silently treat a 10-wide window as the whole ranking."""
    rec = {"query_id": 0, "candidate_gallery_ids": [1, 2],
           "base_score": [0.9, 0.5], "verification_score": [0.1, 0.2],
           "target_pos": 0}
    path = write_cache([rec])
    try:
        with pytest.raises(cm.CacheError, match="CIRCO cache"):
            cm.load_circo_records(path)
    finally:
        os.unlink(path)


def test_loader_rejects_duplicate_ids_in_the_ranking():
    rec = make_record(0, [0.9, 0.5], [0.1, 0.2], gt=[0])
    rec["base_top50_gallery_ids"][5] = rec["base_top50_gallery_ids"][0]
    path = write_cache([rec])
    try:
        with pytest.raises(cm.CacheError, match="duplicate"):
            cm.load_circo_records(path)
    finally:
        os.unlink(path)


def test_loader_rejects_ragged_scores():
    rec = make_record(0, [0.9, 0.5, 0.2], [0.1, 0.2], gt=[0])
    path = write_cache([rec])
    try:
        with pytest.raises(cm.CacheError, match="entries for"):
            cm.load_circo_records(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def test_paired_bootstrap_identical_arms_is_zero():
    a = [0.5, 0.25, 1.0, 0.0] * 10
    r = cm.paired_bootstrap(a, list(a), n_rep=200)
    assert r["delta"] == pytest.approx(0.0)
    assert r["ci_low"] == pytest.approx(0.0)


def test_paired_bootstrap_detects_a_real_difference():
    a = [1.0] * 60 + [0.0] * 10
    b = [0.0] * 70
    r = cm.paired_bootstrap(a, b, n_rep=400)
    assert r["delta"] > 0 and r["ci_low"] > 0 and r["p_two_sided"] < 0.05


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="differ in length"):
        cm.paired_bootstrap([1.0, 0.0], [1.0], n_rep=10)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_analyse_reports_every_rank_and_ranking():
    recs = [make_record(i, [0.9, 0.6, 0.3], [0.2, 0.9, 0.1],
                        gt=[i * 1000 + (i % 3)]) for i in range(30)]
    path = write_cache(recs)
    try:
        res = cm.analyse(path, lam=1.0, n_rep=200)
        for name in ("base", "bounded", "rank"):
            for k in cm.RANKS:
                assert k in res["maps"][name]
    finally:
        os.unlink(path)


def test_lambda_zero_makes_bounded_identical_to_base():
    recs = [make_record(i, [0.9, 0.6, 0.3], [0.2, 0.9, 0.1],
                        gt=[i * 1000]) for i in range(20)]
    path = write_cache(recs)
    try:
        res = cm.analyse(path, lam=0.0, n_rep=100)
        for k in cm.RANKS:
            assert res["maps"]["bounded"][k] == \
                pytest.approx(res["maps"]["base"][k])
    finally:
        os.unlink(path)


def test_map_is_monotone_nondecreasing_in_k():
    """A larger cutoff can only find more relevant images, never fewer."""
    recs = [make_record(i, [0.9, 0.6, 0.3], [0.2, 0.9, 0.1],
                        gt=[i * 1000 + 30]) for i in range(20)]
    path = write_cache(recs)
    try:
        res = cm.analyse(path, lam=1.0, n_rep=100)
        for name in ("base", "bounded", "rank"):
            vals = [res["maps"][name][k] for k in cm.RANKS]
            for a, b in zip(vals, vals[1:]):
                assert b >= a - 1e-9, (name, vals)
    finally:
        os.unlink(path)


def test_report_runs(capsys):
    recs = [make_record(i, [0.9, 0.6, 0.3], [0.2, 0.9, 0.1],
                        gt=[i * 1000]) for i in range(15)]
    path = write_cache(recs)
    try:
        res = cm.analyse(path, lam=1.0, n_rep=100)
        cm.report(res)
        out = capsys.readouterr().out
        assert "mAP@5" in out and "NOT retuned on CIRCO" in out
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# real cache -- the anchor
# ---------------------------------------------------------------------------

def _find_circo():
    for p in [os.path.join(os.path.dirname(__file__), "..", "..",
                           "rv_score_cache_circo_fc_aep_rv.jsonl"),
              os.path.join(os.path.dirname(__file__), "..",
                           "rv_score_cache_circo_fc_aep_rv.jsonl")]:
        if os.path.exists(p):
            return p
    return None


def test_real_cache_base_map5_matches_published():
    """The anchor. The paper reports base CIRCO mAP@5 = 24.49, produced by
    the official evaluator on the full gallery. Reproducing it here from the
    cached top-50 is what says this AP reimplementation agrees with it."""
    path = _find_circo()
    if path is None:
        pytest.skip("CIRCO cache not present")
    res = cm.analyse(path, lam=1.0, n_rep=100)
    assert res["n_queries"] == 220
    assert res["maps"]["base"][5] == pytest.approx(24.49, abs=0.01)


def test_real_cache_mean_ground_truths_matches_the_paper():
    path = _find_circo()
    if path is None:
        pytest.skip("CIRCO cache not present")
    res = cm.analyse(path, lam=1.0, n_rep=50)
    assert res["mean_gt_per_query"] == pytest.approx(4.16, abs=0.01)


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
    print("ALL CIRCO-mAP TESTS PASSED.")
