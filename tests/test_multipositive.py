"""Regression tests for scripts/multipositive_analysis.py.

These are written so that reverting the module to single-target semantics
makes them FAIL. That is the point: the bug being guarded against is not a
crash, it is a silent change of definition that still produces plausible
numbers.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_ROOT), str(_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from multipositive_analysis import (  # noqa: E402
    base_order,
    bounded_fusion_order,
    conditional_usefulness,
    definitional_check,
    error_taxonomy,
    hit_at_1,
    is_multipositive,
    load_cache,
    rank_fuse_order,
    relevant_positions,
    rv_only_order,
    _auroc_within_window,
    _zscore,
)


def _write(recs):
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8")
    for r in recs:
        fh.write(json.dumps(r) + "\n")
    fh.close()
    return fh.name


def test_multipositive_hit_differs_from_single_target():
    """The core guard: a relevant item at rank 0 is a HIT even when the
    designated target_pos is -1. Single-target code scores this a miss."""
    rec = {
        "query_id": 1,
        "candidate_gallery_ids": [10, 11, 12],
        "base_score": [0.9, 0.5, 0.1],
        "verification_score": [0.1, 0.2, 0.3],
        "target_pos": -1,                        # designated target NOT in window
        "relevant_gallery_indices": [10, 99],
        "relevant_positions_in_candidates": [0],  # but candidate 10 IS relevant
    }
    rel = relevant_positions(rec)
    assert rel == {0}, rel
    assert hit_at_1(base_order(rec), rel) == 1, (
        "multi-positive hit@1 must count a relevant item at rank 0 even when "
        "target_pos == -1; single-target logic would return 0 here"
    )

    chk = definitional_check([rec])
    assert chk["single_target_hit_at_1_pct"] == 0.0
    assert chk["multipositive_hit_at_1_pct"] == 100.0
    assert chk["understatement_pp"] == 100.0
    assert chk["queries_designated_missed_but_relevant_present"] == 1
    print("  [ok] multi-positive hit@1 diverges from single-target as required")


def test_relevant_positions_falls_back_to_target_pos_for_cirr():
    """A CIRR-style record (no multi-positive fields) must still work."""
    cirr = {"candidate_gallery_ids": [1, 2], "base_score": [0.9, 0.1],
            "verification_score": [0.1, 0.9], "target_pos": 1}
    assert relevant_positions(cirr) == {1}
    assert not is_multipositive([cirr])

    missed = dict(cirr, target_pos=-1)
    assert relevant_positions(missed) == set()
    assert hit_at_1(base_order(missed), relevant_positions(missed)) == 0
    print("  [ok] CIRR single-target fallback intact")


def test_rank_fuse_matches_hand_traced_production_formula():
    """2*base_rank + adj_rank, hand-traced."""
    rec = {"candidate_gallery_ids": [0, 1, 2],
           "base_score": [0.9, 0.8, 0.7],       # base ranks 0,1,2
           "verification_score": [0.1, 0.9, 0.5],  # rv ranks   2,0,1
           "target_pos": 1}
    # fused = 2*b + r  ->  c0: 0+2=2 | c1: 2+0=2 | c2: 4+1=5
    # c0 and c1 tie at 2; stable tie-break keeps lower index first -> [0,1,2]
    assert rank_fuse_order(rec) == [0, 1, 2], rank_fuse_order(rec)

    rec2 = dict(rec, verification_score=[0.1, 0.99, 0.5],
                base_score=[0.9, 0.85, 0.7])
    # identical ranks -> identical result; confirms magnitude does not leak in
    assert rank_fuse_order(rec2) == [0, 1, 2]
    print("  [ok] rank fusion matches hand-traced 2*base_rank + adj_rank")


def test_rank_fuse_is_rank_based_not_magnitude_based():
    """A huge RV magnitude gap must not move anything if ranks are unchanged."""
    a = {"candidate_gallery_ids": [0, 1], "base_score": [0.9, 0.8],
         "verification_score": [0.50, 0.51], "target_pos": 0}
    b = dict(a, verification_score=[0.01, 0.99])
    assert rank_fuse_order(a) == rank_fuse_order(b), (
        "rank fusion must depend only on rank order, not score magnitude"
    )
    print("  [ok] rank fusion ignores magnitude, as designed")


def test_bounded_fusion_lambda_zero_equals_base_and_large_lambda_follows_rv():
    rec = {"candidate_gallery_ids": [0, 1, 2],
           "base_score": [0.9, 0.8, 0.7],
           "verification_score": [0.1, 0.2, 0.95],
           "target_pos": 2}
    assert bounded_fusion_order(rec, 0.0) == base_order(rec), (
        "lambda=0 must reduce exactly to the base ordering"
    )
    assert bounded_fusion_order(rec, 50.0)[0] == 2, (
        "very large lambda must let RV dominate the ordering"
    )
    print("  [ok] bounded fusion endpoints behave correctly")


def test_zscore_handles_zero_variance_without_nan():
    z = _zscore([0.5, 0.5, 0.5])
    assert all(v == 0.0 for v in z), z
    assert not any(v != v for v in z), "z-score produced NaN on constant input"
    print("  [ok] z-score is NaN-safe on constant input")


def test_auroc_is_within_query_and_handles_ties():
    """Perfect separation -> 1.0; reversed -> 0.0; all-tied -> 0.5."""
    perfect = {"candidate_gallery_ids": [0, 1], "base_score": [0.9, 0.1],
               "verification_score": [0.9, 0.1], "target_pos": 0,
               "relevant_positions_in_candidates": [0],
               "relevant_gallery_indices": [0]}
    assert _auroc_within_window([perfect]) == 1.0

    reversed_ = dict(perfect, verification_score=[0.1, 0.9])
    assert _auroc_within_window([reversed_]) == 0.0

    tied = dict(perfect, verification_score=[0.5, 0.5])
    assert _auroc_within_window([tied]) == 0.5, "ties must score exactly 0.5"

    # a query with no relevant item in the window contributes nothing
    none_rel = dict(perfect, relevant_positions_in_candidates=[])
    assert _auroc_within_window([none_rel]) is None
    print("  [ok] within-query AUROC correct incl. ties and empty cases")


def test_conditional_usefulness_separates_covered_from_uncovered():
    """Coverage C_K and the covered-only subset must be computed on the
    multi-positive definition, and uncovered queries cannot change."""
    covered = {"candidate_gallery_ids": [0, 1], "base_score": [0.9, 0.8],
               "verification_score": [0.1, 0.9], "target_pos": 1,
               "relevant_gallery_indices": [1],
               "relevant_positions_in_candidates": [1]}
    uncovered = {"candidate_gallery_ids": [2, 3], "base_score": [0.9, 0.8],
                 "verification_score": [0.9, 0.1], "target_pos": -1,
                 "relevant_gallery_indices": [77],
                 "relevant_positions_in_candidates": []}
    cu = conditional_usefulness([covered, uncovered])
    assert cu["n_queries"] == 2
    assert cu["n_covered"] == 1
    assert cu["candidate_coverage_C_K"] == 0.5
    # overall base hit@1 = 0/2; covered-only = 0/1
    assert cu["variants"]["rv_only"]["overall"]["n"] == 2
    assert cu["variants"]["rv_only"]["covered_only"]["n"] == 1
    # rv_only puts candidate 1 first on the covered query -> a rescue
    assert cu["variants"]["rv_only"]["covered_only"]["help"] == 1
    print("  [ok] conditional usefulness isolates the covered subset")


def test_error_taxonomy_buckets_are_exhaustive_and_hand_traced():
    """One query of each type, verified by construction."""
    # A: base wrong, rv_only right
    a = {"candidate_gallery_ids": [0, 1], "base_score": [0.9, 0.1],
         "verification_score": [0.1, 0.9], "target_pos": 1,
         "relevant_gallery_indices": [1], "relevant_positions_in_candidates": [1]}
    # B: base right, rv_only wrong
    b = {"candidate_gallery_ids": [0, 1], "base_score": [0.9, 0.1],
         "verification_score": [0.1, 0.9], "target_pos": 0,
         "relevant_gallery_indices": [0], "relevant_positions_in_candidates": [0]}
    # C: both wrong (relevant item sits at index 1, both rank index 0 first)
    c = {"candidate_gallery_ids": [0, 1], "base_score": [0.9, 0.1],
         "verification_score": [0.9, 0.1], "target_pos": 1,
         "relevant_gallery_indices": [1], "relevant_positions_in_candidates": [1]}
    # D: both right
    d = {"candidate_gallery_ids": [0, 1], "base_score": [0.9, 0.1],
         "verification_score": [0.9, 0.1], "target_pos": 0,
         "relevant_gallery_indices": [0], "relevant_positions_in_candidates": [0]}

    tax = error_taxonomy([a, b, c, d], variant="rv_only")
    counts = {k: tax["types"][k]["count"] for k in "ABCD"}
    assert counts == {"A": 1, "B": 1, "C": 1, "D": 1}, counts
    assert sum(counts.values()) == 4, "taxonomy must partition every query"
    print("  [ok] error taxonomy A/B/C/D hand-traced and exhaustive")


def test_rank_fuse_order_matches_production_rank_fuse_randomized():
    """Guard against silent drift between the analysis reimplementation and
    the production `_rank_fuse`.

    The audit flagged this as the highest-risk maintenance problem, and it
    is a fair concern: this module reimplements the fusion maths in pure
    Python so it can run on a login node without torch on the hot path.
    Equivalence is therefore asserted rather than assumed.

    Tie pressure is deliberate. `_rank_fuse`'s own docstring records that
    its RETURNED fused_rank values can still tie across candidates, so the
    caller's argsort must also pass stable=True; this test drives coarse
    quantisation (q=1,2,3) specifically to manufacture those ties.
    """
    import random

    import torch

    from revua_cir.models.revua_cir import _rank_fuse

    rng = random.Random(0)
    mismatches = []
    for _ in range(2000):
        k = rng.choice([2, 3, 5, 10])
        q = rng.choice([1, 2, 3, 1000])   # low q => many exact ties
        base = [rng.randint(0, q) / q for _ in range(k)]
        rv = [rng.randint(0, q) / q for _ in range(k)]

        fused = _rank_fuse(torch.tensor(base, dtype=torch.float64),
                           torch.tensor(rv, dtype=torch.float64))
        production = fused.argsort(descending=True, stable=True).tolist()
        analysis = rank_fuse_order({"base_score": base, "verification_score": rv})
        if production != analysis:
            mismatches.append((base, rv, production, analysis))

    assert not mismatches, (
        "analysis rank_fuse_order diverged from production _rank_fuse on "
        "%d/2000 cases; first: %r" % (len(mismatches), mismatches[0])
    )
    print("  [ok] rank_fuse_order matches production _rank_fuse on 2000 "
          "randomized cases incl. heavy ties")


def test_load_cache_rejects_ragged_and_malformed_records():
    ok = {"candidate_gallery_ids": [0, 1], "base_score": [0.9, 0.1],
          "verification_score": [0.5, 0.5], "target_pos": 0}
    path = _write([ok])
    assert len(load_cache(path)) == 1

    ragged = {"candidate_gallery_ids": [0, 1], "base_score": [0.9],
              "verification_score": [0.5, 0.5], "target_pos": 0}
    try:
        load_cache(_write([ragged]))
        raise AssertionError("ragged record must be rejected")
    except ValueError as exc:
        assert "ragged" in str(exc)

    missing = {"candidate_gallery_ids": [0, 1], "base_score": [0.9, 0.1]}
    try:
        load_cache(_write([missing]))
        raise AssertionError("record missing verification_score must be rejected")
    except ValueError as exc:
        assert "verification_score" in str(exc)
    print("  [ok] cache loader rejects ragged and incomplete records")


def test_real_circo_cache_reproduces_cached_window_hit1():
    """End-to-end check against job 6486's CACHED-WINDOW diagnostics.

    NOT the headline CIRCO metrics. Job 6486's published mAP@5 figures are
    base 24.49 / rerank 22.49, computed by the official evaluator over the
    full 123k gallery. The 31.818 / 27.273 asserted here are multi-positive
    hit@1 within the cached top-10 window -- a different metric entirely.
    Conflating the two is exactly how a wrong number reaches a paper table.

    Skips cleanly when the cache is not present locally, so the suite stays
    runnable on the cluster and on a fresh checkout.
    """
    cache = _ROOT.parent / "rv_score_cache_circo_fc_aep_rv.jsonl"
    if not cache.exists():
        print("  [skip] CIRCO cache not present at %s" % cache)
        return

    recs = load_cache(cache)
    assert len(recs) == 220, len(recs)
    assert is_multipositive(recs)

    base = [hit_at_1(base_order(r), relevant_positions(r)) for r in recs]
    fused = [hit_at_1(rank_fuse_order(r), relevant_positions(r)) for r in recs]
    b_pct = 100.0 * sum(base) / len(base)
    f_pct = 100.0 * sum(fused) / len(fused)
    help_ = sum(1 for x, y in zip(base, fused) if not x and y)
    harm = sum(1 for x, y in zip(base, fused) if x and not y)

    assert abs(b_pct - 31.818) < 0.01, b_pct
    assert abs(f_pct - 27.273) < 0.01, f_pct
    assert (help_, harm) == (9, 19), (help_, harm)

    chk = definitional_check(recs)
    assert abs(chk["understatement_pp"] - 8.636) < 0.01, chk["understatement_pp"]
    assert chk["queries_with_multiple_relevant_in_window"] == 78
    assert chk["queries_designated_missed_but_relevant_present"] == 42
    print("  [ok] real CIRCO cache reproduces job-6486 cached-window diagnostics")


if __name__ == "__main__":
    test_multipositive_hit_differs_from_single_target()
    test_relevant_positions_falls_back_to_target_pos_for_cirr()
    test_rank_fuse_matches_hand_traced_production_formula()
    test_rank_fuse_is_rank_based_not_magnitude_based()
    test_bounded_fusion_lambda_zero_equals_base_and_large_lambda_follows_rv()
    test_zscore_handles_zero_variance_without_nan()
    test_auroc_is_within_query_and_handles_ties()
    test_conditional_usefulness_separates_covered_from_uncovered()
    test_error_taxonomy_buckets_are_exhaustive_and_hand_traced()
    test_rank_fuse_order_matches_production_rank_fuse_randomized()
    test_load_cache_rejects_ragged_and_malformed_records()
    test_real_circo_cache_reproduces_cached_window_hit1()
    print("\nALL MULTI-POSITIVE TESTS PASSED.")
