"""Tests for scripts/fc_vs_bounded_paired.py (the CPU parts). No pytest needed."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))

from fc_vs_bounded_paired import (  # noqa: E402
    PairingError, cache_outcomes, check_alignment, check_anchors,
)
from ablation_significance import compare  # noqa: E402


def rec(i, base, ver, tp, tgi):
    return {"query_index": i, "candidate_gallery_ids": list(range(len(base))),
            "base_score": base, "verification_score": ver, "target_pos": tp,
            "target_gallery_index": tgi}


def raises(fn, *a):
    try:
        fn(*a)
    except PairingError:
        return True
    return False


def test_cache_outcomes_orders_by_query_index_and_scores_both_rules():
    # Query 1 listed first: output must still be in query_index order.
    r1 = rec(1, [0.9, 0.1], [0.0, 1.0], 1, 70)   # verifier lifts the target
    r0 = rec(0, [0.9, 0.1], [0.5, 0.5], 0, 50)   # base already right
    base, bounded, targets = cache_outcomes([r1, r0], 5.0)
    assert targets == [50, 70]
    assert base == [1, 0]
    assert bounded == [1, 1]


def test_cache_must_cover_every_index_once():
    assert raises(cache_outcomes, [rec(0, [1.0], [1.0], 0, 1), rec(2, [1.0], [1.0], 0, 1)], 1.0)
    assert raises(cache_outcomes, [rec(0, [1.0], [1.0], 0, 1), rec(0, [1.0], [1.0], 0, 1)], 1.0)


def test_alignment_requires_identical_targets_per_query():
    check_alignment([5, 9, 2], [5, 9, 2])
    assert raises(check_alignment, [5, 9, 2], [5, 2, 9])
    assert raises(check_alignment, [5, 9], [5, 9, 2])


def test_anchors_are_exact_counts():
    fc = [1] * 1505 + [0] * 2676
    base = [1] * 1550 + [0] * 2631
    bounded = [1] * 1646 + [0] * 2535
    check_anchors(fc, base, bounded)
    assert raises(check_anchors, fc[:-1] + [1], base, bounded)


def test_compare_gives_the_end_to_end_delta():
    # 141 more hits on 4181 queries -> +3.372pp, the Table 3 figure.
    fc = [1] * 1505 + [0] * 2676
    bounded = [1] * 1646 + [0] * 2535
    valid = [1] * 4181
    out = compare(fc, bounded, valid, valid, "FC", "bounded")
    assert round(out["delta_pp"], 3) == 3.372
    assert out["help"] == 141 and out["harm"] == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok  " + f.__name__)
    print("\n%d tests passed" % len(fns))
