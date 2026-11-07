"""Tests for scripts/bounded_recall_at_k.py. Runs without pytest."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))

from bounded_recall_at_k import check_anchors, hit_at_k, recall_table  # noqa: E402
from fusion_controls import hit_at_1  # noqa: E402


def rec(base, ver, tp):
    return {"candidate_gallery_ids": list(range(len(base))),
            "base_score": base, "verification_score": ver, "target_pos": tp}


def test_hit_at_k_basic():
    s = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    assert hit_at_k(s, {0}, 1) == 1
    assert hit_at_k(s, {4}, 5) == 1
    assert hit_at_k(s, {5}, 5) == 0
    assert hit_at_k(s, set(), 10) == 0


def test_k1_matches_the_published_hit_at_1_including_ties():
    # Ties must break by original position, as in fusion_controls.
    for s, rel in (([0.5, 0.5, 0.1], {1}), ([0.5, 0.5, 0.1], {0}),
                   ([0.2, 0.9, 0.9], {2}), ([0.3, 0.1], set())):
        assert hit_at_k(s, rel, 1) == hit_at_1(s, rel)


def test_recall_at_10_is_unchanged_by_reranking_a_ten_window():
    import random
    rng = random.Random(0)
    recs = []
    for q in range(200):
        tp = rng.choice([-1] + list(range(10)))
        recs.append(rec([rng.random() for _ in range(10)],
                        [rng.random() for _ in range(10)], tp))
    t = recall_table(recs, False, 1.0)
    assert t[("bounded", 10)] == t[("base", 10)]


def test_reranking_can_change_recall_at_5():
    # Target sits at base position 6; a strong verifier lifts it into the top 5.
    base = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    ver = [0.0] * 10
    ver[6] = 1.0
    t = recall_table([rec(base, ver, 6)], False, 5.0)
    assert t[("base", 5)][0] == 0
    assert t[("bounded", 5)][0] == 1


def test_uncovered_queries_count_as_misses():
    t = recall_table([rec([0.5, 0.4], [0.1, 0.2], -1)], False, 1.0)
    assert all(v[0] == 0 for v in t.values())


def test_anchor_check_flags_a_mismatch():
    good = {("base", 1): (1550, 37.072), ("base", 5): (3024, 72.327),
            ("base", 10): (3536, 84.573), ("bounded", 1): (1646, 39.369)}
    assert check_anchors(good) == []
    bad = dict(good)
    bad[("bounded", 1)] = (1645, 39.345)
    assert len(check_anchors(bad)) == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok  " + f.__name__)
    print("\n%d tests passed" % len(fns))
