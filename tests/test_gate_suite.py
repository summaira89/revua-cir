"""Tests for scripts/gate_suite.py. Runs without pytest."""
import json
import math
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import gate_suite as G  # noqa: E402
from fusion_suite import r_bounded  # noqa: E402


def rec(base, ver, tp):
    return {"candidate_gallery_ids": list(range(len(base))), "base_score": base,
            "verification_score": ver, "target_pos": tp}


def test_query_view_outcomes_and_features():
    # verifier strongly prefers position 2; bounded at large lambda moves top-1 there
    r = rec([0.9, 0.8, 0.7], [0.1, 0.2, 0.9], 2)
    v = G.query_view(r, {2}, r_bounded, 5.0)
    assert v["changed"] and v["hit_fused"] == 1 and v["hit_base"] == 0
    assert v["feat"]["fused_base_rank"] == 2.0
    assert v["feat"]["ver_gap"] > 0 and v["feat"]["base_gap"] > 0
    assert v["feat"]["base_margin"] > 0


def test_unchanged_query_is_never_affected_by_the_gate():
    v = {"changed": False, "hit_base": 1, "hit_fused": 1, "feat": {}}
    assert G.gated_hits([v], lambda _: False) == [1]
    assert G.gated_hits([v], lambda _: True) == [1]


def test_gated_hits_switch_between_fused_and_base():
    vs = [{"changed": True, "hit_base": 0, "hit_fused": 1, "feat": {}},
          {"changed": True, "hit_base": 1, "hit_fused": 0, "feat": {}}]
    assert G.gated_hits(vs, lambda _: True) == [1, 0]
    assert G.gated_hits(vs, lambda _: False) == [0, 1]


def test_threshold_choice_blocks_a_separable_harmful_change():
    # helped changes have large verifier gap, broken ones small: the gate should
    # accept the first group and block the second.
    views = []
    for i in range(40):
        helped = i % 2 == 0
        views.append({"changed": True, "hit_base": 0 if helped else 1,
                      "hit_fused": 1 if helped else 0,
                      "feat": {"ver_gap": (2.0 if helped else 0.1)}})
    t, n_acc = G.choose_threshold(views, lambda v: v["feat"]["ver_gap"], "ge", 1.0)
    hits = G.gated_hits(views, lambda v: v["feat"]["ver_gap"] >= t)
    assert sum(hits) == 40 and n_acc == 20


def test_threshold_can_choose_accept_none_when_fusion_only_hurts():
    views = [{"changed": True, "hit_base": 1, "hit_fused": 0, "feat": {"x": float(i)}}
             for i in range(10)]
    t, n_acc = G.choose_threshold(views, lambda v: v["feat"]["x"], "ge", 1.0)
    assert n_acc == 0 and t == math.inf


def test_logistic_learns_the_separating_direction():
    rng = random.Random(0)
    X, y = [], []
    for _ in range(300):
        lab = rng.random() < 0.5
        f = [rng.gauss(0, 1) for _ in G.FEATURES]
        f[1] += 1.5 if lab else -1.5          # ver_gap separates
        X.append(f)
        y.append(1 if lab else 0)
    score, model = G.fit_logistic(X, y)
    hi = dict(zip(G.FEATURES, [0, 3, 0, 0, 0]))
    lo = dict(zip(G.FEATURES, [0, -3, 0, 0, 0]))
    assert score(hi) > 0.9 and score(lo) < 0.1 and model["weights"][2] > 0


def test_risk_x_matches_urisk_definition():
    hits, base = [1, 0, 1, 0], [0, 1, 1, 0]
    assert G.risk_x(hits, base, 1.0) == [1, -2.0, 0, 0]


def _cache(n=200, seed=5):
    rng = random.Random(seed)
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for q in range(n):
        k, tp = 6, rng.choice([0, 0, 1, 2, -1])
        ver = [rng.random() for _ in range(k)]
        if tp >= 0:
            ver[tp] += 0.5
        fh.write(json.dumps({"query_id": q, "candidate_gallery_ids": list(range(q * 10, q * 10 + k)),
                             "base_score": sorted([rng.random() for _ in range(k)], reverse=True),
                             "verification_score": ver, "target_pos": tp}) + "\n")
    fh.close()
    return fh.name


def test_analyse_end_to_end_and_none_gate_equals_fusion():
    path = _cache()
    try:
        res = G.analyse(path, n_rep=200)
        for r in res["rules"].values():
            none = r["gates"]["none"]
            assert none["reverted_share_of_changed"] == 0.0
            assert none["helped_kept"] in (1.0, None) and none["broken_blocked"] in (0.0, None)
            for s in r["gates"].values():
                assert 0.0 <= s["reverted_share_of_changed"] <= 1.0
                assert "vs_weaker_urisk1" in s and "vs_weaker_r1" in s
            assert "weaker" in r["gates"]
        assert res["rules"]["bounded_frozen"]["weaker_param"] in G.RISK_GRIDS["bounded_frozen"]
        assert res["rules"]["convex"]["weaker_param"] in G.RISK_GRIDS["convex"]
        assert 0.0 in G.RISK_GRIDS["bounded_frozen"] and 0.0 in G.RISK_GRIDS["convex"]
    finally:
        os.unlink(path)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok  " + f.__name__)
    print("\n%d tests passed" % len(fns))
