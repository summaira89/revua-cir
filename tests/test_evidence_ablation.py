"""Tests for scripts/evidence_ablation.py. Runs without pytest."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, _HERE)

import evidence_ablation as E  # noqa: E402
import gate_suite as G  # noqa: E402
from test_apply_frozen_policy import _cache, _policy, _write  # noqa: E402


def _files(n=300, seed=7):
    return _write(_policy(), ".json"), _write(_cache(n, seed), ".jsonl")


def _cleanup(*paths):
    for p in paths:
        os.unlink(p)


def test_full_gate_reproduces_gate_suite_learned_gate():
    pol, cache = _files()
    try:
        ref = G.analyse(cache, n_rep=50)["rules"]["bounded_frozen"]["gates"]["learned"]
        res = E.analyse(pol, cache, [], n_rep=50)
        got = res["rules"]["bounded_frozen"]["directions"]["fit_tune_eval_report"]["full"]
        assert abs(got["r1"] - ref["r1"]) < 1e-12
        assert (got["helped"], got["broken"]) == (ref["helped"], ref["broken"])
    finally:
        _cleanup(pol, cache)


def test_crossfit_is_the_union_of_both_directions():
    pol, cache = _files()
    try:
        r = E.analyse(pol, cache, [], n_rep=50)["rules"]["bounded_frozen"]
        d1 = r["directions"]["fit_tune_eval_report"]
        d2 = r["directions"]["fit_report_eval_tune"]
        for fs in E.FEATURE_SETS:
            for k in ("accepted", "helped", "broken", "changed", "H", "B"):
                assert r["crossfit"][fs][k] == d1[fs][k] + d2[fs][k], (fs, k)
    finally:
        _cleanup(pol, cache)


def test_threshold_frontier_endpoints_and_ties():
    vs = [{"changed": True, "hit_fused": 1, "hit_base": 0, "s": 0.9},
          {"changed": True, "hit_fused": 0, "hit_base": 1, "s": 0.5},
          {"changed": True, "hit_fused": 1, "hit_base": 0, "s": 0.5},
          {"changed": False, "hit_fused": 1, "hit_base": 1, "s": 0.99},
          {"changed": True, "hit_fused": 0, "hit_base": 0, "s": 0.1}]
    pts = E.threshold_frontier(vs, lambda v: v["s"])
    assert pts[0] == (0, 0) and pts[-1] == (1, 2)
    assert (0, 2) not in pts            # cannot take the tied rescue without the tied break
    assert pts == [(0, 0), (0, 1), (1, 2), (1, 2)]


def test_best_at_budgets_is_monotone():
    b = E.best_at_budgets([(0, 0), (3, 10), (12, 20), (40, 30)])
    vals = [b[k] for k in E.BREAK_BUDGETS]
    assert vals == sorted(vals) and b[5] == 10 and b[15] == 20 and b[30] == 20


def test_random_gate_expectation():
    o = {"accepted": 10, "changed": 40, "H": 20, "B": 8, "n": 100}
    r = E.random_gate_expectation(o)
    assert r["rate"] == 0.25 and r["exp_helped"] == 5.0 and r["exp_broken"] == 2.0
    assert abs(r["exp_urisk1"] - (5.0 - 4.0) / 100) < 1e-15


def test_single_class_training_falls_back_like_gate_suite():
    # all rescues: a constant gate; the URisk threshold rule then accepts all
    views = [{"changed": True, "hit_base": 0, "hit_fused": 1,
              "feat": {k: 0.0 for k in E.FEATURE_SETS["full"]}} for _ in range(5)]
    score, t, n = E.fit_gate(views, E.FEATURE_SETS["full"])
    assert n == 5 and E.decisions(views, score, t) == [True] * 5
    # all breaks: the same rule accepts none
    views = [dict(v, hit_base=1, hit_fused=0) for v in views]
    score, t, n = E.fit_gate(views, E.FEATURE_SETS["full"])
    assert E.decisions(views, score, t) == [False] * 5


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok  " + f.__name__)
    print("\n%d tests passed" % len(fns))
