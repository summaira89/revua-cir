"""Tests for scripts/operating_points.py. Runs without pytest."""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, _HERE)

import apply_frozen_policy as A  # noqa: E402
import operating_points as O  # noqa: E402
from fusion_controls import CacheError, load_records, relevant_set  # noqa: E402
from test_apply_frozen_policy import _cache, _policy, _write  # noqa: E402


def _files(n=200, seed=3, **pol):
    return _write(_policy(**pol), ".json"), _write(_cache(n, seed), ".jsonl")


def _cleanup(*paths):
    for p in paths:
        os.unlink(p)


def test_reproduces_apply_frozen_policy():
    pol, cache = _files(th_learned=0.5, th_base=0.4, th_vgap=0.3)
    wfile = _write({"matched_weights": {"bounded_frozen": {"matched_damage": 0.5,
                                                           "matched_gain": 0.7},
                                        "convex": {"matched_damage": 0.2,
                                                   "matched_gain": 0.3}}}, ".json")
    try:
        ref = A.analyse(pol, [cache], n_rep=50)
        res = O.analyse(pol, [cache], cirr=False, weights_path=wfile, n_rep=50)
        for rname in ("bounded_frozen", "convex"):
            for k in ("base", "fusion", "weaker", "gate_base", "gate_vgap", "gate_learned"):
                a, b = ref["rules"][rname][k], res["rules"][rname][k]
                assert abs(a["r1"] - b["r1"]) < 1e-12, (rname, k)
                assert (a["helped"], a["broken"]) == (b["helped"], b["broken"]), (rname, k)
    finally:
        _cleanup(pol, cache, wfile)


def test_accepted_splits_into_helped_broken_neutral():
    pol, cache = _files()
    try:
        res = O.analyse(pol, [cache], cirr=True, weights_path=None, n_rep=50)
        for rows in res["rules"].values():
            for e in rows.values():
                assert e["neutral"] >= 0
                assert e["helped"] + e["broken"] + e["neutral"] == e["accepted"]
            assert rows["base"]["accepted"] == 0
    finally:
        _cleanup(pol, cache)


def test_matched_weights_respect_their_targets_on_tune():
    pol, cache = _files(n=300, seed=5)
    try:
        res = O.analyse(pol, [cache], cirr=True, weights_path=None, n_rep=50)
        for w in res["matched_weights"].values():
            t = w["tune_target"]
            if w["matched_damage"] is not None:
                assert w["tune_matched_damage"]["broken"] <= t["broken"]
            if w["matched_gain"] is not None:
                assert w["tune_matched_gain"]["helped"] >= t["helped"]
    finally:
        _cleanup(pol, cache)


def test_weight_zero_is_base_and_frozen_weight_is_fusion():
    pol, cache = _files()
    try:
        recs, multi = load_records(cache)
        rels = [relevant_set(r, multi) for r in recs]
        spec = json.load(open(pol))["rules"]["bounded_frozen"]
        prep = O.prepare(recs, rels, spec, json.load(open(pol))["features"])
        fn = A.FUSIONS["bounded"]
        h0, m0 = O.weight_policy(prep, fn, 0.0)
        assert h0 == [A.hit_at(q["ob"], q["rel"], 1) for q in prep] and not any(m0)
        h1, _ = O.weight_policy(prep, fn, spec["param"])
        assert h1 == [A.hit_at(q["of"], q["rel"], 1) for q in prep]
    finally:
        _cleanup(pol, cache)


def test_select_matched_rules_and_infeasible_gain():
    pol, cache = _files()
    try:
        recs, multi = load_records(cache)
        rels = [relevant_set(r, multi) for r in recs]
        p = json.load(open(pol))
        prep = O.prepare(recs, rels, p["rules"]["bounded_frozen"], p["features"])
        fn = A.FUSIONS["bounded"]
        s = O.select_matched(prep, fn, O.grid_for(1.0), target_h=10 ** 6, target_b=0)
        assert s["matched_gain"] is None                  # nothing reaches a million rescues
        assert s["matched_damage"] is not None            # weight 0 always breaks nothing
        assert s["tune_matched_damage"]["broken"] == 0
    finally:
        _cleanup(pol, cache)


def test_grid_contains_zero_frozen_weight_and_weaker_controls():
    g = O.grid_for(1.0)
    assert len(g) == 101 and g[0] == 0.0 and g[-1] == 1.0 and 0.25 in g
    g = O.grid_for(0.5)
    assert 0.1 in g and g[-1] == 0.5


def test_non_cirr_requires_weights():
    pol, cache = _files()
    try:
        try:
            O.analyse(pol, [cache], cirr=False, weights_path=None, n_rep=10)
        except CacheError:
            pass
        else:
            raise AssertionError("expected CacheError")
    finally:
        _cleanup(pol, cache)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok  " + f.__name__)
    print("\n%d tests passed" % len(fns))
