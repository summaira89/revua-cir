"""Tests for scripts/apply_frozen_policy.py. Runs without pytest."""
import json
import math
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import apply_frozen_policy as A  # noqa: E402

FEATS = ["base_margin", "ver_gap", "base_gap", "fused_base_rank", "ver_margin"]


def _policy(th_learned=0.5, th_base=math.inf, th_vgap=-math.inf):
    spec = {"param": 1.0, "weaker_param": 0.25,
            "logistic": {"weights": [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                         "mean": [0.0] * 5, "sd": [1.0] * 5},
            "thresholds": {"learned": th_learned, "base": th_base, "vgap": th_vgap}}
    return {"features": FEATS, "rules": {"bounded_frozen": dict(spec, fusion="bounded"),
                                         "convex": dict(spec, fusion="convex", param=0.5,
                                                        weaker_param=0.1)}}


def _write(obj, suffix):
    fh = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8")
    if suffix == ".jsonl":
        for r in obj:
            fh.write(json.dumps(r) + "\n")
    else:
        json.dump(obj, fh)
    fh.close()
    return fh.name


def _cache(n=80, seed=2):
    rng = random.Random(seed)
    out = []
    for q in range(n):
        k, tp = 8, rng.choice([0, 1, 2, 3, -1])
        ver = [rng.random() for _ in range(k)]
        if tp >= 0:
            ver[tp] += 0.7
        out.append({"query_id": q, "candidate_gallery_ids": list(range(q * 10, q * 10 + k)),
                    "base_score": sorted([rng.random() for _ in range(k)], reverse=True),
                    "verification_score": ver, "target_pos": tp})
    return out


def test_logistic_scorer_formula():
    s = A.logistic_scorer({"weights": [0.5, 1.0], "mean": [2.0], "sd": [4.0]}, ["x"])
    assert abs(s({"x": 6.0}) - 1 / (1 + math.exp(-(0.5 + 1.0)))) < 1e-15


def test_hit_at_k():
    assert A.hit_at([3, 1, 2], {1}, 1) == 0 and A.hit_at([3, 1, 2], {1}, 2) == 1
    assert A.hit_at([0, 1], set(), 5) == 0


def test_fisher_matches_known_value():
    # classic tea-tasting table [[3,1],[1,3]]: two-sided p = 0.4857
    assert abs(A.fisher_two_sided(3, 1, 1, 3) - 0.4857142857) < 1e-9
    assert A.fisher_two_sided(0, 0, 5, 5) == 1.0


def test_accept_all_gate_equals_fusion_and_accept_none_equals_base():
    pol, cache = _write(_policy(th_learned=-1.0), ".json"), _write(_cache(), ".jsonl")
    try:
        res = A.analyse(pol, [cache], n_rep=100)
        r = res["rules"]["bounded_frozen"]
        assert r["gate_learned"]["r1"] == r["fusion"]["r1"]           # threshold -1: accept all
        assert r["gate_base"]["r1"] == r["fusion"]["r1"]              # base thr +inf: accept all
        assert r["gate_vgap"]["r1"] == r["fusion"]["r1"]              # vgap thr -inf: accept all
    finally:
        os.unlink(pol)
        os.unlink(cache)
    pol, cache = _write(_policy(th_learned=2.0, th_base=-math.inf, th_vgap=math.inf), ".json"), \
        _write(_cache(), ".jsonl")
    try:
        r = A.analyse(pol, [cache], n_rep=100)["rules"]["bounded_frozen"]
        for g in ("gate_learned", "gate_base", "gate_vgap"):
            assert r[g]["r1"] == r["base"]["r1"] and r[g]["helped"] == 0 and r[g]["broken"] == 0
    finally:
        os.unlink(pol)
        os.unlink(cache)


def test_multiple_caches_are_pooled():
    pol = _write(_policy(), ".json")
    c1, c2 = _write(_cache(40, 1), ".jsonl"), _write(_cache(30, 2), ".jsonl")
    try:
        res = A.analyse(pol, [c1, c2], n_rep=100)
        assert res["n"] == 70 and len(res["sources"]) == 2
    finally:
        for p in (pol, c1, c2):
            os.unlink(p)


def test_anchor_check_reports_mismatches():
    rules = {}
    for (rname, k), v in A.CIRR_REPORT_ANCHORS.items():
        rules.setdefault(rname, {})[k] = {"r1": v}
    assert A.check_cirr_anchors({"rules": rules}) == []
    rules["convex"]["gate_learned"]["r1"] = 38.70
    assert len(A.check_cirr_anchors({"rules": rules})) == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok  " + f.__name__)
    print("\n%d tests passed" % len(fns))
