"""Tests for scripts/fusion_suite.py. Runs without pytest."""
import json
import math
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import fusion_suite as F  # noqa: E402
from fusion_controls import _ranks_desc  # noqa: E402


def test_rrf_matches_hand_values():
    b, r = [0.9, 0.5, 0.1], [0.1, 0.9, 0.5]
    s = F.r_rrf60(b, r, 0.0)
    # base ranks 1,2,3 ; verifier ranks 3,1,2
    want = [1 / 61 + 1 / 63, 1 / 62 + 1 / 61, 1 / 63 + 1 / 62]
    assert all(abs(x - y) < 1e-15 for x, y in zip(s, want))


def test_zero_weight_reproduces_base_order_for_every_tunable_rule():
    rng = random.Random(0)
    for _ in range(50):
        b = [rng.random() for _ in range(8)]
        r = [rng.random() for _ in range(8)]
        base_order = sorted(range(8), key=lambda i: (-b[i], i))
        for fn in (F.r_wrrf, F.r_linear, F.r_bounded, F.r_convex):
            s = fn(b, r, 0.0)
            assert sorted(range(8), key=lambda i: (-s[i], i)) == base_order, fn.__name__


def test_convex_handles_a_constant_window():
    assert F.r_convex([0.5, 0.5], [0.2, 0.9], 0.0) == [0.0, 0.0]
    assert F.r_convex([0.1, 0.9], [0.4, 0.4], 0.5) == [0.0, 0.5]


def test_rank_rules_are_invariant_to_monotone_transforms():
    rng = random.Random(1)
    for _ in range(50):
        b = [rng.random() for _ in range(7)]
        cos = [rng.uniform(-1, 1) for _ in range(7)]
        cal = [1 / (1 + math.exp(-(3.0 * c - 0.4))) for c in cos]   # Platt, slope > 0
        for fn in (F.r_rank_wsum, F.r_rrf60):
            assert _ranks_desc(fn(b, cos, 0.0)) == _ranks_desc(fn(b, cal, 0.0))
        assert _ranks_desc(F.r_wrrf(b, cos, 2.0)) == _ranks_desc(F.r_wrrf(b, cal, 2.0))


def test_urisk_trisk_hand_values():
    rule = [1, 1, 0, 0, 1]
    base = [0, 1, 1, 0, 1]          # one win, one loss
    out = F.urisk_trisk(rule, base, 1.0)
    xs = [1, 0, -2, 0, 0]
    u = sum(xs) / 5
    se = (sum((x - u) ** 2 for x in xs) / 4) ** 0.5 / 5 ** 0.5
    assert abs(out["urisk"] - u) < 1e-12 and abs(out["trisk"] - u / se) < 1e-12
    # binary outcomes: URisk = (helped - (1+alpha) * broken) / n
    assert abs(F.urisk_trisk(rule, base, 5.0)["urisk"] - (1 - 6 * 1) / 5) < 1e-12


def _cache(n=120, seed=3):
    rng = random.Random(seed)
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for q in range(n):
        k = 6
        tp = rng.choice([0, 0, 1, 2, 3, -1])
        cos = [rng.uniform(-1, 1) for _ in range(k)]
        if tp >= 0:
            cos[tp] += 0.6                      # an informative verifier
        rec = {"query_id": q, "candidate_gallery_ids": list(range(q * 10, q * 10 + k)),
               "base_score": sorted([rng.random() for _ in range(k)], reverse=True),
               "cos_sim": cos,
               "verification_score": [1 / (1 + math.exp(-(2.5 * c + 0.1))) for c in cos],
               "target_pos": tp}
        fh.write(json.dumps(rec) + "\n")
    fh.close()
    return fh.name


def test_analyse_end_to_end_and_invariance_check():
    path = _cache()
    try:
        res = F.analyse(path, n_rep=200)
        assert res["n_tune"] + res["n_report"] == 120
        assert all(res["rank_rules_invariant_to_calibration"].values())
        assert res["rows"]["base"]["helped"] == 0 and res["rows"]["base"]["broken"] == 0
        assert res["rows"]["bounded_frozen[calibrated]"]["param"] == 1.0
        assert res["strongest_on_tune"] in res["rows"]
        assert res["strongest_on_tune"] not in res["vs_strongest"]
        for v in res["rows"].values():
            assert len(v["risk"]) == 2
    finally:
        os.unlink(path)


def test_strongest_is_chosen_on_tune_not_report():
    # The selection key uses tune_r1 only; perturbing report hits cannot change it.
    path = _cache(seed=7)
    try:
        res = F.analyse(path, n_rep=100)
        tuned = [k for k, v in res["rows"].items() if F.RULES[v["rule"]][1] is not None]
        best_tune = max(res["rows"][k]["tune_r1"] for k in tuned)
        assert res["rows"][res["strongest_on_tune"]]["tune_r1"] == best_tune
    finally:
        os.unlink(path)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok  " + f.__name__)
    print("\n%d tests passed" % len(fns))
