"""revision_followups4.py (amendment §23): helper contracts on synthetic inputs."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import revision_followups3 as r3  # noqa: E402
import revision_followups4 as r4  # noqa: E402

RNG = np.random.default_rng(23)


def test_profile_hand_example():
    tv = np.array([0.0, 0.9, 0.5, 0.7, 0.2, 0.8, 0.1])
    # target 2; W_B brings in row 1 (T 0.9), W_I has row 4 instead
    p = r4.query_profile([2, 1, 3], [3, 2, 4], 2, tv)
    assert p["shared"] == 1 and p["hb"] == 1 and p["hb_swapped_in"] == 1
    assert p["d_b"] == 0.9 and p["d_i"] == 0.7 and p["hi_in_wb"] == 1 and p["di_subset_wb"] == 0
    assert p["t_b"] == 0 and p["t_i"] == 0
    assert p["only_b"] == [1] and p["only_i"] == [4]
    assert p["m_b"] == pytest.approx(-0.4) and p["m_i"] == pytest.approx(-0.2)


def test_profile_tie_goes_to_lowest_position_like_hit_at_1():
    tv = np.array([0.5, 0.5, 0.5])
    p = r4.query_profile([1, 0, 2], [0, 1, 2], 0, tv)   # target 0 at position 1 in W_B, 0 in W_I
    assert p["hb"] == 1 and p["m_b"] == 0.0 and p["t_b"] == 0  # distractor 1 precedes the target
    assert p["t_i"] == 1                                       # target precedes in W_I


def test_profile_requires_target_in_both():
    with pytest.raises(ValueError):
        r4.query_profile([0, 1], [1, 2], 0, np.zeros(3))


def test_nesting_implies_no_fixed_without_ties():
    for _ in range(500):
        tv = RNG.uniform(size=40)
        rows = RNG.permutation(40)
        t, wb, extra = int(rows[0]), [int(x) for x in rows[:10]], [int(x) for x in rows[10:15]]
        wi = [t] + [int(x) for x in RNG.choice(wb[1:] + extra, 9, replace=False)]
        p = r4.query_profile(wb, wi, t, tv)
        if p["hi_in_wb"]:
            assert p["d_b"] >= p["d_i"] and not (p["t_b"] and not p["t_i"])
        assert len(p["only_b"]) == len(p["only_i"])


def test_gallery_ranks_match_rank_of():
    for _ in range(100):
        tv = np.round(RNG.uniform(size=30), 2)
        ex = int(RNG.integers(30))
        rk = r4.gallery_ranks(tv, ex)
        assert rk[ex] == -1
        for row in range(30):
            if row != ex:
                assert rk[row] == r3.rank_of(tv, row, ex)


def test_boot_mean_deterministic_and_brackets_mean():
    x = RNG.normal(size=200)
    a, b = r4.boot_mean(x, 2000, 0), r4.boot_mean(x, 2000, 0)
    assert a == b and a["ci_low"] < a["mean"] < a["ci_high"] and a["n"] == 200


def test_preflight_aborts_on_mismatch():
    P = {0: {"t_b": 0, "t_i": 1}, 2: {"t_b": 1, "t_i": 1}}
    want = {"both_covered": {"n": 2, "T_WB_minus_WI": {"fixed": 0, "broken": 1}}}
    r4.preflight(P, [0, 1, 2], want)
    want["both_covered"]["T_WB_minus_WI"]["broken"] = 2
    with pytest.raises(SystemExit):
        r4.preflight(P, [0, 1, 2], want)


def test_summarise_is_json_serialisable_and_consistent():
    import json
    P = []
    for _ in range(60):
        tv = RNG.uniform(size=40)
        rows = RNG.permutation(40)
        t, wb = int(rows[0]), [int(x) for x in rows[:10]]
        wi = [t] + [int(x) for x in RNG.choice(wb[1:] + [int(x) for x in rows[10:20]], 9, replace=False)]
        p = r4.query_profile(wb, wi, t, tv)
        rank = r4.gallery_ranks(tv, int(rows[-1]))
        p["hb_rank"] = int(rank[p["hb"]])
        p["only_b_t"] = [float(tv[r]) for r in p["only_b"]]
        p["only_i_t"] = [float(tv[r]) for r in p["only_i"]]
        p["only_b_rank"] = [int(rank[r]) for r in p["only_b"]]
        p["only_i_rank"] = [int(rank[r]) for r in p["only_i"]]
        P.append(p)
    s = r4.summarise(P)
    json.dumps(s)
    assert s["n"] == 60 and sum(s["overlap"]["shared_hist"]) == 60
    td = s["t_discordance"]
    assert td["fixed_hi_in_wb"] == 0
    if td["broken"]:
        assert s["h_b_source"]["share_swapped_in_broken"] == 1.0
