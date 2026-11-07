"""revision_followups2.py (amendment §21): helper contracts on synthetic inputs."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import fusion_controls as fc  # noqa: E402
import fusion_suite as fs  # noqa: E402
import revision_followups2 as r2  # noqa: E402
import selective_retrieval_analysis as sra  # noqa: E402

RNG = np.random.default_rng(7)


def _recs(n=60, k=10, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for q in range(n):
        b = sorted(rng.uniform(0.4, 0.7, k).tolist(), reverse=True)
        out.append({"base_score": b, "verification_score": rng.uniform(0.6, 0.7, k).tolist(),
                    "recon_cos": rng.uniform(0.8, 1.0, k).tolist(), "target_pos": int(rng.integers(-1, k)),
                    "candidate_gallery_ids": list(range(q * k, q * k + k)), "target_gallery_index": q * k})
    return out


def test_three_term_reduces_to_bounded_when_lx_is_zero():
    b, t, x = RNG.uniform(size=10).tolist(), RNG.uniform(size=10).tolist(), RNG.uniform(size=10).tolist()
    assert r2.three_term(b, t, x, 2.0, 0.0) == pytest.approx(fc.rule_bounded(b, t, 2.0))
    s = r2.three_term(b, t, x, 1.0, 0.5)
    zb, zt, zx = fc._zscore(b), fc._zscore(t), fc._zscore(x)
    assert s[3] == pytest.approx(zb[3] + math.tanh(zt[3]) + 0.5 * math.tanh(zx[3]))


def test_select_three_tie_rule_prefers_no_increment():
    # X constant within every window -> z(X) = 0 -> every lx ties; the smallest (0) must win
    recs = _recs()
    rel = [fc.relevant_set(r, False) for r in recs]
    T = [r["recon_cos"] for r in recs]
    X = [[0.5] * 10 for _ in recs]
    base = [fc.hit_at_1(r["base_score"], rel[i]) for i, r in enumerate(recs)]
    lt, lx = r2.select_three(recs, rel, T, X, list(range(len(recs))), base, "r1")
    assert lx == 0.0
    # and lt equals the bounded-rule tuning with ties to the smaller lambda
    assert lt == r2.select_rule(recs, rel, T, fc.rule_bounded, fc.LAMBDA_GRID, list(range(len(recs))), base, "r1")


def test_select_rule_parameter_free_and_ties():
    recs = _recs(5)
    rel = [fc.relevant_set(r, False) for r in recs]
    assert r2.select_rule(recs, rel, None, None, None, [0], [0], "r1") == 0.0
    # a constant second signal makes every lambda tie -> smallest grid value
    T = [[0.3] * 10 for _ in recs]
    base = [fc.hit_at_1(r["base_score"], rel[i]) for i, r in enumerate(recs)]
    assert r2.select_rule(recs, rel, T, fc.rule_bounded, fc.LAMBDA_GRID, [0, 1, 2], base, "u") == fc.LAMBDA_GRID[0]


def test_utility_is_urisk_alpha1():
    h, b = [1, 0, 1, 0], [0, 1, 1, 0]
    assert r2.utility(h, b) == pytest.approx((1 - 2) / 4)
    assert r2.utility(h, b) == fs.urisk_trisk(h, b, 1.0)["urisk"]


def test_mean_midranks_matches_combine_by_rank_and_three_way():
    a, b, c = RNG.normal(size=50).tolist(), RNG.normal(size=50).tolist(), RNG.normal(size=50).tolist()
    assert r2.mean_midranks(a, b) == sra.combine_by_rank(a, b)
    ra, rb, rc = sra._midranks(a), sra._midranks(b), sra._midranks(c)
    assert r2.mean_midranks(a, b, c) == pytest.approx([(x + y + z) / 3 for x, y, z in zip(ra, rb, rc)])


def test_text_cosines_alignment_checks():
    recs = [{"target_gallery_index": 1, "candidate_gallery_ids": [0, 1, 2]}]
    img = RNG.normal(size=(5, 4)).astype(np.float32)
    feats = {"image_emb": img, "text_emb": RNG.normal(size=(1, 4)).astype(np.float32),
             "ref_row": np.array([4]), "tgt_row": np.array([1])}
    T = r2.text_cosines(recs, feats, 5)
    u = img[[0, 1, 2]] / np.linalg.norm(img[[0, 1, 2]], axis=1, keepdims=True)
    t = feats["text_emb"][0] / np.linalg.norm(feats["text_emb"][0])
    assert T[0] == pytest.approx((u @ t).tolist(), abs=1e-6)
    with pytest.raises(SystemExit):
        r2.text_cosines(recs, feats, 6)                       # wrong gallery size
    with pytest.raises(SystemExit):
        r2.text_cosines(recs, dict(feats, ref_row=np.array([2])), 5)   # reference in window
    with pytest.raises(SystemExit):
        r2.text_cosines(recs, dict(feats, tgt_row=np.array([0])), 5)   # misaligned target


def test_part_a_and_b_run_on_synthetic_windows():
    recs = _recs(80)
    rel = [fc.relevant_set(r, False) for r in recs]
    T = [np.random.default_rng(i).uniform(0.1, 0.3, 10).tolist() for i in range(len(recs))]
    base = [fc.hit_at_1(r["base_score"], rel[i]) for i, r in enumerate(recs)]
    tune, report = list(range(0, 80, 2)), list(range(1, 80, 2))
    a = r2.part_a(recs, rel, T, tune, report, base)
    for q in ("Q1", "Q2"):
        c = a["contrasts"][q]
        assert c["excluded_by_tuning"] == (c["lambda_X"] == 0.0)
        assert 0.0 <= c["vs_BT"]["p_holm"] <= 1.0
        if c["excluded_by_tuning"] and c["lambda_T"] == r2.LAMBDA_T_S20:
            assert c["vs_BT"]["delta_pp"] == 0.0 and c["vs_BT"]["fixed"] == c["vs_BT"]["broken"] == 0
    b = r2.part_b(recs, rel, T, tune, report, base)
    assert b["r1"]["reference_rule"] in r2.SCORE_RULES and b["u"]["status"] == "descriptive"
    assert set(b["r1"]["rank_vs_reference"]) == set(r2.RANK_RULES)
    assert b["r1"]["rules"]["rank_wsum"]["param"] == 0.0


def test_part_risk_pools_every_query_and_checks_clusters():
    recs = _recs(40)
    rel = [fc.relevant_set(r, False) for r in recs]
    T = [np.random.default_rng(i).uniform(0.1, 0.3, 10).tolist() for i in range(40)]
    base = [fc.hit_at_1(r["base_score"], rel[i]) for i, r in enumerate(recs)]
    cluster = [i // 2 for i in range(40)]
    pooled = {g: {"u": 0.0, "r1": 0.0, "rescues": 0, "breaks": 0} for g in ("FW", "control", "A")}
    out = r2.part_risk(recs, rel, T, list(range(20)), list(range(20, 40)), base, cluster,
                       {"pooled": pooled, "n_clusters": 20})
    assert out["utility_ci"]["diff"] == pytest.approx(out["utility"])
    assert out["rescues"] + out["breaks"] <= 40
    with pytest.raises(SystemExit):
        r2.part_risk(recs, rel, T, list(range(20)), list(range(20, 40)), base, cluster,
                     {"pooled": pooled, "n_clusters": 21})
