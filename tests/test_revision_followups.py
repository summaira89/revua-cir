"""revision_followups.py: the helpers behind amendment §20, on synthetic inputs only."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import fusion_controls as fc  # noqa: E402
import revision_followups as rf  # noqa: E402


def test_holm_adjusted_matches_step_down():
    adj = rf.holm_adjusted([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])          # 3*0.01; max(2*0.03, 0.03)=0.06; max(1*0.04, 0.06)
    assert rf.holm_adjusted([0.9, 0.8]) == [1.0, 1.0]         # capped at 1
    assert rf.holm_adjusted([0.02]) == [0.02]


def test_holm_agrees_with_project_boolean_version():
    from revua_cir.significance import holm_bonferroni
    rng = np.random.default_rng(1)
    for _ in range(200):
        ps = rng.uniform(0, 0.1, rng.integers(1, 6)).tolist()
        assert [p <= 0.05 for p in rf.holm_adjusted(ps)] == holm_bonferroni(ps, 0.05)


def test_paired_counts_and_direction():
    new, ref = [1, 1, 0, 0, 1], [0, 1, 1, 0, 0]
    d = rf.paired(new, ref)
    assert (d["fixed"], d["broken"]) == (2, 1)
    assert d["delta_pp"] == pytest.approx(20.0)
    assert d["ci_low"] <= d["delta_pp"] <= d["ci_high"]
    with pytest.raises(ValueError):
        rf.paired([1], [1, 0])


def test_outcome_utility_weights_breaks_twice():
    o = rf.outcome([1, 0, 1, 1], [0, 1, 1, 0])
    assert (o["rescues"], o["breaks"]) == (2, 1)
    assert o["utility"] == pytest.approx((2 - 2) / 4)
    assert o["r1"] == pytest.approx(75.0)


def test_top1_ties_to_lower_index_like_hit_at_1():
    s = [0.5, 0.9, 0.9, 0.1]
    assert rf.top1(s) == 1
    assert fc.hit_at_1(s, {1}) == 1 and fc.hit_at_1(s, {2}) == 0


def test_tr_scores_endpoints_are_the_single_signals():
    T, R = [[0.1, 0.5, 0.3]], [[0.9, 0.2, 0.4]]
    assert rf.tr_scores(T, R, 1.0)[0] == pytest.approx(fc._zscore(T[0]))
    assert rf.tr_scores(T, R, 0.0)[0] == pytest.approx(fc._zscore(R[0]))


def test_tune_prefers_smaller_lambda_on_ties():
    # one query whose fused top-1 never changes: every lambda ties, so the smallest wins
    recs = [{"base_score": [0.9, 0.1]}]
    assert rf.tune(recs, [{0}], [[0.9, 0.1]], [0], (0.0, 1.0, 2.0)) == 0.0


def test_x6_counts_negatives(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"candidate_gallery_ids": [1,2,3,4,5]}\n{"candidate_gallery_ids": [1,2,3,4,5,6]}\n'
                 '{"candidate_gallery_ids": [7,8,9,10,11]}\n', encoding="utf-8")
    assert rf.x6(str(p)) == {"negatives_per_query": {4: 2, 5: 1}, "queries": 3}


def test_training_free_signals_rejects_misaligned_features():
    recs = [{"target_gallery_index": 0, "candidate_gallery_ids": [0, 1]}]
    feats = {"image_emb": np.eye(2297, 4, dtype=np.float32) + 1e-3, "text_emb": np.ones((1, 4), np.float32),
             "ref_row": np.array([5]), "tgt_row": np.array([1])}
    with pytest.raises(SystemExit):
        rf.training_free_signals(recs, feats)
    feats["tgt_row"], feats["ref_row"] = np.array([0]), np.array([1])      # reference inside the window
    with pytest.raises(SystemExit):
        rf.training_free_signals(recs, feats)
    feats["ref_row"] = np.array([5])
    sig = rf.training_free_signals(recs, feats)
    assert len(sig["T"][0]) == 2 and len(sig["R"][0]) == 2


def _row(k, full, base, ov):
    return {"k": k, "full": {"p50": full}, "base": {"p50": base}, "overhead_paired": {"p50": ov}}


def test_latency_table_never_mixes_sessions():
    lat = {"rows": [_row(3, 60, 20, 40), _row(5, 85, 20, 65), _row(7, 110, 20, 90), _row(10, 160, 20, 140)]}
    lat12 = {"rows": [_row(1, 38, 18, 20), _row(2, 50, 18, 32), _row(10, 168, 18, 150)]}
    t, anchor = rf.latency_table(lat, lat12)
    assert sorted(t) == [1, 2, 3, 5, 7, 10]
    assert t[5]["overhead_share_of_k10"] == pytest.approx(65 / 140) and t[5]["session"] == "main"
    assert t[1]["overhead_share_of_k10"] == pytest.approx(20 / 150) and t[1]["session"] == "second"
    assert t[10]["total_p50_ms"] == 160                       # the main session's k=10, not the anchor
    assert anchor["second_vs_main"] == pytest.approx(168 / 160 - 1)
    with pytest.raises(SystemExit):
        rf.latency_table({"rows": lat["rows"][:3]}, lat12)
