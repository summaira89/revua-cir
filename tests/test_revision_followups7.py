"""revision_followups7.py and circo_towers.py (amendment §27): helper contracts."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import circo_bounded_map as cbm  # noqa: E402
import circo_towers as ct  # noqa: E402
import revision_followups7 as r7  # noqa: E402

RNG = np.random.default_rng(27)


def _rec(i, cand, top50_tail, gt):
    return {"query_id": i, "candidate_gallery_ids": cand, "base_top50_gallery_ids": cand + top50_tail,
            "relevant_gallery_indices": gt, "base_score": list(np.linspace(1, 0, len(cand))),
            "verification_score": list(RNG.uniform(size=len(cand)))}


def test_rerank_top50_reorders_window_keeps_tail_and_ties_low():
    r = _rec(0, [5, 6, 7], [8, 9], [7])
    assert r7.rerank_top50(r, [0.0, 1.0, 1.0]) == [6, 7, 5, 8, 9]
    with pytest.raises(SystemExit):
        r7.rerank_top50(r, [0.0, 1.0])
    bad = dict(r, base_top50_gallery_ids=[6, 5, 7, 8, 9])
    with pytest.raises(SystemExit):
        r7.rerank_top50(bad, [1, 0, 0])


def test_base_scores_reproduce_base_ranking_map():
    recs = [_rec(i, [10 * i + j for j in range(10)], [10 * i + 10 + j for j in range(40)],
                 [10 * i + 3, 10 * i + 20]) for i in range(5)]
    for r in recs:
        base = cbm.average_precision_at_k(r["base_top50_gallery_ids"], r["relevant_gallery_indices"], 5)
        assert cbm.average_precision_at_k(r7.rerank_top50(r, r["base_score"]), r["relevant_gallery_indices"], 5) == base


def test_circo_cosines_maps_rows():
    rows = np.array([3, 7, 11, 20], dtype=np.int64)
    img = RNG.normal(size=(4, 5)).astype(np.float32)
    txt = RNG.normal(size=(2, 5)).astype(np.float32)
    recs = [{"candidate_gallery_ids": [11, 3]}, {"candidate_gallery_ids": [20, 7, 3]}]
    T = r7.circo_cosines({"rows": rows, "image_emb": img, "text_emb": txt}, recs)
    u = lambda a: a / np.linalg.norm(a)  # noqa: E731
    assert T[0][0] == pytest.approx(float(u(img[2].astype(float)) @ u(txt[0].astype(float))))
    assert T[1][1] == pytest.approx(float(u(img[1].astype(float)) @ u(txt[1].astype(float))))


def test_evaluate_runs_and_constant_signals_change_nothing(monkeypatch):
    recs = [_rec(i, [100 * i + j for j in range(10)], [100 * i + 10 + j for j in range(40)],
                 [100 * i + int(RNG.integers(10)), 100 * i + 15]) for i in range(30)]
    B = [r["base_score"] for r in recs]
    const = [[0.5] * 10 for _ in recs]
    T = {"pretrained": const, "fc_only": const, "ours": const}
    lam = {"T_pretrained": 4.0, "T_fc_only": 4.0, "T_ours": 4.0, "V": 1.0, "T_ours+V": [4.0, 1.0]}
    out = r7.evaluate(recs, r7.arm_scores(B, T, const, lam), 200)
    json.dumps(out, default=str)
    for a, m in out["maps"].items():
        assert m == out["maps"]["base"], a          # constant second signals cannot reorder
    assert out["primary"]["P1_T_fc_only_vs_base"][5]["delta"] == 0.0
    assert "mAP@5_p_holm" in out["primary"]["P2_T_ours+V_vs_T_ours"]


def test_check_alignment_and_base_parity():
    gallery = ["/x/%012d.jpg" % i for i in (1, 2, 3, 4, 5, 9)]
    ann = [{"id": 0, "gt_img_ids": [3, 4], "reference_img_id": 9}]
    recs = [{"query_id": 0, "relevant_gallery_indices": [2, 3], "candidate_gallery_ids": [0, 2],
             "base_top50_gallery_ids": [0, 2, 1], "base_score": [0.0, 0.0]}]
    assert ct.check_alignment(ann, recs, gallery) == [0, 2]
    with pytest.raises(ct.WindowError):
        ct.check_alignment([dict(ann[0], gt_img_ids=[3, 5])], recs, gallery)
    with pytest.raises(ct.WindowError):
        ct.check_alignment([dict(ann[0], reference_img_id=1)], recs, gallery)
    img = np.array([[1.0, 0.0], [0.0, 1.0]])
    recs[0]["base_score"] = [0.6, 0.8]
    assert ct.base_parity(np.array([[0.6, 0.8]]), [0, 2], img, recs) == pytest.approx(0.0)
    with pytest.raises(ct.WindowError):
        ct.base_parity(np.array([[0.8, 0.6]]), [0, 2], img, recs)


def _ctowers(tmp_path, name, source, cache_sha, rows, n_q=2, parity=None, ckpt=None):
    d = tmp_path / name
    d.mkdir()
    np.savez(d / "t.npz", rows=np.asarray(rows, np.int64), query_id=np.arange(n_q),
             image_emb=np.ones((len(rows), 3), np.float32), text_emb=np.ones((n_q, 3), np.float32))
    res = {"schema": r7.CIRCO_SCHEMA, "output_name": "t.npz", "output_files": {"t.npz": r7.rf.sha256(str(d / "t.npz"))},
           "source": source, "parity_base": parity, "manifest": {"cache_sha256": cache_sha, "checkpoint_sha256": ckpt}}
    (d / "results.json").write_text(json.dumps(res), encoding="utf-8")
    return "%s=%s" % (name, d)


def test_load_circo_towers_provenance(tmp_path):
    rows = [1, 2, 3]
    assert r7.load_circo_towers(_ctowers(tmp_path, "pretrained", "pretrained", "c", rows), "c", rows, 2)[0] == "pretrained"
    with pytest.raises(SystemExit):
        r7.load_circo_towers(_ctowers(tmp_path, "fc_only", "checkpoint", "other", rows), "c", rows, 2)
    with pytest.raises(SystemExit):
        r7.load_circo_towers(_ctowers(tmp_path, "deployed", "checkpoint", "c", rows), "c", rows, 2)   # no parity
    dep = r7.CHECKPOINT_SHA["deployed"]
    ok = _ctowers(tmp_path, "dep2", "checkpoint", "c", rows, parity={"base_score_worst": 1e-6}, ckpt=dep)
    assert r7.load_circo_towers(ok.replace("dep2=", "deployed="), "c", rows, 2)[0] == "deployed"
    wrong = _ctowers(tmp_path, "dep3", "checkpoint", "c", rows, parity={"base_score_worst": 1e-6},
                     ckpt=r7.CHECKPOINT_SHA["fc_only"])
    with pytest.raises(SystemExit):
        r7.load_circo_towers(wrong.replace("dep3=", "deployed="), "c", rows, 2)          # wrong checkpoint
    with pytest.raises(SystemExit):
        r7.load_circo_towers(_ctowers(tmp_path, "pretrained2", "pretrained", "c", [1, 2]).replace(
            "pretrained2=", "pretrained="), "c", rows, 2)
