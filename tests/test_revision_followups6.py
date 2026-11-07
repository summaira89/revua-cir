"""revision_followups6.py and encode_towers.py (amendment §26): helper contracts."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import encode_towers as et  # noqa: E402
import fusion_controls as fc  # noqa: E402
import revision_followups6 as r6  # noqa: E402

RNG = np.random.default_rng(26)


def _sig(n=80):
    return {"cand": np.array([RNG.permutation(200)[:10] for _ in range(n)]),
            "pos": [int(RNG.integers(10)) if RNG.uniform() < 0.8 else None for _ in range(n)],
            "tune": list(range(0, n, 2)), "report": list(range(1, n, 2)),
            "B": [list(RNG.normal(size=10)) for _ in range(n)],
            "T_C": [list(RNG.normal(size=10)) for _ in range(n)],
            "T_O": [list(RNG.normal(size=10)) for _ in range(n)]}


def test_utility_interval_matches_hand_count():
    hits, ref = [1, 0, 1, 0, 1, 1], [0, 1, 1, 0, 0, 1]   # 2 rescues, 1 break over 6 queries
    u = r6.utility_interval(hits, ref, list(range(6)))
    assert u["diff"] == pytest.approx((2 - 2 * 1) / 6)


def test_tune_v_only_constant_verifier_is_excluded():
    s = _sig()
    V = [[0.3] * 10 for _ in s["pos"]]
    assert r6.tune_v_only(s["B"], s["T_O"], V, 2.0, s["pos"], s["tune"]) == fc.LAMBDA_GRID[0]


def test_part_posthoc_runs_and_aborts_on_lambda_drift(monkeypatch):
    monkeypatch.setattr(r6, "N_REP", 200)
    monkeypatch.setattr(r6.rf, "N_REP", 200)
    s = _sig()
    V = [list(RNG.normal(size=10)) for _ in s["pos"]]
    lam3, lam4 = r6.r5.tune_one(s["B"], s["T_O"], s["pos"], s["tune"]), r6.r5.tune_one(s["B"], V, s["pos"], s["tune"])
    lt5, lx5 = r6.r5.tune_three(s["B"], s["T_O"], V, s["pos"], s["tune"])
    s25c = {"lambda_tuned": {"A3": lam3, "A4": lam4, "A5": {"lambda_T": lt5, "lambda_V": lx5}}}
    cluster = list(range(len(s["pos"])))
    p = r6.part_posthoc(s, V, s25c, cluster)
    json.dumps(p, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    assert p["lambda"]["A5_fixedT"][0] == lam3
    assert set(p["report"]["utility_vs_A1"]) == {"A3", "A4", "A5", "A5_fixedT"}
    s25c["lambda_tuned"]["A4"] = 99.0
    with pytest.raises(SystemExit):
        r6.part_posthoc(s, V, s25c, cluster)


def test_part_encoders_runs_with_pairs(monkeypatch):
    monkeypatch.setattr(r6.rf, "N_REP", 200)
    s = _sig()
    enc = {"clip4cir": s["T_C"], "ours": s["T_O"], "pretrained": [list(RNG.normal(size=10)) for _ in s["pos"]]}
    e = r6.part_encoders(s, enc, r6.encoder_pairs(list(enc)))
    json.dumps(e, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    assert set(e["report"]["fused_contrasts"]) == {"ours_vs_pretrained", "pretrained_vs_clip4cir"}
    assert "text_pretrained" in e["report"]["arms"] and "fused_ours" in e["report"]["arms"]


def test_encoder_pairs_requires_named_encoders():
    assert ("ours", "fc_only") in r6.encoder_pairs(["clip4cir", "ours", "pretrained", "fc_only"])
    with pytest.raises(SystemExit):
        r6.encoder_pairs(["clip4cir", "ours"])


def _towers(tmp_path, name, source, gallery_sha="g", parity=None, n=4, m=6):
    d = tmp_path / name
    d.mkdir()
    np.savez(d / "towers.npz", image_emb=np.ones((m, 3), np.float32), text_emb=np.ones((n, 3), np.float32))
    res = {"schema": r6.TOWERS_SCHEMA, "output_name": "towers.npz",
           "output_files": {"towers.npz": r6.rf.sha256(str(d / "towers.npz"))}, "source": source,
           "gallery_sha256": gallery_sha, "parity_vs_phase_b": parity, "manifest": {"checkpoint_sha256": None}}
    (d / "results.json").write_text(json.dumps(res), encoding="utf-8")
    return "%s=%s" % (name, d)


def test_load_towers_provenance(tmp_path):
    ok = _towers(tmp_path, "pretrained", "pretrained")
    assert r6.load_towers(ok, 4, 6, "g")[0] == "pretrained"
    with pytest.raises(SystemExit):
        r6.load_towers(ok, 4, 6, "other")                                      # different gallery
    with pytest.raises(SystemExit):
        r6.load_towers(_towers(tmp_path, "fc_only", "pretrained"), 4, 6, "g")  # source mismatch
    with pytest.raises(SystemExit):
        r6.load_towers(_towers(tmp_path, "deployed", "checkpoint", parity={"image_emb": 1e-3, "text_emb": 0}), 4, 6, "g")
    good_dep = _towers(tmp_path, "deployed2", "checkpoint", parity={"image_emb": 1e-7, "text_emb": 1e-7})
    assert r6.load_towers(good_dep.replace("deployed2=", "deployed="), 4, 6, "g")[0] == "deployed"


def test_max_one_minus_cos():
    a = np.eye(3, dtype=np.float32)
    assert et.max_one_minus_cos(a, a) == pytest.approx(0.0)
    with pytest.raises(Exception):
        et.max_one_minus_cos(a, a[:2])
