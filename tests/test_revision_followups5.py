"""revision_followups5.py and cirr_external_windows.py (amendment §25): helper contracts."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import cirr_external_windows as cew  # noqa: E402
import fusion_controls as fc  # noqa: E402
import revision_followups2 as r2  # noqa: E402
import revision_followups3 as r3  # noqa: E402
import revision_followups5 as r5  # noqa: E402

RNG = np.random.default_rng(25)


def test_build_windows_matches_top_k_excluding_and_scores():
    img = RNG.normal(size=(30, 8)).astype(np.float32)
    comp = RNG.normal(size=(5, 8)).astype(np.float32)
    ref = [0, 3, 7, 7, 29]
    cand, base = r5.build_windows(img, comp, ref)
    imgn = img / np.linalg.norm(img, axis=1, keepdims=True)
    for i in range(5):
        s = imgn @ comp[i].astype(np.float64)
        assert cand[i].tolist() == r3.top_k_excluding(s, ref[i], r5.K)
        assert np.array_equal(base[i], s[cand[i]]) and ref[i] not in cand[i]


def test_window_cosines_bit_identical_to_section21():
    img = RNG.normal(size=(40, 6)).astype(np.float32)
    txt = RNG.normal(size=(4, 6)).astype(np.float32)
    cand = np.array([RNG.choice(40, 10, replace=False) for _ in range(4)])
    recs = [{"candidate_gallery_ids": c.tolist(), "target_gallery_index": int(c[0])} for c in cand]
    feats = {"image_emb": img, "text_emb": txt, "tgt_row": cand[:, 0], "ref_row": np.full(4, 99)}
    assert np.array_equal(r5.window_cosines(txt, img, cand), np.asarray(r2.text_cosines(recs, feats, 40)))


def test_target_positions():
    cand = np.array([[5, 6, 7], [1, 2, 3]])
    assert r5.target_positions(cand, [7, 9]) == [2, None]


def test_tune_one_prefers_smaller_lambda_on_ties():
    B = [[1.0, 0.0, -1.0]] * 4
    X = [[0.0, 0.0, 0.0]] * 4          # constant signal: every lambda gives the same hits
    assert r5.tune_one(B, X, [0, 0, 1, None], [0, 1, 2, 3]) == fc.LAMBDA_GRID[0]


def test_tune_three_with_useless_x_excludes_it():
    B = [list(RNG.normal(size=10)) for _ in range(60)]
    T = [list(RNG.normal(size=10)) for _ in range(60)]
    X = [[0.5] * 10 for _ in range(60)]  # constant: z(X) = 0, so lx cannot change any ranking
    pos = [int(np.argmax(np.array(t))) for t in T]
    lt, lx = r5.tune_three(B, T, X, pos, list(range(60)))
    assert lx == fc.LAMBDA_GRID[0]
    best_lt = r5.tune_one(B, T, pos, list(range(60)))
    assert lt == best_lt


def test_contrasts_holm_family():
    h = {"a": [1] * 30 + [0] * 70, "b": [0] * 100, "c": [1] * 31 + [0] * 69}
    c = r5.contrasts(h, (("X", "a", "b"), ("Y", "c", "a")))
    assert c["X"]["fixed"] == 30 and c["Y"]["fixed"] == 1
    assert c["X"]["p_holm"] >= c["X"]["p_mcnemar"] and c["X"]["holm_significant"]


# --------------------------------------------------------------------------- GPU-script helpers
def _recs(n=3, k=4):
    out = []
    for i in range(n):
        ids = [10 * i + j for j in range(k)]
        out.append({"query_id": i, "candidate_gallery_ids": ids, "base_score": [1.0 - 0.1 * j for j in range(k)],
                    **{f: [float(10 * i + j) for j in range(k)] for f in cew.FIELDS}})
    return out


def test_check_windows_rejects_reference_repeats_and_order():
    recs = _recs()
    cand = np.array([[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24]])
    ref = np.array([0, 10, 20])
    cew.check_windows(cand, np.arange(3), ref, recs, 4, 100)
    with pytest.raises(cew.WindowError):
        cew.check_windows(cand, np.array([0, 2, 1]), ref, recs, 4, 100)
    bad = cand.copy()
    bad[1, 0] = 10
    with pytest.raises(cew.WindowError):
        cew.check_windows(bad, np.arange(3), ref, recs, 4, 100)
    bad = cand.copy()
    bad[2, 1] = 21
    with pytest.raises(cew.WindowError):
        cew.check_windows(bad, np.arange(3), ref, recs, 4, 100)


def test_check_own_windows():
    recs = _recs()
    top = np.array([r["candidate_gallery_ids"] for r in recs])
    s = np.array([r["base_score"] for r in recs])
    assert cew.check_own_windows(top, s + 1e-5, recs) == pytest.approx(1e-5)
    with pytest.raises(cew.WindowError):
        cew.check_own_windows(top[:, ::-1], s, recs)
    with pytest.raises(cew.WindowError):
        cew.check_own_windows(top, s + 1e-2, recs)


def test_shared_pair_parity_maps_positions_and_aborts():
    recs = _recs()
    cand = np.array([[3, 99, 1, 98], [97, 96, 95, 10], [20, 21, 94, 93]])
    fields = {f: np.zeros(cand.shape, dtype=np.float32) for f in cew.FIELDS}
    for i, w in enumerate(cand):                     # copy the cache values at the matching positions
        for jc, g in enumerate(w):
            if g in recs[i]["candidate_gallery_ids"]:
                for f in cew.FIELDS:
                    fields[f][i, jc] = recs[i][f][recs[i]["candidate_gallery_ids"].index(g)]
    p = cew.shared_pair_parity(cand, fields, recs, 5)
    assert p["n_pairs"] == 5 and p["n_queries"] == 3 and max(p["worst_error"].values()) == 0.0
    with pytest.raises(cew.WindowError):
        cew.shared_pair_parity(cand, fields, recs, 6)
    fields["verification_score"][0, 0] += 1e-3
    with pytest.raises(cew.WindowError):
        cew.shared_pair_parity(cand, fields, recs, 5)


def _gpu_dir(tmp_path, cand, **over):
    import json
    v = RNG.uniform(size=cand.shape).astype(np.float32)
    np.savez(tmp_path / "w.npz", cand=cand, verification_score=v)
    res = {"schema": r5.GPU_SCHEMA, "n_queries": len(cand), "output_name": "w.npz",
           "output_files": {"w.npz": r5.rf.sha256(str(tmp_path / "w.npz"))}, "parity": {},
           "manifest": {"checkpoint_sha256": r5.GPU_CHECKPOINT_SHA, "config_sha256": r5.GPU_CONFIG_SHA}}
    for k, val in over.items():
        (res["manifest"] if k.endswith("sha256") else res)[k] = val
    (tmp_path / "results.json").write_text(json.dumps(res), encoding="utf-8")
    return str(tmp_path), v


def test_load_gpu_run_accepts_matching_run(tmp_path):
    cand = np.arange(20).reshape(2, 10)
    d, v = _gpu_dir(tmp_path, cand)
    arrays, _ = r5.load_gpu_run(d, cand)
    assert np.array_equal(arrays["verification_score"], v)


@pytest.mark.parametrize("over", [{"checkpoint_sha256": "0" * 64}, {"config_sha256": "0" * 64},
                                  {"n_queries": 1}, {"schema": "other/1"}])
def test_load_gpu_run_rejects_wrong_provenance(tmp_path, over):
    cand = np.arange(20).reshape(2, 10)
    d, _ = _gpu_dir(tmp_path, cand, **over)
    with pytest.raises(SystemExit):
        r5.load_gpu_run(d, cand)


def test_load_gpu_run_rejects_other_windows_and_tampering(tmp_path):
    cand = np.arange(20).reshape(2, 10)
    d, _ = _gpu_dir(tmp_path, cand)
    with pytest.raises(SystemExit):
        r5.load_gpu_run(d, cand[::-1])
    np.savez(tmp_path / "w.npz", cand=cand, verification_score=np.zeros(cand.shape, np.float32))
    with pytest.raises(SystemExit):
        r5.load_gpu_run(d, cand)


def _synthetic_sig(n=80):
    cand = np.array([RNG.permutation(200)[:10] for _ in range(n)])
    pos = [int(RNG.integers(10)) if RNG.uniform() < 0.8 else None for _ in range(n)]
    tune, report = list(range(0, n, 2)), list(range(1, n, 2))
    return {"cand": cand, "pos": pos, "tune": tune, "report": report,
            "B": [list(RNG.normal(size=10)) for _ in range(n)],
            "T_C": [list(RNG.normal(size=10)) for _ in range(n)],
            "T_O": [list(RNG.normal(size=10)) for _ in range(n)]}


def test_parts_run_end_to_end_and_serialise(monkeypatch):
    import json
    monkeypatch.setattr(r5.rf, "N_REP", 200)
    sig = _synthetic_sig()
    a = r5.part_windows(sig)
    json.dumps(a, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    assert set(a["report"]["contrasts"]) == {"C1", "C2", "C3"} and a["report"]["arms"]["A1"]["vs_A1"] is None
    # a constant verifier score cannot change any ranking: tuning must exclude it and C5 must be 0/0
    V = [[0.7] * 10 for _ in sig["pos"]]
    v = r5.part_verifier(sig, V)
    json.dumps(v, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    assert v["A5_verifier_excluded_by_tuning"] is True
    for half in ("report", "full"):
        c5 = v[half]["contrasts"]["C5"]
        assert (c5["fixed"], c5["broken"]) == (0, 0)
    assert v["lambda_tuned"]["A5"]["lambda_T"] == v["lambda_tuned"]["A3"]
