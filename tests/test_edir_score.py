"""bbnv_score.py EDIR branches with the REAL frozen banks and heads on synthetic
768-d EDIR features: identity-based rows equal the positional formulation
exactly, both protocols run end to end, and a broken hash chain is refused."""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import bbnv_extract as ext  # noqa: E402
import bbnv_provenance as prov  # noqa: E402
import bbnv_score as sc  # noqa: E402
import bbnv_train as trn  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TF, MD, HD = (os.path.join(ROOT, p) for p in ("bbnv_feats/v1/cirr_train", "bbnv_runs/manifest_groups_cluster",
                                              "bbnv_heads/v1"))
pytestmark = pytest.mark.skipif(not all(os.path.isdir(p) for p in (TF, MD, HD)), reason="frozen BBNV inputs absent")
N_C, N_Q = 40, 3
FIELDS = ("base_score", "cos_sim", "recon_cos", "invariant_cos", "edit_cos", "verification_score")


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _unit(rng, n):
    x = rng.standard_normal((n, 768)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _jsonl(path, rows):
    with open(path, "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("edir_chain")
    rng = np.random.default_rng(7)
    ckpt = json.load(open(os.path.join(TF, "results.json")))["manifest"]["checkpoint_sha256"]
    tgt = np.array([3, 17, 30])
    ref_index = np.array([2, 0, 1])                      # deliberately not query order
    feat = {"image_emb": _unit(rng, N_C), "text_emb": _unit(rng, N_Q), "ref_emb": _unit(rng, N_Q),
            "image_ids": np.array(["corpus:%d_s_%d.png" % (i // 5, i % 5) for i in range(N_C)]),
            "ref_ids": np.array(["ref:%d_s.jpg" % i for i in range(N_Q)]), "ref_index": ref_index,
            "tgt_row": tgt, "texts": np.array(["t"] * N_Q),
            "image_paths": np.array(["imgs_mc/x%d.png" % i for i in range(N_C)]),
            "ref_paths": np.array(["imgs_mc/r%d.jpg" % i for i in range(N_Q)])}
    vf = str(tmp / "feat")
    prov.prepare_fresh_dir(vf)
    ext.publish_npz_run(vf, feat, {"schema": ext.SCHEMA, "target": "edir_test", "category": None,
                                   "vectors_sha256": "V" * 64, "gallery_sha256": "G" * 64,
                                   "manifest": {"checkpoint_sha256": ckpt}})
    windows = [list(range(0, 10)), list(range(10, 20)), list(range(20, 30))]   # q2's target (30) not retrieved
    cache = str(tmp / "edir_cache.jsonl")
    _jsonl(cache, [{"query_id": q, "target_gallery_index": int(tgt[q]), "candidate_gallery_ids": w,
                    "target_pos": w.index(int(tgt[q])) if int(tgt[q]) in w else -1,
                    **{f: rng.uniform(0.2, 0.8, 10).tolist() for f in FIELDS}} for q, w in enumerate(windows)])
    vman = str(tmp / "vec_edir.manifest.json")
    json.dump({"checkpoint_sha256": ckpt, "npz_sha256": "V" * 64, "cache_sha256": _sha(cache)}, open(vman, "w"))
    sets = [[2, 3, 4, 5, 6, 7], [15, 16, 17, 18, 19, 14], [30, 31, 32, 33, 34]]
    ann = str(tmp / "edir_annotated.jsonl")
    _jsonl(ann, [{"protocol": "edir_annotated", "query_id": q, "target_gallery_index": int(tgt[q]),
                  "candidate_gallery_ids": sorted(s), "target_pos": sorted(s).index(int(tgt[q])),
                  **{f: rng.uniform(0.2, 0.8, len(s)).tolist() for f in FIELDS}} for q, s in enumerate(sets)])
    aman = {"protocol": "edir_annotated", "cache_sha256": _sha(ann), "checkpoint_sha256": ckpt,
            "retrieval_cache_sha256": _sha(cache), "gallery_sha256": "G" * 64,
            "cross_check": {"shared_pairs": 9, "violations": {}}}
    json.dump(aman, open(ann[:-6] + ".manifest.json", "w"))
    return {"tmp": tmp, "vf": vf, "cache": cache, "vman": vman, "ann": ann, "feat": feat, "windows": windows}


def _common(c):
    return ["--train-features", TF, "--manifest-dir", MD, "--heads-dir", HD, "--val-features", c["vf"],
            "--vectors-manifest", c["vman"], "--threads", "4"]


def test_retrieval_windows_and_identity_equivalence(chain):
    out = str(chain["tmp"] / "win")
    assert sc.main(["windows", "--cache", chain["cache"], "--out-dir", out] + _common(chain)) == 0
    res = prov.verify_run(out)
    assert res["protocol"] == "edir_retrieval" and res["target"] == "edir_test"
    rows = [json.loads(line) for line in open(os.path.join(out, "windows.jsonl"))]
    assert [r["window"] for r in rows] == chain["windows"] and [r["target_pos"] for r in rows] == [3, 7, -1]
    # the positional formulation: table [B_R | B_C | corpus | refs], r at 2n + N_C + ref_index
    tr = trn.load_inputs(TF, MD)
    heads = sc.load_heads(HD, TF)
    banks = sc.make_banks(tr["feat"]["image_ids"], tr["feat"]["partition"])
    f, nb = chain["feat"], sc.BANK_SIZE
    emb = np.concatenate([tr["feat"]["image_emb"][banks["B_R"]], tr["feat"]["image_emb"][banks["B_C"]],
                          f["image_emb"], f["ref_emb"]])
    for q in range(N_Q):
        want = sc.score_query(heads, emb, f["text_emb"][q].astype(np.float64), 2 * nb + N_C + int(f["ref_index"][q]),
                              np.arange(nb), np.arange(nb, 2 * nb), 2 * nb + np.asarray(chain["windows"][q]))
        assert rows[q]["heads"] == json.loads(json.dumps(want))          # bit-identical


def test_annotated_protocol(chain):
    out = str(chain["tmp"] / "ann_out")
    assert sc.main(["annotated", "--annotated-cache", chain["ann"], "--out-dir", out] + _common(chain)) == 0
    res = prov.verify_run(out)
    assert res["protocol"] == "edir_annotated" and res["window_sizes"] == {"5": 1, "6": 2}
    rows = [json.loads(line) for line in open(os.path.join(out, "windows.jsonl"))]
    assert [len(r["heads"]["F0"]["raw"]) for r in rows] == [6, 6, 5]
    assert all(r["window"][r["target_pos"]] == int(chain["feat"]["tgt_row"][i]) for i, r in enumerate(rows))


def test_broken_chain_refused(chain, tmp_path):
    bad = str(tmp_path / "ann.jsonl")
    open(bad, "w").write(open(chain["ann"]).read())
    m = json.load(open(chain["ann"][:-6] + ".manifest.json"))
    m.update(cache_sha256=_sha(bad), retrieval_cache_sha256="0" * 64)        # a different retrieval cache
    json.dump(m, open(bad[:-6] + ".manifest.json", "w"))
    ckpt = m["checkpoint_sha256"]
    with pytest.raises(sc.ScoreError):
        sc.load_annotated(chain["vf"], bad, chain["vman"], ckpt)


def test_rows_refuse_bank_overlap(chain, monkeypatch):
    row_of = {"ref:0_s.jpg": 1, "corpus:0_s_0.png": 2048 + 0}
    vf = {"ref_ids": np.array(["ref:0_s.jpg"]), "ref_index": np.array([0]),
          "image_ids": np.array(["corpus:0_s_0.png"])}
    with pytest.raises(sc.ScoreError):                     # a reference row inside B_R
        sc.edir_rows(row_of, vf, 0, [0])


def test_edir_analysis_end_to_end(chain, monkeypatch):
    import edir_analysis as ea
    monkeypatch.setattr(ea, "EXPECTED_Q", N_Q)
    monkeypatch.setattr(ea, "EXPECTED_SOURCES", N_Q)
    win, ann, out = (str(chain["tmp"] / n) for n in ("e2e_win", "e2e_ann", "e2e_analysis"))
    assert sc.main(["windows", "--cache", chain["cache"], "--out-dir", win] + _common(chain)) == 0
    assert sc.main(["annotated", "--annotated-cache", chain["ann"], "--out-dir", ann] + _common(chain)) == 0
    analysis = os.path.join(ROOT, "bbnv_runs", "analysis_cirr_v1")
    assert ea.main(["--annotated", ann, "--retrieval", win, "--features", chain["vf"],
                    "--analysis", analysis, "--out-dir", out]) == 0
    res = prov.verify_run(out)
    steps = res["track_A"]["steps"]
    assert set(steps) == {"Q1a", "Q1b", "Q2"} and all(isinstance(steps[k]["status"], str) for k in steps)
    assert res["track_C"]["primary"]["ci_level"] == 0.975 and res["bootstrap"]["n_groups"] == N_Q
    with pytest.raises(ea.EdirAnalysisError):            # protocols swapped are refused
        ea.load_inputs(win, ann, chain["vf"], analysis)
