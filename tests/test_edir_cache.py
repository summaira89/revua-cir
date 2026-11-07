"""edir_cache.py end to end on CPU with the tiny test model and a synthetic EDIR."""
import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
import edir_cache as ec  # noqa: E402
from test_smoke import build_tiny_config  # noqa: E402

N_Q, PER_SOURCE = 12, 3                      # 12 queries x 3 edits = 36 corpus images


def _synthetic_edir(root):
    os.makedirs(os.path.join(root, "imgs_mc"))
    rng = np.random.default_rng(0)
    q, c, inst = [], [], []
    for i in range(N_Q):
        src = "%d_src" % i
        Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(os.path.join(root, "imgs_mc", src + ".jpg"))
        edits = ["%s_%d.png" % (src, k) for k in range(1, PER_SOURCE + 1)]
        for e in edits:
            Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(os.path.join(root, "imgs_mc", e))
        q.append({"id": str(i), "image": src + ".jpg", "text": "make it red", "metadata": {"subcategory": "color"}})
        c += edits
        inst.append({"qid": str(i), "pos": [edits[1]], "neg": [edits[0], edits[2]]})
    for name, rows in (("queries", q), ("corpus", [{"id": x, "image": x} for x in c]), ("instances", inst)):
        with open(os.path.join(root, name + ".jsonl"), "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    import torch
    from revua_cir.models.revua_cir import ReVUACIR
    from revua_cir.utils import save_checkpoint
    edir = str(tmp_path / "edir")
    _synthetic_edir(edir)
    cfg = build_tiny_config()
    cfg.reverse_verifier.top_k = 10
    cfg_path = str(tmp_path / "tiny.yaml")
    cfg.to_yaml(cfg_path)
    torch.manual_seed(0)
    ckpt = str(tmp_path / "tiny.pt")
    save_checkpoint(ckpt, ReVUACIR(cfg.apply_module_ablation("fc+aep+rv")), epoch=0, best_metric=0.0)
    monkeypatch.setattr(ec, "EXPECTED", {"queries": N_Q, "corpus": N_Q * PER_SOURCE})
    return {"edir": edir, "cfg": cfg_path, "ckpt": ckpt, "out": str(tmp_path / "edir_cache.jsonl")}


def _args(s):
    return ["--config", s["cfg"], "--checkpoint", s["ckpt"], "--module-combo", "fc+aep+rv",
            "--edir-root", s["edir"], "--meta", s["edir"], "--out", s["out"], "--eval-batch-size", "5"]


def test_end_to_end(setup):
    assert ec.main(_args(setup)) == 0
    recs = [json.loads(line) for line in open(setup["out"])]
    assert [r["query_id"] for r in recs] == list(range(N_Q))
    assert all(len(r["candidate_gallery_ids"]) == 10 for r in recs)
    for i, r in enumerate(recs):               # target = the query's annotated positive, by corpus row
        assert r["target_gallery_index"] == i * PER_SOURCE + 1
    m = json.load(open(setup["out"][:-6] + ".manifest.json"))
    assert m["protocol"] == "edir_retrieval" and m["complete"] and m["n_queries"] == N_Q
    assert m["gallery_size"] == N_Q * PER_SOURCE and m["eval_batch_size"] == 5
    assert set(m["edir_meta_sha256"]) == set(ec.META_FILES) and len(m["corpus_ids_sha256"]) == 64
    with pytest.raises(SystemExit):            # a second run must refuse to overwrite
        ec.main(_args(setup))


def test_refuses_existing_output(setup):
    open(setup["out"], "w").close()
    with pytest.raises(SystemExit):
        ec.main(_args(setup))


def test_check_cache_catches_wrong_target(setup):
    from revua_cir.data.datasets import EDIRDataset
    assert ec.main(_args(setup)) == 0
    ds = EDIRDataset(root=setup["edir"], split="test", transform=None)
    recs = [json.loads(line) for line in open(setup["out"])]
    recs[3]["target_gallery_index"] += 1
    bad = setup["out"] + ".bad.jsonl"
    with open(bad, "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in recs)
    with pytest.raises(RuntimeError):
        ec.check_cache(bad, ds, ds.corpus_paths)


def test_vector_dump_reproduces_edir_cache(setup, tmp_path):
    import dump_verifier_vectors as dv
    assert ec.main(_args(setup)) == 0
    out = str(tmp_path / "vec_edir.npz")
    common = ["--config", setup["cfg"], "--checkpoint", setup["ckpt"], "--module-combo", "fc+aep+rv",
              "--dataset", "edir", "--edir-root", setup["edir"], "--edir-meta", setup["edir"],
              "--cache", setup["out"], "--batch-size", "4"]
    assert dv.main(common + ["--out", out]) == 0              # nothing is written unless it reproduces
    man = json.load(open(out[:-4] + ".manifest.json"))
    assert man["dataset"] == "edir" and man["split"] == "test" and man["category"] is None
    assert man["n_queries"] == N_Q and man["gallery_size"] == N_Q * PER_SOURCE
    assert all(man["worst_reproduction_error"][f] <= dv.VECTOR_TOL[f] for f in dv.VECTOR_TOL)
    with pytest.raises(SystemExit):                           # --category is refused for edir
        dv.main(common + ["--out", str(tmp_path / "v2.npz"), "--category", "dress"])
    with pytest.raises(SystemExit):                           # edir needs its root and metadata
        dv.main([a for a in common if a not in ("--edir-root", "--edir-meta", setup["edir"])]
                + ["--out", str(tmp_path / "v3.npz")])


def test_split_of_keeps_existing_datasets_on_val():
    import dump_verifier_vectors as dv
    assert dv.split_of("cirr") == "val" and dv.split_of("fashioniq") == "val" and dv.split_of("edir") == "test"


def test_feature_extraction_edir(setup, tmp_path):
    """cache -> vectors -> features for edir_test: separate, namespaced reference
    block, no ref_row, and the features reproduce the dump (checked inside)."""
    import dump_verifier_vectors as dv
    import bbnv_extract as ext
    assert ec.main(_args(setup)) == 0
    vec = str(tmp_path / "vec_edir.npz")
    assert dv.main(["--config", setup["cfg"], "--checkpoint", setup["ckpt"], "--module-combo", "fc+aep+rv",
                    "--dataset", "edir", "--edir-root", setup["edir"], "--edir-meta", setup["edir"],
                    "--cache", setup["out"], "--out", vec, "--batch-size", "4"]) == 0
    out = str(tmp_path / "feat_edir")
    assert ext.main(["--config", setup["cfg"], "--checkpoint", setup["ckpt"], "--module-combo", "fc+aep+rv",
                     "--target", "edir_test", "--edir-root", setup["edir"], "--edir-meta", setup["edir"],
                     "--vectors", vec, "--out-dir", out, "--batch-size", "8", "--workers", "0",
                     "--device", "cpu"]) == 0
    z = np.load(os.path.join(out, ext.FEATURES_NAME))
    assert "ref_row" not in z.files
    assert z["image_emb"].shape[0] == N_Q * PER_SOURCE and z["ref_emb"].shape[0] == N_Q
    assert all(i.startswith("corpus:") for i in z["image_ids"]) and all(r.startswith("ref:") for r in z["ref_ids"])
    assert list(z["ref_ids"]) == sorted(z["ref_ids"])
    for q in range(N_Q):                         # each query's reference and target, by identity
        assert z["ref_ids"][z["ref_index"][q]] == "ref:%d_src.jpg" % q
        assert z["image_ids"][z["tgt_row"][q]] == "corpus:%d_src_2.png" % q
    assert list(z["ref_paths"])[0].startswith("imgs_mc/") and list(z["image_paths"])[0].startswith("imgs_mc/")
    res = json.load(open(os.path.join(out, "results.json")))
    assert res["target"] == "edir_test" and res["n_refs"] == N_Q


def _annotated_args(setup, tmp_path, out_name="ann.jsonl", ckpt=None):
    return ["--config", setup["cfg"], "--checkpoint", ckpt or setup["ckpt"], "--module-combo", "fc+aep+rv",
            "--edir-root", setup["edir"], "--meta", setup["edir"], "--retrieval-cache", setup["out"],
            "--out", str(tmp_path / out_name), "--query-batch", "5"]


def test_annotated_cache(setup, tmp_path, monkeypatch):
    import edir_annotated_cache as ea
    monkeypatch.setattr(ea, "EXPECTED", ec.EXPECTED)      # the fixture's synthetic counts, restored after
    assert ec.main(_args(setup)) == 0
    assert ea.main(_annotated_args(setup, tmp_path)) == 0
    recs = [json.loads(line) for line in open(str(tmp_path / "ann.jsonl"))]
    assert len(recs) == N_Q and all(r["protocol"] == "edir_annotated" for r in recs)
    for i, r in enumerate(recs):
        ids = r["candidate_gallery_ids"]
        assert ids == sorted(ids) == [i * PER_SOURCE + k for k in range(PER_SOURCE)]   # its own 3 edits
        assert ids[r["target_pos"]] == r["target_gallery_index"] == i * PER_SOURCE + 1
        assert all(len(r[f]) == PER_SOURCE for f in ea.FIELDS)
    m = json.load(open(str(tmp_path / "ann.manifest.json")))
    assert m["cross_check"]["shared_pairs"] > 0 and not m["cross_check"]["violations"]


def test_annotated_cache_with_relative_root(setup, tmp_path, monkeypatch):
    """The real command passes ../edir_data: candidate paths must match the
    gallery's absolute paths whatever form the root was given in."""
    import edir_annotated_cache as ea
    monkeypatch.setattr(ea, "EXPECTED", ec.EXPECTED)
    assert ec.main(_args(setup)) == 0
    monkeypatch.chdir(str(tmp_path))
    args = _annotated_args(setup, tmp_path)
    args[args.index("--edir-root") + 1] = os.path.relpath(setup["edir"]).replace(os.sep, "/")
    assert ea.main(args) == 0
    assert len(open(str(tmp_path / "ann.jsonl")).readlines()) == N_Q


def test_annotated_cross_check_catches_disagreement():
    import edir_annotated_cache as ea
    a = [{"candidate_gallery_ids": [1, 2], **{f: [0.5, 0.5] for f in ea.FIELDS}}]
    r = [{"candidate_gallery_ids": [9, 2, 7], **{f: [0.0, 0.5, 0.0] for f in ea.FIELDS}}]
    ok = ea.cross_check(a, r)
    assert ok["shared_pairs"] == 1 and not ok["violations"]
    r[0]["edit_cos"][1] = 0.5 + 2e-4                 # beyond 1e-4
    assert "edit_cos" in ea.cross_check(a, r)["violations"]


def test_interrupted_cache_leaves_no_final_file(setup, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated failure after evaluation")
    monkeypatch.setattr(ec, "check_cache", boom)
    with pytest.raises(RuntimeError):
        ec.main(_args(setup))
    assert not os.path.exists(setup["out"]) and os.path.exists(setup["out"] + ".partial")
    monkeypatch.undo()
    monkeypatch.setattr(ec, "EXPECTED", {"queries": N_Q, "corpus": N_Q * PER_SOURCE})
    assert ec.main(_args(setup)) == 0                  # a rerun clears the partial and completes
    assert os.path.exists(setup["out"]) and not os.path.exists(setup["out"] + ".partial")


def test_annotated_refuses_other_config_before_gpu(setup, tmp_path, monkeypatch):
    import edir_annotated_cache as ea
    monkeypatch.setattr(ea, "EXPECTED", ec.EXPECTED)
    assert ec.main(_args(setup)) == 0
    other = str(tmp_path / "other.yaml")
    open(other, "w").write(open(setup["cfg"]).read() + "\n# different file\n")
    monkeypatch.setattr(ea, "ReVUACIR", None, raising=False)    # never reached
    args = _annotated_args(setup, tmp_path)
    args[args.index("--config") + 1] = other
    with pytest.raises(SystemExit):
        ea.main(args)
    assert not os.path.exists(str(tmp_path / "ann.jsonl"))


def test_edir_paths_are_absolute_whatever_the_root_form(setup, monkeypatch):
    from revua_cir.data.datasets import EDIRDataset
    a = EDIRDataset(root=setup["edir"], split="test", transform=None)
    monkeypatch.chdir(os.path.dirname(setup["edir"]))
    b = EDIRDataset(root=os.path.basename(setup["edir"]), split="test", transform=None)
    assert a.corpus_paths == b.corpus_paths and os.path.isabs(b.corpus_paths[0]) and a.root == b.root
