"""EDIRDataset: identities, targets, annotated sets, and every refusal case."""
import json
import os

import pytest

from revua_cir.data.datasets import DATASET_REGISTRY, EDIRDataset


def _edir(tmp, queries=None, corpus=None, instances=None, skip_file=None):
    root = str(tmp)
    os.makedirs(os.path.join(root, "imgs_mc"), exist_ok=True)
    queries = queries or [{"id": "0", "image": "1_a.jpg", "text": "make it red", "metadata": {"subcategory": "color"}},
                          {"id": "1", "image": "2_b.jpg", "text": "add a hat", "metadata": {"subcategory": "add"}}]
    corpus = corpus or ["1_a_%d.png" % k for k in range(1, 4)] + ["2_b_%d.png" % k for k in range(1, 4)]
    instances = instances or [{"qid": "0", "pos": ["1_a_2.png"], "neg": ["1_a_1.png", "1_a_3.png"]},
                              {"qid": "1", "pos": ["2_b_1.png"], "neg": ["2_b_2.png", "2_b_3.png"]}]
    for name, rows in (("queries", queries), ("corpus", [{"id": c, "image": c} for c in corpus]),
                       ("instances", instances)):
        with open(os.path.join(root, name + ".jsonl"), "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
    for f in [q["image"] for q in queries] + corpus:
        if f != skip_file:
            open(os.path.join(root, "imgs_mc", f), "wb").close()
    return root


def test_structure(tmp_path):
    ds = DATASET_REGISTRY["edir"](root=_edir(tmp_path), split="test", transform=None)
    assert isinstance(ds, EDIRDataset) and len(ds) == 2 and len(ds.corpus_paths) == 6
    t = ds.triplets[0]
    assert t["target_path"].endswith("1_a_2.png") and t["ref_path"].endswith("1_a.jpg")
    assert t["meta"] == {"dataset": "edir", "query_id": "0", "reference_id": "ref:1_a.jpg",
                         "target_id": "corpus:1_a_2.png", "source": "1_a", "subcategory": "color"}
    assert ds.corpus_ids[0] == "corpus:1_a_1.png"
    assert ds.annotated[1] == {"pos": "2_b_1.png", "neg": ["2_b_2.png", "2_b_3.png"]}
    assert not {t["ref_path"] for t in ds.triplets} & set(ds.corpus_paths)


def test_wrong_split(tmp_path):
    with pytest.raises(ValueError):
        EDIRDataset(root=_edir(tmp_path), split="val", transform=None)


def test_missing_image_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        EDIRDataset(root=_edir(tmp_path, skip_file="2_b_3.png"), split="test", transform=None)


@pytest.mark.parametrize("case", ["ref_in_corpus", "two_pos", "foreign_candidate", "not_in_corpus", "qid_mismatch",
                                  "duplicate_candidate"])
def test_refusals(tmp_path, case):
    q = [{"id": "0", "image": "1_a.jpg", "text": "t"}]
    c = ["1_a_1.png", "1_a_2.png", "1_a_3.png", "9_z_1.png"]
    i = [{"qid": "0", "pos": ["1_a_1.png"], "neg": ["1_a_2.png", "1_a_3.png"]}]
    if case == "ref_in_corpus":
        c = c + ["1_a.jpg"]
    elif case == "two_pos":
        i = [{"qid": "0", "pos": ["1_a_1.png", "1_a_2.png"], "neg": ["1_a_3.png"]}]
    elif case == "foreign_candidate":
        i = [{"qid": "0", "pos": ["1_a_1.png"], "neg": ["9_z_1.png"]}]
    elif case == "not_in_corpus":
        i = [{"qid": "0", "pos": ["1_a_9.png"], "neg": ["1_a_2.png"]}]
    elif case == "qid_mismatch":
        i = [{"qid": "7", "pos": ["1_a_1.png"], "neg": ["1_a_2.png"]}]
    else:
        i = [{"qid": "0", "pos": ["1_a_1.png"], "neg": ["1_a_2.png", "1_a_2.png"]}]
    with pytest.raises(ValueError):
        EDIRDataset(root=_edir(tmp_path, queries=q, corpus=c, instances=i), split="test", transform=None)
