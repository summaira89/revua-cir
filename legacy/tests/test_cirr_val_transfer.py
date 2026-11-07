"""cirr_val_transfer.py: pack -> unzip -> verify round trip on a fake CIRR layout."""
import json
import os
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import cirr_val_transfer as cvt  # noqa: E402

FIVE_KB = 5000 / (1024 * 1024)      # part limit in MiB that forces several parts


def _fake_cirr(root, n=6, dev_dir="dev"):
    for d in ("captions", "image_splits", dev_dir):
        os.makedirs(os.path.join(root, d))
    ids = ["dev-%d-0-img%d" % (i // 2, i % 2) for i in range(n)]
    for k, i in enumerate(ids):
        with open(os.path.join(root, dev_dir, i + ".png"), "wb") as fh:
            fh.write(os.urandom(2000 + k))
    with open(os.path.join(root, "image_splits", "split.rc2.val.json"), "w") as fh:
        json.dump({i: "./dev/%s.png" % i for i in ids}, fh)
    with open(os.path.join(root, "captions", "cap.rc2.val.json"), "w") as fh:
        json.dump([{"reference": ids[0], "target_hard": ids[1], "caption": "c", "pairid": 0,
                    "img_set": {"members": ids[:2]}}], fh)
    return ids


def _pack_and_unzip(tmp_path, part_mb=FIVE_KB, dev_dir="dev"):
    ids = _fake_cirr(str(tmp_path / "src"), dev_dir=dev_dir)
    out, dest = str(tmp_path / "pack"), str(tmp_path / "laptop")
    m = cvt.pack(str(tmp_path / "src"), out, part_mb)
    for name in sorted(m["parts"]):
        with zipfile.ZipFile(os.path.join(out, name)) as zf:
            zf.extractall(dest)
    return ids, m, os.path.join(out, "cirr_val_manifest.json"), dest


@pytest.mark.parametrize("dev_dir", ["dev", "dev_img"])
def test_round_trip_multi_part(tmp_path, dev_dir):
    from revua_cir.data.datasets import CIRRDataset
    ids, m, man, dest = _pack_and_unzip(tmp_path, dev_dir=dev_dir)
    assert m["n_images"] == len(ids) and m["n_files"] == len(ids) + 2
    assert len(m["parts"]) >= 2
    assert cvt.verify(man, dest) == []
    # the unpacked tree loads with the project's own dataset class, map naming
    ds = CIRRDataset(root=os.path.join(dest, "cirr_val"), split="val", transform=None)
    assert all(os.path.isfile(ds._img_path(i)) for i in ids)


def test_verify_catches_corruption_missing_and_extra(tmp_path):
    _, _, man, dest = _pack_and_unzip(tmp_path)
    dev = os.path.join(dest, "cirr_val", "dev")
    first, second = sorted(os.listdir(dev))[:2]
    with open(os.path.join(dev, first), "r+b") as fh:
        fh.write(b"X")
    os.remove(os.path.join(dev, second))
    with open(os.path.join(dev, "stray.png"), "wb") as fh:
        fh.write(b"0")
    problems = cvt.verify(man, dest)
    assert any(p.startswith("content differs") for p in problems)
    assert any(p.startswith("missing") for p in problems)
    assert any(p.startswith("extra image") for p in problems)


def test_pack_refuses_missing_image(tmp_path):
    ids = _fake_cirr(str(tmp_path / "src"))
    os.remove(os.path.join(str(tmp_path / "src"), "dev", ids[3] + ".png"))
    with pytest.raises(RuntimeError):
        cvt.pack(str(tmp_path / "src"), str(tmp_path / "pack"), FIVE_KB)


def test_arcname_rejects_traversal():
    assert cvt.arcname_for("./dev/a.png") == "cirr_val/dev/a.png"
    with pytest.raises(ValueError):
        cvt.arcname_for("../etc/x.png")
