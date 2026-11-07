"""edir_unpack.py: split-tar streaming, path safety and completeness check."""
import io
import json
import os
import sys
import tarfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import edir_unpack as eu  # noqa: E402


def _tar_bytes(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _split(raw, d, n=3):
    os.makedirs(d)
    step = len(raw) // n + 1
    for k in range(n):
        with open(os.path.join(d, "imgs_mc.tar.parta%s" % "abcdefg"[k]), "wb") as fh:
            fh.write(raw[k * step:(k + 1) * step])


def _meta(d, refs, corpus, pos, neg):
    os.makedirs(d)
    with open(os.path.join(d, "queries.jsonl"), "w") as fh:
        for i, r in enumerate(refs):
            fh.write(json.dumps({"id": str(i), "image": r, "text": "t"}) + "\n")
    with open(os.path.join(d, "corpus.jsonl"), "w") as fh:
        for c in corpus:
            fh.write(json.dumps({"id": c, "image": c}) + "\n")
    with open(os.path.join(d, "instances.jsonl"), "w") as fh:
        fh.write(json.dumps({"qid": "0", "pos": pos, "neg": neg}) + "\n")


def test_split_archive_round_trip(tmp_path):
    names = ["a.jpg", "a_1.png", "a_2.png"]
    raw = _tar_bytes([("imgs_mc/" + n, os.urandom(3000)) for n in names])
    _split(raw, str(tmp_path / "parts"))
    _meta(str(tmp_path / "meta"), ["a.jpg"], ["a_1.png", "a_2.png"], ["a_1.png"], ["a_2.png"])
    assert eu.main(["--parts-dir", str(tmp_path / "parts"), "--dest", str(tmp_path / "out"),
                    "--meta", str(tmp_path / "meta")]) == 0
    for n in names:
        assert os.path.getsize(str(tmp_path / "out" / "imgs_mc" / n)) == 3000


def test_missing_image_reported(tmp_path):
    raw = _tar_bytes([("imgs_mc/a.jpg", b"x" * 10)])
    _split(raw, str(tmp_path / "parts"), 1)
    _meta(str(tmp_path / "meta"), ["a.jpg"], ["a_1.png"], ["a_1.png"], [])
    assert eu.main(["--parts-dir", str(tmp_path / "parts"), "--dest", str(tmp_path / "out"),
                    "--meta", str(tmp_path / "meta")]) == 1


@pytest.mark.parametrize("bad", ["../evil.png", "other/x.png", "/abs.png"])
def test_unsafe_member_stops(tmp_path, bad):
    raw = _tar_bytes([("imgs_mc/a.jpg", b"x"), (bad, b"y")])
    _split(raw, str(tmp_path / "parts"), 1)
    with pytest.raises(SystemExit):
        eu.unpack(str(tmp_path / "parts"), str(tmp_path / "out"))
    assert not os.path.exists(str(tmp_path / "evil.png"))


def test_refuses_existing_destination(tmp_path):
    raw = _tar_bytes([("imgs_mc/a.jpg", b"x")])
    _split(raw, str(tmp_path / "parts"), 1)
    os.makedirs(str(tmp_path / "out" / "imgs_mc"))
    with pytest.raises(SystemExit):
        eu.unpack(str(tmp_path / "parts"), str(tmp_path / "out"))
