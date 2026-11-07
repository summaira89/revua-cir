"""Unpack the hash-verified EDIR image archive (v3.1 §19) and check completeness.

The release is one tar split into parts (imgs_mc.tar.partaa ... partag). They
are streamed in name order as a single archive; every member must be a plain
file or directory under imgs_mc/ (anything else aborts before writing it).
Afterwards every image named by queries.jsonl (references), corpus.jsonl and
instances.jsonl (pos and neg) must exist; the counts are printed.

  python scripts/edir_unpack.py --parts-dir ../edir_data --dest ../edir_data --meta ../edir_meta/72581ea0
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import posixpath
import tarfile
from typing import List, Optional, Sequence

TOP = "imgs_mc"


class ConcatReader(io.RawIOBase):
    """Read several files back to back as one stream."""

    def __init__(self, paths: Sequence[str]):
        self.paths, self.i, self.fh = list(paths), 0, None

    def readable(self) -> bool:
        return True

    def readinto(self, buf) -> int:
        while self.i < len(self.paths):
            if self.fh is None:
                self.fh = open(self.paths[self.i], "rb")
            n = self.fh.readinto(buf)
            if n:
                return n
            self.fh.close()
            self.fh, self.i = None, self.i + 1
        return 0


def safe_member(m: tarfile.TarInfo) -> bool:
    name = m.name
    parts = name.split("/")
    return ((m.isfile() or m.isdir()) and not posixpath.isabs(name) and ".." not in parts
            and parts[0] == TOP)


def required_images(meta: str) -> dict:
    def rows(name):
        with open(os.path.join(meta, name), encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    refs = {r["image"] for r in rows("queries.jsonl")}
    corpus = {r["image"] for r in rows("corpus.jsonl")}
    annotated = set()
    for r in rows("instances.jsonl"):
        annotated.update(r["pos"])
        annotated.update(r["neg"])
    return {"references": refs, "corpus": corpus, "annotated": annotated}


def check(dest: str, meta: str) -> List[str]:
    root = os.path.join(dest, TOP)
    present = set(os.listdir(root)) if os.path.isdir(root) else set()
    problems = []
    for kind, names in required_images(meta).items():
        missing = sorted(names - present)
        print("%-10s %6d required, %d missing" % (kind, len(names), len(missing)))
        problems += ["missing %s image %s" % (kind, n) for n in missing]
    return problems


def unpack(parts_dir: str, dest: str) -> int:
    parts = sorted(glob.glob(os.path.join(parts_dir, "imgs_mc.tar.part*")))
    if not parts:
        raise SystemExit("no imgs_mc.tar.part* in %s" % parts_dir)
    if os.path.exists(os.path.join(dest, TOP)):
        raise SystemExit("%s already exists; refusing to unpack over it" % os.path.join(dest, TOP))
    n = 0
    with tarfile.open(fileobj=io.BufferedReader(ConcatReader(parts), 1 << 20), mode="r|") as tar:
        for m in tar:
            if not safe_member(m):
                raise SystemExit("unsafe archive member %r; stopped" % m.name)
            tar.extract(m, dest, filter="data")
            n += m.isfile()
            if n and n % 5000 == 0 and m.isfile():
                print("%d files" % n, flush=True)
    return n


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parts-dir", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--meta", required=True, help="folder with queries/corpus/instances.jsonl")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.check_only:
        print("unpacked %d files" % unpack(args.parts_dir, args.dest), flush=True)
    problems = check(args.dest, args.meta)
    print("COMPLETE: every referenced EDIR image is present" if not problems
          else "INCOMPLETE: %d missing (first: %s)" % (len(problems), problems[0]))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
