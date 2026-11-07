"""Move the CIRR validation split (images + annotations) from the cluster to the
laptop for the §19 equivalence test, verified file by file.

  pack    (cluster)  resolve every validation image through the project's own
                     CIRRDataset / build_cirr_full_gallery (which refuse missing
                     images), write zip parts of at most --part-mb each, and a
                     manifest of every file's path, size and SHA-256.
  verify  (laptop)   after unzipping all parts into one folder, require every
                     manifest file present with the same size and SHA-256, and
                     no extra image. A missing image would otherwise become a
                     silent gray placeholder at load time.

Files are stored under cirr_val/ in the split map's own naming
(cirr_val/dev/<name>.png, cirr_val/captions/..., cirr_val/image_splits/...), so
CIRRDataset(root=<folder>/cirr_val) resolves them directly.

Usage
  python scripts/cirr_val_transfer.py pack   --data-root ./datasets/cirr --out ./cirr_val_pack
  python scripts/cirr_val_transfer.py verify --manifest cirr_val_manifest.json --root path/to/cirr_val_unpacked
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), HERE]

SCHEMA = "cirr-val-transfer/1"
TOP = "cirr_val"
IMAGE_EXT = (".png", ".jpg", ".jpeg")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def arcname_for(rel: str) -> str:
    """The split map's relative path ('./dev/x.png') as a zip member under TOP."""
    parts = [p for p in rel.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        raise ValueError("unsafe or empty split-map path %r" % rel)
    return "/".join([TOP] + parts)


def collect(data_root: str) -> List[Tuple[str, str]]:
    """(source path, arcname) for every validation image and both annotation files."""
    from revua_cir.data.datasets import CIRRDataset
    from evaluate_cirr import build_cirr_full_gallery

    ds = CIRRDataset(root=data_root, split="val", transform=None)
    gallery = set(build_cirr_full_gallery(ds))      # raises on any unresolved image
    out: Dict[str, str] = {}
    for image_id, rel in ds.id_to_path.items():
        src = ds._img_path(image_id)
        if src not in gallery or not os.path.isfile(src):
            raise RuntimeError("image %s did not resolve to a real file (%s)" % (image_id, src))
        arc = arcname_for(rel)
        if arc in out and out[arc] != src:
            raise RuntimeError("two images map to %s" % arc)
        out[arc] = src
    for t in ds.triplets:                            # every reference and target is in the gallery
        for key in ("ref_path", "target_path"):
            if t[key] not in gallery:
                raise RuntimeError("%s %s is not a gallery image" % (key, t[key]))
    for sub, name in (("captions", "cap.rc2.val.json"), ("image_splits", "split.rc2.val.json")):
        out["%s/%s/%s" % (TOP, sub, name)] = os.path.join(ds.ann_root, sub, name)
    return sorted((src, arc) for arc, src in out.items())


def pack(data_root: str, out_dir: str, part_mb: float) -> dict:
    files = collect(data_root)
    os.makedirs(out_dir, exist_ok=False)
    limit = int(part_mb * 1024 * 1024)
    entries, parts, zf, part_bytes = [], [], None, 0
    for src, arc in files:
        size = os.path.getsize(src)
        if size > limit:
            raise RuntimeError("%s (%d B) exceeds the part limit" % (src, size))
        if zf is None or part_bytes + size > limit:
            if zf is not None:
                zf.close()
            parts.append("cirr_val_part%02d.zip" % (len(parts) + 1))
            zf = zipfile.ZipFile(os.path.join(out_dir, parts[-1]), "w", zipfile.ZIP_STORED)
            part_bytes = 0
        zf.write(src, arc)
        part_bytes += size
        entries.append({"path": arc, "bytes": size, "sha256": sha256_file(src), "part": parts[-1]})
    if zf is not None:
        zf.close()
    n_images = sum(1 for e in entries if e["path"].lower().endswith(IMAGE_EXT))
    manifest = {"schema": SCHEMA, "n_images": n_images, "n_files": len(entries),
                "parts": {p: {"bytes": os.path.getsize(os.path.join(out_dir, p)),
                              "sha256": sha256_file(os.path.join(out_dir, p))} for p in parts},
                "files": entries}
    with open(os.path.join(out_dir, "cirr_val_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def verify(manifest_path: str, root: str) -> List[str]:
    """Problems found (empty list = verified). `root` is the folder holding cirr_val/."""
    m = json.load(open(manifest_path, encoding="utf-8"))
    if m.get("schema") != SCHEMA:
        return ["manifest schema %r, expected %r" % (m.get("schema"), SCHEMA)]
    problems, expected = [], set()
    for e in m["files"]:
        expected.add(e["path"])
        p = os.path.join(root, *e["path"].split("/"))
        if not os.path.isfile(p):
            problems.append("missing %s" % e["path"])
        elif os.path.getsize(p) != e["bytes"] or sha256_file(p) != e["sha256"]:
            problems.append("content differs %s" % e["path"])
    for dirpath, _, names in os.walk(os.path.join(root, TOP)):
        for n in names:
            rel = os.path.relpath(os.path.join(dirpath, n), root).replace(os.sep, "/")
            if rel not in expected and n.lower().endswith(IMAGE_EXT):
                problems.append("extra image %s" % rel)
    if sum(1 for e in m["files"] if e["path"].lower().endswith(IMAGE_EXT)) != m["n_images"]:
        problems.append("manifest n_images inconsistent")
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--data-root", required=True, help="CIRR dataset root as passed to CIRRDataset")
    p.add_argument("--out", required=True, help="new folder for the zip parts and manifest")
    p.add_argument("--part-mb", type=float, default=900)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", required=True)
    v.add_argument("--root", required=True, help="folder the parts were unzipped into (holds cirr_val/)")
    args = ap.parse_args(argv)
    if args.cmd == "pack":
        m = pack(os.path.expanduser(args.data_root), os.path.expanduser(args.out), args.part_mb)
        print("packed %d images + %d annotation files" % (m["n_images"], m["n_files"] - m["n_images"]))
        for name, info in m["parts"].items():
            print("%s %.1f MB sha256 %s" % (name, info["bytes"] / 1e6, info["sha256"]))
        return 0
    problems = verify(args.manifest, args.root)
    for line in problems[:20]:
        print(line)
    print("VERIFIED: all %s files match" % json.load(open(args.manifest, encoding="utf-8"))["n_files"]
          if not problems else "FAILED: %d problem(s)" % len(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
