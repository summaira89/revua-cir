"""Preflight check on a local dataset root -- run this BEFORE any training.

The loaders in `revua_cir/data/datasets.py` are written against each dataset's
*official* schema, and `safe_load_transform` substitutes a gray placeholder for
any image it cannot open. Those two facts combine badly: point `--data-root` one
directory too high and every image silently becomes gray, every JSON key lookup
fails, and training still runs to completion producing meaningless numbers.

This script resolves each configured dataset, reports triplet counts, samples
actual image paths off disk, and fails loudly on anything that would have been
swallowed at runtime.

    python scripts/verify_data.py --data-root D:/path/to/datasets
    python scripts/verify_data.py --data-root D:/path/to/datasets --datasets fashioniq cirr
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from revua_cir.data.datasets import DATASET_REGISTRY  # noqa: E402

# What each adapter needs on disk, relative to {data_root}/{name}/.
# `{split}` is filled from the split being checked.
EXPECTED_LAYOUT = {
    "fashioniq": {
        "splits": ["train", "val"],
        "annotations": ["captions/cap.dress.{split}.json",
                        "captions/cap.shirt.{split}.json",
                        "captions/cap.toptee.{split}.json"],
        # fashion-iq-master/fashion-iq-master/ shows up on third-party mirrors
        # (e.g. the Kaggle dataset binhhuunguyen/fashioniq), a "Download ZIP"
        # artifact from zipping an already-extracted GitHub repo.
        "annotation_prefixes": ["", os.path.join("fashion-iq-master", "fashion-iq-master")],
        "image_dir": "images",
        "keys": ["candidate", "target", "captions"],
    },
    "cirr": {
        "splits": ["train", "val"],
        "annotations": ["captions/cap.rc2.{split}.json",
                        "image_splits/split.rc2.{split}.json"],
        # A cloned CIRR_repo/ nested inside the data dir is a common local
        # layout; CIRR/cirr/ shows up on third-party mirrors (e.g. the Kaggle
        # dataset anhngzv/cirr-cir), where the official repo's own cirr/
        # subfolder gets preserved under an extra CIRR/ wrapper.
        "annotation_prefixes": ["", "CIRR_repo", os.path.join("CIRR", "cirr")],
        "image_dir": ".",  # paths come from the split map, relative to root
        "keys": ["reference", "target_hard", "caption"],
    },
    "circo": {
        "splits": ["val"],
        "annotations": ["annotations/{split}.json"],
        "image_dir": "COCO2017_unlabeled/unlabeled2017",
        "keys": ["reference_img_id", "relative_caption"],
    },
    "lasco": {
        "splits": ["train", "val"],
        "annotations": ["{split}.json"],
        "image_dir": "images",
        "keys": ["query-image", "target-image", "query-text"],
    },
    "genecis": {
        "splits": ["focus_attribute", "change_attribute", "focus_object", "change_object"],
        "annotations": ["{split}.json"],
        "image_dir": "images",
        "keys": ["reference", "target", "condition"],
    },
}


def check_dataset(root: str, name: str, sample_n: int) -> bool:
    spec = EXPECTED_LAYOUT[name]
    ds_root = os.path.join(root, name)
    ok = True

    print(f"\n=== {name} ===")
    print(f"  root: {ds_root}")
    if not os.path.isdir(ds_root):
        print(f"  FAIL: directory does not exist.")
        print(f"        Expected {{data-root}}/{name}/ -- train.py resolves each")
        print(f"        dataset as os.path.join(cfg.data.root, name).")
        return False

    prefixes = spec.get("annotation_prefixes", [""])

    def locate(rel: str):
        """Annotations may sit at the dataset root or one level down in a
        nested repo checkout; return the first prefix where the file exists."""
        for pre in prefixes:
            cand = os.path.join(pre, rel) if pre else rel
            if os.path.exists(os.path.join(ds_root, cand)):
                return cand
        return None

    for split in spec["splits"]:
        located = {rel.format(split=split): locate(rel.format(split=split))
                   for rel in spec["annotations"]}
        missing_ann = [rel for rel, hit in located.items() if hit is None]
        present_ann = [hit for hit in located.values() if hit is not None]

        if not present_ann:
            print(f"  [{split}] FAIL: none of the expected annotation files exist:")
            for rel in spec["annotations"]:
                print(f"           {rel.format(split=split)}")
            ok = False
            continue
        if missing_ann:
            # FashionIQ splits across 3 category files; some being absent is survivable.
            print(f"  [{split}] WARN: missing {len(missing_ann)} annotation file(s): {missing_ann}")

        # Schema check on the first present annotation file.
        first = os.path.join(ds_root, present_ann[0])
        with open(first, "rb") as f:
            head = f.read(2048)
        if b"<!DOCTYPE html" in head or b"<html" in head:
            print(f"  [{split}] FAIL: {present_ann[0]} is an HTML page, not JSON.")
            print(f"           This is a GitHub *web page* saved as .json -- download the")
            print(f"           raw file instead (use the 'Raw' button, or a raw.githubusercontent")
            print(f"           .com URL). Size on disk looks plausible, so this passes a")
            print(f"           casual file-listing check.")
            ok = False
            continue

        try:
            with open(first, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [{split}] FAIL: cannot parse {present_ann[0]}: {type(e).__name__}: {e}")
            ok = False
            continue

        if isinstance(data, dict):
            print(f"  [{split}] {present_ann[0]}: dict with {len(data)} entries (split map)")
        elif isinstance(data, list) and data:
            sample = data[0]
            absent = [k for k in spec["keys"] if k not in sample]
            print(f"  [{split}] {present_ann[0]}: {len(data)} records")
            if absent:
                print(f"           FAIL: expected key(s) {absent} not in the first record.")
                print(f"           Found keys: {sorted(sample.keys())}")
                print(f"           -> fix the key names in revua_cir/data/datasets.py ({name}).")
                ok = False
        else:
            print(f"  [{split}] FAIL: {present_ann[0]} is empty or an unexpected type.")
            ok = False

    # Instantiate the real adapter and sample images off disk -- this is the check
    # that catches a wrong root, because it exercises the same path construction
    # training will use.
    probe_split = spec["splits"][0]
    try:
        cls = DATASET_REGISTRY[name]
        kwargs = {"template": probe_split} if name == "genecis" else {"split": probe_split}
        ds = cls(root=ds_root, transform=None, **kwargs)
    except Exception as e:
        print(f"  FAIL: adapter raised on construction: {type(e).__name__}: {e}")
        return False

    n = len(ds)
    print(f"  adapter built: {n} triplets in '{probe_split}'")
    if n == 0:
        print("  FAIL: zero triplets -- annotations parsed but produced nothing.")
        return False

    rng = random.Random(0)
    idxs = rng.sample(range(n), min(sample_n, n))
    missing = []
    for i in idxs:
        for key in ("ref_path", "target_path"):
            p = ds.triplets[i][key]
            if not os.path.exists(p):
                missing.append(p)

    checked = len(idxs) * 2
    if missing:
        rate = len(missing) / checked
        print(f"  images: {checked - len(missing)}/{checked} found ({rate:.0%} MISSING)")
        for p in missing[:5]:
            print(f"           missing: {p}")
        if rate > 0.5:
            print("  FAIL: majority of sampled images missing -- this is almost certainly")
            print("        a wrong path, not link rot. Training would run on gray placeholders.")
            ok = False
        else:
            print("  WARN: some images missing (link rot). They become gray placeholders;")
            print("        report the final usable triplet count in the paper.")
    else:
        print(f"  images: {checked}/{checked} found")

    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="parent dir containing fashioniq/, cirr/, ...")
    ap.add_argument("--datasets", nargs="*", default=["fashioniq", "cirr", "circo"],
                    help="which to check (default: the three the config trains/tests on)")
    ap.add_argument("--sample", type=int, default=200, help="triplets to sample for image existence")
    args = ap.parse_args()

    print(f"data root: {args.data_root}")
    if not os.path.isdir(args.data_root):
        print("FAIL: --data-root does not exist.")
        sys.exit(1)
    print(f"contains: {sorted(os.listdir(args.data_root))}")

    results = {name: check_dataset(args.data_root, name, args.sample) for name in args.datasets}

    print("\n=== summary ===")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    if not all(results.values()):
        print("\nFix the FAILs before training -- the loaders fail soft, so these")
        print("would not have stopped a run, only corrupted its results.")
        sys.exit(1)
    print("\nAll checked datasets look correct.")


if __name__ == "__main__":
    main()
