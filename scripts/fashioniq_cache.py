"""FashionIQ validation score caches for the frozen-policy test (plan step 6).

Writes one cache per category, ranking each category's queries against that
category's official validation gallery (split.{category}.val.json), with the
reference image excluded -- the standard FashionIQ protocol. Model setup
mirrors scripts/evaluate_cirr.py exactly (same config, module combo,
load_checkpoint, evaluate_dataset with rerank=True and cache_path), so these
caches are produced by the same pipeline as the CIRR cache the policy was
fitted on. No policy is applied here; apply_frozen_policy.py does that.

Base-ranking R@10/R@50 per category are printed as context only.

Usage
-----
  python scripts/fashioniq_cache.py --config CONFIG --checkpoint CKPT \\
      --module-combo fc+aep+rv --out-dir ../fiq_caches
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CATEGORIES = ("dress", "shirt", "toptee")
EXPECTED = {"dress": (2017, 3817), "shirt": (2038, 6346), "toptee": (1961, 5373)}


def official_gallery(dataset, category: str):
    split = os.path.join(dataset.ann_root, "image_splits", "split.%s.val.json" % category)
    ids = json.load(open(split, encoding="utf-8"))
    return [dataset._img_path(i) for i in ids]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--module-combo", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    import torch
    from revua_cir.config import ReVUAConfig
    from revua_cir.data.datasets import DATASET_REGISTRY
    from revua_cir.data.tokenizer import build_tokenizer
    from revua_cir.data.transforms import build_image_transform
    from revua_cir.evaluate import evaluate_dataset
    from revua_cir.models.revua_cir import ReVUACIR
    from revua_cir.utils import load_checkpoint

    cfg = ReVUAConfig.from_yaml(args.config).apply_module_ablation(args.module_combo)
    transform = build_image_transform(cfg.data.image_size, train=False)
    tokenizer = build_tokenizer(cfg.backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.join(cfg.data.root, "fashioniq")

    os.makedirs(args.out_dir, exist_ok=True)
    plans = []
    for cat in CATEGORIES:
        ds = DATASET_REGISTRY["fashioniq"](root=root, split="val", transform=transform,
                                           categories=[cat])
        gallery = official_gallery(ds, cat)
        missing = [p for p in gallery if not os.path.exists(p)]
        cache = os.path.join(args.out_dir, "fiq_%s_val.jsonl" % cat)
        print("[fiq] %-7s queries %d | gallery %d | missing image files %d"
              % (cat, len(ds), len(gallery), len(missing)), flush=True)
        if (len(ds), len(gallery)) != EXPECTED[cat] or missing:
            print("FATAL: %s does not match the expected val split %s, or images are missing"
                  % (cat, EXPECTED[cat]), file=sys.stderr)
            return 1
        if os.path.exists(cache):
            print("FATAL: %s already exists; refusing to overwrite" % cache, file=sys.stderr)
            return 1
        plans.append((cat, ds, gallery, cache))

    model = ReVUACIR(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    if model.rv is None:
        print("FATAL: model has no verifier (module combo must include rv)", file=sys.stderr)
        return 1
    print("[fiq] loaded %s (%s) on %s" % (args.checkpoint, args.module_combo, device), flush=True)

    for cat, ds, gallery, cache in plans:
        t0 = time.time()
        m = evaluate_dataset(model, ds, cfg, tokenizer, transform, device,
                             ks=(1, 5, 10, 50), use_map=False, use_group_recall=False,
                             full_gallery_paths=gallery, max_queries=len(ds),
                             rerank=True, cache_path=cache)
        with open(cache, encoding="utf-8") as fh:
            n_lines = sum(1 for line in fh if line.strip())
        print("[fiq] %-7s done in %.0fs | cache lines %d (expect %d) | base-context R@10 %.2f R@50 %.2f"
              % (cat, time.time() - t0, n_lines, len(ds), m.get("recall_at_10"),
                 m.get("recall_at_50")), flush=True)
        if n_lines != len(ds):
            print("FATAL: %s has %d lines for %d queries" % (cache, n_lines, len(ds)), file=sys.stderr)
            return 1
    print("[fiq] all three caches written to %s" % args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
