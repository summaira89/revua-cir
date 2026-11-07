"""Per-query base-retrieval hit@1 for one checkpoint, saved for paired tests.

Uses the exact evaluation path of scripts/ablation_significance.py
(`_evaluate`, `_hit_vector`: full official gallery, all CIRR validation
queries, same validity mask), so the output can be paired with any other
checkpoint evaluated the same way -- in particular FC's saved
fc_per_query_hits.json from job 7246.

Usage
-----
  python scripts/per_query_hits.py --config CONFIG --checkpoint CKPT \\
      --module-combo fc+aep --out ../fc_aep_per_query_hits.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--module-combo", required=True,
                    help="must match the checkpoint; load_checkpoint is non-strict")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    from revua_cir.config import ReVUAConfig
    from revua_cir.data.datasets import DATASET_REGISTRY
    from revua_cir.data.transforms import build_image_transform
    from ablation_significance import _evaluate, _hit_vector
    from evaluate_cirr import build_cirr_full_gallery

    cfg = ReVUAConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DATASET_REGISTRY["cirr"](
        root=os.path.join(cfg.data.root, "cirr"), split="val",
        transform=build_image_transform(cfg.data.image_size, train=False))
    gallery = build_cirr_full_gallery(dataset)
    print("device %s | combo %s | queries %d | gallery %d"
          % (device, args.module_combo, len(dataset), len(gallery)))

    res = _evaluate(cfg, args.checkpoint, args.module_combo, dataset, gallery, device)
    hits, valid = _hit_vector(res["per_query"])
    targets = [int(t) for t in res["per_query"]["targets"]]
    print("R@1 = %.4f  (%d / %d hits, %d invalid)"
          % (res["metrics"]["recall_at_1"], sum(hits), len(hits), len(hits) - sum(valid)))

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"checkpoint": args.checkpoint, "module_combo": args.module_combo,
                   "query_ids": res["per_query"].get("query_ids"),
                   "targets": targets, "valid": valid, "hit_at_1": hits}, fh)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
