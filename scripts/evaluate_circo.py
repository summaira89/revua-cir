"""
CIRCO stress-test evaluation (Section 5.1 / 6 of the paper framework): scores
an already-trained CIRR checkpoint against CIRCO val's full official gallery,
reporting mAP@5/10/25/50 -- the framework's designated headline metric for
"false-negative-robust open-domain retrieval."

This is a SEPARATE script from run_ablation_matrix.py, not a mode of it,
because CIRCO evaluation is a genuinely different shape of problem:
  * gallery is the full COCO2017-unlabeled set (123,403 images, confirmed on
    disk 2026-09-06), not the few-thousand-image per-split union CIRR uses --
    encode_gallery_paths() has ~54x more images to embed.
  * metric is multi-ground-truth mAP@K, not single-target recall -- no
    group_recall (CIRR-only concept), ks must include 25 (the default
    (1,5,10,50) omits it; Section 6 explicitly wants 5/10/25/50).
  * this evaluates checkpoints already trained on CIRR (cross-dataset
    zero-shot transfer, Ablation #7), it does not train anything itself.

Critical bug this script fixes (README.md's own "before publication" item
#4, evaluate.py's `full_gallery_paths=` parameter was never actually wired
up anywhere): `evaluate_dataset()`'s default gallery
(`build_default_gallery`) only contains each triplet's OWN primary target
image, never CIRCO's other ~3 ground-truth images per query (avg 4.16 GT).
Running CIRCO through the default gallery would structurally exclude most
true positives from ever being retrievable, silently deflating mAP to a
number that means nothing. This script always passes the full COCO2017
image list explicitly.

    python scripts/evaluate_circo.py \
        --config configs/stage1_finetune_full_unfreeze_gradcache.yaml \
        --checkpoint runs/ablation_partial_unfreeze_v2/fc_aep_rv_ug/epoch016.pt \
        --module-combo fc+aep+rv+ug
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from revua_cir.config import ReVUAConfig
from revua_cir.data.datasets import DATASET_REGISTRY
from revua_cir.data.tokenizer import build_tokenizer
from revua_cir.data.transforms import build_image_transform
from revua_cir.evaluate import evaluate_dataset
from revua_cir.models.revua_cir import ReVUACIR
from revua_cir.utils import load_checkpoint


def build_circo_full_gallery(circo_root: str) -> list[str]:
    """The full COCO2017-unlabeled image set -- CIRCO's official gallery,
    per README.md item #4 and the CIRCO paper itself. NOT the per-triplet
    union build_default_gallery() would produce (see module docstring).
    """
    image_dir = os.path.join(circo_root, "COCO2017_unlabeled", "unlabeled2017")
    if not os.path.isdir(image_dir):
        image_dir = os.path.join(circo_root, "COCO2017_unlabeled")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"CIRCO image directory not found at either "
            f"{os.path.join(circo_root, 'COCO2017_unlabeled', 'unlabeled2017')} or "
            f"{os.path.join(circo_root, 'COCO2017_unlabeled')} -- check --data-root."
        )
    paths = sorted(
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not paths:
        raise RuntimeError(f"{image_dir} exists but contains no image files.")
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data-root", type=str, default=None, help="overrides data.root from the config")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="a single checkpoint path, e.g. runs/.../fc_aep_rv_ug/epoch016.pt")
    p.add_argument("--module-combo", type=str, required=True,
                   help="e.g. 'fc' | 'fc+aep' | 'fc+aep+rv' | 'fc+aep+rv+ug' -- must match the "
                        "combo the checkpoint was actually trained with, or load_checkpoint will "
                        "either error on shape mismatch or silently skip the mismatched submodules.")
    p.add_argument("--eval-split", type=str, default="val", choices=["val"],
                   help="CIRCO 'test' has no public ground truth (gt_img_ids absent, labels held on "
                        "the authors' server) -- only 'val' is locally scoreable, so it's the only choice.")
    p.add_argument("--max-queries", type=int, default=None,
                   help="cap queries for a fast smoke run; omit for the full 220-query val set")
    p.add_argument("--rerank", action="store_true",
                   help="also run with RV/UG reranking and report both, matching Table 2's format "
                        "(the CIRR ablation already does this by default; CIRCO needs it opted in "
                        "explicitly here since it costs a second full-gallery-adjacent pass)")
    p.add_argument("--cache-path", type=str, default=None,
                   help="only used together with --rerank: appends one JSON line per query to this "
                        "path with the per-candidate base score and RV sub-scores (recon_cos, "
                        "invariant_cos, edit_cos, verification_score) already computed during "
                        "reranking, for OFFLINE fusion-rule experimentation without a second GPU "
                        "pass per variant -- see evaluate_dataset()'s docstring for the schema.")
    args = p.parse_args()

    cfg = ReVUAConfig.from_yaml(args.config)
    if args.data_root:
        cfg.data.root = args.data_root
    cfg = cfg.apply_module_ablation(args.module_combo)

    eval_transform = build_image_transform(cfg.data.image_size, train=False)
    tokenizer = build_tokenizer(cfg.backbone)
    circo_root = os.path.join(cfg.data.root, "circo")
    eval_dataset = DATASET_REGISTRY["circo"](root=circo_root, split=args.eval_split, transform=eval_transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[evaluate_circo] CIRCO {args.eval_split}: {len(eval_dataset)} queries")
    t0 = time.time()
    full_gallery = build_circo_full_gallery(circo_root)
    print(f"[evaluate_circo] full official gallery: {len(full_gallery)} images "
          f"(listed in {time.time() - t0:.1f}s)")

    model = ReVUACIR(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    print(f"[evaluate_circo] loaded {args.checkpoint} (module_combo={args.module_combo})")

    n_eval = len(eval_dataset) if args.max_queries is None else min(args.max_queries, len(eval_dataset))
    common = dict(
        ks=(5, 10, 25, 50),  # Section 6: CIRCO reports mAP@5/10/25/50, not the CIRR-default (1,5,10,50)
        use_map=True,
        use_group_recall=False,  # CIRR-only concept -- CIRCO has no 6-way distractor subset protocol
        full_gallery_paths=full_gallery,
        max_queries=n_eval,
    )

    t0 = time.time()
    base = evaluate_dataset(model, eval_dataset, cfg, tokenizer, eval_transform, device, rerank=False, **common)
    print(f"[evaluate_circo] base eval done in {time.time() - t0:.1f}s")
    print(f"[evaluate_circo] BASE: map_at_5={base.get('map_at_5'):.2f} map_at_10={base.get('map_at_10'):.2f} "
          f"map_at_25={base.get('map_at_25'):.2f} map_at_50={base.get('map_at_50'):.2f}")
    print(f"[evaluate_circo] full base metrics -> {base}")

    if args.rerank and model.rv is not None:
        t0 = time.time()
        rr = evaluate_dataset(model, eval_dataset, cfg, tokenizer, eval_transform, device, rerank=True,
                               cache_path=args.cache_path, **common)
        print(f"[evaluate_circo] rerank eval done in {time.time() - t0:.1f}s")
        print(f"[evaluate_circo] RERANK: map_at_5={rr.get('map_at_5'):.2f} map_at_10={rr.get('map_at_10'):.2f} "
              f"map_at_25={rr.get('map_at_25'):.2f} map_at_50={rr.get('map_at_50'):.2f}")
        for k in (5, 10, 25, 50):
            b, a = base.get(f"map_at_{k}"), rr.get(f"map_at_{k}")
            if b is not None and a is not None:
                print(f"[evaluate_circo] GAIN map_at_{k}: {a - b:+.2f}")
        if "rv_reranking_latency_sec" in rr:
            print(f"[evaluate_circo] LATENCY: base_retrieval={rr['base_retrieval_latency_sec']*1000:.2f}ms "
                  f"rv_reranking={rr['rv_reranking_latency_sec']*1000:.2f}ms "
                  f"total_pipeline={rr['total_pipeline_latency_sec']*1000:.2f}ms per query "
                  f"(report total_pipeline, not just base_retrieval, as 'ReVUA-CIR inference latency')")
        print(f"[evaluate_circo] full rerank metrics -> {rr}")


if __name__ == "__main__":
    main()
