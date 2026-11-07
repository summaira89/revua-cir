"""
Direct base-vs-rerank comparison against ONE specific checkpoint, using the
new rank-fusion reranker (_rank_fuse in models/revua_cir.py, 2026-09-03) that
replaced the old magnitude-based `score * (0.5 + 0.5*adj)` rescale -- that
formula couldn't move rankings because RV's calibrated verification_score
spread within a query's own top-k collapsed to ~0.00007 even after the
hard-negative-mining fix improved its *ordering* ability
(topk_target_rv_argmax_rate 16%->23%, chance=10%). Rank fusion uses ordering
directly, independent of that tiny magnitude.

    python scripts/verify_rerank_gain.py \
        --config configs/stage1_finetune_full_unfreeze_gradcache.yaml \
        --checkpoint runs/verify_rv_hardneg/epoch006.pt \
        --max-queries 300
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from revua_cir.config import ReVUAConfig
from revua_cir.data.datasets import DATASET_REGISTRY
from revua_cir.data.tokenizer import build_tokenizer
from revua_cir.data.transforms import build_image_transform
from revua_cir.evaluate import evaluate_dataset
from revua_cir.models.revua_cir import ReVUACIR
from revua_cir.utils import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--module-combo", type=str, default="fc+aep+rv")
    p.add_argument("--eval-dataset", type=str, default="cirr", choices=["cirr", "fashioniq", "circo"])
    p.add_argument("--eval-split", type=str, default="val")
    p.add_argument("--max-queries", type=int, default=300)
    args = p.parse_args()

    cfg = ReVUAConfig.from_yaml(args.config)
    cfg = cfg.apply_module_ablation(args.module_combo)

    eval_transform = build_image_transform(cfg.data.image_size, train=False)
    tokenizer = build_tokenizer(cfg.backbone)
    eval_root = os.path.join(cfg.data.root, args.eval_dataset)
    eval_dataset = DATASET_REGISTRY[args.eval_dataset](
        root=eval_root, split=args.eval_split, transform=eval_transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ReVUACIR(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    print(f"[verify_rerank_gain] loaded {args.checkpoint}")

    n_eval = min(len(eval_dataset), args.max_queries)
    common = dict(
        use_group_recall=(args.eval_dataset == "cirr"),
        max_queries=n_eval,
    )

    base = evaluate_dataset(model, eval_dataset, cfg, tokenizer, eval_transform, device, rerank=False, **common)
    rr = evaluate_dataset(model, eval_dataset, cfg, tokenizer, eval_transform, device, rerank=True, **common)

    print()
    print("[verify_rerank_gain] BASE (embedding similarity only):")
    print(f"  recall_at_1={base.get('recall_at_1'):.2f} recall_at_5={base.get('recall_at_5'):.2f} "
          f"group_recall_at_1={base.get('group_recall_at_1'):.2f}")
    print("[verify_rerank_gain] RERANK (new rank-fusion RV rescale):")
    print(f"  recall_at_1={rr.get('recall_at_1'):.2f} recall_at_5={rr.get('recall_at_5'):.2f} "
          f"group_recall_at_1={rr.get('group_recall_at_1'):.2f}")
    print()
    print(f"[verify_rerank_gain] GAIN: recall_at_1 {rr.get('recall_at_1') - base.get('recall_at_1'):+.2f}  "
          f"recall_at_5 {rr.get('recall_at_5') - base.get('recall_at_5'):+.2f}  "
          f"group_recall_at_1 {rr.get('group_recall_at_1') - base.get('group_recall_at_1'):+.2f}")
    print(f"[verify_rerank_gain] (R@10/@50 are invariant to reranking within top-k by construction, "
          f"same as before -- not reported here)")
    print()
    print(f"[verify_rerank_gain] full base metrics -> {base}")
    print(f"[verify_rerank_gain] full rerank metrics -> {rr}")


if __name__ == "__main__":
    main()
