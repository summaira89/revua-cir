"""
Quick verification pass for the RV hard-negative-mining fix (2026-09-01):
trains fc+aep+rv for a small, capped number of epochs into a FRESH output
dir (does not touch the existing runs/ablation_partial_unfreeze/fc_aep_rv
checkpoints), then runs the same within-top-k diagnostics as
scripts/diagnose_rv.py against the result.

This is a direction check, not a final number: enough epochs to see whether
diag_rv_cos_gap widens and topk_verification_score_spread /
topk_target_rv_argmax_rate improve, before committing to a full run.

    python scripts/verify_rv_hardneg.py \
        --config configs/stage1_finetune_full_unfreeze_gradcache.yaml \
        --output-dir runs/verify_rv_hardneg \
        --max-epochs 5 \
        --max-queries 300
"""
from __future__ import annotations

import argparse
import glob
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
from revua_cir.train import train
from revua_cir.utils import load_checkpoint


def best_checkpoint(run_dir: str):
    best_path, best_metric = None, None
    for path in sorted(glob.glob(os.path.join(run_dir, "epoch*.pt"))):
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        m = state.get("best_metric")
        if m is not None and (best_metric is None or m > best_metric):
            best_path, best_metric = path, m
    return best_path, best_metric


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True,
                   help="fresh dir -- must NOT be the existing ablation matrix output dir")
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--max-queries", type=int, default=300)
    p.add_argument("--eval-dataset", type=str, default="cirr")
    p.add_argument("--eval-split", type=str, default="val")
    p.add_argument("--early-stopping", action="store_true",
                   help="let training converge naturally (config's own patience/monitor) instead "
                        "of stopping at a fixed epoch count. Needed to check whether a fix holds up "
                        "at CONVERGENCE, not just in an early snapshot -- the pure-argmax hard-negative "
                        "fix (2026-09-01) looked great at 6-8 epochs and only revealed its collapse "
                        "at full 20-24 epoch convergence (scripts/verify_rerank_gain.py, 2026-09-05/06).")
    args = p.parse_args()

    cfg = ReVUAConfig.from_yaml(args.config)
    cfg = cfg.apply_module_ablation("fc+aep+rv")
    cfg.optim.max_epochs = args.max_epochs
    cfg.run.output_dir = args.output_dir
    cfg.run.experiment_name = "verify_rv_hardneg"
    cfg.early_stopping.enabled = args.early_stopping
    # Cluster home filesystem hit 100% full mid-run once already (each
    # checkpoint here is ~1.3G; save_top_k=3 plus the mid-epoch snapshot adds
    # up fast on a disposable diagnostic run). Keep only the single best
    # checkpoint on disk at a time -- this script only ever reads the best one
    # anyway (best_checkpoint() below), so there is no reason to retain more.
    cfg.run.save_top_k = 1
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[verify_rv_hardneg] training fc+aep+rv for {args.max_epochs} epochs -> {args.output_dir}")
    train(cfg)

    eval_transform = build_image_transform(cfg.data.image_size, train=False)
    tokenizer = build_tokenizer(cfg.backbone)
    eval_root = os.path.join(cfg.data.root, args.eval_dataset)
    eval_dataset = DATASET_REGISTRY[args.eval_dataset](
        root=eval_root, split=args.eval_split, transform=eval_transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt, ckpt_metric = best_checkpoint(args.output_dir)
    model = ReVUACIR(cfg).to(device)
    if ckpt:
        load_checkpoint(ckpt, model, map_location=device)
        print(f"[verify_rv_hardneg] loaded {os.path.basename(ckpt)} (val monitor={ckpt_metric:.4f})")
    model.eval()

    n_eval = min(len(eval_dataset), args.max_queries)
    metrics = evaluate_dataset(
        model, eval_dataset, cfg, tokenizer, eval_transform, device,
        use_group_recall=(args.eval_dataset == "cirr"),
        max_queries=n_eval, rerank=False,
    )

    print(f"[verify_rv_hardneg] AFTER hard-negative fix, {args.max_epochs} epochs:")
    print(f"[verify_rv_hardneg]   recall_at_1={metrics.get('recall_at_1'):.2f} "
          f"verification_auroc={metrics.get('verification_auroc'):.4f}")
    print(f"[verify_rv_hardneg]   topk_verification_score_spread={metrics.get('topk_verification_score_spread'):.4f} "
          f"topk_verification_score_std={metrics.get('topk_verification_score_std'):.4f}")
    print(f"[verify_rv_hardneg]   topk_target_present_rate={metrics.get('topk_target_present_rate'):.4f} "
          f"topk_target_rv_argmax_rate={metrics.get('topk_target_rv_argmax_rate'):.4f} "
          f"(chance ~= {100.0/max(cfg.reverse_verifier.top_k,1):.1f}%)")
    print(f"[verify_rv_hardneg] full metrics -> {metrics}")
    print()
    print("[verify_rv_hardneg] COMPARE against the pre-fix numbers (scripts/diagnose_rv.py, "
          "fc+aep+rv/epoch020.pt, 20 full epochs): spread=0.0249 std=0.0081 "
          "topk_target_rv_argmax_rate=0.1603 (chance=10%). This run has fewer epochs, so "
          "look for DIRECTION (spread/argmax_rate improving) not an exact match.")


if __name__ == "__main__":
    main()
