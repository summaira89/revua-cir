"""
Standalone RV discrimination diagnostic -- does NOT touch run_ablation_matrix.py's
summary files. Loads existing checkpoints for the given combos and evaluates
`verification_auroc` (does RV actually score the true target higher
than a known-wrong candidate, pooled into a true ROC-AUC across queries? --
renamed 2026-09-09 from `verification_discrimination_auc`, which was actually
a per-query matched-pair win rate, not pooled AUROC; see
metrics.verification_auroc's docstring) alongside the usual
recall/verification_agreement_rate numbers, on a small query subset for a
fast turnaround.

    python scripts/diagnose_rv.py \
        --config configs/stage1_finetune_full_unfreeze_gradcache.yaml \
        --output-dir runs/ablation_partial_unfreeze \
        --combos fc+aep+rv fc+aep+rv+ug \
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
                   help="ablation matrix output dir containing <combo>/ subdirs with epoch*.pt")
    p.add_argument("--combos", nargs="*", default=["fc+aep+rv", "fc+aep+rv+ug"])
    p.add_argument("--eval-dataset", type=str, default="cirr", choices=["cirr", "fashioniq", "circo"])
    p.add_argument("--eval-split", type=str, default="val")
    p.add_argument("--max-queries", type=int, default=300)
    args = p.parse_args()

    base_cfg = ReVUAConfig.from_yaml(args.config)
    eval_transform = build_image_transform(base_cfg.data.image_size, train=False)
    tokenizer = build_tokenizer(base_cfg.backbone)
    eval_root = os.path.join(base_cfg.data.root, args.eval_dataset)
    eval_dataset = DATASET_REGISTRY[args.eval_dataset](
        root=eval_root, split=args.eval_split, transform=eval_transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diagnose_rv] eval on {args.eval_dataset}/{args.eval_split}: {len(eval_dataset)} triplets, "
          f"using {min(len(eval_dataset), args.max_queries)}")

    for combo in args.combos:
        run_dir = os.path.join(args.output_dir, combo.replace("+", "_"))
        cfg = base_cfg.apply_module_ablation(combo)
        cfg.run.output_dir = run_dir

        ckpt, ckpt_metric = best_checkpoint(run_dir)
        if not ckpt:
            print(f"[diagnose_rv] {combo}: NO CHECKPOINT FOUND in {run_dir}, skipping")
            continue

        model = ReVUACIR(cfg).to(device)
        load_checkpoint(ckpt, model, map_location=device)
        model.eval()
        print(f"[diagnose_rv] {combo}: loaded {os.path.basename(ckpt)} (val monitor={ckpt_metric:.4f})")

        n_eval = min(len(eval_dataset), args.max_queries)
        metrics = evaluate_dataset(
            model, eval_dataset, cfg, tokenizer, eval_transform, device,
            use_group_recall=(args.eval_dataset == "cirr"),
            max_queries=n_eval, rerank=False,
        )

        auc = metrics.get("verification_auroc")
        agreement = metrics.get("verification_agreement_rate")
        r1 = metrics.get("recall_at_1")
        spread = metrics.get("topk_verification_score_spread")
        std = metrics.get("topk_verification_score_std")
        present = metrics.get("topk_target_present_rate")
        argmax_rate = metrics.get("topk_target_rv_argmax_rate")
        print(f"[diagnose_rv] {combo}: recall_at_1={r1:.2f} "
              f"verification_agreement_rate={agreement:.4f} "
              f"verification_auroc={auc:.4f} "
              f"(0.5=chance, 1.0=perfect separation)")
        print(f"[diagnose_rv] {combo}: within-top-k spread={spread:.4f} std={std:.4f} "
              f"(near 0 = RV assigns ~identical scores within a query's own top-k, "
              f"which would explain why reranking can't reorder anything even "
              f"though the global discrimination AUC above is healthy)")
        if present is not None:
            print(f"[diagnose_rv] {combo}: true target present in top-k for {present*100:.1f}% of queries; "
                  f"when present, RV ranks it #1 among the top-k {argmax_rate*100:.1f}% of the time "
                  f"(chance level ~= {100.0/max(cfg.reverse_verifier.top_k,1):.1f}%)")
        print(f"[diagnose_rv] {combo}: full metrics -> {metrics}")


if __name__ == "__main__":
    main()
