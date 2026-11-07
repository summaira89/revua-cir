"""
Efficient hyperparameter search (user's explicit request: "add efficient
algorithms to get best hyperparameters").

Uses Optuna with:
  * a TPE sampler (Bayesian, sample-efficient vs. grid/random search),
  * a Hyperband/ASHA-style pruner (`optuna.pruners.HyperbandPruner`) that
    kills unpromising trials early using the per-epoch validation metric
    reported mid-trial, instead of always paying for `max_epochs_per_trial`
    full epochs on every trial,
  * a cheap-proxy objective: each trial trains on a fixed-size random
    subset of the training data for a small number of epochs on a held-out
    slice of the validation set (both configurable) -- this is the standard
    "search cheap, retrain expensive" HPO pattern, since running full
    30-epoch trials for a 20+ trial search is not compute-rational.

Search space covers exactly the knobs Section 4/8 of the paper framework
identifies as consequential: optimizer (lr, weight decay, backbone LR
multiplier, warmup), the three connective-loss weights (lambda1-3, Section
4.5 / Ablation #2), fusion depth/width, and batch size.

Usage:
    python -m revua_cir.hpo --config configs/default.yaml --n-trials 30 \
        --data-root /path/to/datasets --study-name revua_cir_search
"""
from __future__ import annotations

import argparse
import copy
import os
import random

import optuna
import torch
from torch.utils.data import DataLoader, Subset

from revua_cir.config import ReVUAConfig
from revua_cir.data.collate import build_collate_fn
from revua_cir.data.tokenizer import build_tokenizer
from revua_cir.data.transforms import build_image_transform
from revua_cir.evaluate import evaluate_dataset
from revua_cir.models.revua_cir import ReVUACIR
from revua_cir.train import build_optimizer, build_scheduler, build_train_dataset, build_val_dataset
from revua_cir.utils import get_logger, set_seed

logger = get_logger("revua_cir.hpo")


def suggest_config(trial: optuna.Trial, base_cfg: ReVUAConfig) -> ReVUAConfig:
    cfg = copy.deepcopy(base_cfg)
    cfg.optim.lr = trial.suggest_float("lr", 1e-6, 5e-4, log=True)
    cfg.optim.backbone_lr_mult = trial.suggest_float("backbone_lr_mult", 0.01, 1.0, log=True)
    cfg.optim.weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.5, log=True)
    cfg.optim.warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.15)
    cfg.data.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    cfg.loss.lambda_disentangle = trial.suggest_float("lambda_disentangle", 0.0, 1.0)
    cfg.loss.lambda_reverse = trial.suggest_float("lambda_reverse", 0.0, 1.0)
    cfg.loss.lambda_uncertainty = trial.suggest_float("lambda_uncertainty", 0.0, 1.0)

    cfg.fusion.num_layers = trial.suggest_int("fusion_num_layers", 1, 4)
    cfg.fusion.num_heads = trial.suggest_categorical("fusion_num_heads", [4, 8, 16])
    cfg.fusion.dropout = trial.suggest_float("fusion_dropout", 0.0, 0.3)
    # This is the parameter that actually controls the contrastive logit scale
    # (symmetric_info_nce receives model.composer.temperature, which is seeded
    # from this value, not a separate loss.retrieval_temperature -- that field
    # was declared but never wired into the loss and has been removed).
    cfg.fusion.init_temperature = trial.suggest_float("init_temperature", 0.01, 0.2)
    return cfg


def run_trial(trial: optuna.Trial, base_cfg: ReVUAConfig, max_epochs_per_trial: int, train_subset_size: int,
              val_subset_size: int, device: torch.device) -> float:
    cfg = suggest_config(trial, base_cfg)
    set_seed(cfg.run.seed + trial.number)

    train_transform = build_image_transform(cfg.data.image_size, train=True)
    eval_transform = build_image_transform(cfg.data.image_size, train=False)
    tokenizer = build_tokenizer(cfg.backbone)
    collate_fn = build_collate_fn(tokenizer)

    full_train = build_train_dataset(cfg, train_transform)
    full_val = build_val_dataset(cfg, eval_transform)

    train_idx = random.sample(range(len(full_train)), min(train_subset_size, len(full_train)))
    train_subset = Subset(full_train, train_idx)
    train_loader = DataLoader(train_subset, batch_size=cfg.data.batch_size, shuffle=True, drop_last=True,
                               num_workers=cfg.data.num_workers, collate_fn=collate_fn)

    model = ReVUACIR(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader) * max_epochs_per_trial)

    best_metric = 0.0
    for epoch in range(max_epochs_per_trial):
        model.train()
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch)
            loss = out["losses"]["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip_norm)
            optimizer.step()
            scheduler.step()

        metrics = evaluate_dataset(model, full_val, cfg, tokenizer, eval_transform, device,
                                    max_queries=val_subset_size)
        metric = metrics.get("recall_at_10", metrics.get("recall_at_5", 0.0))
        best_metric = max(best_metric, metric)

        trial.report(metric, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_metric


def run_search(base_cfg: ReVUAConfig, n_trials: int = 30, max_epochs_per_trial: int = 3,
                train_subset_size: int = 5000, val_subset_size: int = 500,
                study_name: str = "revua_cir_search", storage: str = None) -> optuna.Study:
    device = torch.device(base_cfg.run.device if torch.cuda.is_available() else "cpu")

    sampler = optuna.samplers.TPESampler(seed=base_cfg.run.seed, multivariate=True, group=True)
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=max_epochs_per_trial, reduction_factor=3)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                                 study_name=study_name, storage=storage, load_if_exists=True)

    study.optimize(
        lambda trial: run_trial(trial, base_cfg, max_epochs_per_trial, train_subset_size, val_subset_size, device),
        n_trials=n_trials, show_progress_bar=True,
    )

    logger.info("Best trial: value=%.4f params=%s", study.best_value, study.best_params)
    return study


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--max-epochs-per-trial", type=int, default=3)
    p.add_argument("--train-subset-size", type=int, default=5000)
    p.add_argument("--val-subset-size", type=int, default=500)
    p.add_argument("--study-name", type=str, default="revua_cir_search")
    p.add_argument("--storage", type=str, default=None, help="e.g. sqlite:///hpo.db for a resumable/parallel study")
    p.add_argument("--output-config", type=str, default="configs/best_from_hpo.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    base_cfg = ReVUAConfig.from_yaml(args.config) if args.config else ReVUAConfig()
    if args.data_root:
        base_cfg.data.root = args.data_root

    study = run_search(base_cfg, args.n_trials, args.max_epochs_per_trial, args.train_subset_size,
                        args.val_subset_size, args.study_name, args.storage)

    best_cfg = suggest_config(optuna.trial.FixedTrial(study.best_params), base_cfg)
    os.makedirs(os.path.dirname(args.output_config) or ".", exist_ok=True)
    best_cfg.to_yaml(args.output_config)
    logger.info("Best config written to %s", args.output_config)


if __name__ == "__main__":
    main()
