"""
Section 8 / Ablation #1 -- the progressive module ablation the paper framework
calls "the single most important table in the paper".

Trains FC-only, FC+AEP, FC+AEP+RV, and the full FC+AEP+RV+UG model from the
*same* base config and seed, evaluates every one of them through one shared
evaluation path, and writes a summary CSV/markdown table.

    python scripts/run_ablation_matrix.py --config configs/fc_cirr_only.yaml \
        --output-dir ./runs/ablation

Design notes (each of these was a bug in the first version of this script):

* **Evaluate the *best* checkpoint, not the returned model.** `train()` returns
  the raw model at the final epoch, but validation and early stopping score the
  EMA shadow, and the final epoch is usually not the best one. Selecting on the
  metric recorded in each checkpoint keeps the table consistent with the
  early-stopping criterion that was actually used.
* **Never evaluate on CIRCO `test`.** That split ships without `gt_img_ids`
  (labels are held on the authors' server), so mAP against it is meaningless.
  Use `val` (220 queries, avg 4.16 GT each).
* **One evaluation path for every row.** The comparison is a *difference*
  between rows; if rows were scored by different code the deltas would be
  silently invalid.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
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

# Single source of truth is ReVUAConfig.VALID_MODULE_ABLATIONS (config.py) --
# apply_module_ablation() itself now enforces this list, rejecting anything
# else (in particular "fc+ug", a deliberately-unsupported degenerate combo --
# see that constant's own comment). Referenced here, not duplicated, so this
# script's default can never silently drift from what's actually supported.
#
# The default is the progressive ladder only. "fc+aep+rv+noinv" (2026-10-11)
# is a valid combo but a branch, not a rung, and costs a full training run;
# it must be requested explicitly with --combos so that re-running an old
# command that relied on the default does not start an extra paid job.
MODULE_COMBOS = [c for c in ReVUAConfig.VALID_MODULE_ABLATIONS
                 if "noinv" not in c.split("+")]


def best_checkpoint(run_dir: str):
    """Pick the checkpoint with the best recorded monitor metric.

    `TopKCheckpointKeeper` leaves the top-k files on disk and each one stores the
    `best_metric` it was saved with, so the best surviving file is authoritative
    -- more reliable than re-parsing the log or trusting the returned model.
    """
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
    p.add_argument("--data-root", type=str, default=None,
                   help="overrides data.root from the config")
    p.add_argument("--output-dir", type=str, default="./runs/ablation")
    p.add_argument("--eval-dataset", type=str, default="cirr",
                   choices=["cirr", "fashioniq", "circo"],
                   help="held-out set every row is scored on")
    p.add_argument("--eval-split", type=str, default="val")
    p.add_argument("--max-queries", type=int, default=None,
                   help="cap queries for a fast smoke run. DEFAULT CHANGED to None (= all "
                        "queries) on 2026-09-13: it used to default to 2000, which silently "
                        "evaluated under half of CIRR val (4181) and produced R@1 figures that "
                        "could NOT be placed in the same table as the headline full-val numbers.")
    p.add_argument("--full-gallery", action="store_true",
                   help="evaluate against the FULL official CIRR gallery (every image in "
                        "split.rc2.<split>.json, 2297 on val) instead of evaluate.py's reduced "
                        "default gallery (the union of this split's own ref+target images). "
                        "REQUIRED for numbers comparable to scripts/evaluate_cirr.py's -- the two "
                        "galleries differ (2297 vs ~2265) and mixing them across a results table "
                        "is exactly the inconsistency a reviewer would catch.")
    p.add_argument("--combos", nargs="*", default=MODULE_COMBOS)
    p.add_argument("--skip-trained", action="store_true",
                   help="reuse an existing run dir instead of retraining it")
    p.add_argument("--seed", type=int, default=None,
                   help="override run.seed of the config for every combo (seed study, added "
                        "2026-10-24); default keeps the config's seed (42 for the published runs)")
    args = p.parse_args()

    if args.eval_dataset == "circo" and args.eval_split == "test":
        raise SystemExit(
            "CIRCO 'test' has no public ground truth (gt_img_ids is absent; labels live on "
            "the authors' server). Use --eval-split val.")

    base_cfg = ReVUAConfig.from_yaml(args.config)
    if args.data_root:
        base_cfg.data.root = args.data_root
    if args.seed is not None:
        base_cfg.run.seed = args.seed
    print(f"[matrix] training seed {base_cfg.run.seed}")
    os.makedirs(args.output_dir, exist_ok=True)

    eval_transform = build_image_transform(base_cfg.data.image_size, train=False)
    tokenizer = build_tokenizer(base_cfg.backbone)
    eval_root = os.path.join(base_cfg.data.root, args.eval_dataset)
    eval_dataset = DATASET_REGISTRY[args.eval_dataset](
        root=eval_root, split=args.eval_split, transform=eval_transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[matrix] eval on {args.eval_dataset}/{args.eval_split}: {len(eval_dataset)} triplets")

    # Build the official gallery ONCE, outside the combo loop, so every row
    # of the table is scored against the IDENTICAL candidate pool. Building
    # it per-combo would risk silently different galleries across rows,
    # which is the kind of inconsistency that invalidates a whole table.
    full_gallery = None
    if args.full_gallery:
        if args.eval_dataset != "cirr":
            raise SystemExit(f"[matrix] --full-gallery is implemented for CIRR only, "
                              f"got --eval-dataset {args.eval_dataset}")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from evaluate_cirr import build_cirr_full_gallery
        full_gallery = build_cirr_full_gallery(eval_dataset)
        print(f"[matrix] FULL official gallery: {len(full_gallery)} images "
              f"(matches scripts/evaluate_cirr.py, so these numbers are directly comparable "
              f"to the headline full-val result)")
    else:
        print("[matrix] WARNING: using evaluate.py's REDUCED default gallery. These R@K values "
              "are NOT comparable to scripts/evaluate_cirr.py's full-gallery numbers. Pass "
              "--full-gallery for a publication table.")

    rows = []
    for combo in args.combos:
        run_dir = os.path.join(args.output_dir, combo.replace("+", "_"))
        cfg = base_cfg.apply_module_ablation(combo)
        cfg.run.output_dir = run_dir
        cfg.run.experiment_name = f"ablation_{combo}"

        # Validate any reusable checkpoint before trusting it (added
        # 2026-09-15). `--skip-trained` used to accept the mere EXISTENCE
        # of an epoch*.pt, which is unsafe in two real ways seen on job
        # 6439: (1) a checkpoint truncated by a disk-full torch.save --
        # 2.5GB where a real one is 5.6GB -- and (2) a combo that only
        # reached epoch 0 before crashing, which would then be evaluated
        # against another combo's fully-converged model and silently
        # produce a meaningless ablation row. Both are worse than
        # retraining, because both yield a NUMBER rather than an error.
        existing = sorted(glob.glob(os.path.join(run_dir, "epoch*.pt"))) if os.path.isdir(run_dir) else []
        usable = []
        for ck in existing:
            size_gb = os.path.getsize(ck) / 1e9
            try:
                torch.load(ck, map_location="cpu", weights_only=False)
                usable.append(ck)
            except Exception as e:
                print(f"[matrix] {combo}: IGNORING unreadable checkpoint "
                      f"{os.path.basename(ck)} ({size_gb:.2f}GB): {type(e).__name__}: {e}")
        if usable and args.skip_trained:
            print(f"[matrix] {combo}: reusing existing {run_dir} "
                  f"({len(usable)} valid checkpoint(s): "
                  f"{', '.join(os.path.basename(c) for c in usable)})")
        else:
            if existing and not usable:
                print(f"[matrix] {combo}: {len(existing)} checkpoint(s) present but NONE loadable "
                      f"-- retraining from scratch rather than evaluating a corrupt model")
            print(f"[matrix] {combo}: training -> {run_dir}")
            train(cfg)

        ckpt, ckpt_metric = best_checkpoint(run_dir)
        model = ReVUACIR(cfg).to(device)
        if ckpt:
            load_checkpoint(ckpt, model, map_location=device)
            print(f"[matrix] {combo}: loaded {os.path.basename(ckpt)} (val monitor={ckpt_metric:.4f})")
        else:
            print(f"[matrix] {combo}: WARNING no checkpoint found, evaluating freshly-built model")
        model.eval()

        n_eval = len(eval_dataset) if args.max_queries is None else min(len(eval_dataset),
                                                                         args.max_queries)
        common = dict(use_map=(args.eval_dataset == "circo"),
                      use_group_recall=(args.eval_dataset == "cirr"),
                      max_queries=n_eval)
        if full_gallery is not None:
            common["full_gallery_paths"] = full_gallery

        metrics = evaluate_dataset(model, eval_dataset, cfg, tokenizer,
                                   eval_transform, device, rerank=False, **common)
        row = {"module_combo": combo, "checkpoint": os.path.basename(ckpt) if ckpt else "",
               "val_monitor": ckpt_metric, **metrics}

        # Second pass with inference-time RV/UG reranking. Only meaningful when
        # the Reverse Verifier exists; for fc/fc+aep the two passes are identical
        # by construction, so we skip the work and mirror the values.
        if model.rv is not None:
            rr = evaluate_dataset(model, eval_dataset, cfg, tokenizer,
                                  eval_transform, device, rerank=True, **common)
            for k, v in rr.items():
                row[f"rr_{k}"] = v
        else:
            for k, v in metrics.items():
                row[f"rr_{k}"] = v

        rows.append(row)
        print(f"[matrix] {combo}: {row}")

        # Write after every row so a crash on row 4 does not lose rows 1-3.
        _write(args.output_dir, rows)

    _write(args.output_dir, rows)
    print(f"[matrix] done -> {os.path.join(args.output_dir, 'ablation_summary.csv')}")


def _write(output_dir: str, rows):
    if not rows:
        return
    fieldnames = ["module_combo", "checkpoint", "val_monitor"] + sorted(
        {k for r in rows for k in r.keys()} - {"module_combo", "checkpoint", "val_monitor"})
    with open(os.path.join(output_dir, "ablation_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(output_dir, "ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # Two tables. Table 1 is the progressive ablation (deltas vs the fc row);
    # Table 2 isolates the inference-time reranking gain, which is the claim
    # Section 3 actually makes and which Table 1 alone cannot support.
    # Renamed 2026-09-09 (code-audit finding #5): `mean_query_latency_sec`
    # (evaluate.py's legacy key, kept only for old-log compatibility) is the
    # BASE retrieval pass only -- the `metrics` row here always comes from
    # `evaluate_dataset(..., rerank=False)` above. Reporting it under a
    # generic "latency" header in a table that also has reranking-enabled
    # combos invites reading it as the full ReVUA-CIR pipeline latency,
    # which it structurally cannot be (rerank=False never runs RV/UG).
    # `base_retrieval_latency_sec` is the honestly-named field
    # evaluate_dataset() now also populates alongside the legacy key; RV/UG
    # reranking overhead and total pipeline latency belong in Table 2 below
    # (the rerank-enabled table), not here.
    keys = [k for k in ("recall_at_1", "recall_at_5", "recall_at_10", "recall_at_50",
                        "group_recall_at_1", "group_recall_at_2", "group_recall_at_3",
                        "map_at_5", "map_at_10", "ece", "base_retrieval_latency_sec")
            if any(k in r for r in rows)]
    base = rows[0]
    lines = ["## Table 1 - progressive module ablation (no inference reranking)",
             "",
             "Deltas are against the `fc` row. These isolate what each module's",
             "*joint-loss* term contributes to the trained composer.",
             "",
             "| model | " + " | ".join(keys) + " |",
             "|---" * (len(keys) + 1) + "|"]
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k)
            if v is None:
                cells.append("--")
            elif r is base or base.get(k) is None:
                cells.append(f"{v:.2f}")
            else:
                cells.append(f"{v:.2f} ({v - base[k]:+.2f})")
        lines.append(f"| {r['module_combo']} | " + " | ".join(cells) + " |")

    rr_keys = [k for k in ("recall_at_1", "recall_at_5", "group_recall_at_1",
                           "group_recall_at_2", "group_recall_at_3")
               if any(f"rr_{k}" in r for r in rows)]
    lines += ["", "## Table 2 - inference-time RV/UG reranking gain", "",
              "`base` = ranking by embedding similarity; `rerank` = after the",
              "Reverse Verifier / Uncertainty Gate rescale the top-k. NOTE: R@10 and",
              "R@50 are omitted because reranking permutes only within the top-k",
              "(k=10), so they are invariant to it by construction.",
              "",
              "| model | " + " | ".join(f"{k} base / rerank / gain" for k in rr_keys) + " |",
              "|---" * (len(rr_keys) + 1) + "|"]
    for r in rows:
        cells = []
        for k in rr_keys:
            b, a = r.get(k), r.get(f"rr_{k}")
            cells.append("--" if b is None or a is None else f"{b:.2f} / {a:.2f} / {a - b:+.2f}")
        lines.append(f"| {r['module_combo']} | " + " | ".join(cells) + " |")

    # Latency breakdown (added 2026-09-09, code-audit finding #5): reported
    # separately from the recall/mAP "base / rerank / gain" columns above --
    # latency isn't a "gain" metric (RV/UG reranking is strictly SLOWER,
    # never faster, than base retrieval alone), so folding it into the same
    # base/rerank/delta format would misleadingly frame reranking overhead
    # as if it could ever be a negative-is-good number. `rr_rv_reranking_
    # latency_sec`/`rr_total_pipeline_latency_sec` only exist on rows where
    # rerank actually ran with RV enabled (evaluate_dataset only populates
    # them under rerank=True) -- rows without RV (fc/fc+aep) show `--`.
    if any("rr_rv_reranking_latency_sec" in r for r in rows):
        lines += ["", "## Table 3 - inference latency breakdown", "",
                  "`base_retrieval` = query encode + score + topk only (no RV/UG). "
                  "`rv_reranking` = RV (+UG, when enabled) candidate-rescoring overhead "
                  "on top of base retrieval. `total_pipeline` = base_retrieval + "
                  "rv_reranking -- this is the number that should be cited as "
                  "'ReVUA-CIR inference latency', not base_retrieval alone.",
                  "",
                  "| model | base_retrieval_sec | rv_reranking_sec | total_pipeline_sec |",
                  "|---|---|---|---|"]
        for r in rows:
            base_lat = r.get("base_retrieval_latency_sec")
            rr_lat = r.get("rr_rv_reranking_latency_sec")
            total_lat = r.get("rr_total_pipeline_latency_sec")
            cells = [f"{base_lat:.4f}" if base_lat is not None else "--",
                     f"{rr_lat:.4f}" if rr_lat is not None else "--",
                     f"{total_lat:.4f}" if total_lat is not None else "--"]
            lines.append(f"| {r['module_combo']} | " + " | ".join(cells) + " |")

    with open(os.path.join(output_dir, "ablation_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
