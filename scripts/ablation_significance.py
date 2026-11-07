"""Paired significance test for the FC vs FC+AEP+RV ablation gap.

**The gap this closes.** Job 6453 reports FC R@1 = 35.996 and FC+AEP+RV
R@1 = 37.072 on the same 4181 CIRR val queries -- a +1.08pp gain, quoted in
the paper with NO p-value. Aggregate R@1 provably cannot supply one: a net
gain of 45 queries is equally consistent with help=45/harm=0 (p ~ 1e-13)
and with help=445/harm=400 (p ~ 0.14). Those are opposite conclusions. Only
the per-query discordant counts distinguish them, and only a paired test
can use them.

**Why this is cheap.** It needs no training and no RV pass -- only each
checkpoint's per-query base-retrieval hit vector. `evaluate_dataset`'s
`per_query_out` hook (evaluate.py:856) already exposes exactly the three
tensors `recall_at_k` consumes, and unlike the `cache_path` route it is NOT
gated on `rerank=True`, so it works for FC, which has no RV module at all.
Two forward evaluation passes, no gradient, no training.

**What it does NOT do.** It does not test individual module attribution.
The comparison is FC vs FC+AEP+RV -- two modules added together -- so a
significant result supports "the auxiliary components collectively improve
the system" and nothing finer. The FC+AEP row needed for per-module
attribution was never trained, and one seed per configuration could not
separate a module effect from epoch luck even if it had been.

**Reproducibility requirement.** Both evaluations must use identical
gallery, query set and ordering, or the pairing is meaningless. This script
shares ONE dataset object between both runs and asserts that the query-ID
and target vectors match element-for-element before any test is computed.
A mismatch aborts rather than reporting a number. Target equality alone is
NOT sufficient -- targets repeat across queries, so two different query
sequences can share a target sequence.

**Pre-specification, for manuscript validity.** McNemar at alpha=0.05 is
correct here only because FC vs FC+AEP+RV is the PRE-SPECIFIED primary
ablation hypothesis: these are the only two rows job 6453 trained under the
final corrected configuration, and the comparison was fixed by the
experiment's design before any per-query outcome was examined. It was not
chosen by scanning many checkpoint pairs and keeping the most favourable.
If further ablation comparisons are ever reported as confirmatory, the
family must be declared and a multiplicity correction applied (Holm, as
used elsewhere in this project); otherwise they are exploratory and must be
labelled as such.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from revua_cir.config import ReVUAConfig  # noqa: E402
from revua_cir.significance import (  # noqa: E402
    mcnemar_exact,
    mcnemar_midp,
    paired_bootstrap_ci,
)

# Bootstrap replicate count for reported CIs. 1000 is adequate as a
# diagnostic, but measured seed-to-seed drift at 1000 is ~0.024pp --
# exactly one query on a 4181-query set (100/4181 = 0.0239pp), so the
# endpoints can wobble by a full quantum of the metric. At 10000 the
# CI is stable across seeds and costs ~25s on CPU, which is free
# relative to a GPU eval pass. Paper numbers should not visibly depend
# on Monte-Carlo noise.
N_BOOTSTRAP = 10000



def _hit_vector(per_query: dict) -> tuple[list[int], list[int]]:
    """Per-query R@1 hit/miss plus the valid mask, from a per_query_out dict.

    Returns (hits, valid) as parallel 0/1 lists. A query whose target never
    resolved in the gallery is marked invalid and excluded downstream --
    the same queries `recall_at_k` excludes via `valid_mask`, so the
    denominator here matches the reported metric exactly. Getting this
    wrong is a real hazard: dividing a valid-filtered numerator by the
    unfiltered n produced a phantom 1.62pp-21pt mismatch earlier in this
    project.
    """
    ranked = per_query["ranked"]
    targets = per_query["targets"]
    valid = per_query["valid_mask"]
    hits, vmask = [], []
    for i in range(ranked.shape[0]):
        v = int(valid[i])
        vmask.append(v)
        hits.append(1 if (v and int(ranked[i][0]) == int(targets[i])) else 0)
    return hits, vmask


def compare(hits_a: list[int], hits_b: list[int], valid_a: list[int],
            valid_b: list[int], label_a: str, label_b: str) -> dict:
    """Paired McNemar + bootstrap on the queries valid under BOTH systems."""
    if len(hits_a) != len(hits_b):
        raise ValueError(
            "query counts differ (%d vs %d) -- the runs are not paired"
            % (len(hits_a), len(hits_b))
        )
    keep = [i for i in range(len(hits_a)) if valid_a[i] and valid_b[i]]
    a = [hits_a[i] for i in keep]
    b = [hits_b[i] for i in keep]
    n = len(keep)
    if n == 0:
        raise ValueError("no query is valid under both systems")

    help_ = sum(1 for x, y in zip(a, b) if not x and y)
    harm = sum(1 for x, y in zip(a, b) if x and not y)
    r1_a = 100.0 * sum(a) / n
    r1_b = 100.0 * sum(b) / n
    boot = paired_bootstrap_ci(a, b, n_replicates=N_BOOTSTRAP, seed=0)

    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_paired_valid": n,
        "n_total": len(hits_a),
        "n_dropped_invalid": len(hits_a) - n,
        "r1_a_pct": r1_a,
        "r1_b_pct": r1_b,
        "delta_pp": r1_b - r1_a,
        "help": help_,
        "harm": harm,
        "discordant": help_ + harm,
        "net": help_ - harm,
        "mcnemar_exact_p": mcnemar_exact(help_, harm),
        "mcnemar_midp_p": mcnemar_midp(help_, harm),
        "ci95_low_pp": boot["ci_low"],
        "ci95_high_pp": boot["ci_high"],
        "ci_straddles_zero": boot["straddles_zero"],
    }


def checkpoint_metadata(path: str) -> dict:
    """Training provenance for one checkpoint, read from its own payload.

    `save_checkpoint` stores `epoch`, `best_metric` and the full `config`
    (which carries `run.seed` and the early-stopping rule), so none of this
    has to be supplied by hand or trusted from a filename. Loaded onto CPU
    with the tensors discarded -- this is metadata only.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state.get("config") or {}

    def _dig(*keys, default=None):
        cur = cfg
        for k in keys:
            if isinstance(cur, dict):
                cur = cur.get(k)
            else:
                cur = getattr(cur, k, None)
            if cur is None:
                return default
        return cur

    es_monitor = _dig("early_stopping", "monitor", default="?")
    es_mode = _dig("early_stopping", "mode", default="?")
    top_k = _dig("run", "save_top_k", default="?")
    return {
        "path": path,
        "epoch": state.get("epoch"),
        "best_metric": state.get("best_metric"),
        "seed": _dig("run", "seed"),
        "backbone": _dig("backbone", "pretrained_name"),
        "lr": _dig("optim", "lr"),
        "backbone_lr_mult": _dig("optim", "backbone_lr_mult"),
        "patience": _dig("early_stopping", "patience"),
        "selection_rule": "top-%s on %s (%s)" % (top_k, es_monitor, es_mode),
    }


def _evaluate(base_cfg, checkpoint: str, combo: str, dataset, full_gallery,
              device) -> dict:
    """One base-retrieval pass, returning metrics plus the per_query_out dict.

    `--module-combo` is load-bearing, not cosmetic: `load_checkpoint` uses
    `strict=False` by design (so frozen backbone weights come from model
    construction rather than disk), which means a combo that does not match
    the checkpoint builds extra modules, loads them SILENTLY with random
    weights, and produces well-formed, worthless numbers.
    """
    from revua_cir.data.tokenizer import build_tokenizer
    from revua_cir.data.transforms import build_image_transform
    from revua_cir.evaluate import evaluate_dataset
    from revua_cir.models.revua_cir import ReVUACIR
    from revua_cir.utils import load_checkpoint

    cfg = base_cfg.apply_module_ablation(combo)
    tokenizer = build_tokenizer(cfg.backbone)
    transform = build_image_transform(cfg.data.image_size, train=False)

    model = ReVUACIR(cfg).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()

    per_query: dict = {}
    metrics = evaluate_dataset(
        model, dataset, cfg, tokenizer, transform, device,
        ks=(1, 5, 10, 50),
        use_map=False,
        use_group_recall=False,
        full_gallery_paths=full_gallery,
        max_queries=len(dataset),
        rerank=False,
        per_query_out=per_query,
    )
    return {"metrics": metrics, "per_query": per_query}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Paired significance test for the collective ablation gap.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-a", required=True, help="baseline, e.g. fc epoch003")
    ap.add_argument("--combo-a", required=True, help="e.g. fc")
    ap.add_argument("--checkpoint-b", required=True, help="e.g. fc_aep_rv epoch004")
    ap.add_argument("--combo-b", required=True, help="e.g. fc+aep+rv")
    ap.add_argument("--eval-split", default="val")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    import os
    from revua_cir.data.datasets import DATASET_REGISTRY
    from revua_cir.data.transforms import build_image_transform
    # `scripts/` has no __init__.py, so `scripts.evaluate_cirr` resolves only
    # as a namespace package and only when cwd happens to be the repo root.
    # _HERE is on sys.path (set at module top), so the flat import is the
    # one that works regardless of where the job was launched from.
    from evaluate_cirr import build_cirr_full_gallery

    cfg = ReVUAConfig.from_yaml(args.config)
    if args.data_root:
        cfg.data.root = args.data_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ONE dataset object, shared by both evaluations. This is what makes the
    # comparison paired: identical queries in identical order.
    eval_transform = build_image_transform(cfg.data.image_size, train=False)
    dataset = DATASET_REGISTRY["cirr"](
        root=os.path.join(cfg.data.root, "cirr"),
        split=args.eval_split, transform=eval_transform)
    full_gallery = build_cirr_full_gallery(dataset)

    meta_a = checkpoint_metadata(args.checkpoint_a)
    meta_b = checkpoint_metadata(args.checkpoint_b)

    print("=" * 74)
    print("Collective ablation significance test")
    print("=" * 74)
    print("  %-22s %-28s %-28s" % ("", "A", "B"))
    print("  %-22s %-28s %-28s" % ("combo", args.combo_a, args.combo_b))
    print("  %-22s %-28s %-28s"
          % ("checkpoint", Path(args.checkpoint_a).name, Path(args.checkpoint_b).name))
    for key, label in (("seed", "training seed"), ("epoch", "epoch"),
                       ("best_metric", "val monitor"), ("lr", "head lr"),
                       ("backbone_lr_mult", "backbone_lr_mult"),
                       ("patience", "patience"), ("backbone", "backbone")):
        print("  %-22s %-28s %-28s" % (label, meta_a.get(key), meta_b.get(key)))
    print("  %-22s %s" % ("selection rule A", meta_a.get("selection_rule")))
    print("  %-22s %s" % ("selection rule B", meta_b.get("selection_rule")))
    print("  %-22s %s" % ("config", args.config))
    print("  %-22s %s / %s" % ("split, queries", args.eval_split, len(dataset)))
    print("  %-22s %d images (full official)" % ("gallery", len(full_gallery)))
    print("  %-22s %s" % ("device", device))

    # --- comparability warnings ---------------------------------------
    # A paired test removes EVALUATION noise (same queries, same gallery).
    # It does NOT remove TRAINING noise. If the two checkpoints differ in
    # seed or were selected under different rules, part of the measured gap
    # is training variance, not the added modules.
    warnings = []
    if meta_a.get("seed") != meta_b.get("seed"):
        warnings.append("training seeds DIFFER (%s vs %s) -- the gap mixes the "
                        "module effect with seed variance"
                        % (meta_a.get("seed"), meta_b.get("seed")))
    if meta_a.get("selection_rule") != meta_b.get("selection_rule"):
        warnings.append("checkpoint-selection rules DIFFER (%s vs %s)"
                        % (meta_a.get("selection_rule"), meta_b.get("selection_rule")))
    for k, label in (("lr", "head lr"), ("backbone_lr_mult", "backbone_lr_mult"),
                     ("patience", "patience"), ("backbone", "backbone")):
        if meta_a.get(k) != meta_b.get(k):
            warnings.append("%s DIFFERS (%s vs %s)"
                            % (label, meta_a.get(k), meta_b.get(k)))
    if warnings:
        print("\n  " + "!" * 68)
        for w in warnings:
            print("  !! %s" % w)
        print("  " + "!" * 68)
    else:
        print("\n  [ok] seed, selection rule, LRs, patience and backbone all match")
    print("=" * 74)

    print("\n[1/2] evaluating A (%s) ..." % args.combo_a)
    ra = _evaluate(cfg, args.checkpoint_a, args.combo_a, dataset, full_gallery, device)
    print("      R@1 = %.4f" % ra["metrics"]["recall_at_1"])

    print("\n[2/2] evaluating B (%s) ..." % args.combo_b)
    rb = _evaluate(cfg, args.checkpoint_b, args.combo_b, dataset, full_gallery, device)
    print("      R@1 = %.4f" % rb["metrics"]["recall_at_1"])

    # --- pairing guard -------------------------------------------------
    # Target equality alone is NOT proof of identical query order: targets
    # repeat across queries, so two different query sequences can share a
    # target sequence. Both runs here share one dataset object and
    # evaluate_dataset iterates Subset(dataset, range(n)) with shuffle=False
    # and num_workers=0, so reordering is structurally impossible -- but the
    # guard should prove what it claims rather than rely on that invariant
    # holding forever. Query IDs (CIRR `pairid`) close the loophole.
    ta = ra["per_query"]["targets"]
    tb = rb["per_query"]["targets"]
    if ta.shape != tb.shape or not torch.equal(ta, tb):
        raise SystemExit(
            "ABORT: the two runs did not see the same target vector. The "
            "pairing would be invalid, so no test is reported. Check that "
            "both used the same dataset/split/gallery."
        )

    qa = ra["per_query"].get("query_ids")
    qb = rb["per_query"].get("query_ids")
    if not qa or not qb:
        raise SystemExit(
            "ABORT: evaluate_dataset did not return query_ids. This build "
            "cannot prove the two runs saw the same queries in the same "
            "order, and target equality alone is not sufficient. Update "
            "revua_cir/evaluate.py to populate per_query_out['query_ids']."
        )
    if len(qa) != len(qb) or qa != qb:
        first = next((i for i, (x, y) in enumerate(zip(qa, qb)) if x != y), None)
        raise SystemExit(
            "ABORT: the two runs saw DIFFERENT queries or a different query "
            "order (first divergence at index %s: %r vs %r). The target "
            "vectors matched, which is exactly the silent failure this check "
            "exists to catch. No test is reported."
            % (first, qa[first] if first is not None else None,
               qb[first] if first is not None else None)
        )
    n_unique = len(set(qa))
    if n_unique != len(qa):
        print("  [warn] query IDs are not unique (%d unique of %d) -- the "
              "pairing is still positional and valid, but the dataset "
              "repeats identifiers" % (n_unique, len(qa)))
    print("\n  [ok] query IDs identical: YES (%d queries, %d unique)"
          % (len(qa), n_unique))
    print("  [ok] target IDs identical: YES")

    hits_a, valid_a = _hit_vector(ra["per_query"])
    hits_b, valid_b = _hit_vector(rb["per_query"])

    # Self-check: the per-query hit vector must reproduce the reported R@1.
    for name, hits, valid, res in (
        (args.combo_a, hits_a, valid_a, ra), (args.combo_b, hits_b, valid_b, rb)
    ):
        denom = sum(valid)
        recomputed = 100.0 * sum(hits) / denom if denom else 0.0
        reported = res["metrics"]["recall_at_1"]
        # Tolerance rationale: one query is 100/4181 = 0.0239pp, so 0.01pp
        # is 0.42 of a single query -- the check cannot admit an off-by-one
        # error. Tightening further (e.g. 0.005pp = 0.21 queries) would not
        # detect anything this does not already catch; the only strictly
        # better check is an integer hit-count comparison, which
        # evaluate_dataset does not currently expose.
        if abs(recomputed - reported) > 0.01:
            raise SystemExit(
                "ABORT: %s per-query R@1 (%.4f, denominator=%d) disagrees with "
                "the reported %.4f. The hit vector does not match the metric, "
                "so any test built on it would be wrong."
                % (name, recomputed, denom, reported)
            )
        print("  [ok] %-12s per-query R@1 %.4f reproduces reported %.4f "
              "(denominator %d)" % (name, recomputed, reported, denom))

    res = compare(hits_a, hits_b, valid_a, valid_b, args.combo_a, args.combo_b)

    print("\n" + "-" * 74)
    print("PAIRED RESULT")
    print("-" * 74)
    print("  queries paired & valid : %d  (dropped %d unresolved)"
          % (res["n_paired_valid"], res["n_dropped_invalid"]))
    print("  %-14s R@1 : %.4f" % (res["label_a"], res["r1_a_pct"]))
    print("  %-14s R@1 : %.4f" % (res["label_b"], res["r1_b_pct"]))
    print("  delta                  : %+.4f pp" % res["delta_pp"])
    print("  help / harm            : %d / %d  (discordant %d, net %+d)"
          % (res["help"], res["harm"], res["discordant"], res["net"]))
    print("  McNemar exact p        : %.6g" % res["mcnemar_exact_p"])
    print("  McNemar mid-p          : %.6g" % res["mcnemar_midp_p"])
    print("  95%% bootstrap CI       : [%+.3f, %+.3f] pp%s"
          % (res["ci95_low_pp"], res["ci95_high_pp"],
             "  (INCLUDES zero)" if res["ci_straddles_zero"] else ""))
    verdict = ("SIGNIFICANT at alpha=0.05" if res["mcnemar_exact_p"] < 0.05
               else "NOT significant at alpha=0.05")
    print("  verdict                : %s" % verdict)
    print("\n  " + "-" * 70)
    print("  SCOPE -- what this p-value does and does not mean")
    print("  " + "-" * 70)
    print("  DOES: paired evidence that these two checkpoints produce")
    print("        different R@1 outcomes on the same CIRR val queries.")
    print("        This is EVALUATION uncertainty, handled correctly by")
    print("        McNemar + the paired bootstrap.")
    print("  DOES NOT: estimate TRAINING uncertainty. Both arms are single")
    print("        checkpoints from one seed each, so this cannot show that")
    print("        AEP+RV has an effect independent of training seed. That")
    print("        needs independent training seeds (measured cost: ~6.0")
    print("        GPU-hours per seed).")
    print("  DOES NOT: attribute anything to AEP or RV individually. Two")
    print("        modules were added together; this estimates a collective")
    print("        configuration difference, not an independent effect of")
    print("        either module.")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"comparison": res,
                       "checkpoint_a": meta_a,
                       "checkpoint_b": meta_b,
                       "comparability_warnings": warnings,
                       "config": args.config,
                       "eval_split": args.eval_split,
                       "n_queries": len(dataset),
                       "gallery_size": len(full_gallery),
                       "scope": ("Paired evaluation comparison of two single-seed "
                                 "checkpoints. Estimates a collective configuration "
                                 "difference under evaluation uncertainty only; does "
                                 "not estimate training-seed variance and does not "
                                 "attribute effects to AEP or RV individually."),
                       "metrics_a": ra["metrics"],
                       "metrics_b": rb["metrics"]}, fh, indent=2)
        print("\nwrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
