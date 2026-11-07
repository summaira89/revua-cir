"""
Official full-gallery CIRR evaluation (README.md item B.6 / paper framework
Section 6): scores an already-trained checkpoint against CIRR's OFFICIAL,
full open-domain image gallery for a given split, reporting the numbers a
CIRR paper table actually reports -- R@1/R@5/R@10/R@50 and Subset R@1/R@2/R@3
-- rather than the reduced development gallery `evaluate_dataset()` builds by
default during training-time validation (`build_default_gallery`, the union
of just this split's own reference+target images -- fast for iterative
monitoring, but a query's true nearest neighbour from the FULL dev image set
can easily be excluded from that reduced pool, silently inflating recall
relative to what a real CIRR submission would score).

This is the CIRR analogue of `scripts/evaluate_circo.py` -- same shape of
fix (README.md's own "before publication" item), same reason it exists as a
separate script rather than a flag on `run_ablation_matrix.py`: the full
CIRR gallery for a split is not the training loop's per-batch gallery, and
building/encoding it is a genuinely separate, slower pass.

**`--eval-split train` (added 2026-09-12) is for FITTING HYPERPARAMETERS,
never for reporting results.** The motivating problem: selecting a fusion
weight (lambda) on half of CIRR val and reporting on the other half left
only ~2091 queries for the reported comparison -- and the verdict flipped
between p=0.037 and p=0.061 purely on which lambda the tuning half chose.
Fixing lambda a priori on a DIFFERENT dataset instead keeps all 4181 val
queries available for the reported comparison, at zero extra GPU cost.
CIRR train carries public ground truth (`target_hard`, read identically
to val -- CIRRDataset is fully generic over split), so it is a legitimate
source for that fitting.

Stated precisely (tightened 2026-09-12 after external review): the benefit
is that the evaluation sample is no longer divided between tuning and
reporting -- roughly 2091 independent queries become roughly 4181. That is
a statement about sample size, not a claim to have "fixed power". A priori
power calculations (which suggested ~0.49 vs ~0.78 for a +1.4pp effect)
are useful for PLANNING this experiment, but they rest on assumptions
about the true effect size and discordant-pair rate, and post-hoc observed
power should NOT be reported as a result. Report the effect estimate, its
95% CI, the paired test, and the sample size instead -- the CI carries the
precision information that a power figure only gestures at.

The checkpoint was of course TRAINED on this split, so numbers measured
here are optimistically biased and MUST NOT be reported as retrieval
results. The precise statement of what this protocol does and does not
guarantee (tightened 2026-09-12 after external review flagged the earlier
wording as too categorical): a lambda selected using train labels MAY fit
the training distribution rather than the general one, and nothing here
rules that out. What the protocol does guarantee is that this possibility
cannot CONTAMINATE the validation evaluation, provided lambda is frozen
before any validation result is inspected. The validation number then
evaluates the complete procedure -- fixed checkpoint plus train-selected
fusion weight -- as a single pre-specified system. State the protocol
explicitly in any write-up: using a training split at all invites
(reasonable) reviewer suspicion, and the defence is the freezing
discipline, not a claim that the bias is harmless.

**Which splits are locally scoreable.** CIRR's official repo
(github.com/Cuberick-Orion/CIRR) ships ground truth for `train` and `val`
splits only (the file the repo and this codebase's own `cap.rc2.val.json`
both call "val" is what the CIRR paper text calls the "dev" split -- same
data, two names in circulation). `test1` has NO public ground truth: its
`cap.rc2.test1.json` entries have no `target_hard`/known-correct answer --
official test1 numbers only come from submitting a ranking file to CIRR's
CodaLab evaluation server. This script evaluates `val` (equivalently "dev")
by default and refuses `test1` with an explicit error rather than silently
producing a meaningless number against absent ground truth -- exactly the
same policy `evaluate_circo.py` already applies to CIRCO's `test` split.
`--dump-test1-predictions <experiment_name>` writes TWO files matching the
shape CIRR's CodaLab evaluation server documents for a submission
(code-audit document finding #10):
  * `recall_<experiment_name>.json` -- `{"version": "rc2", "metric":
    "recall", "predictions": {pairid: [top-50 ranked image ids]}}`, ranked
    over the FULL gallery.
  * `subset_<experiment_name>.json` -- `{"version": "rc2", "metric":
    "recall_subset", "predictions": {pairid: [top-3 ranked image ids]}}`,
    ranked over ONLY the query's own 5 non-reference subset candidates (per
    `meta["group_members"]` minus `meta["reference_id"]`, the same official
    six-member-image-set semantics `evaluate.py`'s group-recall block uses
    -- see that module's docstring).
This still does NOT claim byte-exact certainty against CodaLab's current
live upload validator, since that has not been (and cannot be, from this
codebase alone) verified against the actual server -- the two-file
structure and field names follow the code-audit document's own
specification of the documented schema, but CHECK THE CURRENT SCHEMA AT THE
CIRR CODALAB PAGE before uploading, in case it has since changed.

**Subset (group) recall.** CIRR's `img_set.members` field in the official
captions file is the official six-member image set for each query,
INCLUDING the reference image itself -- `CIRRDataset` (revua_cir/data/
datasets.py) threads it through as `meta["group_members"]` plus a separate
`meta["reference_id"]`, so `evaluate.py`'s group-recall block can exclude
the reference and rank only the 5 non-reference candidates, exactly the
official subset protocol (fixed wording 2026-09-09, code-audit finding #2
-- this used to say "6-way distractor group", which undersold that the
reference is structurally excluded from ranking, not merely a 6th
candidate). `evaluate_dataset(use_group_recall=True)` computes this with no
extra plumbing here. `group_ks=(1,2,3)` matches what CIRR papers report
(any k>=5 saturates the 5-candidate subset).

    python scripts/evaluate_cirr.py \
        --config configs/stage1_finetune_full_unfreeze_gradcache.yaml \
        --checkpoint runs/ablation_partial_unfreeze_v2/fc_aep_rv_ug/epoch016.pt \
        --module-combo fc+aep+rv+ug
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from revua_cir.config import ReVUAConfig
from revua_cir.data.datasets import CIRRDataset, DATASET_REGISTRY
from revua_cir.data.tokenizer import build_tokenizer
from revua_cir.data.transforms import build_image_transform
from revua_cir.evaluate import encode_gallery_paths, evaluate_dataset
from revua_cir.models.revua_cir import ReVUACIR
from revua_cir.utils import load_checkpoint


def strict_preflight_check(dataset: CIRRDataset, gallery_paths: list[str], max_queries=None) -> None:
    """Fail loudly on ANY malformed/incomplete CIRR subset-protocol query
    before running the official evaluation (added 2026-09-09, code-audit
    finding #1). `evaluate_dataset()`'s own group-recall block is
    deliberately TOLERANT -- it excludes a malformed/incomplete query from
    the subset-recall denominator rather than crashing, which is the right
    behavior for training-time monitoring (a single corrupted annotation
    shouldn't kill a multi-hour run) but is too permissive for a number a
    paper table cites as official.

    In particular, `evaluate_dataset()`'s tolerant path has one edge case
    this preflight catches that a purely "did the candidates resolve"
    check would not: a query whose group has 6 members, whose reference is
    present, and whose 5 non-reference candidates all resolve in the
    gallery -- but where the TRUE TARGET is not actually one of those 5
    candidates (malformed/inconsistent annotation). `evaluate_dataset()`
    silently omits such a query from the metric (correct for a
    monitoring pass); official evaluation must not silently accept that.

    Requires, for EVERY query in the (possibly `max_queries`-truncated) split:
      - exactly 6 unique group members
      - the query's reference_id is among those 6 members
      - exactly 5 non-reference candidates remain after excluding it
      - the query's own target_id is among those 5 candidates
      - all 5 candidates resolve to a real file within `gallery_paths`
    Raises RuntimeError with the first failure found (and a count of how
    many queries fail overall) rather than stopping at the first one blind
    to the total scope of the problem.
    """
    path_to_idx = {p: i for i, p in enumerate(gallery_paths)}
    id_to_idx = {}
    for image_id in dataset.id_to_path:
        p = dataset._img_path(image_id)
        if p in path_to_idx:
            id_to_idx[str(image_id)] = path_to_idx[p]

    n = len(dataset.triplets) if max_queries is None else min(max_queries, len(dataset.triplets))
    failures = []
    for i in range(n):
        triplet = dataset.triplets[i]
        meta = triplet["meta"]
        pairid = meta.get("pairid", i)
        ref_id = str(meta.get("reference_id", ""))
        target_id = str(meta.get("target_id", ""))
        members = meta.get("group_members", [])
        all_member_ids = {str(m) for m in members}

        if len(all_member_ids) != 6:
            failures.append(f"pairid={pairid}: group has {len(all_member_ids)} unique members, expected 6")
            continue
        if ref_id not in all_member_ids:
            failures.append(f"pairid={pairid}: reference_id={ref_id!r} is not among the group's 6 members")
            continue
        candidate_ids = all_member_ids - {ref_id}
        if len(candidate_ids) != 5:
            failures.append(f"pairid={pairid}: {len(candidate_ids)} non-reference candidates after "
                             f"excluding the reference, expected 5")
            continue
        if target_id not in candidate_ids:
            failures.append(f"pairid={pairid}: target_id={target_id!r} is not among the 5 non-reference "
                             f"candidates {sorted(candidate_ids)}")
            continue
        unresolved = [c for c in candidate_ids if c not in id_to_idx]
        if unresolved:
            failures.append(f"pairid={pairid}: candidate(s) {sorted(unresolved)} do not resolve to a "
                             f"real file in the gallery")

    if failures:
        preview = "\n  ".join(failures[:10])
        raise RuntimeError(
            f"Strict official-evaluation preflight failed for {len(failures)}/{n} quer"
            f"{'y' if len(failures) == 1 else 'ies'} -- official CIRR subset evaluation requires "
            f"every query to have a valid six-member group (reference + 5 candidates, target among "
            f"them, all 5 candidates present in the gallery). First failures:\n  {preview}\n"
            f"Fix the annotation/gallery, or pass --no-strict to fall back to evaluate_dataset()'s "
            f"tolerant behavior (excludes malformed queries from the metric instead of failing --"
            f" NOT appropriate for numbers cited in a paper).")


def build_cirr_full_gallery(dataset: CIRRDataset) -> list[str]:
    """The full official image gallery for `dataset`'s split: every image
    listed in that split's `split.rc2.{split}.json` map (already loaded as
    `dataset.id_to_path` by CIRRDataset.__init__), resolved through the same
    `_img_path` fallback logic the dataset itself uses for ref/target images
    -- so a mirror's `dev_img/`-suffixed layout (see CIRRDataset's own
    docstring) is handled identically here, not re-derived.

    This is the ENTIRE dev/val image split (thousands of images), not the
    reduced `build_default_gallery` union of just this split's own
    reference+target images used during fast training-time validation.
    """
    paths = sorted({dataset._img_path(image_id) for image_id in dataset.id_to_path})
    if not paths:
        raise RuntimeError(
            "CIRR split map (split.rc2.<split>.json) resolved to zero image paths -- "
            "check --data-root and the split name.")
    # Fail hard on any unresolved image (added 2026-09-09, code-audit finding
    # #8): CIRRDataset._img_path falls back to returning a non-existent path
    # when no layout candidate matches (a deliberate design choice for
    # TRAINING-time validation, where safe_load_transform substitutes a gray
    # placeholder rather than crashing a long run over a handful of missing
    # files -- see safe_load_transform's own docstring). That tolerance is
    # wrong for OFFICIAL evaluation: a paper's reported R@K/Subset-R@K must
    # be computed against the real images, not silently against gray
    # placeholders for whatever fraction of the gallery failed to resolve.
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise RuntimeError(
            f"{len(missing)}/{len(paths)} CIRR gallery images did not resolve to a real file on "
            f"disk -- refusing to run an official evaluation against a gallery with silent gray-"
            f"placeholder substitutions. Check --data-root and the CIRR image layout (see "
            f"CIRRDataset._resolve_image_root's docstring for the layouts this repo knows how to "
            f"find, including the official img_raw/ release layout). First examples: {missing[:5]}")
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
    p.add_argument("--eval-split", type=str, default="val", choices=["val", "test1", "train"],
                   help="'train' added 2026-09-12 for HYPERPARAMETER FITTING ONLY (see below); "
                        "CIRR 'test1' has no public ground truth (ground-truth target withheld, "
                        "scored only by CIRR's CodaLab server) -- only 'val' (== the paper's 'dev' "
                        "split) is locally scoreable for R@K/Subset-R@K. Pass 'test1' only together "
                        "with --dump-test1-predictions; this script refuses to compute metrics for it.")
    p.add_argument("--max-queries", type=int, default=None,
                   help="cap queries for a fast smoke run; omit for the full split")
    p.add_argument("--rerank", action="store_true",
                   help="also run with RV/UG reranking and report both -- the difference IS the "
                        "reranking gain the paper's Section 6 ablation reports.")
    p.add_argument("--cache-path", type=str, default=None,
                   help="only used together with --rerank: appends one JSON line per query to this "
                        "path with the per-candidate base score and RV sub-scores (recon_cos, "
                        "invariant_cos, edit_cos, verification_score) already computed during "
                        "reranking, for OFFLINE fusion-rule experimentation (bounded score fusion, "
                        "a tiny cached-feature reranker, a help/harm trust gate) without a second "
                        "GPU pass per variant -- see evaluate_dataset()'s docstring for the schema.")
    p.add_argument("--dump-test1-predictions", type=str, default=None,
                   help="output DIRECTORY for --eval-split test1 submission files. Writes "
                        "recall_<experiment-name>.json (top-50, full gallery) and "
                        "subset_<experiment-name>.json (top-3, 5 non-reference subset candidates "
                        "only), each shaped {'version': 'rc2', 'metric': ..., 'predictions': "
                        "{pairid: [ranked_image_id, ...]}} per the code-audit document's "
                        "specification of CIRR's documented submission schema. NOT verified against "
                        "CodaLab's live upload validator -- check the current schema on the CodaLab "
                        "page before uploading.")
    p.add_argument("--experiment-name", type=str, default="revua_cir",
                   help="used to name the two --dump-test1-predictions output files "
                        "(recall_<name>.json / subset_<name>.json).")
    p.add_argument("--no-strict", action="store_true",
                   help="skip the strict official-evaluation preflight check (code-audit finding "
                        "#1) and fall back to evaluate_dataset()'s tolerant behavior, which silently "
                        "excludes malformed/incomplete subset-protocol queries from the metric "
                        "instead of failing. Default is strict (fail loudly) since this script's "
                        "whole purpose is producing publication-grade numbers -- only pass this for "
                        "a quick exploratory run against data you know is incomplete.")
    args = p.parse_args()

    if args.eval_split == "test1" and not args.dump_test1_predictions:
        raise SystemExit(
            "--eval-split test1 has no public ground truth: R@K/Subset-R@K cannot be computed "
            "locally. Pass --dump-test1-predictions <path> to get a ranking file instead, or use "
            "--eval-split val (the only locally-scoreable split) for real metrics.")

    cfg = ReVUAConfig.from_yaml(args.config)
    if args.data_root:
        cfg.data.root = args.data_root
    cfg = cfg.apply_module_ablation(args.module_combo)

    eval_transform = build_image_transform(cfg.data.image_size, train=False)
    tokenizer = build_tokenizer(cfg.backbone)
    cirr_root = os.path.join(cfg.data.root, "cirr")
    eval_dataset: CIRRDataset = DATASET_REGISTRY["cirr"](root=cirr_root, split=args.eval_split, transform=eval_transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[evaluate_cirr] CIRR {args.eval_split}: {len(eval_dataset)} queries")
    t0 = time.time()
    full_gallery = build_cirr_full_gallery(eval_dataset)
    print(f"[evaluate_cirr] full official gallery: {len(full_gallery)} images "
          f"(listed in {time.time() - t0:.1f}s)")

    if args.eval_split == "val" and not args.no_strict:
        t0 = time.time()
        strict_preflight_check(eval_dataset, full_gallery, args.max_queries)
        print(f"[evaluate_cirr] strict preflight check passed for "
              f"{args.max_queries or len(eval_dataset)} queries ({time.time() - t0:.1f}s)")

    model = ReVUACIR(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    print(f"[evaluate_cirr] loaded {args.checkpoint} (module_combo={args.module_combo})")

    if args.eval_split == "test1":
        _dump_test1_predictions(model, eval_dataset, cfg, tokenizer, eval_transform, device,
                                 full_gallery, args.dump_test1_predictions, args.experiment_name,
                                 args.max_queries)
        return

    n_eval = len(eval_dataset) if args.max_queries is None else min(args.max_queries, len(eval_dataset))
    common = dict(
        ks=(1, 5, 10, 50),
        use_map=False,
        use_group_recall=True,
        group_ks=(1, 2, 3),
        full_gallery_paths=full_gallery,
        max_queries=n_eval,
    )

    t0 = time.time()
    base = evaluate_dataset(model, eval_dataset, cfg, tokenizer, eval_transform, device, rerank=False, **common)
    print(f"[evaluate_cirr] base eval done in {time.time() - t0:.1f}s")
    print(f"[evaluate_cirr] BASE: R@1={base.get('recall_at_1'):.2f} R@5={base.get('recall_at_5'):.2f} "
          f"R@10={base.get('recall_at_10'):.2f} R@50={base.get('recall_at_50'):.2f} | "
          f"SubsetR@1={base.get('group_recall_at_1'):.2f} SubsetR@2={base.get('group_recall_at_2'):.2f} "
          f"SubsetR@3={base.get('group_recall_at_3'):.2f} "
          f"(complete groups: {base.get('group_queries_complete'):.0f}/{base.get('group_queries_total'):.0f})")
    print(f"[evaluate_cirr] full base metrics -> {base}")

    if args.rerank and model.rv is not None:
        t0 = time.time()
        rr = evaluate_dataset(model, eval_dataset, cfg, tokenizer, eval_transform, device, rerank=True,
                               cache_path=args.cache_path, **common)
        print(f"[evaluate_cirr] rerank eval done in {time.time() - t0:.1f}s")
        print(f"[evaluate_cirr] RERANK: R@1={rr.get('recall_at_1'):.2f} R@5={rr.get('recall_at_5'):.2f} "
              f"R@10={rr.get('recall_at_10'):.2f} R@50={rr.get('recall_at_50'):.2f} | "
              f"SubsetR@1={rr.get('group_recall_at_1'):.2f} SubsetR@2={rr.get('group_recall_at_2'):.2f} "
              f"SubsetR@3={rr.get('group_recall_at_3'):.2f}")
        for k in (1, 5, 10, 50):
            b, a = base.get(f"recall_at_{k}"), rr.get(f"recall_at_{k}")
            if b is not None and a is not None:
                print(f"[evaluate_cirr] GAIN R@{k}: {a - b:+.2f}")
        for k in (1, 2, 3):
            b, a = base.get(f"group_recall_at_{k}"), rr.get(f"group_recall_at_{k}")
            if b is not None and a is not None:
                print(f"[evaluate_cirr] GAIN SubsetR@{k}: {a - b:+.2f}")
        if "rv_reranking_latency_sec" in rr:
            print(f"[evaluate_cirr] LATENCY: base_retrieval={rr['base_retrieval_latency_sec']*1000:.2f}ms "
                  f"rv_reranking={rr['rv_reranking_latency_sec']*1000:.2f}ms "
                  f"total_pipeline={rr['total_pipeline_latency_sec']*1000:.2f}ms per query "
                  f"(report total_pipeline, not just base_retrieval, as 'ReVUA-CIR inference latency')")
        print(f"[evaluate_cirr] full rerank metrics -> {rr}")


@torch.no_grad()
def _dump_test1_predictions(model, dataset: CIRRDataset, cfg, tokenizer, transform, device,
                             gallery_paths: list[str], out_dir: str, experiment_name: str, max_queries) -> None:
    """Writes the two-file CIRR test-server submission (fixed 2026-09-09,
    code-audit document finding #10):
      * `recall_<experiment_name>.json` -- global recall submission, top-50
        predictions ranked over the FULL gallery.
      * `subset_<experiment_name>.json` -- subset recall submission, top-3
        predictions ranked over ONLY the query's 5 non-reference subset
        candidates.
    No ground truth exists to score against locally for test1 -- see this
    module's docstring for the exact schema this follows and why it is not
    asserted to be byte-exact against CodaLab's live validator.

    A query whose `group_members` doesn't resolve to a well-formed 6-member
    (reference + 5 candidates, all 5 present in the gallery) set is skipped
    from the SUBSET file specifically (there is no valid 5-candidate pool to
    rank) but still included in the global RECALL file, which only needs the
    full gallery, not group metadata -- consistent with `evaluate.py`'s own
    "malformed group excluded from subset accounting, not from the whole
    evaluation" policy.
    """
    from revua_cir.data.collate import build_collate_fn
    from torch.utils.data import DataLoader

    path_to_id = {p: img_id for img_id, p in ((i, dataset._img_path(i)) for i in dataset.id_to_path)}
    path_to_idx = {p: i for i, p in enumerate(gallery_paths)}
    # Same id_to_idx construction evaluate.py's group-recall block and
    # strict_preflight_check both use: CIRR image ids are NOT recoverable by
    # parsing the resolved filename (see evaluate.py's own comment on this),
    # so this must go through dataset.id_to_path/_img_path directly.
    id_to_idx = {}
    for image_id in dataset.id_to_path:
        p = dataset._img_path(image_id)
        if p in path_to_idx:
            id_to_idx[str(image_id)] = path_to_idx[p]

    gallery_embeds = encode_gallery_paths(model, gallery_paths, transform, device, cfg.data.eval_batch_size).to(device)

    n = len(dataset) if max_queries is None else min(max_queries, len(dataset))
    collate_fn = build_collate_fn(tokenizer)
    loader = DataLoader(torch.utils.data.Subset(dataset, range(n)), batch_size=cfg.data.eval_batch_size,
                         shuffle=False, collate_fn=collate_fn, num_workers=0)

    recall_predictions: dict = {}
    subset_predictions: dict = {}
    skipped_subset = 0
    query_offset = 0
    for batch in loader:
        reference = batch["reference_pixel_values"].to(device)
        input_ids = batch["edit_input_ids"].to(device)
        attn = batch["edit_attention_mask"].to(device)
        q = model.encode_query(reference, input_ids, attn)
        scores = q["query"] @ gallery_embeds.t()

        for i in range(reference.size(0)):
            ref_idx = path_to_idx.get(dataset.triplets[query_offset + i]["ref_path"], -1)
            if ref_idx >= 0:
                scores[i, ref_idx] = float("-inf")

        top_idx = scores.topk(min(50, len(gallery_paths)), dim=-1).indices.cpu().tolist()
        for i, meta in enumerate(batch["meta"]):
            pairid = meta.get("pairid")
            recall_predictions[str(pairid)] = [path_to_id.get(gallery_paths[j]) for j in top_idx[i]]

            ref_id = str(meta.get("reference_id", ""))
            all_member_ids = {str(m) for m in meta.get("group_members", [])}
            candidate_ids = all_member_ids - {ref_id}
            candidate_idxs = [id_to_idx[c] for c in candidate_ids if c in id_to_idx]
            if len(all_member_ids) == 6 and ref_id in all_member_ids and len(candidate_idxs) == 5:
                cand_scores = scores[i, candidate_idxs]
                order = cand_scores.argsort(descending=True).tolist()[:3]
                subset_predictions[str(pairid)] = [path_to_id.get(gallery_paths[candidate_idxs[j]]) for j in order]
            else:
                skipped_subset += 1
        query_offset += reference.size(0)

    os.makedirs(out_dir, exist_ok=True)
    recall_path = os.path.join(out_dir, f"recall_{experiment_name}.json")
    subset_path = os.path.join(out_dir, f"subset_{experiment_name}.json")
    with open(recall_path, "w", encoding="utf-8") as f:
        json.dump({"version": "rc2", "metric": "recall", "predictions": recall_predictions}, f)
    with open(subset_path, "w", encoding="utf-8") as f:
        json.dump({"version": "rc2", "metric": "recall_subset", "predictions": subset_predictions}, f)

    print(f"[evaluate_cirr] wrote {len(recall_predictions)} global-recall predictions to {recall_path}")
    print(f"[evaluate_cirr] wrote {len(subset_predictions)} subset-recall predictions to {subset_path} "
          f"({skipped_subset} quer{'y' if skipped_subset == 1 else 'ies'} skipped -- no well-formed "
          f"6-member/5-candidate group)")
    print(f"[evaluate_cirr] VERIFY both files against CIRR's current CodaLab submission schema "
          f"before uploading -- this has not been confirmed against the live server.")


if __name__ == "__main__":
    main()
