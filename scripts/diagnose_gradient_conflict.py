"""
Gradient-conflict diagnostic (root-cause analysis document, Section 6/16
Check 3; expanded 2026-09-10 per ReVUA_CIR_Final_PreTraining_Recommendations.
md findings #3/#4): measures whether the auxiliary objectives (AEP/
disentangle, RV/reverse) pull the SHARED representation in a direction that
conflicts with the retrieval objective, or a direction that's roughly
aligned/orthogonal -- and, separately, whether any such conflict is large
enough in MAGNITUDE to actually matter for optimization.

Why this matters: job 6300's real training log showed a real, unexplained
pattern -- train retrieval loss fell monotonically and smoothly (2.95->0.97
over 8 epochs) while validation R@1 peaked at epoch 2 and then
plateaued/declined, despite training loss continuing to improve. The
strongest, externally-verified explanation (checked against CLIP4Cir's own
published/code-default LR for the identical full-CLIP-fine-tune-on-CIRR
scenario) is that the backbone LR was ~6x too aggressive, causing
overfitting -- but a SEPARATE, complementary hypothesis is that the
auxiliary losses may ALSO be actively pulling the shared representation
away from what benefits retrieval, independent of the LR issue. This
script measures that second hypothesis directly and cheaply (a handful of
real batches from the real CIRR training set through a real trained
checkpoint), rather than assuming either way.

**Two shared parameter spaces, not one.** The Reverse Verifier's Tier 1
(`tier1_inverse_consistency`, the default) reuses the SAME `ForwardComposer`
instance the retriever itself uses (`self.rv.shared_composer is
self.composer`, confirmed directly in revua_cir.py's `__init__` and
reverse_verifier.py's forward()) -- so RV can conflict with retrieval in
TWO places: the backbone (`backbone.*` params) AND the composer
(`composer.*` params, weight-shared, not covered by `backbone.*`). AEP has
no such composer-sharing (it reads the composer's *output*, not the
composer itself), so AEP's only possible conflict surface is the backbone.

**What is actually measured.** For a batch of real CIRR triplets, run one
real forward pass (no gradcache -- this script does NOT need the large
effective batch size gradcache exists for; it needs individual per-loss
gradients on a small batch, which gradcache's whole design specifically
avoids ever materializing). Compute retrieval loss and each enabled
auxiliary loss SEPARATELY, back-propagate each one independently (with
`retain_graph=True` between them, zeroing gradients after each), and
collect the resulting gradient on every `backbone.*` parameter and,
separately, every `composer.*` parameter. Flatten each loss's gradient
(per parameter space) into one vector and compute: cosine similarity with
`grad_retrieval`, the raw gradient norm, and the ratio of the *weighted*
auxiliary gradient norm to the retrieval gradient norm (using this run's
ACTUAL configured `loss.lambda_*` weight, not the raw/unweighted loss --
weight is applied to the loss scalar before backward(), so a lambda-scaled
gradient is exactly `lambda * grad_of_raw_loss` by linearity of
differentiation).

**Why magnitude matters as much as cosine** (code-audit finding #3): a
cosine of -0.4 barely matters if the auxiliary gradient's norm is
near-zero relative to retrieval's -- the optimizer step is still
dominated by retrieval regardless of the (small) conflicting component.
The weighted-norm ratio tells you whether a negative cosine is
optimization-relevant or just noise-level.

**Only two of the four auxiliary losses can even reach the backbone**
(confirmed 2026-09-10 by reading revua_cir.py's forward() directly, not
assumed): `rv_calibration` is computed on `.detach()`'d cos_sim values (a
deliberate 2026-09-08 fix so training the RV calibrator does not
re-perturb candidate_decomp/backbone a second time -- `reverse` already
does that), and `uncertainty` is computed entirely from `query.detach(),
ref_out.pooled.detach(), text_out.pooled.detach()` (UG's own leak fix,
covered by test_ug_calibration_loss_does_not_leak_into_rv_or_backbone).
Both are ARCHITECTURALLY INCAPABLE of conflicting with retrieval on either
the backbone or the composer, regardless of what their loss VALUE does
during training. Only `disentangle` (AEP) and `reverse` (RV) have a real,
non-detached gradient path into the backbone; only `reverse` (RV) has one
into the composer.

**Training-time augmentation, not eval-time** (code-audit finding #4): uses
`build_image_transform(..., train=True)` by default -- the SAME
augmentation pipeline the real optimizer sees during training (RandomCrop/
flip/color-jitter), not the deterministic eval-time center-crop. A
diagnostic measuring the eval-time transform would be measuring gradients
on a systematically different input distribution than the one that
actually produced job 6300's training curve. Pass `--eval-transform` to
additionally run a second, deterministic pass (useful for exact
reproducibility across invocations) -- this does NOT replace the
training-transform measurement, which is always run.

**Interpretation.** A strongly negative cosine (e.g. < -0.1) combined with
a non-negligible weighted-norm ratio means that objective is measurably
PULLING the shared representation away from where retrieval's own
gradient wants to go, AND that pull is large enough to matter -- direct,
actionable evidence of objective conflict. A cosine near 0, or a
near-negligible weighted-norm ratio regardless of cosine sign, means the
objectives are not meaningfully fighting on that parameter space. This is
inherently noisy on a SMALL batch (a few dozen triplets, not the
gradcache-scale 192) and a few batches -- treat the sign and rough
magnitude as informative, not a precise number; report the mean and spread
across multiple batches, not a single value.

    python scripts/diagnose_gradient_conflict.py \
        --config configs/stage1_finetune_full_unfreeze_gradcache.yaml \
        --checkpoint runs/stage1_finetune_full_unfreeze_gradcache_v2/epoch002.pt \
        --module-combo fc+aep+rv+ug --n-batches 5 --batch-size 16
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from revua_cir.config import ReVUAConfig
from revua_cir.data.collate import build_collate_fn
from revua_cir.data.datasets import DATASET_REGISTRY
from revua_cir.data.tokenizer import build_tokenizer
from revua_cir.data.transforms import build_image_transform
from revua_cir.models.revua_cir import ReVUACIR
from revua_cir.utils import load_checkpoint, set_seed


# Which of the four auxiliary losses can even reach which shared parameter
# space, by construction -- see this module's own docstring for the full
# explanation of each. Used both to skip a measurement that's guaranteed to
# be zero (saving a wasted backward pass) and to report WHY it's N/A rather
# than just printing an empty/nan result.
_REACHES_BACKBONE = {"disentangle", "reverse"}
_REACHES_COMPOSER = {"reverse"}  # only RV shares the composer weight-wise; AEP does not
_ARCHITECTURALLY_DETACHED = {"rv_calibration", "uncertainty"}

# The config field name holding each loss's lambda weight -- used to report
# the WEIGHTED gradient norm ratio (code-audit finding #3), matching what
# the optimizer actually sees (loss.lambda_* is applied to the scalar loss
# before backward(), so by linearity the weighted gradient is exactly
# lambda times the raw-loss gradient -- no need to re-run backward() on a
# re-scaled loss to get this, just scale the already-computed norm).
_LAMBDA_FIELD = {
    "disentangle": "lambda_disentangle",
    "reverse": "lambda_reverse",
}


def _flat_grad(model: ReVUACIR, prefix: str) -> torch.Tensor:
    """Concatenates every `{prefix}*` parameter's CURRENT .grad into one 1-D
    vector, in a fixed (named_parameters iteration) order -- same order every
    call, so vectors from different backward() passes are directly comparable
    element-for-element. Missing gradients (a parameter this particular loss
    term never touched) are treated as exactly zero, which is the correct
    mathematical contribution of an untouched parameter to that loss's true
    (if sparse) gradient vector.
    """
    parts = []
    for name, p in model.named_parameters():
        if name.startswith(prefix) and p.requires_grad:
            # torch.zeros(p.numel()) (no device=) always creates a CPU
            # tensor -- on a real CUDA run, real gradients are on cuda:0, so
            # torch.cat would fail with a device-mismatch RuntimeError the
            # moment any parameter in this space had a missing gradient
            # (confirmed: this crashed in exactly that way the first time
            # this function was exercised with a real CUDA-resident model,
            # not caught by an earlier CPU-only test). zeros_like(p) matches
            # both device AND dtype automatically.
            parts.append(p.grad.detach().flatten() if p.grad is not None else torch.zeros_like(p).flatten())
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return float("nan")
    return float((a @ b) / (na * nb))


def _measure_one_pass(model: ReVUACIR, batch: dict, cfg: ReVUAConfig, aux_names: list[str]) -> dict:
    """Runs one forward pass and the retrieval + each enabled auxiliary
    loss's independent backward pass, returning per-space (backbone/
    composer), per-loss cosine/norm/weighted-ratio results for this single
    batch. Kept as its own function so main() can call it twice (train-
    transform batch, optional eval-transform batch) with identical logic.
    """
    model.zero_grad(set_to_none=True)
    out = model(batch)
    losses = out["losses"]
    if "retrieval" not in losses:
        raise RuntimeError("model(batch) did not produce a 'retrieval' loss -- check module_combo")
    losses["retrieval"].backward(retain_graph=True)
    grad_retrieval_backbone = _flat_grad(model, "backbone.")
    grad_retrieval_composer = _flat_grad(model, "composer.")
    retrieval_backbone_norm = float(grad_retrieval_backbone.norm())
    retrieval_composer_norm = float(grad_retrieval_composer.norm())

    result = {"retrieval_backbone_norm": retrieval_backbone_norm,
              "retrieval_composer_norm": retrieval_composer_norm, "aux": {}}

    for name in aux_names:
        if name not in losses:
            continue
        entry = {}
        if name in _ARCHITECTURALLY_DETACHED:
            entry["status"] = "detached_by_design"
            result["aux"][name] = entry
            continue

        model.zero_grad(set_to_none=True)
        losses[name].backward(retain_graph=True)
        lam = getattr(cfg.loss, _LAMBDA_FIELD.get(name, ""), 1.0) if name in _LAMBDA_FIELD else 1.0

        if name in _REACHES_BACKBONE:
            grad_aux_backbone = _flat_grad(model, "backbone.")
            raw_norm = float(grad_aux_backbone.norm())
            entry["backbone_cosine"] = _cosine(grad_retrieval_backbone, grad_aux_backbone)
            entry["backbone_aux_grad_norm"] = raw_norm
            entry["backbone_weighted_aux_grad_norm"] = lam * raw_norm
            entry["backbone_weighted_ratio"] = (
                (lam * raw_norm) / retrieval_backbone_norm if retrieval_backbone_norm > 0 else float("nan"))

        if name in _REACHES_COMPOSER:
            grad_aux_composer = _flat_grad(model, "composer.")
            raw_norm_c = float(grad_aux_composer.norm())
            entry["composer_cosine"] = _cosine(grad_retrieval_composer, grad_aux_composer)
            entry["composer_aux_grad_norm"] = raw_norm_c
            entry["composer_weighted_aux_grad_norm"] = lam * raw_norm_c
            entry["composer_weighted_ratio"] = (
                (lam * raw_norm_c) / retrieval_composer_norm if retrieval_composer_norm > 0 else float("nan"))

        result["aux"][name] = entry

    return result


def _print_report(label: str, n_done: int, batch_size: int, split: str, aux_names: list[str],
                   collected: dict) -> None:
    print()
    print(f"[diagnose_gradient_conflict] === {label} ({n_done} batches of {batch_size} real "
          f"CIRR {split} triplets) ===")
    for name in aux_names:
        if name in _ARCHITECTURALLY_DETACHED:
            print(f"[diagnose_gradient_conflict]   {name:16s}: N/A -- architecturally detached from "
                  f"both backbone and composer by design (trains only its own module's parameters), "
                  f"cannot conflict with retrieval regardless of its loss VALUE.")
            continue
        for space in ("backbone", "composer"):
            key_cos = f"{space}_cosine"
            key_ratio = f"{space}_weighted_ratio"
            vals_cos = collected.get(name, {}).get(key_cos, [])
            vals_ratio = collected.get(name, {}).get(key_ratio, [])
            if not vals_cos:
                if name in _REACHES_COMPOSER or space == "backbone":
                    # Only print N/A for combinations that COULD have data
                    # (e.g. don't print an AEP-composer N/A row -- AEP never
                    # shares the composer at all, that's not a measurement
                    # gap, it's a nonexistent question).
                    if space == "composer" and name not in _REACHES_COMPOSER:
                        continue
                    print(f"[diagnose_gradient_conflict]   {name:16s} [{space:9s}]: N/A "
                          f"(module disabled or loss absent)")
                continue
            mean_cos = float(np.mean(vals_cos))
            std_cos = float(np.std(vals_cos))
            mean_ratio = float(np.nanmean(vals_ratio)) if vals_ratio else float("nan")
            flag = ""
            if mean_cos < -0.1 and mean_ratio > 0.05:
                flag = "  <-- NEGATIVE + non-negligible magnitude: measurable, optimization-relevant conflict"
            elif mean_cos < -0.1:
                flag = "  <-- negative cosine but small relative magnitude -- likely not optimization-relevant"
            elif mean_cos > 0.1:
                flag = "  <-- POSITIVE: this objective reinforces retrieval's gradient direction"
            else:
                flag = "  <-- near-orthogonal: neither reinforcing nor conflicting"
            print(f"[diagnose_gradient_conflict]   {name:16s} [{space:9s}]: cos={mean_cos:+.4f} "
                  f"std={std_cos:.4f} weighted_ratio={mean_ratio:.4f} (n={len(vals_cos)}){flag}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--module-combo", type=str, required=True)
    p.add_argument("--eval-dataset", type=str, default="cirr")
    p.add_argument("--split", type=str, default="train",
                   help="real triplets to measure gradients on -- 'train' matches what the "
                        "optimizer actually saw during training (the scenario this diagnostic "
                        "is investigating), not 'val'.")
    p.add_argument("--n-batches", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16,
                   help="small on purpose -- this measures individual per-loss gradients, which "
                        "gradcache's whole design avoids ever materializing at the real "
                        "training batch_size. Not meant to reproduce the training batch scale.")
    p.add_argument("--eval-transform", action="store_true",
                   help="ALSO run a second, deterministic pass using the eval-time (center-crop, "
                        "no augmentation) transform, for exact reproducibility across invocations. "
                        "Does not replace the training-transform measurement (always run) -- see "
                        "this module's own docstring for why train=True is the primary measurement.")
    args = p.parse_args()

    cfg = ReVUAConfig.from_yaml(args.config)
    if args.data_root:
        cfg.data.root = args.data_root
    cfg = cfg.apply_module_ablation(args.module_combo)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = build_image_transform(cfg.data.image_size, train=True)
    tokenizer = build_tokenizer(cfg.backbone)
    collate_fn = build_collate_fn(tokenizer)

    root = os.path.join(cfg.data.root, args.eval_dataset)
    dataset = DATASET_REGISTRY[args.eval_dataset](root=root, split=args.split, transform=train_transform)
    print(f"[diagnose_gradient_conflict] {args.eval_dataset}/{args.split}: {len(dataset)} triplets")

    model = ReVUACIR(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.train()  # dropout/etc. active -- matches the actual training-time gradient, not eval-mode
    print(f"[diagnose_gradient_conflict] loaded {args.checkpoint} (module_combo={args.module_combo})")
    print(f"[diagnose_gradient_conflict] loss weights in use: lambda_disentangle="
          f"{cfg.loss.lambda_disentangle} lambda_reverse={cfg.loss.lambda_reverse}")

    aux_names = ["disentangle", "reverse", "rv_calibration", "uncertainty"]

    def run_pass(transform, label: str):
        ds = DATASET_REGISTRY[args.eval_dataset](root=root, split=args.split, transform=transform)
        set_seed(0)
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        collected: dict = {}
        n_done = 0
        for batch in loader:
            if n_done >= args.n_batches:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            per_batch = _measure_one_pass(model, batch, cfg, aux_names)
            for name, entry in per_batch["aux"].items():
                dst = collected.setdefault(name, {})
                for k, v in entry.items():
                    if k == "status":
                        continue
                    if np.isfinite(v):
                        dst.setdefault(k, []).append(v)
            n_done += 1
            print(f"[diagnose_gradient_conflict] [{label}] batch {n_done}/{args.n_batches} done")
        _print_report(label, n_done, args.batch_size, args.split, aux_names, collected)

    run_pass(train_transform, "train-time augmentation (primary measurement)")
    if args.eval_transform:
        eval_transform = build_image_transform(cfg.data.image_size, train=False)
        run_pass(eval_transform, "eval-time deterministic transform (secondary, reproducibility check)")

    print()
    print("[diagnose_gradient_conflict] NOTE: small-batch, few-batch estimate -- treat sign/rough "
          "magnitude as informative, not a precise number. A batch_size={} gradient is noisier than "
          "the real training batch_size={} gradient this is approximating.".format(
              args.batch_size, cfg.data.batch_size))


if __name__ == "__main__":
    main()
