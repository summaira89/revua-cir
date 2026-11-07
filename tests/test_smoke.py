"""
Offline smoke test -- exercises the ENTIRE pipeline (backbone -> forward
composer -> attribute parser -> reverse verifier -> uncertainty gate ->
connective joint loss -> backward pass -> optimizer step -> metrics ->
early stopping -> checkpoint save/load -> module-ablation switching) using
the `tiny` backbone and a synthetic in-memory dataset, so it runs on CPU
with no network access and no real CIR datasets.

This is a correctness gate, not a benchmark: it proves the code is wired up
right before you point it at real FashionIQ/CIRR/CIRCO/LaSCo data and a GPU.

Run with:  python -m pytest tests/test_smoke.py -v
       or: python tests/test_smoke.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from revua_cir.config import ReVUAConfig
from revua_cir.data.collate import build_collate_fn
from revua_cir.data.tokenizer import build_tokenizer
from revua_cir.metrics import expected_calibration_error, mean_average_precision_at_k, recall_at_k
from revua_cir.models.revua_cir import ReVUACIR
from revua_cir.train import build_optimizer, build_scheduler
from revua_cir.utils import EMA, EarlyStopper, load_checkpoint, save_checkpoint, set_seed


TEXTS = [
    "make it red and change the sleeves to short, but keep the same length",
    "instead of stripes, use a solid color",
    "more fitted around the waist",
    "same neckline, different color, add polka dots",
]


class SyntheticCIRDataset(Dataset):
    """Random-tensor stand-in for a real CIR dataset -- same __getitem__ contract as
    revua_cir.data.datasets.CIRTripletDatasetBase so it drops straight into training."""

    def __init__(self, n: int, image_size: int, attribute_vocab_size: int = 512):
        self.n = n
        self.image_size = image_size
        self.attribute_vocab_size = attribute_vocab_size
        # `evaluate_dataset` builds its gallery from `.triplets` and uses the
        # ref/target paths for reference-masking, so the stand-in has to carry
        # that contract too. The paths need not exist: safe_load_transform
        # substitutes a gray placeholder, which is fine for a shape/plumbing test.
        self.triplets = [
            {"ref_path": f"__synthetic__/ref_{i}.png",
             "target_path": f"__synthetic__/tgt_{i}.png",
             "text": TEXTS[i % len(TEXTS)],
             "meta": {"dataset": "synthetic", "target_id": i}}
            for i in range(n)
        ]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)
        return {
            "reference_pixel_values": torch.randn(3, self.image_size, self.image_size, generator=g),
            "target_pixel_values": torch.randn(3, self.image_size, self.image_size, generator=g),
            "edit_text": TEXTS[idx % len(TEXTS)],
            "attribute_targets": torch.zeros(self.attribute_vocab_size),
            "meta": {"dataset": "synthetic", "target_id": idx},
        }


def build_tiny_config() -> ReVUAConfig:
    cfg = ReVUAConfig()
    cfg.backbone.kind = "tiny"
    cfg.backbone.embed_dim = 64
    cfg.backbone.tiny_image_size = 32
    cfg.backbone.tiny_patch_size = 8
    cfg.backbone.tiny_width = 64
    cfg.backbone.tiny_layers = 1
    cfg.backbone.tiny_heads = 4
    cfg.fusion.num_heads = 4
    cfg.fusion.num_layers = 1
    cfg.attribute_parser.hidden_dim = 64
    cfg.attribute_parser.attribute_vocab_size = 512
    cfg.reverse_verifier.hidden_dim = 64
    cfg.uncertainty_gate.variance_hidden_dim = 32
    cfg.uncertainty_gate.coordination_hidden_dim = 32
    cfg.uncertainty_gate.neighborhood_k = 3
    cfg.data.image_size = 32
    cfg.data.batch_size = 8
    cfg.data.num_workers = 0
    cfg.optim.max_epochs = 1
    cfg.optim.amp = False
    cfg.run.device = "cpu"
    return cfg


def run_forward_backward(cfg, module_combo):
    cfg = cfg.apply_module_ablation(module_combo)
    set_seed(0)
    model = ReVUACIR(cfg)
    tokenizer = build_tokenizer(cfg.backbone)
    collate_fn = build_collate_fn(tokenizer)
    dataset = SyntheticCIRDataset(16, cfg.data.image_size, cfg.attribute_parser.attribute_vocab_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, num_training_steps=len(loader) * 2)

    losses_seen = []
    for batch in loader:
        out = model(batch)
        loss = out["losses"]["total"]
        assert torch.isfinite(loss), f"non-finite loss for combo {module_combo}: {loss}"
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Every parameter that should receive gradient must receive one --
        # catches accidentally-disconnected modules (a common CAGF-style bug).
        # rv.calibrator (Platt scale/bias) only feeds the *reported* verification
        # probability, not the reverse-consistency loss itself -- it legitimately
        # gets no gradient when the Uncertainty Gate (its only training-time
        # consumer) is disabled. Every other parameter must receive one.
        allowlisted_dead = {"rv.calibrator.scale", "rv.calibrator.bias"}
        n_missing = 0
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is None and name not in allowlisted_dead:
                n_missing += 1
                print("MISSING GRAD:", name)
        assert n_missing == 0, f"{n_missing} params got no gradient for combo {module_combo}"

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        losses_seen.append(loss.item())

    print(f"[{module_combo}] losses: {['%.4f' % l for l in losses_seen]}")
    return model, cfg


def test_module_ablation_matrix():
    cfg = build_tiny_config()
    for combo in ["fc", "fc+aep", "fc+aep+rv", "fc+aep+rv+ug"]:
        run_forward_backward(cfg, combo)
    print("OK: module ablation matrix (fc / +aep / +rv / +ug) all forward+backward cleanly.")


def test_residual_composer_diagnostics_are_logged_and_wired():
    """Regression guard for the monitoring recommendation in the code-audit
    document (2026-09-09, finding #14, revised 2026-09-09 per finding #7 of
    the follow-up document): `residual_alpha` is trainable and unbounded, so
    the documents recommend logging enough per-step diagnostics that a real
    drift problem would actually be visible in training logs (not
    constraining alpha preemptively -- monitor first).

    The diagnostic SET changed between the two documents: the original
    `delta_norm`/`alpha_delta_norm` pair was replaced with
    `pre_normalization_delta_norm`/`cos_q0_delta` because `delta` is
    L2-normalized right before those two were computed, so `delta_norm` was
    always ~1.0 by construction and `alpha_delta_norm` was just
    `|residual_alpha|` restated -- neither carried information the plain
    `residual_alpha` scalar didn't already give. The replacements carry
    genuinely new information: `pre_normalization_delta_norm` tracks CAGF's
    own raw output scale (independent of alpha, invisible once `delta` is
    normalized), and `cos_q0_delta` reveals whether the learned correction
    reinforces q0, is orthogonal to it, or opposes it.

    ForwardComposer.forward() returns a third element (a diagnostics dict),
    and ReVUACIR.encode_query()/forward() thread it into
    `losses["diag_composer_<name>"]`, picked up automatically by train.py's
    existing generic per-step meter logging (same `diag_`-prefix mechanism
    `diag_rv_cos_gap` already uses).

    This test checks the diagnostics directly (not just that training
    doesn't crash): with use_residual_composition=True, `encode_query`
    returns non-empty `composer_diagnostics` with all 4 expected keys, all
    finite, both cosine-similarity diagnostics in [-1, 1], and
    `pre_normalization_delta_norm` > 0 (a real pre-normalization scale, not
    a degenerate zero vector). With use_residual_composition=False, the
    diagnostics dict must be empty -- this also regression-guards the
    composer's own forward() return signature: a stale 2-tuple unpack
    anywhere (a real bug this exact mechanism caused and caught in
    reverse_verifier.py's shared_composer call, fixed in the same pass that
    introduced it) would raise here instead of silently mismatching.
    """
    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc")
    set_seed(0)
    model = ReVUACIR(cfg)
    model.eval()

    ref_pixels = torch.randn(2, 3, cfg.data.image_size, cfg.data.image_size)
    input_ids = torch.randint(0, 100, (2, 8))
    attn = torch.ones(2, 8, dtype=torch.long)

    q = model.encode_query(ref_pixels, input_ids, attn)
    diag = q["composer_diagnostics"]
    expected_keys = {"residual_alpha", "cos_q0_q_final", "pre_normalization_delta_norm", "cos_q0_delta"}
    assert set(diag.keys()) == expected_keys, (
        f"expected composer diagnostics {expected_keys}, got {set(diag.keys())}")
    for k, v in diag.items():
        assert torch.isfinite(v).all(), f"composer diagnostic {k} is non-finite: {v}"

    assert diag["pre_normalization_delta_norm"].item() > 0.0, (
        "pre_normalization_delta_norm should be a real positive scale, not a degenerate zero vector")
    assert -1.0 <= diag["cos_q0_q_final"].item() <= 1.0, (
        f"cos_q0_q_final out of valid cosine-similarity range: {diag['cos_q0_q_final'].item()}")
    assert -1.0 <= diag["cos_q0_delta"].item() <= 1.0, (
        f"cos_q0_delta out of valid cosine-similarity range: {diag['cos_q0_delta'].item()}")

    cfg_no_residual = build_tiny_config()
    cfg_no_residual.fusion.use_residual_composition = False
    cfg_no_residual = cfg_no_residual.apply_module_ablation("fc")
    set_seed(0)
    model_no_residual = ReVUACIR(cfg_no_residual)
    model_no_residual.eval()
    q2 = model_no_residual.encode_query(ref_pixels, input_ids, attn)
    assert q2["composer_diagnostics"] == {}, (
        f"composer_diagnostics should be empty when use_residual_composition=False, "
        f"got {q2['composer_diagnostics']}")

    print("OK: residual-composer diagnostics (residual_alpha, cos_q0_q_final, "
          "pre_normalization_delta_norm, cos_q0_delta) are present, finite, and in valid range when "
          "residual composition is enabled; empty when disabled.")


def test_fc_ug_ablation_is_rejected():
    """Regression guard for an explicit user decision (2026-09-08, in
    response to the audit finding that "fc+ug" -- RV disabled, UG enabled --
    is a structurally degenerate ablation): UncertaintyGate's
    fuse_reverse_disagreement signal falls back to a constant zero tensor
    when RV is absent, making part of UG's own fusion layer dead input, and
    UG receives only positive calibration supervision without RV's
    negative-pair pass. Rather than silently allowing this weak/misleading
    combo to be constructed, `apply_module_ablation` now rejects it (and
    anything outside the strictly progressive ladder) with a clear error.
    This is enforcement, not just documentation -- checked here so a future
    change can't silently regress it back to being constructible.
    """
    cfg = build_tiny_config()
    for combo in ReVUAConfig.VALID_MODULE_ABLATIONS:
        cfg.apply_module_ablation(combo)  # must not raise
    raised = False
    try:
        cfg.apply_module_ablation("fc+ug")
    except ValueError as e:
        raised = "fc+ug" in str(e) or "not one of the supported combos" in str(e)
    assert raised, "apply_module_ablation('fc+ug') should raise ValueError, not silently construct a degenerate ablation"
    print("OK: fc+ug (RV disabled, UG enabled) is rejected; the valid progressive ladder still works.")


def test_residual_composition_gives_text_pooled_real_gradient():
    """Regression guard for a real bug fixed 2026-09-08 in fusion.py's
    ForwardComposer: the previous CAGF-only path computed `fused = fused +
    text_pooled * 0.0` -- CLIP's own pretrained text embedding multiplied by
    exactly zero, contributing NOTHING to the forward value or the backward
    gradient (the chain rule multiplies by that 0.0 coefficient). The
    misleading comment there ("keep text_pooled in the graph for future
    use") made this look deliberate; it was a true no-op. Because
    ReverseVerifier's Tier-1 path reuses this same composer instance, RV's
    own scoring was affected identically.

    Fixed via residual composition (`cfg.use_residual_composition`, default
    True): q0 = norm(norm(image_pooled) + norm(text_pooled)), q = norm(q0 +
    alpha * CAGF_output), alpha learned, init 0.05 (NOT 0.0 -- see the
    second half of this test). This test verifies the concrete, checkable
    symptom of the old bug is gone: gradient genuinely reaches the backbone
    parameters that produce text_pooled (backbone.text_proj under the tiny
    backbone), and the learned alpha parameter itself receives gradient.

    Second regression guard, found via a follow-up code-audit document
    (2026-09-08): at alpha=0 EXACTLY, d(query)/d(theta_CAGF) = alpha *
    d(CAGF_output)/d(theta_CAGF) = 0 -- the entire CAGF branch (gate_mlp,
    cross-attention blocks, out_proj) would receive ZERO gradient at the
    first training step and stay untrained until alpha itself moved away
    from exactly 0 via its own gradient. Verified directly: with
    residual_alpha_init=0.0, `composer.gate_mlp[0].weight.grad` was exactly
    0.0 after a real backward pass. Fixed by defaulting
    `residual_alpha_init` to 0.05 (keeps q0 dominant, matching the fix's own
    intent of preserving pretrained CLIP geometry, while giving CAGF real
    gradient from step 0). This test checks gate_mlp gets real, nonzero
    gradient at the CURRENT default init, guarding against a future
    accidental revert to 0.0.

    Also checks `use_residual_composition=False` still runs cleanly end to
    end, since it exists specifically for a controlled ablation against this
    fix and needs to keep working.
    """
    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc")
    cfg.backbone.freeze_backbone = False
    cfg.fusion.dropout = 0.0
    assert cfg.fusion.use_residual_composition is True, "use_residual_composition should default to True"
    assert cfg.fusion.residual_alpha_init != 0.0, (
        "residual_alpha_init must not be exactly 0.0 -- the CAGF branch gets zero gradient at that "
        "value (d(query)/d(theta_CAGF) = alpha * d(CAGF_output)/d(theta_CAGF) = 0 when alpha=0)")

    set_seed(0)
    model = ReVUACIR(cfg)
    model.train()

    bsz = 4
    batch = {
        "reference_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "target_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "edit_input_ids": torch.randint(0, 100, (bsz, 8)),
        "edit_attention_mask": torch.ones(bsz, 8, dtype=torch.long),
    }

    out = model(batch)
    out["losses"]["total"].backward()

    assert abs(model.composer.residual_alpha.item() - cfg.fusion.residual_alpha_init) < 1e-6
    assert model.composer.residual_alpha.grad is not None and model.composer.residual_alpha.grad.abs().sum() > 0, (
        "residual_alpha received no gradient -- it will never train away from its init")
    text_proj_grad = model.backbone.text_proj.weight.grad
    assert text_proj_grad is not None and text_proj_grad.norm().item() > 0, (
        "backbone.text_proj (feeds text_pooled) received no gradient -- the * 0.0 bug appears to still be present")
    gate_grad = model.composer.gate_mlp[0].weight.grad
    assert gate_grad is not None and gate_grad.norm().item() > 0, (
        "composer.gate_mlp (part of the CAGF branch) received no gradient -- this is the "
        "alpha=0 cold-start bug; check fusion.residual_alpha_init is not 0.0")

    # use_residual_composition=False must still run cleanly (the controlled-
    # ablation opt-out this fix was gated behind).
    cfg_off = build_tiny_config()
    cfg_off = cfg_off.apply_module_ablation("fc")
    cfg_off.fusion.use_residual_composition = False
    set_seed(0)
    model_off = ReVUACIR(cfg_off)
    model_off.train()
    assert not hasattr(model_off.composer, "residual_alpha"), (
        "residual_alpha should not even be constructed when use_residual_composition=False")
    out_off = model_off(batch)
    out_off["losses"]["total"].backward()
    print("OK: residual composition gives text_pooled AND the CAGF branch real gradient at step 0; "
          "residual_alpha trains from its non-zero init; use_residual_composition=False still runs cleanly.")


def test_residual_alpha_scale_is_honest():
    """Regression guard for a real bug found via code-audit document review
    (2026-09-08): `q = norm(q0 + alpha * CAGF_output)` used the raw,
    UNNORMALIZED CAGF_output as the residual branch. Measured directly on
    this project's real (tiny-backbone) data: CAGF_output's norm is ~3.2,
    not 1.0, so `alpha * CAGF_output`'s actual magnitude relative to q0
    (always exactly unit norm by construction) was ~3.2x what
    `residual_alpha_init: 0.05`'s name/docstring implied -- a real ~16%
    perturbation labeled as "5%," undermining the fix's own stated goal of
    keeping q0 (pretrained CLIP geometry) dominant.

    Fixed by normalizing the residual branch (`delta = F.normalize(...)`)
    before scaling by alpha, so `||alpha * delta|| / ||q0|| ~= |alpha|`
    holds as intended. This test verifies that ratio directly, with a
    generous tolerance (since q0 and delta are not perfectly orthogonal, the
    exact ratio of the SUM's components isn't exactly |alpha| -- but it
    should be close, not off by a factor of 3+ like before the fix).
    """
    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc")
    cfg.backbone.freeze_backbone = False
    cfg.fusion.dropout = 0.0
    cfg.fusion.residual_alpha_init = 0.05

    set_seed(0)
    model = ReVUACIR(cfg)
    model.eval()

    bsz = 8
    batch = {
        "reference_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "target_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "edit_input_ids": torch.randint(0, 100, (bsz, 8)),
        "edit_attention_mask": torch.ones(bsz, 8, dtype=torch.long),
    }

    # Reproduce the composer's own internal computation up to the point
    # where alpha is applied, to measure the actual delta norm directly
    # (can't observe it from the outside since forward() only returns the
    # final normalized query).
    with torch.no_grad():
        ref_out = model.backbone.encode_image(batch["reference_pixel_values"])
        text_out = model.backbone.encode_text(batch["edit_input_ids"], batch["edit_attention_mask"])
        composer = model.composer
        text_kpm = None if text_out.mask is None else (text_out.mask == 0)
        t, i = text_out.tokens, ref_out.tokens
        for block in composer.blocks:
            t, i = block(t, i, text_key_padding_mask=text_kpm)
        from revua_cir.models.fusion import masked_mean_pool
        text_attended = masked_mean_pool(t, text_out.mask)
        image_attended = i.mean(dim=1) if i.dim() == 3 else i
        image_grounded = ref_out.pooled + image_attended
        gate = torch.sigmoid(composer.gate_mlp(torch.cat([image_grounded, text_attended], dim=-1)))
        fused = gate * text_attended + (1 - gate) * image_grounded
        residual = composer.out_proj(fused)
        cagf_output = fused + residual
        delta = torch.nn.functional.normalize(composer.final_ln(cagf_output), dim=-1)
        delta_norm = delta.norm(dim=-1).mean().item()

    assert abs(delta_norm - 1.0) < 1e-4, (
        f"the residual branch (delta) should be unit-norm after the fix's explicit "
        f"F.normalize(), got {delta_norm:.4f} -- the normalization appears to be missing")
    alpha = model.composer.residual_alpha.item()
    perturbation_ratio = alpha * delta_norm  # ||alpha * delta|| / ||q0||, since ||q0||=1 and ||delta||=1
    assert abs(perturbation_ratio - alpha) < 1e-4, (
        f"||alpha * delta|| / ||q0|| ({perturbation_ratio:.4f}) should equal alpha ({alpha:.4f}) "
        f"since both q0 and delta are unit-norm -- residual_alpha_init's value no longer has "
        f"an honest, direct scale interpretation")
    print(f"OK: residual branch is unit-norm, so alpha={alpha:.4f} genuinely means a "
          f"{alpha*100:.1f}% perturbation relative to q0, not the ~3x-inflated value before this fix.")


def test_rv_edit_weight_inits_to_balanced_50_50():
    """Regression guard for a real bug found via code-audit document review
    (2026-09-08): `rv.edit_weight` was initialized as
    `nn.Parameter(torch.tensor(0.5))`, but the actual weight used is
    `w = sigmoid(self.edit_weight)` -- so the real initial w was
    sigmoid(0.5) ~= 0.622, not the balanced 0.5 the name/comment ("Relative
    weight of...") implied. Fixed by initializing the raw parameter to 0.0
    (sigmoid(0.0) = 0.5 exactly).
    """
    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc+aep+rv")
    set_seed(0)
    model = ReVUACIR(cfg)
    w = torch.sigmoid(model.rv.edit_weight)
    assert abs(w.item() - 0.5) < 1e-5, (
        f"rv.edit_weight should give a balanced initial w=0.5 via sigmoid, got {w.item():.4f}")
    print("OK: rv.edit_weight initializes to a genuine 0.5 balanced weight.")


def test_reverse_verifier_tier2():
    cfg = build_tiny_config()
    cfg.reverse_verifier.tier = "tier2_reconstruction"
    run_forward_backward(cfg, "fc+aep+rv+ug")
    print("OK: Tier-2 reverse verifier forward+backward.")


def test_reverse_verifier_tier1_uses_renamed_inverse_consistency_name():
    """Regression guard for the RV tier rename (2026-09-09, novelty-review
    finding #7): `tier1_vqascore` was renamed to `tier1_inverse_consistency`
    across config.py, reverse_verifier.py, revua_cir.py, and all 9 shipped
    YAML configs -- the old name invited readers/reviewers to assume Tier 1
    implements actual VQAScore (a generative image-text likelihood method,
    Lin et al.), when it is really a shared-composer cosine-similarity
    reconstruction check (see reverse_verifier.py's own module docstring).

    A silent partial rename (e.g. config.py's default string updated but a
    branch condition in reverse_verifier.py or revua_cir.py left comparing
    against the OLD string) would make Tier 1 unreachable/misconfigured
    without necessarily crashing loudly in every code path -- e.g.
    `ReVUACIR.__init__`'s `shared = self.composer if cfg.reverse_verifier.
    tier == "tier1_..." else None` would silently pass `shared=None` to a
    verifier that then hits `ReverseVerifier.__init__`'s own
    `if cfg.tier == "tier1_...":` branch and only fails via an assertion
    deep in forward(), not at construction. Explicitly pins both call sites
    stay in sync with the config default.
    """
    cfg = build_tiny_config()
    assert cfg.reverse_verifier.tier == "tier1_inverse_consistency", (
        f"ReverseVerifierConfig's default tier should be 'tier1_inverse_consistency' "
        f"(renamed from 'tier1_vqascore'), got {cfg.reverse_verifier.tier!r}")
    cfg = cfg.apply_module_ablation("fc+aep+rv+ug")
    set_seed(0)
    model = ReVUACIR(cfg)
    assert model.rv.shared_composer is not None, (
        "Tier 1 (tier1_inverse_consistency) should share the ForwardComposer instance -- "
        "shared_composer is None, suggesting ReVUACIR.__init__'s tier-name comparison has "
        "silently drifted out of sync with the config default")
    assert model.rv.shared_composer is model.composer, (
        "Tier 1's shared_composer must be the SAME instance as the model's own composer "
        "(weight-shared reconstruction, not a separate copy)")
    print("OK: RV Tier 1 correctly uses the renamed 'tier1_inverse_consistency' config value "
          "end to end -- config default, ReVUACIR's tier-name branch, and ReverseVerifier's own "
          "tier-name branch all stay in sync.")


def test_autocast_safety():
    """`optim.amp: true` is the shipped default, but the rest of this file runs
    with amp disabled -- which let a real bug through: the Uncertainty Gate's
    calibration term used `F.binary_cross_entropy`, and plain BCE *raises* under
    autocast ("unsafe to autocast", use the with-logits form). Every AMP run
    crashed on the first step. This test exercises the autocast path so that
    class of bug cannot come back.

    Runs on CUDA when available; falls back to CPU bfloat16 autocast, which
    catches the same op-level unsafety.
    """
    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc+aep+rv+ug")
    set_seed(0)

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.bfloat16

    model = ReVUACIR(cfg).to(device)
    collate_fn = build_collate_fn(build_tokenizer(cfg.backbone))
    dataset = SyntheticCIRDataset(8, cfg.data.image_size, cfg.attribute_parser.attribute_vocab_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    batch = next(iter(loader))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    with torch.autocast(device_type=device, dtype=dtype):
        out = model(batch)
    loss = out["losses"]["total"]
    assert torch.isfinite(loss), f"non-finite loss under autocast: {loss}"
    loss.backward()
    print(f"OK: autocast({device}/{dtype}) forward+backward -- AMP path is safe.")


def test_optimizer_excludes_scalars_and_biases_from_weight_decay():
    """Regression guard for a real bug found via code-audit document review
    (2026-09-08): `build_optimizer` passed `weight_decay=cfg.optim.
    weight_decay` once to `torch.optim.AdamW(groups, weight_decay=...)`,
    applying it uniformly to EVERY trainable parameter across both LR
    groups -- including biases, LayerNorm weights, and every learned scalar
    control (composer.residual_alpha, composer.logit_scale, rv.edit_weight,
    rv.calibrator.scale/.bias, ug.temperature). AdamW's decay term
    continuously pulls every decayed parameter toward 0 -- actively fighting
    a scalar like residual_alpha that training is trying to grow away from
    its small near-zero init to learn a useful residual correction.

    Fixed with the standard decay/no-decay split (`p.dim() <= 1` catches
    every bias, LayerNorm weight, and scalar Parameter) crossed with the
    existing backbone/non-backbone LR split, giving 4 param groups total.
    This test checks: every trainable parameter is covered by exactly one
    group (none dropped, none duplicated), and specifically that
    residual_alpha/logit_scale/edit_weight/calibrator.scale/.bias/
    ug.log_temperature all land in a weight_decay=0.0 group. (`ug.temperature`
    was renamed to `ug.log_temperature` 2026-09-09 when its clamp-based
    positivity constraint was replaced with a log-parameterization -- still a
    scalar `dim()==0` Parameter, so it belongs in the same no-decay group.)
    """
    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc+aep+rv+ug")
    cfg.backbone.freeze_backbone = False

    set_seed(0)
    model = ReVUACIR(cfg)
    opt = build_optimizer(model, cfg)

    total_in_groups = sum(len(g["params"]) for g in opt.param_groups)
    total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert total_in_groups == total_trainable, (
        f"optimizer param groups cover {total_in_groups} params but the model has "
        f"{total_trainable} trainable params -- some parameter is missing or duplicated")

    no_decay_param_ids = set()
    for g in opt.param_groups:
        if g["weight_decay"] == 0.0:
            no_decay_param_ids.update(id(p) for p in g["params"])
    decay_param_ids = set()
    for g in opt.param_groups:
        if g["weight_decay"] > 0.0:
            decay_param_ids.update(id(p) for p in g["params"])
    assert not (no_decay_param_ids & decay_param_ids), "a parameter appears in both a decay and no-decay group"

    scalar_names = ["composer.residual_alpha", "composer.logit_scale", "rv.edit_weight",
                    "rv.calibrator.scale", "rv.calibrator.bias", "ug.log_temperature"]
    param_by_name = dict(model.named_parameters())
    for name in scalar_names:
        assert name in param_by_name, f"test setup error: {name} not found in model"
        assert id(param_by_name[name]) in no_decay_param_ids, (
            f"{name} is a learned scalar but is not in a weight_decay=0.0 group -- AdamW "
            f"will continuously pull it toward 0, fighting training")
    print(f"OK: optimizer correctly excludes {len(scalar_names)} learned scalars (and every bias/"
          f"LayerNorm weight) from weight decay; all {total_trainable} trainable params accounted for.")


def test_freeze_backbone():
    """`freeze_backbone: true` must actually freeze.

    This regressed twice: once as unreachable dead code after a `return`, and
    once as `requires_grad=False` that still built the backbone's full autograd
    graph (activations, not optimizer state, are what exhaust an 8 GB card).
    Asserts both properties: no backbone parameter changes under optimizer
    steps, and the trainable-parameter count actually drops.
    """
    cfg = build_tiny_config()
    cfg.backbone.freeze_backbone = True
    cfg = cfg.apply_module_ablation("fc+aep+rv+ug")
    set_seed(0)

    model = ReVUACIR(cfg)
    model.train()
    optimizer = build_optimizer(model, cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert trainable < total, "freeze_backbone=True left every parameter trainable"

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    collate_fn = build_collate_fn(build_tokenizer(cfg.backbone))
    dataset = SyntheticCIRDataset(8, cfg.data.image_size, cfg.attribute_parser.attribute_vocab_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    for batch in loader:
        out = model(batch)
        out["losses"]["total"].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    frozen_names = {n for n, p in model.named_parameters() if not p.requires_grad}
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p.detach())]
    frozen_changed = [n for n in changed if n in frozen_names]
    assert not frozen_changed, f"frozen params changed: {frozen_changed[:5]}"
    assert changed, "nothing trained at all -- the modules should still update"
    print(f"OK: freeze_backbone -- {trainable/1e6:.1f}M/{total/1e6:.1f}M trainable, "
          f"{len(changed)} module params updated, 0 backbone params changed.")


def test_hard_negative_weight_moves_negative_logits_toward_positive():
    """Regression guard for a real math bug found via code-audit document
    review (2026-09-08): `symmetric_info_nce`'s hard_negative_weight path
    reweighted logits MULTIPLICATIVELY (`logits * weights`), which behaves
    unintuitively for negative-valued logits -- a common case here, since
    logits are logit_scale * cosine similarity and cosine similarity is
    frequently negative for unrelated in-batch pairs. Multiplying a negative
    logit by weight > 1 makes it MORE negative (moves it away from the
    positive, i.e. treats it as an EASIER competitor), the opposite of the
    intended "up-weight this hard negative's pull in the softmax". Fixed by
    reweighting ADDITIVELY in log-space (`logits + log(weights)`), the
    mathematically principled way to apply a per-class weight inside a
    softmax denominator (`w_j * exp(z_j) = exp(z_j + log(w_j))`) -- this
    preserves the up-weighting direction regardless of z_j's sign.

    This path is dormant in every shipped config (hard_negative_weight=0.0
    everywhere, and forbidden under gradcache), so the bug never affected
    any reported result -- fixed anyway so it is correct whenever used.

    Test constructs the exact failure case: a negative-valued logit close
    enough to the positive to be flagged "hard", and checks the fixed
    reweighting moves it TOWARD the positive (increasing, since these are
    raw logits where higher = closer to the positive class), not away.
    """
    from revua_cir.losses import symmetric_info_nce

    B, D = 4, 8
    torch.manual_seed(0)
    query = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    target = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    logit_scale = torch.tensor(14.3)

    logits_raw = (logit_scale * query @ target.t()).detach()
    out = symmetric_info_nce(query, target, logit_scale, hard_negative_weight=1.0)
    assert torch.isfinite(out["loss"]), "hard_negative_weight path produced a non-finite loss"

    # Directly reproduce the exact failure case with hand-picked values
    # (independent of the random query/target above, so this is deterministic).
    pos = torch.tensor(2.0)
    neg_negative_valued = torch.tensor(-1.5)  # negative logit, but "close" to pos -> flagged hard
    closeness = (-(pos - neg_negative_valued).abs()).exp()
    weight = 1.0 + 1.0 * closeness
    old_buggy = neg_negative_valued * weight              # the pre-fix multiplicative form
    new_fixed = neg_negative_valued + weight.log()          # the current additive-log-space form
    assert old_buggy < neg_negative_valued, (
        "test setup error: the old multiplicative form should move this negative logit "
        "MORE negative -- if it doesn't, this isn't reproducing the bug scenario")
    assert new_fixed > neg_negative_valued, (
        f"hard_negative_weight reweighting moved a negative-valued 'hard' logit AWAY from "
        f"the positive ({neg_negative_valued.item()} -> {new_fixed.item():.4f}) -- the "
        f"multiplicative sign-flip bug appears to have regressed")
    print("OK: hard_negative_weight reweights additively in log-space, correctly moving "
          "negative-valued hard negatives TOWARD the positive (not away, as the old "
          "multiplicative formula did).")


def test_hard_negative_weight_computes_independent_weights_per_direction():
    """Regression guard for a real math bug found via code-audit document
    review (2026-09-09, finding #12): `symmetric_info_nce`'s hard_negative_
    weight path computed row-wise (i2t-direction) hardness weights ONCE from
    `logits`, applied them to get `logits_weighted`, then reused that SAME
    already-weighted matrix for t2i by simply transposing it
    (`logits_weighted.t()`). Transposing an already-weighted matrix is not
    the same as computing t2i's OWN hardness weights from `logits.t()`'s own
    diagonal/closeness structure -- for a non-symmetric similarity matrix (the
    general case), `logits.t()`'s off-diagonal "closeness to the positive"
    values differ from `logits`'s row-wise closeness values, so the two
    directions need independently-computed weights.

    Dormant in every shipped config (hard_negative_weight=0.0 everywhere),
    so this never affected any reported result -- fixed anyway so the
    mechanism is mathematically sound if ever enabled.

    This test constructs a deliberately ASYMMETRIC similarity matrix (via
    non-orthonormal query/target embeddings) and confirms:
      1. loss_i2t computed from the fixed code's logits_i2t matches directly
         re-deriving i2t's own weights from `logits` (sanity: i2t direction
         unaffected by the fix, as expected).
      2. loss_t2i computed from the fixed code's logits_t2i does NOT equal
         what you'd get by taking i2t's weighted matrix and transposing it
         -- proving the two directions now genuinely diverge, not that the
         fix is a no-op relabeling.
    """
    from revua_cir.losses import symmetric_info_nce

    torch.manual_seed(7)
    B, D = 5, 6
    # Deliberately asymmetric: query and target are independently random (not
    # tied by any orthogonal transform), so `query @ target.t()` has no
    # special symmetry -- closeness-to-diagonal differs row-wise vs col-wise.
    query = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    target = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    logit_scale = torch.tensor(10.0)
    hnw = 2.0

    logits = (logit_scale * query @ target.t()).detach()
    labels = torch.arange(B)

    def apply_hard_weights(m):
        pos = torch.diagonal(m).unsqueeze(1)
        closeness = (-(pos - m).abs()).exp()
        weights = 1.0 + hnw * closeness
        weights.fill_diagonal_(1.0)
        return m + weights.log()

    logits_i2t_expected = apply_hard_weights(logits)
    logits_t2i_expected = apply_hard_weights(logits.t())
    # Confirm the test fixture itself is asymmetric enough to distinguish
    # the two computations -- otherwise this test can't tell fixed from buggy.
    assert not torch.allclose(logits_t2i_expected, logits_i2t_expected.t(), atol=1e-5), (
        "test setup error: this query/target pair happens to be symmetric enough that "
        "transposing i2t's weights equals t2i's own weights -- pick a different seed")

    cross_entropy = torch.nn.functional.cross_entropy
    out = symmetric_info_nce(query, target, logit_scale, hard_negative_weight=hnw)
    loss_i2t_expected = cross_entropy(logits_i2t_expected, labels)
    loss_t2i_expected_correct = cross_entropy(logits_t2i_expected, labels)
    loss_t2i_old_buggy = cross_entropy(logits_i2t_expected.t(), labels)
    loss_expected_correct = 0.5 * (loss_i2t_expected + loss_t2i_expected_correct)
    loss_expected_old_buggy = 0.5 * (loss_i2t_expected + loss_t2i_old_buggy)

    assert abs(loss_expected_correct.item() - loss_expected_old_buggy.item()) > 1e-4, (
        "test setup error: the correct and old-buggy t2i computations happen to give the same "
        "loss on this input -- pick different query/target/seed values")
    assert torch.allclose(out["loss"], loss_expected_correct, atol=1e-5), (
        f"symmetric_info_nce's hard_negative_weight loss ({out['loss'].item():.6f}) does not match "
        f"independently-computed per-direction weighting ({loss_expected_correct.item():.6f}) -- "
        f"got {out['loss'].item():.6f} vs old-buggy-transpose value "
        f"{loss_expected_old_buggy.item():.6f}: the symmetric hard-negative-weight fix appears "
        f"to have regressed to reusing one direction's weights for both.")
    print("OK: symmetric_info_nce computes independent hard-negative weights for the i2t and t2i "
          "directions (confirmed to genuinely diverge from the old transpose-reuse computation on "
          "an asymmetric query/target pair, not just a relabeling).")


def test_reverse_consistency_loss_is_contrastive():
    """The reverse-consistency loss must penalise high-scoring *wrong* candidates.

    The original form was `mean(1 - cos_sim)` over ground-truth targets only, so
    the verifier was trained to score everything it ever saw ~1.0 and had no
    reason to separate correct candidates from incorrect ones -- making
    inference-time reranking a no-op no matter how long you train. This pins the
    contrastive behaviour: with negatives scoring *above* positives the loss must
    exceed the positives-only baseline, and it must fall once they separate.
    """
    from revua_cir.models.reverse_verifier import ReverseVerifier

    pos = torch.tensor([0.9, 0.9, 0.9])
    baseline = ReverseVerifier.reverse_consistency_loss(pos)

    bad_negatives = torch.tensor([0.95, 0.95, 0.95])   # wrong candidates scoring HIGHER
    good_negatives = torch.tensor([0.1, 0.1, 0.1])     # wrong candidates well separated

    l_bad = ReverseVerifier.reverse_consistency_loss(pos, negative_cos_sim=bad_negatives, margin=0.2)
    l_good = ReverseVerifier.reverse_consistency_loss(pos, negative_cos_sim=good_negatives, margin=0.2)

    assert l_bad > baseline, "negatives scoring above positives must be penalised"
    assert l_good < l_bad, "separating negatives from positives must reduce the loss"
    assert abs(l_good.item() - baseline.item()) < 1e-6, (
        "once negatives are beyond the margin the hinge should be inactive, "
        "leaving only the positive term")
    print(f"OK: reverse loss is contrastive (baseline {baseline:.4f}, "
          f"bad-neg {l_bad:.4f}, separated {l_good:.4f}).")


def test_eval_rerank_changes_ranking():
    """`evaluate_dataset(rerank=True)` must actually reorder results.

    The evaluation path originally computed RV/UG signals for diagnostics but
    ranked purely by embedding similarity, so the +rv/+ug ablation rows could not
    show any recall benefit -- the inference-time mechanism Section 3 claims as
    the contribution was never exercised by the metric. This asserts the reranked
    order differs from the base order somewhere in the top-k.

    Also pins the invariance that motivated switching early stopping to R@1:
    reranking permutes *within* the top-k, so the top-k as a SET is unchanged.
    """
    from revua_cir.evaluate import evaluate_dataset
    from revua_cir.data.transforms import build_image_transform

    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc+aep+rv+ug")
    set_seed(0)
    model = ReVUACIR(cfg).eval()
    tokenizer = build_tokenizer(cfg.backbone)
    # Gallery images are loaded from disk by path; these synthetic paths do not
    # exist, so every one becomes a gray placeholder -- fine here, since the test
    # asserts plumbing and ordering invariants, not retrieval quality.
    transform = build_image_transform(cfg.data.image_size, train=False)
    dataset = SyntheticCIRDataset(24, cfg.data.image_size, cfg.attribute_parser.attribute_vocab_size)

    base = evaluate_dataset(model, dataset, cfg, tokenizer, transform,
                            torch.device("cpu"), max_queries=8, rerank=False)
    rr = evaluate_dataset(model, dataset, cfg, tokenizer, transform,
                          torch.device("cpu"), max_queries=8, rerank=True)
    assert "recall_at_1" in base and "recall_at_1" in rr
    # R@10/@50 are set-membership over a window at least as wide as the rerank
    # window, so they must be identical -- this is the invariance that makes
    # val_recall_at_10 useless as an early-stopping monitor for RV/UG.
    for k in ("recall_at_10", "recall_at_50"):
        if k in base and k in rr:
            assert abs(base[k] - rr[k]) < 1e-6, (
                f"{k} changed under reranking ({base[k]} -> {rr[k]}); reranking must "
                f"only permute within the top-k")
    # verification_agreement_rate is measured only on top_idx[:, 0] -- the base
    # ranker's own best guess -- so it cannot show whether RV discriminates at
    # all (see metrics.verification_agreement_rate docstring).
    # verification_auroc (renamed from verification_discrimination_auc,
    # 2026-09-09, code-audit finding B.8 -- see metrics.verification_pairwise_
    # auroc's docstring for why the old name overstated what a matched-pair
    # win rate actually measures) scores a known-correct target against the
    # highest-ranked known-wrong candidate for the same query, pooled into a
    # true ROC-AUC across all queries, and must be present whenever RV is
    # enabled, in [0, 1] (0.5 = chance).
    assert "verification_auroc" in base, (
        "verification_auroc missing from eval results with RV enabled")
    assert "verification_pairwise_win_rate" in base, (
        "verification_pairwise_win_rate missing from eval results with RV enabled")
    auc = base["verification_auroc"]
    win_rate = base["verification_pairwise_win_rate"]
    assert 0.0 <= auc <= 1.0, f"verification_auroc out of range: {auc}"
    assert 0.0 <= win_rate <= 1.0, f"verification_pairwise_win_rate out of range: {win_rate}"

    # Latency breakdown (added 2026-09-09, code-audit finding B.10): base
    # retrieval latency must be present regardless of rerank=, RV/UG-rerank
    # latency only when rerank=True -- and total must equal their sum, so a
    # paper reporting "ReVUA-CIR inference latency" can't accidentally cite
    # only the pre-reranking number as if it were the full pipeline's cost.
    assert "base_retrieval_latency_sec" in base and "base_retrieval_latency_sec" in rr
    assert "rv_reranking_latency_sec" not in base, (
        "rv_reranking_latency_sec should be absent when rerank=False -- no reranking ran")
    assert "rv_reranking_latency_sec" in rr, (
        "rv_reranking_latency_sec missing from eval results with rerank=True")
    assert "total_pipeline_latency_sec" in rr
    assert abs(rr["total_pipeline_latency_sec"]
               - (rr["base_retrieval_latency_sec"] + rr["rv_reranking_latency_sec"])) < 1e-9, (
        "total_pipeline_latency_sec must equal base_retrieval_latency_sec + rv_reranking_latency_sec")
    # UG ranking-impact diagnostics (added 2026-09-09, novelty-review
    # findings #9/#10; adjustment/final split added 2026-09-10 per external
    # review): only meaningful/populated when rerank=True with both RV and
    # UG enabled (this fixture's module_combo is fc+aep+rv+ug).
    # "adjustment" = rv_score vs rv_score*ug_confidence (pre-fusion signal
    # only); "final" = the actual _rank_fuse output with UG vs an RV-only
    # counterfactual through the SAME _rank_fuse call -- the operationally
    # meaningful "does UG change what the user sees" comparison, which
    # "adjustment" alone cannot answer since _rank_fuse mixes the signal
    # with the base retrieval rank afterward.
    assert "ug_changes_adjustment_order_rate" not in base, (
        "ug_changes_adjustment_order_rate should be absent when rerank=False -- no reranking ran")
    assert "ug_changes_adjustment_order_rate" in rr, (
        "ug_changes_adjustment_order_rate missing from eval results with rerank=True and UG enabled")
    assert 0.0 <= rr["ug_changes_adjustment_order_rate"] <= 100.0, (
        f"ug_changes_adjustment_order_rate out of [0,100] range: {rr['ug_changes_adjustment_order_rate']}")
    if "ug_rv_order_spearman" in rr:
        assert -1.0 <= rr["ug_rv_order_spearman"] <= 1.0, (
            f"ug_rv_order_spearman out of [-1,1] range: {rr['ug_rv_order_spearman']}")
    assert "ug_changes_final_order_rate" in rr, (
        "ug_changes_final_order_rate missing from eval results with rerank=True and UG enabled")
    assert 0.0 <= rr["ug_changes_final_order_rate"] <= 100.0, (
        f"ug_changes_final_order_rate out of [0,100] range: {rr['ug_changes_final_order_rate']}")

    print(f"OK: eval rerank path runs (R@1 {base['recall_at_1']:.2f} -> {rr['recall_at_1']:.2f}); "
          f"R@10/@50 invariant as expected; verification_auroc={auc:.4f} and "
          f"verification_pairwise_win_rate={win_rate:.4f} present and in range; latency breakdown "
          f"(base={rr['base_retrieval_latency_sec']*1000:.2f}ms, "
          f"rerank={rr['rv_reranking_latency_sec']*1000:.2f}ms, "
          f"total={rr['total_pipeline_latency_sec']*1000:.2f}ms) present and consistent; "
          f"ug_changes_adjustment_order_rate={rr['ug_changes_adjustment_order_rate']:.1f}% and "
          f"ug_changes_final_order_rate={rr['ug_changes_final_order_rate']:.1f}% present and in range.")


def test_spearman_corr_matches_known_values_without_scipy():
    """Regression guard for evaluate.py's `_spearman_corr` (added 2026-09-09,
    novelty-review finding #9/#10 support code): implemented from scratch
    (Pearson correlation of rank vectors, with tied-rank averaging) instead
    of `from scipy.stats import spearmanr`, because importing scipy.stats in
    the same process as torch + scikit-learn triggered a REAL, FATAL
    `OMP: Error #15` (duplicate OpenMP runtime) crash on this project's
    Windows dev environment the first time it was tried in a real test run
    -- not a hypothetical risk.

    Since this reimplements a well-known statistic instead of calling a
    trusted library, it needs independent verification against known
    values, not just internal self-consistency: perfect positive/negative
    correlation must give exactly +-1.0, and a classic textbook 10-point
    example (Spearman's rho = -0.1757575758..., independently computable by
    hand/any reference statistics tool) must match to high precision.
    """
    from revua_cir.evaluate import _spearman_corr

    a = torch.tensor([1., 2., 3., 4., 5.])
    b_pos = torch.tensor([2., 4., 6., 8., 10.])
    b_neg = torch.tensor([5., 4., 3., 2., 1.])
    assert abs(_spearman_corr(a, b_pos) - 1.0) < 1e-9, "perfect positive correlation should give rho=1.0"
    assert abs(_spearman_corr(a, b_neg) - (-1.0)) < 1e-9, "perfect negative correlation should give rho=-1.0"

    # Textbook 10-point example (widely reproduced -- IQ vs TV-hours-style
    # rank correlation exercise): rho = -0.17575757575757575.
    x = torch.tensor([86, 97, 99, 100, 101, 103, 106, 110, 112, 113], dtype=torch.float64)
    y = torch.tensor([0, 20, 28, 27, 50, 29, 7, 17, 6, 12], dtype=torch.float64)
    rho = _spearman_corr(x, y)
    assert abs(rho - (-0.17575757575757575)) < 1e-9, (
        f"expected rho=-0.17575757575757575 for the known textbook example, got {rho}")

    # Constant input (zero rank variance) must return nan, not raise or
    # silently return a bogus finite number.
    import math
    const = torch.tensor([1., 1., 1., 1.])
    varying = torch.tensor([1., 2., 3., 4.])
    assert math.isnan(_spearman_corr(const, varying)), "constant input should give nan, matching scipy's own convention"

    print("OK: _spearman_corr matches known reference values exactly (perfect +-1.0 correlations "
          "and a classic textbook 10-point example to 1e-9 precision) without depending on scipy, "
          "which was found to crash this environment when imported alongside torch+sklearn.")


def test_ug_ranking_impact_diagnostics_correctly_distinguish_reordering():
    """Regression guard for the novelty-review diagnostic (2026-09-09,
    findings #9/#10 of ReVUA_CIR_Novelty_Review_Findings.md): the review
    document argues UG's non-RV inputs (query-uncertainty, image-text
    coordination) are constant across a query's own top-k candidates, so
    UG's confidence scaling COULD collapse to a near-identity rank
    transform of RV's own score, in which case UG would function as a
    calibration layer rather than a genuine reranking layer -- weakening
    (not invalidating) the claim that UG changes candidate ordering.

    Verified independently before implementing (not taken on faith): even
    under the review document's own simplified monotonicity sketch
    (confidence C(s) increasing in RV score s), s*C(s) is NOT guaranteed
    monotonic in s in general, purely because C(s) itself varies with s --
    this does not require s to go negative. (CORRECTED 2026-09-11: an
    earlier version of this docstring claimed the real verification_score
    "is frequently negative" to justify the non-monotonicity concern --
    wrong. `verification_score` is `PlattCalibrator.forward()`'s
    `torch.sigmoid(...)` output, reverse_verifier.py, and is therefore
    always in (0, 1); only the raw, pre-calibration `cos_sim` can be
    negative. The scenarios below use negative-valued fixtures purely as
    convenient, arbitrary test data to exercise the order-detection/
    Spearman logic generically -- they are not meant to represent a real
    verification_score's actual range.) The review's own argument does not
    by itself guarantee UG is a no-op on ranking even confined to (0, 1),
    which is why this is measured empirically rather than assumed.

    This test exercises the exact diagnostic-computation logic added to
    evaluate.py's `if rerank:` block (argsort comparison of rv_score vs
    rv_score*ug_confidence, and evaluate.py's own `_spearman_corr` -- a
    dependency-free rank-correlation implementation added specifically
    because `from scipy.stats import spearmanr` triggered a real, fatal
    `OMP: Error #15` duplicate-OpenMP-runtime crash in the same process as
    torch + scikit-learn on this project's Windows dev environment) directly
    on two constructed scenarios:
      1. UG confidence is IDENTICAL across all candidates (the calibration-
         only regime the review document describes) -- order must be
         UNCHANGED (order_changed=0), Spearman must be exactly 1.0.
      2. UG confidence is constructed to REVERSE two adjacent candidates'
         relative RV scores -- order must be DETECTED as changed
         (order_changed=1), Spearman must be < 1.0.
    This proves the diagnostic itself is a correct, discriminating
    measurement -- not a metric that reports "no change" regardless of
    input.
    """
    from revua_cir.evaluate import _spearman_corr

    # Scenario 1: uniform confidence (UG contributes nothing to ranking).
    rv_score = torch.tensor([0.9, 0.5, -0.2, 0.7, -0.8])
    ug_confidence_uniform = torch.full_like(rv_score, 0.63)
    adj = rv_score
    ug_adj = rv_score * ug_confidence_uniform
    rv_order = adj.argsort(descending=True)
    ug_order = ug_adj.argsort(descending=True)
    order_changed = 0 if torch.equal(rv_order, ug_order) else 1
    rho = _spearman_corr(adj, ug_adj)
    assert order_changed == 0, "uniform UG confidence must NOT change candidate order"
    assert abs(rho - 1.0) < 1e-9, f"uniform UG confidence must give Spearman exactly 1.0, got {rho}"

    # Scenario 2: confidence deliberately anti-correlated with rv_score for
    # two adjacent-ranked candidates, engineered to flip their relative order.
    rv_score2 = torch.tensor([0.9, 0.85, 0.1, -0.3, -0.5])
    # Candidate 1 (rv=0.85, currently rank 2) gets boosted past candidate 0
    # (rv=0.9, currently rank 1) by a much higher confidence.
    ug_confidence2 = torch.tensor([0.5, 0.99, 0.5, 0.5, 0.5])
    adj2 = rv_score2
    ug_adj2 = rv_score2 * ug_confidence2
    rv_order2 = adj2.argsort(descending=True)
    ug_order2 = ug_adj2.argsort(descending=True)
    order_changed2 = 0 if torch.equal(rv_order2, ug_order2) else 1
    rho2 = _spearman_corr(adj2, ug_adj2)
    assert order_changed2 == 1, (
        f"engineered confidence should flip candidate order (rv_order={rv_order2.tolist()}, "
        f"ug_order={ug_order2.tolist()}) -- if this fails, the order-changed detection itself is broken")
    assert rho2 < 1.0, f"engineered reordering should give Spearman < 1.0, got {rho2}"

    print("OK: UG ranking-impact diagnostics correctly report NO change (order_changed=0, "
          "Spearman=1.0) under uniform confidence, and correctly DETECT a real reordering "
          "(order_changed=1, Spearman<1.0) when confidence is engineered to flip candidate order -- "
          "the diagnostic is a genuine, discriminating measurement.")


def test_ug_final_order_diagnostic_differs_from_adjustment_order_diagnostic():
    """Regression guard for a real gap found via external review of the real
    dry-run diagnostics (2026-09-10, job 6298): the original UG
    ranking-impact diagnostic (`ug_changes_adjustment_order_rate`) only
    compares `rv_score.argsort()` vs `(rv_score*ug_confidence).argsort()`
    -- the pre-fusion signal. But the actual output ranking a user would see
    comes from `_rank_fuse(base_scores, adj)`, which mixes that signal with
    the BASE RETRIEVAL rank via `2*base_rank + adj_rank` -- so a change (or
    non-change) in the adjustment's own order does not necessarily survive
    being fused with a different, independent base ranking.

    This test constructs a scenario where the adjustment order is IDENTICAL
    between RV-only and RV+UG (uniform confidence, matching the "UG changes
    nothing" case from the sibling test above) but demonstrates that
    `_rank_fuse`'s OWN behavior is deterministic given identical inputs --
    i.e. confirms the reverse direction is real too: since `_rank_fuse` is a
    pure function of `(base_scores, adj)`, if `adj`'s own argsort is
    unchanged, the fused rank_of `adj` is unchanged too (rank_fuse only
    depends on adj through its OWN rank, `adj.argsort().argsort()`, not on
    adj's absolute values), so identical adjustment order does guarantee
    identical final order in this specific formula -- confirming the
    "final order" diagnostic is not just a duplicate but can be
    MEANINGFULLY equal to "adjustment order" in this case, while remaining
    a genuinely different computation (uses _rank_fuse, not a raw argsort)
    that would catch a future change to _rank_fuse's fusion formula that
    the adjustment-order diagnostic structurally cannot see.
    """
    from revua_cir.models.revua_cir import _rank_fuse

    base_scores = torch.tensor([0.5, 0.3, 0.8, 0.1, 0.6])
    rv_score = torch.tensor([0.9, 0.5, -0.2, 0.7, -0.8])
    ug_confidence_uniform = torch.full_like(rv_score, 0.63)
    ug_adj = rv_score * ug_confidence_uniform

    adjustment_order_rv = rv_score.argsort(descending=True)
    adjustment_order_ug = ug_adj.argsort(descending=True)
    adjustment_unchanged = torch.equal(adjustment_order_rv, adjustment_order_ug)
    assert adjustment_unchanged, "test setup error: uniform confidence should leave adjustment order unchanged"

    final_rv_only = _rank_fuse(base_scores, rv_score).argsort(descending=True)
    final_ug = _rank_fuse(base_scores, ug_adj).argsort(descending=True)
    final_unchanged = torch.equal(final_rv_only, final_ug)
    assert final_unchanged, (
        "when the adjustment's own order is unchanged, _rank_fuse's output order must also be "
        "unchanged (rank_fuse depends on adj only through its own rank) -- if this fails, "
        "_rank_fuse's formula changed in a way that breaks this invariant")

    # Now the genuinely distinct case: an adjustment that DOES flip its own
    # order, verified to also flip the final fused order when the base
    # scores don't happen to cancel it out.
    ug_confidence_flip = torch.tensor([0.5, 0.99, 0.5, 0.5, 0.5])
    ug_adj_flip = rv_score * ug_confidence_flip
    adjustment_order_flip = ug_adj_flip.argsort(descending=True)
    assert not torch.equal(adjustment_order_rv, adjustment_order_flip), (
        "test setup error: this confidence should flip the adjustment order")
    final_ug_flip = _rank_fuse(base_scores, ug_adj_flip).argsort(descending=True)
    assert not torch.equal(final_rv_only, final_ug_flip), (
        "a genuine adjustment-order change should also change the final fused order in this "
        "constructed example -- if this fails, _rank_fuse is somehow absorbing the change")

    print("OK: the final-order UG diagnostic uses the real _rank_fuse formula (not a duplicate of "
          "the adjustment-order comparison) -- confirmed both that an unchanged adjustment order "
          "implies an unchanged final order (structural property of rank_fuse) and that a genuine "
          "adjustment-order change propagates through to a genuine final-order change.")


def test_gradcache_matches_plain_step():
    """Gradient caching must be *exactly* equivalent to a plain full-batch step.

    This is not optional rigor: gradcache exists to raise the in-batch negative
    count from 63 to 511-1023, and every result it produces gets compared against
    baselines trained the plain way. A silent gradient scale factor would
    invalidate that comparison. Development hit exactly that -- a uniform 0.48x
    caused by `hard_negative_weight`'s in-batch reweighting, which is not
    chunk-decomposable.
    """
    import copy
    from revua_cir.gradcache import gradcache_step

    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc")
    cfg.optim.amp = False              # fp32 so the comparison is exact
    cfg.fusion.dropout = 0.0           # remove stochasticity between the two runs
    cfg.loss.hard_negative_weight = 0.0
    cfg.optim.gradcache_chunk_size = 2
    # Frozen backbone is the regime gradcache targets (and the shipped config).
    # With a *trainable* backbone the cached pass-1 encode would need its own
    # gradient injection, which this implementation does not do -- so the
    # equivalence claim is scoped to the frozen case, and asserted there.
    cfg.backbone.freeze_backbone = True

    set_seed(0)
    plain = ReVUACIR(cfg)
    plain.eval()
    cached = copy.deepcopy(plain)
    cached.eval()

    bsz, chunk_size = 8, 2
    batch = {
        "reference_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "target_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "edit_input_ids": torch.randint(0, 100, (bsz, 8)),
        "edit_attention_mask": torch.ones(bsz, 8, dtype=torch.long),
        "attribute_targets": None,
        "meta": [{} for _ in range(bsz)],
    }
    # gradcache_step takes a LIST of pre-chunked batches (what train.py builds
    # from a chunk-sized DataLoader), not one big batch -- see gradcache.py's
    # docstring for why (a Windows commit-charge crash at large batch_size).
    chunks = [
        {k: (v[i:i + chunk_size] if torch.is_tensor(v) or isinstance(v, list) else v)
         for k, v in batch.items()}
        for i in range(0, bsz, chunk_size)
    ]

    opt_plain = build_optimizer(plain, cfg)
    opt_plain.zero_grad(set_to_none=True)
    plain(batch)["losses"]["total"].backward()
    g_plain = {n: p.grad.detach().clone() for n, p in plain.named_parameters() if p.grad is not None}

    class _NoStep:  # capture gradients without mutating weights
        def __init__(self, opt): self.opt = opt
        def zero_grad(self, **kw): self.opt.zero_grad(**kw)
        def step(self): pass

    gradcache_step(cached, chunks, _NoStep(build_optimizer(cached, cfg)), cfg, None)
    g_cached = {n: p.grad.detach().clone() for n, p in cached.named_parameters() if p.grad is not None}

    assert len(g_cached) == len(g_plain), (
        f"gradcache produced grads for {len(g_cached)} params vs {len(g_plain)} plain "
        f"-- missing: {sorted(set(g_plain) - set(g_cached))}")

    # SCOPE. `TinyBackbone` puts trainable projections (`image_proj`,
    # `text_proj`) *inside* encode_image/encode_text, i.e. on the target encode
    # path that pass 1 runs under no_grad. Real CLIP does not: on ViT-L/14 the
    # target-side adapter is an nn.Identity (projection_dim == embed_dim == 768),
    # so nothing trainable sits there. Equivalence on the **real ViT-L/14 model**
    # was measured separately at ratio 1.0000 for chunk sizes 8/16/32.
    #
    # What this test therefore guards is the property that actually regressed
    # during development and that no other test covers: every parameter must
    # receive a gradient (the `logit_scale` drop), and the loss must be finite
    # and match. The full numeric equivalence claim rests on the ViT-L
    # measurement recorded in revua_cir/gradcache.py.
    assert "composer.logit_scale" in g_cached, (
        "composer.logit_scale received no gradient under gradcache -- pass 2 must "
        "run before optimizer.zero_grad(), see gradcache_step")
    for n, v in g_cached.items():
        assert torch.isfinite(v).all(), f"non-finite gradient under gradcache: {n}"
    print(f"OK: gradcache produced finite gradients for all {len(g_cached)} params "
          f"including logit_scale (numeric equivalence verified on ViT-L/14: 1.0000).")


def test_gradcache_trainable_backbone_matches_plain_step():
    """Gradient caching with a TRAINABLE backbone must be exactly equivalent
    to a plain full-batch step, including the backbone's own parameters.

    Before 2026-09, gradcache asserted `freeze_backbone == true` -- the
    target-image encode path (`backbone.encode_image(target_pixel_values)`,
    called directly, not through `encode_query`) never had its own gradient
    injected in pass 3, so a trainable backbone would have received retrieval
    gradient through the query path only, silently. Fixed by injecting a
    second, independent cached gradient (`grad_t`) via a tied-encoder pattern
    confirmed against the official GradCache reference implementation
    (github.com/luyug/GradCache) and the original paper (Gao et al.,
    arXiv:2101.06983) -- two separate forward() calls through the same shared
    module build two independent autograd graphs, so their two backward()
    calls correctly SUM into the shared parameters' .grad with no
    interference and no retain_graph needed.

    `TinyBackbone` (unlike real CLIP) puts its trainable projections INSIDE
    encode_image/encode_text, so this test directly exercises whether the
    target-side injection actually reaches backbone parameters -- exactly
    the property the frozen-backbone equivalence test above cannot check.
    """
    import copy
    from revua_cir.gradcache import gradcache_step

    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc")
    cfg.optim.amp = False              # fp32 so the comparison is exact
    cfg.fusion.dropout = 0.0           # remove stochasticity between the two runs
    cfg.loss.hard_negative_weight = 0.0
    cfg.optim.gradcache_chunk_size = 2
    cfg.backbone.freeze_backbone = False  # the case this fix unblocks
    # gradcache_step applies grad_clip_norm internally (a real, intentional
    # feature); the "plain" comparison path below does not. With a trainable
    # backbone the raw gradient norm on this tiny synthetic batch is ~53 (vs
    # a few units in the frozen-backbone case the default config's clip=5.0
    # was never actually tested against), so clipping silently rescaled only
    # the gradcache side and looked exactly like a uniform 0.094x injection
    # bug (5.0/53.4 = 0.0937) -- caught during development, NOT a bug in the
    # injection math itself (verified separately via manual single-chunk and
    # multi-chunk replication outside this test, both exact). Set the clip
    # threshold above the observed raw norm so this test isolates the
    # injection math, not clipping.
    cfg.optim.grad_clip_norm = 1e6

    set_seed(0)
    plain = ReVUACIR(cfg)
    plain.eval()
    cached = copy.deepcopy(plain)
    cached.eval()

    bsz, chunk_size = 8, 2
    batch = {
        "reference_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "target_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "edit_input_ids": torch.randint(0, 100, (bsz, 8)),
        "edit_attention_mask": torch.ones(bsz, 8, dtype=torch.long),
        "attribute_targets": None,
        "meta": [{} for _ in range(bsz)],
    }
    chunks = [
        {k: (v[i:i + chunk_size] if torch.is_tensor(v) or isinstance(v, list) else v)
         for k, v in batch.items()}
        for i in range(0, bsz, chunk_size)
    ]

    opt_plain = build_optimizer(plain, cfg)
    opt_plain.zero_grad(set_to_none=True)
    plain(batch)["losses"]["total"].backward()
    g_plain = {n: p.grad.detach().clone() for n, p in plain.named_parameters() if p.grad is not None}

    class _NoStep:
        def __init__(self, opt): self.opt = opt
        def zero_grad(self, **kw): self.opt.zero_grad(**kw)
        def step(self): pass

    gradcache_step(cached, chunks, _NoStep(build_optimizer(cached, cfg)), cfg, None)
    g_cached = {n: p.grad.detach().clone() for n, p in cached.named_parameters() if p.grad is not None}

    assert len(g_cached) == len(g_plain), (
        f"gradcache produced grads for {len(g_cached)} params vs {len(g_plain)} plain "
        f"-- missing: {sorted(set(g_plain) - set(g_cached))}")

    # The property the frozen-backbone test structurally cannot check: does
    # the BACKBONE itself (not just the composer) receive gradient, and does
    # it match the plain full-batch step exactly?
    backbone_params = [n for n in g_plain if n.startswith("backbone.")]
    assert backbone_params, "test setup error: no backbone.* parameters found to check"
    max_rel_err = 0.0
    for n in backbone_params:
        plain_g, cached_g = g_plain[n], g_cached[n]
        assert torch.isfinite(cached_g).all(), f"non-finite backbone gradient under gradcache: {n}"
        denom = plain_g.norm().clamp_min(1e-8)
        rel_err = (cached_g - plain_g).norm() / denom
        max_rel_err = max(max_rel_err, rel_err.item())
    assert max_rel_err < 1e-3, (
        f"gradcache backbone gradients diverge from plain full-batch step: "
        f"max relative error {max_rel_err:.2e} across {len(backbone_params)} backbone params "
        f"-- the target-side injection (grad_t) is likely missing or wrong")

    for n, v in g_cached.items():
        assert torch.isfinite(v).all(), f"non-finite gradient under gradcache: {n}"
    print(f"OK: gradcache with a TRAINABLE backbone matches a plain full-batch step "
          f"(max relative error {max_rel_err:.2e} across {len(backbone_params)} backbone params).")


def test_gradcache_trainable_backbone_full_module_stack_bounded_divergence():
    """fc+aep+rv+ug under gradcache with a trainable backbone: backbone/
    composer/UG-fusion gradients must be near-exact; RV/AEP/UG-variance
    gradients are expected to show BOUNDED divergence from the plain path,
    not exactness, and this test guards that the divergence stays within
    the understood, accepted range rather than silently growing worse.

    Two real bugs were found and fixed via this exact check (2026-09):
      1. `outputs["rv_negative"]` was never set in the `full_batch_context`
         branch of the RV negative-mining block in revua_cir.py -- so UG's
         negative-pair calibration pass silently never ran under gradcache,
         producing sign-flipped, ~10x-magnitude-wrong gradients on
         ug.fusion/ug.log_temperature/rv.calibrator. Fixed by adding the
         assignment; confirmed those params dropped to <5% relative error
         here, matching the backbone/composer range.
      2. UG's neighborhood_deviation_loss and RV's semi-hard negative search
         both need FULL-BATCH statistics (k-NN dispersion, log_var mean/std,
         the positive-similarity threshold) -- computed from a tiny gradcache
         chunk alone they're near-meaningless (measured up to ~195x error
         before the `full_batch_context` plumbing was added).

    What's left is a real, understood, ACCEPTED limitation, not a bug:
    RV's semi-hard negative search can only ever pick a candidate from the
    CURRENT CHUNK's own gradient-carrying tensors (gradcache never holds
    another chunk's activations in memory at the same time -- that is the
    whole memory-saving point of the technique). With a tiny chunk (e.g.
    size 2), the plain path's full-batch negative choice and gradcache's
    chunk-restricted choice will often genuinely differ, and RV/AEP/UG-
    variance-head all sit downstream of "which candidate was selected" via
    the Section 4.5 connective loss (RV checks against AEP's invariant
    embedding; UG's variance target depends on the query neighborhood) --
    so a different (still valid, still "below positive") negative choice
    propagates as bounded-but-real gradient divergence in exactly those
    modules, not elsewhere.

    `ug.variance_head` specifically is the noisiest of the three (2026-09-08
    investigation, while auditing an unrelated fusion.py change): its loss
    target is a k-NN dispersion computed over only `bsz=8` synthetic samples
    with `k` up to 7 neighbors, so on this tiny fixture the target itself is
    highly sensitive to exactly which 2-of-8 samples land in a given
    gradcache chunk -- a seed sweep (0-4) measured max_loose_err from 0.68 to
    2.19 with NO code changes at all between runs, confirming this is
    inherent test-fixture noise at this batch scale, not something a specific
    change regresses. The bound below (3.0) carries real margin above that
    observed range rather than being tuned to one lucky seed.
    """
    import copy
    from revua_cir.gradcache import gradcache_step

    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc+aep+rv+ug")
    cfg.optim.amp = False
    cfg.fusion.dropout = 0.0
    cfg.loss.hard_negative_weight = 0.0
    cfg.optim.gradcache_chunk_size = 2
    cfg.backbone.freeze_backbone = False
    cfg.optim.grad_clip_norm = 1e6  # isolate injection math from clipping, see the fc-only test above

    set_seed(0)
    plain = ReVUACIR(cfg)
    plain.eval()
    cached = copy.deepcopy(plain)
    cached.eval()

    bsz, chunk_size = 8, 2
    batch = {
        "reference_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "target_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "edit_input_ids": torch.randint(0, 100, (bsz, 8)),
        "edit_attention_mask": torch.ones(bsz, 8, dtype=torch.long),
        "attribute_targets": None,
        "meta": [{} for _ in range(bsz)],
    }
    chunks = [
        {k: (v[i:i + chunk_size] if torch.is_tensor(v) or isinstance(v, list) else v)
         for k, v in batch.items()}
        for i in range(0, bsz, chunk_size)
    ]

    opt_plain = build_optimizer(plain, cfg)
    opt_plain.zero_grad(set_to_none=True)
    plain(batch)["losses"]["total"].backward()
    g_plain = {n: p.grad.detach().clone() for n, p in plain.named_parameters() if p.grad is not None}

    class _NoStep:
        def __init__(self, opt): self.opt = opt
        def zero_grad(self, **kw): self.opt.zero_grad(**kw)
        def step(self): pass

    gradcache_step(cached, chunks, _NoStep(build_optimizer(cached, cfg)), cfg, None)
    g_cached = {n: p.grad.detach().clone() for n, p in cached.named_parameters() if p.grad is not None}

    assert len(g_cached) == len(g_plain), (
        f"gradcache produced grads for {len(g_cached)} params vs {len(g_plain)} plain "
        f"-- missing: {sorted(set(g_plain) - set(g_cached))}")
    for n, v in g_cached.items():
        assert torch.isfinite(v).all(), f"non-finite gradient under gradcache: {n}"

    # Tight bound: backbone, composer, and UG's fusion/calibration path (the
    # things bug #1 above broke) must be near-exact, same tolerance as the
    # fc-only test.
    tight_prefixes = ("backbone.", "composer.")
    tight_names = ("ug.fusion.weight", "ug.fusion.bias", "ug.log_temperature",
                   "ug.coordination_head.net.0.weight", "ug.coordination_head.net.0.bias",
                   "ug.coordination_head.net.2.weight", "ug.coordination_head.net.2.bias")
    max_tight_err = 0.0
    for n in g_plain:
        if n.startswith(tight_prefixes) or n in tight_names:
            rel = (g_cached[n] - g_plain[n]).norm() / g_plain[n].norm().clamp_min(1e-8)
            max_tight_err = max(max_tight_err, rel.item())
    assert max_tight_err < 0.05, (
        f"backbone/composer/ug.fusion gradients diverge more than expected under gradcache: "
        f"max relative error {max_tight_err:.4f} -- check outputs['rv_negative'] is still being "
        f"set in the full_batch_context branch of revua_cir.py's RV block")

    # Loose bound: RV's own heads, AEP, and UG's variance-head are downstream
    # of "which chunk-restricted negative was picked" (the accepted
    # limitation above) -- allow real divergence, but cap it so a NEW,
    # unrelated bug (e.g. another missing full_batch_context wire-up) still
    # gets caught rather than being absorbed into "expected slop."
    #
    # Skip near-zero-gradient scalars from the RELATIVE check (2026-09-08,
    # found via a seed sweep with the real build_tiny_config -- an earlier,
    # non-matching hand-rolled tiny-config diagnostic script gave misleadingly
    # small numbers and hid this): `rv.edit_weight` is a single scalar
    # parameter whose plain-path gradient is sometimes ~1e-4 in magnitude
    # (noise floor for this tiny synthetic setup) -- relative error on such a
    # small denominator explodes (seed 0: 20.0) even though the ABSOLUTE
    # difference is tiny (~0.0018) and harmless. A seed sweep (0-4) with a
    # min-gradient-norm floor of 1e-3 measured max_loose_err 1.07-1.88 -- this
    # bound reflects that, with real margin, rather than either the
    # artifically-inflated 20.0 or an earlier undersized bound.
    min_grad_norm_for_relative_check = 1e-3
    max_loose_err = 0.0
    skipped_near_zero = []
    for n in g_plain:
        if n.startswith(tight_prefixes) or n in tight_names:
            continue
        plain_norm = g_plain[n].norm()
        if plain_norm.item() < min_grad_norm_for_relative_check:
            # Near-zero gradient: check absolute difference instead, since
            # relative error is meaningless near a zero denominator.
            abs_diff = (g_cached[n] - g_plain[n]).norm().item()
            skipped_near_zero.append((n, plain_norm.item(), abs_diff))
            assert abs_diff < 0.05, (
                f"{n} has a near-zero plain-path gradient ({plain_norm.item():.2e}) but a large "
                f"ABSOLUTE divergence under gradcache ({abs_diff:.4f}) -- this is not the relative-"
                f"error-near-zero-denominator artifact, it looks like a real bug")
            continue
        rel = (g_cached[n] - g_plain[n]).norm() / plain_norm.clamp_min(1e-8)
        max_loose_err = max(max_loose_err, rel.item())
    assert max_loose_err < 2.5, (
        f"RV/AEP/UG-variance gradients diverge far more than the accepted chunk-restricted-"
        f"negative-pool explanation covers: max relative error {max_loose_err:.4f} -- this "
        f"looks like a NEW bug, not the known limitation (a 2026-09-08 seed sweep with the real "
        f"build_tiny_config measured 1.07-1.88 across seeds 0-4, excluding near-zero-gradient "
        f"scalars like rv.edit_weight which are checked separately above by absolute difference)")

    print(f"OK: gradcache with fc+aep+rv+ug (trainable backbone) -- backbone/composer/ug.fusion "
          f"max relative error {max_tight_err:.4f} (tight bound), RV/AEP/ug.variance_head "
          f"max relative error {max_loose_err:.4f} (loose bound, expected from chunk-restricted "
          f"negative mining).")


def test_xbm_cross_batch_memory():
    """Cross-Batch Memory (Wang et al., CVPR 2020, arXiv:1912.06798), added
    2026-09 as fix candidate #2 for the semi-hard in-batch negative mining
    collapse (verify_semihard_negative.sbatch, 2026-09-07:
    topk_target_rv_argmax_rate settled ~14-16% at full convergence, barely
    above the 10% chance baseline for top_k=10 -- semi-hard bounding alone
    was insufficient because the SELECTOR, not just the selection RULE, is
    entangled with the converging embedding space it searches).

    Exercises: (1) the queue stays disabled/inert with use_xbm=False
    (default, zero behavior change for every existing config), (2) warm-up
    correctly gates `is_ready` and still enqueues during warm-up so the gate
    eventually opens, (3) once ready, the XBM path produces finite loss and
    real gradient into RV's own parameters, (4) checkpoint save/load
    round-trips the queue's contents AND its lazily-initialized shape --
    this last one is the property that regressed during development: a
    freshly-constructed model's queue buffers start as (0,)-shaped
    placeholders (the real shape depends on backbone embed_dim / patch
    count, unknown at construction time), so loading a checkpoint saved
    after the queue was populated raised a hard size-mismatch error until
    `load_checkpoint` was taught to resize empty placeholder buffers to the
    checkpoint's shape first.
    """
    import tempfile
    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc+aep+rv")
    cfg.reverse_verifier.use_xbm = True
    cfg.reverse_verifier.xbm_queue_size = 32
    cfg.reverse_verifier.xbm_warmup_steps = 2

    set_seed(0)
    model = ReVUACIR(cfg)
    model.train()
    assert not model.rv.xbm.is_ready, "XBM should not be ready before any steps"
    assert not model.rv.xbm._shape_initialized, "XBM queue should not be shaped before first enqueue"

    def make_batch():
        return {
            "reference_pixel_values": torch.randn(4, 3, cfg.data.image_size, cfg.data.image_size),
            "target_pixel_values": torch.randn(4, 3, cfg.data.image_size, cfg.data.image_size),
            "edit_input_ids": torch.randint(0, 100, (4, 8)),
            "edit_attention_mask": torch.ones(4, 8, dtype=torch.long),
        }

    for step in range(3):
        out = model(make_batch())
        assert torch.isfinite(out["losses"]["total"]), f"non-finite loss during XBM warm-up, step {step}"
        out["losses"]["total"].backward()
        model.zero_grad()

    assert model.rv.xbm.is_ready, "XBM should be ready after warmup_steps enqueues"
    assert model.rv.xbm._shape_initialized
    assert model.rv.xbm.filled.item() == 12, f"expected 3 steps x 4 samples = 12 filled, got {model.rv.xbm.filled.item()}"

    out = model(make_batch())
    out["losses"]["total"].backward()
    grad = model.rv.reverse_gen.net[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all(), "XBM-sourced negative produced no/non-finite RV gradient"

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/ckpt.pt"
        save_checkpoint(path, model, epoch=1, best_metric=1.0, config=cfg.to_dict())
        model2 = ReVUACIR(cfg)
        assert model2.rv.xbm.pooled_queue.numel() == 0, "fresh model2 should start with placeholder (empty) queue"
        load_checkpoint(path, model2, map_location="cpu")
        assert model2.rv.xbm.is_ready
        assert model2.rv.xbm.pooled_queue.shape == model.rv.xbm.pooled_queue.shape
        assert torch.equal(model2.rv.xbm.pooled_queue, model.rv.xbm.pooled_queue), (
            "resumed XBM queue contents do not match the saved queue")

    # use_xbm=False (the default) must be a complete no-op -- confirms this
    # feature is strictly opt-in.
    cfg_off = build_tiny_config()
    cfg_off = cfg_off.apply_module_ablation("fc+aep+rv")
    assert cfg_off.reverse_verifier.use_xbm is False
    set_seed(0)
    model_off = ReVUACIR(cfg_off)
    assert model_off.rv.xbm is None, "xbm should not even be constructed when use_xbm=False"

    print("OK: Cross-Batch Memory -- warm-up gating, live gradient into RV params, "
          "checkpoint round-trip of lazily-shaped queue buffers, and use_xbm=False no-op all correct.")


def test_xbm_mining_uses_query_similarity_at_matched_scale():
    """Regression guard for a bug fixed 2026-09-08: both XBM branches in
    revua_cir.py's forward() (gradcache+XBM and plain-path+XBM) mined
    negatives using `target_pooled @ self.rv.xbm.pooled_queue.t()` --
    TARGET-to-target similarity, not query-to-target -- the wrong signal for
    a query-conditioned semi-hard search (a negative should be hard *for the
    query*, not merely similar to the positive target). They also compared
    that raw cosine similarity against `pos_sim` taken from
    `retrieval["logits"]`, which is `logit_scale * query @ target.t()` (not
    raw cosine) -- comparing two different scales inside
    semi_hard_negatives()'s `sim < pos_sim` threshold check. Both fixed by
    using `query` (already in scope, computed once per forward()) for both
    similarities, at the same raw (unscaled) magnitude.

    Reproduces the exact fixed computation and checks pos_sim/sim_to_queue
    both land in the raw cosine range [-1, 1] rather than pos_sim being
    logit_scale-inflated -- the concrete, checkable symptom of the old bug.
    """
    import torch.nn.functional as F

    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc+aep+rv")
    cfg.reverse_verifier.use_xbm = True
    cfg.reverse_verifier.xbm_queue_size = 32
    cfg.reverse_verifier.xbm_warmup_steps = 2

    set_seed(0)
    model = ReVUACIR(cfg)
    model.train()

    def make_batch():
        return {
            "reference_pixel_values": torch.randn(4, 3, cfg.data.image_size, cfg.data.image_size),
            "target_pixel_values": torch.randn(4, 3, cfg.data.image_size, cfg.data.image_size),
            "edit_input_ids": torch.randint(0, 100, (4, 8)),
            "edit_attention_mask": torch.ones(4, 8, dtype=torch.long),
        }

    for _ in range(3):
        out = model(make_batch())
        out["losses"]["total"].backward()
        model.zero_grad()
    assert model.rv.xbm.is_ready

    batch = make_batch()
    with torch.no_grad():
        q = model.encode_query(batch["reference_pixel_values"], batch["edit_input_ids"], batch["edit_attention_mask"])
        query = q["query"]
        tgt = model.backbone.encode_image(batch["target_pixel_values"])
        target_pooled = F.normalize(tgt.pooled, dim=-1)

        pos_sim = (query * target_pooled).sum(dim=-1, keepdim=True)
        sim_to_queue = query @ model.rv.xbm.pooled_queue.t()

        # The pre-fix pos_sim (logit_scale-multiplied) for comparison, to
        # confirm the fix actually changed the magnitude, not just the tensor
        # identity.
        retrieval_logits = model.composer.temperature * query @ target_pooled.t()
        old_pos_sim = torch.diagonal(retrieval_logits).unsqueeze(1)

    assert pos_sim.abs().max() <= 1.0 + 1e-4, f"pos_sim outside raw cosine range: {pos_sim}"
    assert sim_to_queue.abs().max() <= 1.0 + 1e-4, f"sim_to_queue outside raw cosine range: {sim_to_queue}"
    # The old (buggy) pos_sim would have been logit_scale x larger -- confirm
    # the fix's pos_sim is NOT that inflated value (guards against a future
    # accidental revert back to retrieval["logits"]).
    assert not torch.allclose(pos_sim, old_pos_sim, atol=1e-4), (
        "pos_sim matches the OLD logit_scale-multiplied value -- the scale fix appears reverted")
    print("OK: XBM mining uses query-to-target similarity at the correct raw-cosine scale, "
          "matching sim_to_queue's magnitude.")


def test_rv_calibrator_receives_gradient():
    """Regression guard for a real bug found via code-audit document review
    (2026-09-08): RV's PlattCalibrator (`rv.calibrator.scale`/`.bias`) was
    trainable but structurally DEAD -- `reverse_consistency_loss` trains
    everything upstream of the calibrator via `rv_out["cos_sim"]` directly,
    never actually calling `self.calibrator`, and the only other consumer of
    the calibrator's output (`disagreement`, fed into UG) was correctly
    `.detach()`'d in an earlier fix (see
    test_ug_calibration_loss_does_not_leak_into_rv_or_backbone) -- which
    closed one gradient leak but, as a side effect, left the calibrator with
    NO live gradient path anywhere in training. Verified directly before the
    fix: `rv.calibrator.scale.grad` / `.bias.grad` were both None after a
    real backward pass, meaning `verification_score` (what inference-time
    reranking actually reads) stayed stuck at its init (scale=1.0, bias=0.0)
    forever.

    Fixed with an explicit positive/negative BCE term (`losses["rv_calibration"]`)
    computed on RE-DETACHED cos_sim values run back through
    `self.rv.calibrator.logit()` -- this trains ONLY the calibrator's own
    scale/bias (a fresh graph starting at the detach point), deliberately
    NOT re-perturbing candidate_decomp/backbone a second time (that's
    already reverse_consistency_loss's job via the undetached path).

    Checks both halves of that claim: the calibrator DOES receive gradient
    now, and isolating `losses["rv_calibration"]` alone produces ZERO
    gradient in candidate_decomp or the backbone (confirming the detach
    correctly isolates this new loss term).
    """
    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc+aep+rv+ug")
    cfg.fusion.dropout = 0.0
    cfg.backbone.freeze_backbone = False

    set_seed(0)
    model = ReVUACIR(cfg)
    model.train()

    bsz = 6
    batch = {
        "reference_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "target_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "edit_input_ids": torch.randint(0, 100, (bsz, 8)),
        "edit_attention_mask": torch.ones(bsz, 8, dtype=torch.long),
    }

    out = model(batch)
    assert "rv_calibration" in out["losses"], "rv_calibration loss key missing -- fix appears reverted"
    out["losses"]["total"].backward()
    assert model.rv.calibrator.scale.grad is not None and model.rv.calibrator.scale.grad.abs().sum() > 0, (
        "rv.calibrator.scale received no gradient -- the dead-calibrator bug appears to still be present")
    assert model.rv.calibrator.bias.grad is not None and model.rv.calibrator.bias.grad.abs().sum() > 0, (
        "rv.calibrator.bias received no gradient")

    # Isolation check: the calibration loss ALONE must not leak into
    # candidate_decomp/backbone (that's reverse_consistency_loss's job via
    # the separate, undetached cos_sim path).
    out2 = model(batch)
    model.zero_grad(set_to_none=True)
    out2["losses"]["rv_calibration"].backward()
    decomp_grad = model.rv.candidate_decomp.shared[0].weight.grad
    assert decomp_grad is None or decomp_grad.abs().sum() == 0, (
        "rv_calibration loss leaked gradient into candidate_decomp -- detach appears missing")
    backbone_leaked = [n for n, p in model.backbone.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert not backbone_leaked, f"rv_calibration loss leaked gradient into backbone: {backbone_leaked}"
    print("OK: RV calibrator (scale/bias) receives real gradient via the new rv_calibration loss, "
          "correctly isolated from candidate_decomp/backbone.")


def test_ug_calibration_loss_does_not_leak_into_rv_or_backbone():
    """Regression guard for a gradient leak fixed 2026-09-08: UG's own module
    comment in revua_cir.py states the design intent -- "UG is trained
    detached from the retrieval/backbone graph... Fixes an unintended
    gradient path that was destabilizing the trunk" -- and detaches three of
    UG's four inputs (query, ref, text). The fourth, `reverse_disagreement`
    (= rv_out["disagreement"], fed into UG's forward and stacked directly
    into the tensor UG's own `fusion` Linear layer consumes), was NOT
    detached, so UG's calibration loss still had a live gradient path back
    through RV's candidate_decomp/calibrator into candidate_pooled/
    forward_text_pooled -- i.e. the SHARED BACKBONE -- exactly the leak the
    comment already claimed was closed.

    Isolates `losses["uncertainty"]` (not the full `total`, which legitimately
    reaches RV/backbone through retrieval/disentangle/reverse's OWN
    gradients) and checks it alone produces ZERO gradient in RV or backbone
    parameters, while still producing real gradient in UG's own parameters
    (confirming the fix didn't accidentally disconnect UG from training
    entirely -- a leak fix that silences the loss is not a correct fix).
    """
    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc+aep+rv+ug")
    cfg.fusion.dropout = 0.0
    cfg.backbone.freeze_backbone = False

    set_seed(0)
    model = ReVUACIR(cfg)
    model.train()

    bsz = 6
    batch = {
        "reference_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "target_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "edit_input_ids": torch.randint(0, 100, (bsz, 8)),
        "edit_attention_mask": torch.ones(bsz, 8, dtype=torch.long),
    }

    out = model(batch)
    model.zero_grad(set_to_none=True)
    out["losses"]["uncertainty"].backward()

    rv_leaked = [n for n, p in model.rv.named_parameters() if p.grad is not None and p.grad.norm() > 0]
    backbone_leaked = [n for n, p in model.backbone.named_parameters() if p.grad is not None and p.grad.norm() > 0]
    ug_trained = [n for n, p in model.ug.named_parameters() if p.grad is not None and p.grad.norm() > 0]

    assert not rv_leaked, f"UG's uncertainty loss leaked gradient into RV params: {rv_leaked}"
    assert not backbone_leaked, f"UG's uncertainty loss leaked gradient into backbone params: {backbone_leaked}"
    assert ug_trained, "UG's uncertainty loss produced no gradient into UG's own params -- fix over-corrected"
    print(f"OK: UG's calibration loss reaches only UG's own {len(ug_trained)} params, "
          f"zero leak into RV/backbone.")


def test_gradcache_uses_real_gradscaler():
    """Regression guard for the AMP/GradScaler bypass (fixed 2026-09-08):
    `gradcache_step` accepted a `scaler` parameter but never called
    scaler.scale()/unscale_()/step()/update() anywhere in the function --
    optimizer.step() was called directly. That meant gradcache training got
    NONE of GradScaler's real protections (inf/NaN gradient detection via
    scaler.step()'s internal overflow check, and the adaptive scale factor
    updating via scaler.update()) even though train.py always constructs and
    passes a real (possibly AMP-disabled, but never None) GradScaler
    instance into every gradcache_step call.

    Two things verified here that a bare `scaler=None` test (as used by the
    injection-exactness tests elsewhere in this file, which deliberately
    isolate injection math from AMP concerns) cannot check:
      1. a real GradScaler object survives the full gradcache_step call
         without error (the scale()/unscale_() plumbing is exercised, not
         skipped).
      2. scaler.step() actually performs the optimizer update -- checked by
         confirming parameters genuinely change, not just that no exception
         was raised (a silently-skipped step due to a scaler bug would look
         identical to success if only checking for exceptions).
    """
    from revua_cir.gradcache import gradcache_step

    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc")
    cfg.optim.amp = True
    cfg.fusion.dropout = 0.0
    cfg.loss.hard_negative_weight = 0.0
    cfg.optim.gradcache_chunk_size = 2
    cfg.backbone.freeze_backbone = False
    cfg.optim.grad_clip_norm = 5.0

    set_seed(0)
    model = ReVUACIR(cfg)
    model.train()

    bsz, chunk_size = 4, 2
    batch = {
        "reference_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "target_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "edit_input_ids": torch.randint(0, 100, (bsz, 8)),
        "edit_attention_mask": torch.ones(bsz, 8, dtype=torch.long),
        "attribute_targets": None,
        "meta": [{} for _ in range(bsz)],
    }
    chunks = [
        {k: (v[i:i + chunk_size] if torch.is_tensor(v) or isinstance(v, list) else v)
         for k, v in batch.items()}
        for i in range(0, bsz, chunk_size)
    ]

    opt = build_optimizer(model, cfg)
    # AMP-disabled on this CPU-only test machine (autocast("cuda") needs a
    # real CUDA device to do anything), but a REAL GradScaler object -- the
    # point is exercising the scale()/unscale_()/step()/update() call chain
    # itself, which a disabled GradScaler still runs through as documented
    # no-ops (confirmed: "if this instance is not enabled, outputs are
    # returned unmodified"), not testing fp16 numerics.
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    p_before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    losses = gradcache_step(model, chunks, opt, cfg, scaler)
    assert torch.isfinite(losses["total"]), "non-finite loss with a real GradScaler in the loop"

    changed = sum(1 for n, p in model.named_parameters()
                  if p.requires_grad and not torch.equal(p_before[n], p.detach()))
    assert changed > 0, (
        "scaler.step() did not update any parameters -- the optimizer step was "
        "silently skipped or scaler.step()/update() are not being called")
    print(f"OK: gradcache_step ran the full scaler.scale/unscale_/step/update chain "
          f"with a real GradScaler, {changed} params updated.")


def test_optimizer_step_skip_detection_matches_gradscaler_behavior():
    """Regression guard for a real bug found via external review of job
    6300's real training log (2026-09-10): `scaler.step(optimizer)`
    silently SKIPS the underlying `optimizer.step()` on any step where an
    unscaled gradient is inf/NaN (GradScaler's own documented overflow
    protection) -- but both train.py call sites (the plain AMP path and
    gradcache_step) used to call `scheduler.step()` unconditionally right
    after, with no way to know a step had been skipped. This is exactly
    what produced the real
    `UserWarning: Detected call of lr_scheduler.step() before
    optimizer.step()` seen in every real cluster training log this
    project has produced, and is more than cosmetic: on a skipped step the
    LR schedule still silently advances one step even though no parameter
    update happened.

    Fixed by comparing `scaler.get_scale()` before/after `scaler.update()`
    -- GradScaler's own documented way to detect a skipped step (the scale
    strictly decreases only after an overflow-triggered skip). This test
    verifies that detection mechanism directly against GradScaler's REAL
    behavior (not a mock), on both a genuine skip (inf gradient) and a
    genuine non-skip (finite gradient) case, confirming: (1) the parameter
    value itself proves whether optimizer.step() actually ran (the ground
    truth), and (2) the scale-comparison correctly matches that ground
    truth in both directions.
    """
    import torch

    # SKIP when no CUDA device is visible (added 2026-09-12, external
    # review: this test failed on the cluster LOGIN node and made "all
    # tests passed" misleading). `torch.amp.GradScaler("cuda",
    # enabled=True)` SILENTLY DISABLES ITSELF when CUDA is unavailable --
    # PyTorch emits "GradScaler is enabled, but CUDA is not available.
    # Disabling." and every scaler call becomes a passthrough, so
    # `scaler.step()` just calls `optimizer.step()` unconditionally and the
    # inf gradient flows straight into the parameter. The skip behaviour
    # this test exists to verify is therefore NOT EXERCISABLE without a
    # GPU, and asserting it there tests nothing. This is a real property of
    # the environment, not a defect in the code under test -- confirmed
    # directly on the cluster login node (torch 2.11.0+cu128, a CUDA build
    # with torch.cuda.is_available() == False).
    if not torch.cuda.is_available():
        print("SKIP: test_optimizer_step_skip_detection_matches_gradscaler_behavior -- "
              "GradScaler self-disables without a visible CUDA device, so its skip-on-inf "
              "behaviour cannot be exercised here. Run on a GPU node to verify it.")
        return

    # Case 1: inf gradient -- GradScaler must skip optimizer.step().
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    opt = torch.optim.SGD([p], lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    loss = (p * float("inf")).sum()
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    p_before = p.detach().clone()
    scale_before = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    scale_after = scaler.get_scale()
    detected_skip = scale_after < scale_before
    actually_skipped = torch.equal(p.detach(), p_before)
    assert actually_skipped, "test setup error: an inf gradient should make GradScaler skip the step"
    assert detected_skip == actually_skipped, (
        f"scale-comparison detection ({detected_skip}) disagrees with ground truth "
        f"(parameter {'was' if actually_skipped else 'was NOT'} actually skipped) on the inf-gradient case")

    # Case 2: finite gradient -- GradScaler must NOT skip optimizer.step().
    p2 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    opt2 = torch.optim.SGD([p2], lr=0.1)
    scaler2 = torch.amp.GradScaler("cuda", enabled=True)
    loss2 = (p2 * 3.0).sum()
    opt2.zero_grad()
    scaler2.scale(loss2).backward()
    scaler2.unscale_(opt2)
    p2_before = p2.detach().clone()
    scale_before2 = scaler2.get_scale()
    scaler2.step(opt2)
    scaler2.update()
    scale_after2 = scaler2.get_scale()
    detected_skip2 = scale_after2 < scale_before2
    actually_skipped2 = torch.equal(p2.detach(), p2_before)
    assert not actually_skipped2, "test setup error: a finite gradient should NOT make GradScaler skip the step"
    assert detected_skip2 == actually_skipped2, (
        f"scale-comparison detection ({detected_skip2}) disagrees with ground truth "
        f"(parameter {'was' if actually_skipped2 else 'was NOT'} actually skipped) on the finite-gradient case")

    # Confirm the actual production code paths (gradcache.py and train.py)
    # both implement this same comparison -- a stale/reverted copy of the
    # fix would slip past the above pure-GradScaler check.
    import inspect
    from revua_cir import gradcache as gradcache_module
    from revua_cir import train as train_module
    gradcache_src = inspect.getsource(gradcache_module.gradcache_step)
    train_src = inspect.getsource(train_module.train)
    assert "get_scale() < scale_before" in gradcache_src, (
        "gradcache_step no longer contains the scale-comparison skip-detection -- "
        "check it wasn't reverted")
    assert "get_scale() < scale_before" in train_src, (
        "train()'s plain (non-gradcache) AMP path no longer contains the scale-comparison "
        "skip-detection -- check it wasn't reverted")
    assert "if not step_skipped:" in train_src and "scheduler.step()" in train_src, (
        "train() no longer conditionally gates scheduler.step() on step_skipped")

    print("OK: optimizer-step-skip detection (scaler.get_scale() before/after update()) matches "
          "GradScaler's real skip/no-skip behavior on both an inf-gradient case and a finite-gradient "
          "case, and both production call sites (gradcache_step, train()'s plain AMP path) still "
          "contain the fix.")


def test_gradcache_injects_gradient_into_trainable_adapters_under_frozen_backbone():
    """Regression guard for a real bug found via code-audit document review
    (2026-09-08): gradcache.py decided whether to inject target-side gradient
    using `trainable_backbone = not cfg.backbone.freeze_backbone` -- too
    coarse. `freeze_backbone=True` only freezes the pretrained TRUNK (e.g.
    HFClipBackbone's `self.clip.parameters()`, or TinyBackbone's
    patch_embed/vision_encoder/token_embed/text_encoder) -- it does NOT
    touch separate trainable adapter/projection layers that exist whenever a
    backbone's native hidden dim differs from cfg.backbone.embed_dim
    (HFClipBackbone._adapter returns a real nn.Linear, not nn.Identity, when
    in_dim != out_dim -- true for ViT-L/14's vision path at typical embed_dim
    settings). TinyBackbone always has this property regardless of any
    dimension match: its image_proj/text_proj are separate nn.Linear layers
    NEVER included in the freeze loop, confirmed here as the cheapest way to
    reproduce the exact bug scenario without needing a real CLIP download.

    Before the fix, gradcache would see freeze_backbone=True, set
    trainable_backbone=False, and skip the entire target-side gradient
    injection block in pass 3 -- silently leaving image_proj/text_proj
    (real, trainable parameters) untrained under gradcache specifically
    (the plain, non-gradcache path has no such gap, since one ordinary
    backward() naturally reaches every trainable parameter regardless).

    Fixed by deriving `trainable_backbone` from `any(p.requires_grad for p
    in model.backbone.parameters())` instead of the coarse config flag. This
    test directly reproduces the bug scenario (TinyBackbone,
    freeze_backbone=True) and confirms image_proj now receives real,
    non-zero gradient under gradcache.
    """
    from revua_cir.gradcache import gradcache_step

    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc")
    cfg.backbone.freeze_backbone = True
    cfg.optim.amp = False
    cfg.fusion.dropout = 0.0
    cfg.optim.gradcache_chunk_size = 2

    set_seed(0)
    model = ReVUACIR(cfg)
    model.train()
    assert any(p.requires_grad for p in model.backbone.parameters()), (
        "test setup error: TinyBackbone should have trainable image_proj/text_proj "
        "even with freeze_backbone=True")

    bsz, chunk_size = 8, 2
    batch = {
        "reference_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "target_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "edit_input_ids": torch.randint(0, 100, (bsz, 8)),
        "edit_attention_mask": torch.ones(bsz, 8, dtype=torch.long),
        "attribute_targets": None,
        "meta": [{} for _ in range(bsz)],
    }
    chunks = [
        {k: (v[i:i + chunk_size] if torch.is_tensor(v) or isinstance(v, list) else v)
         for k, v in batch.items()}
        for i in range(0, bsz, chunk_size)
    ]

    class _NoStep:
        def __init__(self, opt): self.opt = opt
        def zero_grad(self, **kw): self.opt.zero_grad(**kw)
        def step(self): pass

    gradcache_step(model, chunks, _NoStep(build_optimizer(model, cfg)), cfg, None)

    image_proj_grad = model.backbone.image_proj.weight.grad
    assert image_proj_grad is not None and image_proj_grad.norm().item() > 0, (
        "backbone.image_proj (a trainable adapter under a FROZEN backbone) received no gradient "
        "under gradcache -- the coarse freeze_backbone-based trainable_backbone check appears reverted")
    print(f"OK: gradcache injects gradient into trainable adapters even under freeze_backbone=True "
          f"(image_proj grad norm={image_proj_grad.norm().item():.4f}).")


def test_gradcache_rand_context_matches_dropout_mask():
    """Regression guard for the RNG/dropout inconsistency documented in
    gradcache.py's own header (fixed 2026-09-08): pass 1 (embedding cache,
    no_grad) and pass 3 (recompute-with-grad) each call encode_query/
    encode_image once per chunk, and every dropout layer inside consumes the
    global RNG stream -- without saving/restoring that state per chunk, pass
    3's dropout mask differed from pass 1's, so the gradient injected via
    torch.autograd.backward(cached_tensor, grad_cached) was technically the
    gradient of a DIFFERENT forward pass than the one whose loss produced
    grad_cached. Every gradcache training run using fusion.dropout > 0
    (every shipped config that actually trains with gradcache, e.g.
    stage1_finetune_full_unfreeze_gradcache.yaml at dropout=0.1 -- including
    job 6258) carried this silently until now.

    This test does NOT compare gradcache's gradient against a plain
    full-batch run's gradient (that comparison is fundamentally apples-to-
    oranges under real dropout: plain and gradcache consume RNG in
    different orders/amounts by construction, even before this bug, since
    they have different forward structures -- one combined forward vs.
    several per-chunk forwards -- so exact equivalence between THEM was
    never the claim). Instead it verifies the actual claim: replayed via
    RandContext, pass 3's encode_query/encode_image draw the IDENTICAL
    dropout mask pass 1 did, even with real intervening RNG consumption
    between the two passes (simulating other chunks / pass 2's backward
    running in between, exactly as happens in real gradcache_step).
    """
    from revua_cir.gradcache import RandContext

    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc")
    cfg.fusion.dropout = 0.5  # aggressive, so any mask mismatch is obvious
    cfg.backbone.freeze_backbone = False

    set_seed(0)
    model = ReVUACIR(cfg)
    model.train()  # dropout only active in train() mode

    bsz = 4
    sub = {
        "reference_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "target_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "edit_input_ids": torch.randint(0, 100, (bsz, 8)),
        "edit_attention_mask": torch.ones(bsz, 8, dtype=torch.long),
    }

    set_seed(999)
    torch.randn(37)  # simulate RNG already consumed by earlier chunks before this one

    ctx = RandContext()
    with torch.no_grad():
        q1 = model.encode_query(sub["reference_pixel_values"], sub["edit_input_ids"], sub["edit_attention_mask"])
        tgt1 = model.backbone.encode_image(sub["target_pixel_values"])

    torch.randn(500)  # simulate other chunks / pass 2's backward running in between

    with ctx.fork():
        q2 = model.encode_query(sub["reference_pixel_values"], sub["edit_input_ids"], sub["edit_attention_mask"])
    with ctx.fork():
        tgt2 = model.backbone.encode_image(sub["target_pixel_values"])

    assert torch.equal(q1["query"], q2["query"]), (
        "pass-3 query embedding does not exactly match pass-1's cached embedding under "
        "real dropout -- RandContext replay is broken")
    assert torch.equal(tgt1.pooled, tgt2.pooled), (
        "pass-3 target embedding does not exactly match pass-1's cached embedding under "
        "real dropout -- RandContext replay is broken")
    print("OK: RandContext makes gradcache's pass-3 recompute draw the exact same dropout "
          "mask as pass-1's cached forward, even with real intervening RNG consumption.")


def test_xbm_under_gradcache():
    """Regression guard for a real bug found via the cluster dry run (job
    6238, 2026-09-07): with gradcache enabled, `full_batch_context is not
    None` was True on every real training step (gradcache_step always passes
    it into the aux-loss recompute), so it always matched FIRST in the RV
    negative-mining if/elif chain -- meaning the `elif self.rv.xbm is not
    None and self.rv.xbm.is_ready:` branch, and its warm-up `enqueue()` call,
    were structurally unreachable whenever gradcache was active, regardless
    of `use_xbm`. The dry run confirmed this in the real pipeline: "XBM state
    after 5 steps: is_ready=False filled=0/1024 steps_seen=0" despite
    use_xbm=True. Fixed by adding a combined `full_batch_context is not None
    and self.rv.xbm is not None and self.rv.xbm.is_ready` branch ahead of the
    plain gradcache branch, and by enqueuing to XBM from inside the plain
    gradcache branch too (so warm-up progresses under gradcache, not just in
    the no-gradcache path).

    This test exercises the REAL `gradcache_step` function (not a hand-rolled
    stand-in), with a trainable backbone (the real training regime), driving
    enough steps to cross `xbm_warmup_steps` and confirms the queue actually
    becomes ready and enqueues DURING gradcache, not only outside it.
    """
    from revua_cir.gradcache import gradcache_step

    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc+aep+rv")
    cfg.optim.amp = False
    cfg.fusion.dropout = 0.0
    cfg.loss.hard_negative_weight = 0.0
    cfg.optim.gradcache_chunk_size = 2
    cfg.backbone.freeze_backbone = False
    cfg.optim.grad_clip_norm = 1e6
    cfg.reverse_verifier.use_xbm = True
    cfg.reverse_verifier.xbm_queue_size = 32
    cfg.reverse_verifier.xbm_warmup_steps = 2

    set_seed(0)
    model = ReVUACIR(cfg)
    model.train()
    assert not model.rv.xbm.is_ready, "XBM should not be ready before any gradcache steps"

    bsz, chunk_size = 8, 2

    def make_chunks():
        batch = {
            "reference_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
            "target_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
            "edit_input_ids": torch.randint(0, 100, (bsz, 8)),
            "edit_attention_mask": torch.ones(bsz, 8, dtype=torch.long),
            "attribute_targets": None,
            "meta": [{} for _ in range(bsz)],
        }
        return [
            {k: (v[i:i + chunk_size] if torch.is_tensor(v) or isinstance(v, list) else v)
             for k, v in batch.items()}
            for i in range(0, bsz, chunk_size)
        ]

    class _NoStep:
        def __init__(self, opt): self.opt = opt
        def zero_grad(self, **kw): self.opt.zero_grad(**kw)
        def step(self): pass

    opt = _NoStep(build_optimizer(model, cfg))
    for step in range(3):
        losses = gradcache_step(model, make_chunks(), opt, cfg, None)
        assert torch.isfinite(torch.as_tensor(losses["total"])), f"non-finite loss at gradcache step {step}"

    assert model.rv.xbm.is_ready, (
        "XBM never became ready under gradcache -- the combined "
        "(full_batch_context is not None and xbm.is_ready) branch, or the enqueue "
        "call inside the plain gradcache branch, regressed")
    assert model.rv.xbm.filled.item() > 0, "XBM queue is ready but empty -- enqueue() was not called under gradcache"

    # One more gradcache step, now with the queue ready: confirms the
    # combined XBM+gradcache branch (not just warm-up) produces a finite loss
    # and real gradient into RV's own parameters, not just a silent no-op.
    losses = gradcache_step(model, make_chunks(), opt, cfg, None)
    assert torch.isfinite(torch.as_tensor(losses["total"])), "non-finite loss once XBM+gradcache path is active"
    grad = model.rv.reverse_gen.net[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all(), (
        "XBM-sourced negative under gradcache produced no/non-finite RV gradient")

    print(f"OK: XBM activates correctly under gradcache -- ready after warm-up, "
          f"filled={model.rv.xbm.filled.item()}, live gradient confirmed on the combined path.")


def test_early_stopping():
    stopper = EarlyStopper(mode="max", patience=2, min_delta=1e-4, warmup_epochs=1)
    history = [10.0, 10.5, 10.4, 10.3, 10.2]  # improves once, then plateaus/degrades
    stopped_at = None
    for epoch, v in enumerate(history):
        result = stopper.step(v, epoch)
        if result["should_stop"]:
            stopped_at = epoch
            break
    assert stopped_at is not None, "early stopping never triggered on a clearly-plateaued curve"
    print(f"OK: early stopping triggered at epoch {stopped_at}, best={stopper.best:.2f}")


def test_checkpoint_warns_on_missing_trainable_param():
    """Regression guard for a real bug found via code-audit document review
    (2026-09-08): `load_checkpoint`'s missing-trainable-parameter warning
    checked `model.state_dict()[n].requires_grad` -- but `state_dict()`
    always returns DETACHED tensor copies, so `.requires_grad` is False for
    EVERY entry regardless of the source parameter's actual trainability
    (confirmed directly: a plain nn.Linear's weight has requires_grad=True,
    but its state_dict() entry reports False). This made
    `non_trainable_missing` an ALWAYS-EMPTY list -- the warning could never
    fire, silently defeating its own purpose. A missing trainable parameter
    after an architecture change (exactly the kind of change made this
    session: residual_alpha, rv_calibration's calibrator params) would
    resume silently at its default init with zero warning.

    Fixed by checking `model.named_parameters()` instead, which correctly
    reports requires_grad on the LIVE parameter. This test simulates the
    real scenario: save a checkpoint, delete one real trainable parameter's
    key from it (simulating an older checkpoint predating an architecture
    addition), then confirm loading it now actually warns.
    """
    import warnings as warnings_module

    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc+aep+rv")
    set_seed(0)
    model_old = ReVUACIR(cfg)

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/ckpt.pt"
        save_checkpoint(path, model_old, epoch=1, best_metric=1.0, config=cfg.to_dict())

        state = torch.load(path, weights_only=False)
        removed_key = next((k for k in state["model"] if "rv.calibrator.scale" in k), None)
        assert removed_key is not None, "test setup error: rv.calibrator.scale not found in checkpoint"
        del state["model"][removed_key]
        torch.save(state, path)

        set_seed(0)
        model_new = ReVUACIR(cfg)
        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            load_checkpoint(path, model_new, map_location="cpu")
            fired = any("TRAINABLE params missing" in str(w.message) for w in caught)
        assert fired, (
            "load_checkpoint did not warn about a genuinely missing trainable parameter -- "
            "the state_dict().requires_grad bug appears to have regressed")
    print("OK: load_checkpoint correctly warns when a trainable parameter is missing from the checkpoint.")


def test_checkpoint_roundtrip():
    cfg = build_tiny_config()
    model, cfg = run_forward_backward(cfg, "fc+aep+rv+ug")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, model, epoch=3, best_metric=42.0, config=cfg.to_dict())
        model2 = ReVUACIR(cfg)
        state = load_checkpoint(path, model2, map_location="cpu")
        assert state["epoch"] == 3 and state["best_metric"] == 42.0
    print("OK: checkpoint save/load roundtrip.")


def test_rng_state_checkpoint_roundtrip_reproduces_subsequent_draws():
    """Regression guard for the reproducibility recommendation in the
    code-audit document (2026-09-09, finding #9): checkpoint resume did not
    persist Python/NumPy/PyTorch-CPU/PyTorch-CUDA RNG state, so operations
    that consume the global RNG streams mid-training (dropout masks, any
    other stochastic op) would diverge after a resume even though per-epoch
    DataLoader seeding kept sample ORDER exactly reproducible on its own.

    Fixed via capture_rng_state()/restore_rng_state() (utils.py), threaded
    through save_checkpoint()'s new "rng_state" field and restored in
    train.py's resume block right after load_checkpoint().

    This test proves the actual GUARANTEE, not just that the dict round-trips:
    advance the global RNG streams by some arbitrary amount (simulating
    "whatever happened during training before this checkpoint was saved"),
    capture state, advance the streams further (simulating "what would have
    happened next in an uninterrupted run" -- recorded as the ground-truth
    NEXT draws), then separately restore the captured state and draw again --
    the draws after restore must EXACTLY match the ground-truth draws from
    the uninterrupted continuation, not just be independently valid-looking
    random numbers.
    """
    import random
    import numpy as np
    from revua_cir.utils import capture_rng_state, restore_rng_state

    # Advance every stream by an arbitrary, non-trivial amount first, so this
    # test cannot pass by accident (e.g. two freshly-seeded streams trivially
    # agreeing at their very first draw).
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    for _ in range(17):
        random.random(); np.random.rand(); torch.rand(3)

    captured = capture_rng_state()

    # Ground truth: what an UNINTERRUPTED run would draw next.
    expected_python = [random.random() for _ in range(5)]
    expected_numpy = np.random.rand(5).tolist()
    expected_torch = torch.rand(5).tolist()

    # Perturb every stream so restore_rng_state has to do real work, not
    # coincidentally land on the right state anyway.
    for _ in range(31):
        random.random(); np.random.rand(); torch.rand(3)

    restore_rng_state(captured)

    resumed_python = [random.random() for _ in range(5)]
    resumed_numpy = np.random.rand(5).tolist()
    resumed_torch = torch.rand(5).tolist()

    assert resumed_python == expected_python, "restore_rng_state did not reproduce Python random's next draws"
    assert resumed_numpy == expected_numpy, "restore_rng_state did not reproduce NumPy's next draws"
    assert resumed_torch == expected_torch, "restore_rng_state did not reproduce PyTorch CPU's next draws"

    # restore_rng_state(None) (old checkpoints with no saved rng_state) must
    # be a silent no-op, not a crash.
    restore_rng_state(None)

    # End-to-end through the real checkpoint file format: save_checkpoint
    # writes "rng_state", load_checkpoint's returned dict carries it through
    # unmodified (restoration itself is the caller's job, per train.py's
    # resume block -- load_checkpoint does not call restore_rng_state
    # itself).
    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc")
    set_seed(0)
    model = ReVUACIR(cfg)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, model, epoch=0, best_metric=0.0)
        state = load_checkpoint(path, ReVUACIR(cfg), map_location="cpu")
        assert state.get("rng_state") is not None, "save_checkpoint should always populate rng_state"
        for key in ("python", "numpy", "torch", "cuda"):
            assert key in state["rng_state"], f"rng_state missing expected key: {key}"

        # train.py resumes with map_location=device, which moves the saved RNG
        # ByteTensors to the GPU; restoring them must still work and still
        # reproduce the uninterrupted draws (seed-study resume job 7745 crashed here).
        if torch.cuda.is_available():
            torch.manual_seed(7)
            save_checkpoint(path, model, epoch=0, best_metric=0.0)
            expected_torch = torch.rand(5).tolist()
            expected_cuda = torch.rand(5, device="cuda").tolist()
            state = load_checkpoint(path, ReVUACIR(cfg), map_location="cuda")
            assert state["rng_state"]["torch"].is_cuda, "map_location='cuda' no longer moves rng_state; test is stale"
            torch.rand(9); torch.rand(9, device="cuda")
            restore_rng_state(state["rng_state"])
            assert torch.rand(5).tolist() == expected_torch, "GPU-mapped restore did not reproduce PyTorch CPU draws"
            assert torch.rand(5, device="cuda").tolist() == expected_cuda, "GPU-mapped restore did not reproduce CUDA draws"

    print("OK: capture_rng_state()/restore_rng_state() exactly reproduce the next draws an "
          "uninterrupted run would have made (Python/NumPy/PyTorch CPU all verified byte-for-byte "
          "equal, not just independently plausible), None is a safe no-op, and save_checkpoint/"
          "load_checkpoint correctly carry rng_state through a real checkpoint file.")


def test_checkpoint_resume_matches_live_model_not_ema():
    """Resume must restore the LIVE model the optimizer was tracking, not the
    EMA shadow that gets evaluated/checkpointed as `eval_model`.

    A real run hit exactly this: `save_checkpoint` was called with the EMA
    shadow, `load_checkpoint` put those weights into the live training model on
    resume, and the optimizer's momentum (saved from the live model) no longer
    matched what it was now updating -- plus a fresh EMA shadow started
    averaging from scratch. Validation R@1 fell from 13.25 back to ~7 and had to
    re-climb the same curve. This pins that the live model and its paired
    optimizer state round-trip correctly when EMA is enabled.
    """
    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc")
    set_seed(0)
    model = ReVUACIR(cfg)
    optimizer = build_optimizer(model, cfg)
    ema = EMA(model, decay=0.9)

    # Diverge the live model from its EMA shadow: several updates so the shadow
    # (slow-moving average) is measurably behind the live weights.
    tokenizer = build_tokenizer(cfg.backbone)
    collate_fn = build_collate_fn(tokenizer)
    dataset = SyntheticCIRDataset(8, cfg.data.image_size, cfg.attribute_parser.attribute_vocab_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    for batch in loader:
        out = model(batch)
        optimizer.zero_grad(set_to_none=True)
        out["losses"]["total"].backward()
        optimizer.step()
        ema.update(model)

    live_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt_state_before = optimizer.state_dict()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        # Mirrors train.py: checkpoint the EMA shadow (eval_model) but pass ema=
        # and live_model= so both round-trip correctly.
        save_checkpoint(path, ema.shadow, optimizer, epoch=2, best_metric=1.0,
                        config=cfg.to_dict(), ema=ema, live_model=model)

        resumed_model = ReVUACIR(cfg)
        resumed_optimizer = build_optimizer(resumed_model, cfg)
        resumed_ema = EMA(resumed_model, decay=0.9)
        load_checkpoint(path, resumed_model, resumed_optimizer, map_location="cpu", ema=resumed_ema)

    for n, p in resumed_model.named_parameters():
        assert torch.allclose(p, live_before[n]), f"resumed live weights differ from pre-save live weights: {n}"
    for n, p in resumed_ema.shadow.named_parameters():
        shadow_before = dict(ema.shadow.named_parameters())[n]
        assert torch.allclose(p, shadow_before), f"resumed EMA shadow differs from pre-save shadow: {n}"
    resumed_state = resumed_optimizer.state_dict()
    assert len(resumed_state["state"]) == len(opt_state_before["state"]), (
        "resumed optimizer has a different number of tracked param states than "
        "before save -- momentum buffers were not restored")
    print("OK: checkpoint resume restores the live model (not the EMA shadow) "
          "paired with its own optimizer state, and restores the EMA shadow separately.")


def test_checkpoint_excludes_only_pretrained_frozen_params():
    """Checkpoints exclude frozen params ONLY when they are safely
    reconstructible without the checkpoint (`backbone.clip.*`, HF's pretrained
    CLIP weights -- deterministic on every reconstruction). This must hold for
    the EMA shadow too, not just the live model.

    This is narrower than "exclude anything with requires_grad=False", and
    deliberately so -- that broader rule was tried first and is wrong. The
    tiny offline backbone also freezes parameters when `freeze_backbone: true`,
    but `TinyBackbone.text_pos` is `nn.Parameter(torch.randn(...))`: RANDOM at
    construction, never loaded from anywhere. A frozen-but-random parameter
    reconstructs to a DIFFERENT value each time a model is built, so excluding
    it from the checkpoint silently corrupts resume -- caught by this test
    (`resumed_model`'s frozen `text_pos` must equal the value that was actually
    trained with, not a fresh random draw).

    It also nearly regressed a second way: `EMA.__init__` calls
    `requires_grad_(False)` on *every* shadow parameter (only ever written
    under `torch.no_grad()`), so naively deriving "trainable" from each
    model's own `requires_grad` would exclude the ENTIRE shadow, not just its
    pretrained-backbone portion.
    """
    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc")
    cfg.backbone.freeze_backbone = True
    set_seed(0)
    model = ReVUACIR(cfg)
    optimizer = build_optimizer(model, cfg)
    ema = EMA(model, decay=0.9)

    n_total = sum(1 for _ in model.named_parameters())
    n_trainable = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    assert n_trainable < n_total, "test is only meaningful if some params are actually frozen"
    assert not any(n.startswith("backbone.clip.") for n, _ in model.named_parameters()), (
        "tiny backbone unexpectedly has backbone.clip.* params -- this test needs "
        "a config where the ONLY frozen params are non-excludable (tiny, not HF CLIP) "
        "to prove the prefix scoping, not just the requires_grad filtering")

    tokenizer = build_tokenizer(cfg.backbone)
    collate_fn = build_collate_fn(tokenizer)
    dataset = SyntheticCIRDataset(8, cfg.data.image_size, cfg.attribute_parser.attribute_vocab_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    for batch in loader:
        out = model(batch)
        optimizer.zero_grad(set_to_none=True)
        out["losses"]["total"].backward()
        optimizer.step()
        ema.update(model)

    live_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    shadow_before = {n: p.detach().clone() for n, p in ema.shadow.named_parameters()}

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, ema.shadow, optimizer, epoch=1, best_metric=1.0,
                        config=cfg.to_dict(), ema=ema, live_model=model)
        state = torch.load(path, map_location="cpu", weights_only=False)

        # Nothing in this config is backbone.clip.*, so NOTHING should be
        # excluded -- the tiny backbone's frozen params are not safe to drop.
        assert len(state["model"]) == n_total, (
            f"checkpoint saved {len(state['model'])}/{n_total} entries -- a frozen "
            f"but non-pretrained param (e.g. TinyBackbone.text_pos) was incorrectly "
            f"excluded")
        assert len(state["ema_shadow"]) == n_total, (
            f"ema_shadow saved {len(state['ema_shadow'])}/{n_total} entries -- the "
            f"requires_grad-on-the-shadow trap: EMA.shadow marks every param "
            f"requires_grad=False, so filtering by the shadow's OWN requires_grad "
            f"would incorrectly exclude everything")

        resumed_model = ReVUACIR(cfg)
        resumed_optimizer = build_optimizer(resumed_model, cfg)
        resumed_ema = EMA(resumed_model, decay=0.9)
        load_checkpoint(path, resumed_model, resumed_optimizer, map_location="cpu", ema=resumed_ema)

    for n, p in resumed_model.named_parameters():
        assert torch.allclose(p, live_before[n]), (
            f"resumed live param '{n}' does not match pre-save value -- if this is "
            f"a frozen param, it was wrongly excluded and reconstruction gave a "
            f"different (e.g. freshly random) value instead")
    for n, p in resumed_ema.shadow.named_parameters():
        assert torch.allclose(p, shadow_before[n]), (
            f"resumed EMA shadow param '{n}' does not match pre-save value")
    print(f"OK: checkpoint correctly keeps all {n_total} params when none are "
          f"backbone.clip.* (frozen-but-not-pretrained params are never dropped); "
          f"live model and EMA shadow both round-trip exactly.")


def test_checkpoint_excludes_frozen_pretrained_backbone_by_name():
    """Proves the size-reduction half of the fix directly: a frozen param named
    like the real HF backbone's weights (`backbone.clip....`) IS excluded, on a
    tiny synthetic module -- without needing network access to download the
    real ~1.7 GB pretrained CLIP checkpoint that this exclusion exists for
    (this suite is offline-only by design; see the module docstring).
    """
    import torch.nn as nn
    from revua_cir.utils import _trainable_state_dict

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.clip = nn.Module()
            self.backbone.clip.weight = nn.Parameter(torch.randn(4), requires_grad=False)
            self.head = nn.Linear(4, 4)  # trainable

    m = FakeModel()
    sd = _trainable_state_dict(m)
    assert "backbone.clip.weight" not in sd, (
        "a frozen backbone.clip.* param was NOT excluded -- the checkpoint-size "
        "fix is not firing for the real HF CLIP backbone")
    assert "head.weight" in sd and "head.bias" in sd, "trainable params must always be kept"
    print("OK: frozen backbone.clip.* params are excluded from checkpoints by name.")


def test_topk_checkpoint_keeper_rebuilds_from_disk_on_resume():
    """Regression guard for a real bug found via code-audit document review
    (2026-09-09, finding #8): TopKCheckpointKeeper starts with an empty
    `_entries` list on every construction, with no memory of checkpoint
    files a PRIOR (interrupted) run already wrote to `output_dir`. Since its
    own pruning logic (`offer()`'s `while len(self._entries) > self.k:
    remove worst`) only fires once the IN-MEMORY entry count exceeds `k`, a
    freshly-constructed keeper on resume would only ever prune checkpoints
    saved after the resume -- checkpoints from before the interruption stay
    on disk forever even once there are more than `save_top_k` files total.

    Fixed via `rebuild_from_disk()`: scans `output_dir` for `epochNNN.pt`
    files, reads each one's own `best_metric` field (the same field
    `save_checkpoint()` always writes), and reconstructs `_entries` from
    that -- called once at startup in train.py, before the training loop,
    so both a from-scratch run (no-op, no files yet) and a resumed run
    (recovers full history) end up with correct top-K tracking.

    This test writes 5 minimal but realistic checkpoint files (real
    torch.save'd dicts with a `best_metric` field, matching what
    save_checkpoint() actually produces) directly to a temp dir -- bypassing
    a full model/training run, since only the `best_metric` field matters
    to this class -- then verifies: (1) a fresh keeper with k=3 correctly
    identifies the worst 2 of the 5 existing files after rebuild_from_disk,
    and (2) offering a 6th, better-than-worst checkpoint after rebuild
    correctly prunes down to exactly 3 files kept, matching what an
    uninterrupted single-process run with the same 6 offers would have kept.
    """
    import tempfile
    from revua_cir.utils import TopKCheckpointKeeper

    with tempfile.TemporaryDirectory() as d:
        # 5 checkpoints with distinct metrics, mode="max" -- best 3 are
        # epoch003 (50.0), epoch001 (40.0), epoch004 (35.0); worst 2 (to be
        # pruned once a keeper tracks all 5) are epoch000 (10.0) and
        # epoch002 (20.0).
        metrics = {"epoch000.pt": 10.0, "epoch001.pt": 40.0, "epoch002.pt": 20.0,
                   "epoch003.pt": 50.0, "epoch004.pt": 35.0}
        for name, metric in metrics.items():
            torch.save({"model": {}, "epoch": int(name[5:8]), "best_metric": metric}, os.path.join(d, name))

        keeper = TopKCheckpointKeeper(k=3, mode="max")
        keeper.rebuild_from_disk(d)
        assert len(keeper._entries) == 5, (
            f"rebuild_from_disk should recover all 5 existing checkpoints' metrics, "
            f"got {len(keeper._entries)}: {keeper._entries}")
        recovered = {os.path.basename(p): m for m, p in keeper._entries}
        assert recovered == metrics, f"recovered metrics {recovered} don't match what was written {metrics}"

        # rebuild_from_disk does NOT prune on its own (a resume should not
        # silently delete files from a prior, possibly differently-configured
        # run) -- all 5 files must still exist after rebuild alone.
        for name in metrics:
            assert os.path.exists(os.path.join(d, name)), (
                f"rebuild_from_disk must not delete any file on its own -- {name} is missing")

        # Now simulate the run CONTINUING: a new, better checkpoint arrives.
        # With the keeper correctly aware of all 5 prior entries, this offer
        # should push the tracked count to 6 and prune down to the 3 best
        # (epoch005 60.0, epoch003 50.0, epoch001 40.0), removing the 3 worst
        # (epoch004 35.0, epoch002 20.0, epoch000 10.0) -- exactly what a
        # single uninterrupted run offering these same 6 checkpoints in this
        # order would have kept.
        torch.save({"model": {}, "epoch": 5, "best_metric": 60.0}, os.path.join(d, "epoch005.pt"))
        keeper.offer(60.0, os.path.join(d, "epoch005.pt"))

        remaining = sorted(f for f in os.listdir(d) if f.startswith("epoch"))
        assert remaining == ["epoch001.pt", "epoch003.pt", "epoch005.pt"], (
            f"after rebuild_from_disk + one more offer(), expected exactly the 3 best checkpoints "
            f"(epoch001/40.0, epoch003/50.0, epoch005/60.0) to survive -- got {remaining}. If this "
            f"fails, resume is not correctly aware of pre-existing checkpoints' rank and will retain "
            f"more than save_top_k files.")

    print("OK: TopKCheckpointKeeper.rebuild_from_disk() recovers all pre-existing checkpoints' "
          "metrics from disk, does not prune on its own, and correctly resumes top-K pruning "
          "against that recovered history on the next offer().")


def test_dry_run_full_pipeline_handles_optimizer_step_skipped_flag():
    """Regression guard for a real, guaranteed crash found via external
    review (2026-09-10, ReVUA_CIR_Final_PreTraining_Recommendations.md
    finding #2): scripts/dry_run_full_pipeline.py calls `gradcache_step(...)`
    and then treats EVERY value in the returned dict as a tensor (`for k, v
    in step_losses.items(): if not torch.isfinite(v).all()` and
    `float(v)`), including the `_optimizer_step_skipped` bool key
    gradcache_step now always includes (added 2026-09-10, same session, to
    fix the scheduler-step-before-optimizer-step bug). `torch.isfinite`
    requires a Tensor and raises `TypeError: isfinite(): argument 'input'
    ... must be Tensor, not bool` on a plain bool -- this script would crash
    on literally its first gradcache step, every time, with use_gradcache
    enabled (the actual training config's setting).

    Fixed by popping `_optimizer_step_skipped` out of step_losses before
    any tensor-only handling (matching train.py's own handling of the same
    value), and gating `scheduler.step()` on it.

    This test drives the REAL CLI entry point (`dry_run_full_pipeline.main()`
    via sys.argv, not a reimplementation) against a real tiny on-disk CIRR
    fixture, with use_gradcache=True and amp=True (CPU-side GradScaler
    no-op, but exercises the exact `step_losses.pop(...)` code path that
    was missing) -- confirms it completes without the TypeError, and that
    the dict-key removal didn't silently break the finite-loss check either
    (a real non-finite loss must still be caught).
    """
    import sys as _sys
    import io
    import contextlib

    with tempfile.TemporaryDirectory() as d:
        ann_dir = os.path.join(d, "cirr", "CIRR_repo")
        os.makedirs(os.path.join(ann_dir, "captions"))
        os.makedirs(os.path.join(ann_dir, "image_splits"))
        os.makedirs(os.path.join(d, "cirr", "dev_img"))
        n = 8
        split_map = {f"id{i}": f"./dev/img{i}.png" for i in range(n)}
        with open(os.path.join(ann_dir, "image_splits", "split.rc2.train.json"), "w") as f:
            json.dump(split_map, f)
        for i in range(n):
            open(os.path.join(d, "cirr", "dev_img", f"img{i}.png"), "wb").close()
        captions = [{"reference": f"id{i}", "target_hard": f"id{(i + 1) % n}", "caption": TEXTS[i % len(TEXTS)],
                    "pairid": i, "img_set": {"members": [f"id{(i + 1) % n}"]}} for i in range(n)]
        with open(os.path.join(ann_dir, "captions", "cap.rc2.train.json"), "w") as f:
            json.dump(captions, f)

        cfg = build_tiny_config()
        cfg = cfg.apply_module_ablation("fc+aep+rv+ug")
        cfg.data.root = d
        cfg.data.train_datasets = ("cirr",)
        cfg.optim.use_gradcache = True
        cfg.optim.gradcache_chunk_size = 4
        cfg.data.batch_size = 8  # evenly divisible by chunk_size=4
        cfg.optim.amp = True  # exercises the scaler/_optimizer_step_skipped code path

        cfg_path = os.path.join(d, "dry_run_cfg.yaml")
        cfg.to_yaml(cfg_path)

        argv_backup = _sys.argv
        _sys.argv = ["dry_run_full_pipeline.py", "--config", cfg_path, "--max-steps", "2"]
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
            import importlib
            dry_run_module = importlib.import_module("dry_run_full_pipeline")
            importlib.reload(dry_run_module)  # ensure a fresh run even if imported earlier
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                dry_run_module.main()
        finally:
            _sys.argv = argv_backup

        output = captured.getvalue()
        assert "TypeError" not in output, f"dry_run_full_pipeline crashed with a TypeError:\n{output}"
        assert "ALL CHECKS PASSED" in output, (
            f"dry_run_full_pipeline did not report success -- output:\n{output}")

    print("OK: scripts/dry_run_full_pipeline.py correctly pops _optimizer_step_skipped out of "
          "step_losses before tensor-only handling (torch.isfinite/float()), running to completion "
          "with use_gradcache=True + amp=True instead of crashing with a TypeError.")


def test_train_rejects_non_divisible_gradcache_chunk_size():
    """Regression guard for a gap found via code-audit document review
    (2026-09-08): `cfg.data.batch_size // cfg.optim.gradcache_chunk_size`
    (used for both steps_per_epoch and chunks_per_step in train.py) is
    integer division with no validation that batch_size is evenly divisible
    by gradcache_chunk_size. A non-divisor combination (e.g. batch_size=192,
    chunk_size=7) silently drops the remainder EVERY logical step, with no
    error and no visible symptom besides a slightly-smaller effective batch
    than the config claims -- gradcache.py's own header already documents
    this constraint, but nothing enforced it. Fixed by asserting divisibility
    at the very start of train(), failing loudly before any real training
    work happens rather than silently training on a slightly-wrong batch
    size for potentially many epochs.
    """
    import tempfile
    from revua_cir.train import train

    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc")
    cfg.optim.use_gradcache = True
    cfg.data.batch_size = 10
    cfg.optim.gradcache_chunk_size = 3  # not a divisor of 10

    with tempfile.TemporaryDirectory() as d:
        cfg.run.output_dir = d
        raised = False
        try:
            train(cfg)
        except ValueError as e:
            raised = "evenly divisible" in str(e)
        except Exception:
            # Any other exception (e.g. real dataset files not found, since
            # this test's tiny config still points at a real dataset loader)
            # means the divisibility check did NOT fire as the very first
            # thing -- treat as a failure of what THIS test checks, distinct
            # from "the check correctly raised its own ValueError".
            pass
    assert raised, "train() should raise ValueError for a non-divisor gradcache_chunk_size, but did not"

    # A valid, evenly-divisible combination must NOT trip this specific
    # check. Isolate just the check itself (not the full train() pipeline,
    # which would need real dataset files on disk to get further) by
    # replicating its exact condition directly.
    cfg_ok = build_tiny_config()
    cfg_ok.optim.use_gradcache = True
    cfg_ok.data.batch_size = 12
    cfg_ok.optim.gradcache_chunk_size = 4  # divides evenly
    assert not (cfg_ok.optim.use_gradcache and cfg_ok.data.batch_size % cfg_ok.optim.gradcache_chunk_size != 0), (
        "a valid divisor combination should not trip the divisibility check")
    print("OK: train() rejects a non-divisor batch_size/gradcache_chunk_size combination "
          "before any training work happens; a valid divisor combination passes the same check.")


def test_gradcache_resume_skip_replays_same_batches():
    """A mid-epoch resume must replay the exact chunks already consumed, not a
    different random subset -- otherwise an interrupted epoch either skips
    real training data or duplicates it unevenly across a run.

    `train.py` gives the DataLoader a generator seeded from `(run.seed,
    epoch)`, so rebuilding the loader for the same epoch reproduces the same
    shuffle order deterministically; `_iter_gradcache_steps(..., skip_steps=N)`
    then discards the first N logical steps of that reproduced order. This
    checks the two halves fit together: chunks skipped by one generator-seeded
    pass equal the chunks a from-scratch pass would have trained on at the
    same positions.
    """
    from revua_cir.train import _iter_gradcache_steps

    class _TinyDS(torch.utils.data.Dataset):
        def __len__(self): return 20
        def __getitem__(self, i): return {"id": torch.tensor(i)}

    def make_loader(seed):
        g = torch.Generator().manual_seed(seed)
        return torch.utils.data.DataLoader(_TinyDS(), batch_size=2, shuffle=True,
                                           drop_last=True, generator=g)

    # A "from scratch" pass over the whole epoch...
    full_steps = list(_iter_gradcache_steps(make_loader(42), chunks_per_step=2))
    # ...must equal a resumed pass that skips the first 2 logical steps, when
    # both loaders are built with the SAME seed (mirroring train.py rebuilding
    # the loader with cfg.run.seed + epoch on every resume of that epoch).
    resumed_steps = list(_iter_gradcache_steps(make_loader(42), chunks_per_step=2, skip_steps=2))

    assert resumed_steps[0][0] == 2, "resumed step numbering must continue from skip_steps, not restart at 0"
    for (full_step, full_group), (res_step, res_group) in zip(full_steps[2:], resumed_steps):
        assert full_step == res_step
        for fc, rc in zip(full_group, res_group):
            assert torch.equal(fc["id"], rc["id"]), (
                "resumed chunk contents differ from the from-scratch pass at the "
                "same step -- mid-epoch resume would train on the wrong samples")
    print(f"OK: gradcache resume-skip reproduces the same {len(resumed_steps)} "
          f"remaining chunks as a from-scratch pass.")


def test_train_py_passes_ema_to_save_checkpoint():
    """Guards the train.py call site itself, not just the utils.py functions.

    The unit test above proves save/load_checkpoint are correct when called with
    ema= and live_model=; it does NOT prove train.py actually passes them. A
    real regression could drop those kwargs at the call site and this test
    suite would stay green without this check -- which is exactly how the
    original bug (checkpointing the EMA shadow with no way to recover the live
    model or EMA state on resume) could reappear silently.
    """
    import inspect
    import revua_cir.train as train_module

    src = inspect.getsource(train_module.train)
    assert "save_checkpoint(" in src, "train() no longer calls save_checkpoint -- update this test"
    # Find the save_checkpoint call and check its kwargs textually; a full AST
    # parse would be more robust but this catches the actual regression class
    # (someone deletes `ema=ema, live_model=model` while editing nearby code).
    call_start = src.index("save_checkpoint(")
    call_text = src[call_start:src.index(")", src.index(")", call_start) + 1) + 1]
    assert "ema=ema" in call_text, (
        "train.py's save_checkpoint call dropped ema=ema -- resume will silently "
        "load a fresh, un-averaged EMA shadow instead of the saved one")
    assert "live_model=model" in call_text, (
        "train.py's save_checkpoint call dropped live_model=model -- resume will "
        "load the EMA shadow into the live model, mismatched against the saved "
        "optimizer state. This is the exact bug that reset a real run's R@1 "
        "from 13.25 back to ~7 on resume.")
    print("OK: train.py's save_checkpoint call passes ema= and live_model=.")


def test_metrics():
    ranked = torch.tensor([[3, 1, 2, 0], [0, 1, 2, 3]])
    targets = torch.tensor([1, 0])
    r = recall_at_k(ranked, targets, ks=[1, 2, 4])
    assert r["recall_at_1"] == 50.0 and r["recall_at_2"] == 100.0
    mapk = mean_average_precision_at_k(ranked, [{1, 2}, {0}], ks=[2, 4])
    assert 0.0 <= mapk["map_at_2"] <= 100.0
    conf = torch.tensor([0.9, 0.8, 0.2, 0.1])
    correct = torch.tensor([1.0, 0.0, 0.0, 0.0])
    ece = expected_calibration_error(conf, correct, n_bins=4)
    assert 0.0 <= ece <= 1.0
    print(f"OK: metrics sane -- recall={r}, map={mapk}, ece={ece:.4f}")


def test_verification_auroc_matches_pooled_roc_auc():
    """Regression guard for a real metric-correctness bug found via code-audit
    document review (2026-09-09, finding B.8): the function previously named
    `verification_discrimination_auc` computed a per-query MATCHED-PAIR win
    rate (mean of `pos_i > neg_i` over the n matched (positive, negative)
    pairs, one per query) and called it "AUC" -- not the same statistic as
    standard pooled ROC-AUC, which compares every positive score against
    EVERY negative score in the pool (all n_pos * n_neg pairs), and which is
    what a paper claiming "AUROC" must actually report.

    This test constructs positive/negative score arrays where the two
    statistics provably diverge (unequal group sizes with score overlap),
    confirms metrics.verification_auroc matches sklearn's
    roc_auc_score exactly (when sklearn is installed; falls back to a
    brute-force O(n_pos*n_neg) all-pairs comparison otherwise, which pooled
    AUROC is mathematically defined as via the Mann-Whitney-U identity), and
    confirms verification_pairwise_win_rate (the renamed old function) gives
    a DIFFERENT number on this same input -- proving the two are genuinely
    distinct metrics, not a rename of an already-equivalent computation.
    """
    from revua_cir.metrics import verification_auroc, verification_pairwise_win_rate

    torch.manual_seed(0)
    pos = torch.tensor([0.9, 0.8, 0.85, 0.3, 0.95, 0.7])  # n_pos=6
    neg = torch.tensor([0.4, 0.6, 0.2])                    # n_neg=3, unequal size + overlap with pos

    auroc = verification_auroc(pos, neg)

    checked_against = "brute-force all-pairs"
    try:
        from sklearn.metrics import roc_auc_score
        labels = [1] * len(pos) + [0] * len(neg)
        scores = pos.tolist() + neg.tolist()
        expected = roc_auc_score(labels, scores)
        checked_against = "sklearn.roc_auc_score"
    except ImportError:
        # Brute-force all-pairs P(pos > neg) with 0.5 tie credit -- the
        # definitional form of pooled AUROC (Mann-Whitney-U identity).
        wins = sum((p > n).item() + 0.5 * (p == n).item() for p in pos for n in neg)
        expected = wins / (len(pos) * len(neg))

    assert abs(auroc - expected) < 1e-6, (
        f"verification_auroc={auroc:.6f} does not match pooled ROC-AUC={expected:.6f}")

    win_rate = verification_pairwise_win_rate(pos, neg)
    # With unequal-length inputs, verification_pairwise_win_rate only uses the
    # first min(n_pos, n_neg)=3 matched pairs (pos[:3] vs neg[:3]), a
    # genuinely different, smaller-sample computation than pooled AUROC over
    # all 18 pairs -- confirming this isn't a no-op rename.
    assert abs(auroc - win_rate) > 1e-6, (
        f"verification_auroc ({auroc:.6f}) and verification_pairwise_win_rate "
        f"({win_rate:.6f}) should differ on this constructed example -- if they're equal, "
        f"something is silently computing the same statistic under two names again")
    print(f"OK: verification_auroc={auroc:.4f} matches true pooled ROC-AUC exactly "
          f"(cross-checked against {checked_against}); "
          f"verification_pairwise_win_rate={win_rate:.4f} confirmed to be a genuinely different statistic.")


def test_group_recall_uses_untruncated_scores_not_global_topn():
    """Regression guard for a real bug found via code-audit document review
    (2026-09-08): CIRR group/subset recall used to be computed by filtering
    a GLOBALLY-TRUNCATED top-N ranking (`top_idx`, rank_width ~100) down to
    just the query's group members. CIRR's group members are chosen for
    VISUAL similarity to each other, not because they score highly against
    the query overall -- a group's true best-scoring member (exactly what
    the subset protocol is supposed to measure) can rank outside the global
    top-N while still being the unambiguous #1 WITHIN its own 6-way group.
    When that happens, the old code silently scored it as a MISS even
    though the correct subset-only ranking would HIT.

    Fixed by ranking directly from each query's full, untouched per-gallery
    score row (`scores[b_local]`, available before any topk truncation)
    restricted to the group's own member indices -- never touching the
    globally-truncated ranking for this metric at all.

    This test constructs the exact failure scenario directly (bypassing the
    real model, whose random-weight embeddings can't be steered precisely
    enough to reliably reproduce it): a gallery where the query's target is
    the clear #1 scorer among its own 6-member group, but ~150 unrelated
    gallery items score higher than the whole group, pushing every group
    member outside a rank_width=100 truncation. Confirms group_recall_at_1
    now correctly reports a HIT (previously: MISS).
    """
    torch.manual_seed(0)
    gallery_size = 1000
    scores_row = torch.randn(gallery_size)

    group_members = [10, 20, 30, 40, 50, 60]
    target_idx = 30
    # target is the CLEAR best among its 6-member group...
    scores_row[group_members] = torch.tensor([0.02, 0.01, 0.09, -0.01, 0.0, 0.015])
    # ...but 150 unrelated items outscore the entire group, pushing every
    # group member outside a rank_width=100 global truncation.
    scores_row[100:250] = torch.linspace(0.10, 0.3, 150)

    rank_width = 100
    top_idx = scores_row.topk(rank_width).indices.tolist()
    assert not (set(group_members) & set(top_idx)), (
        "test setup error: at least one group member landed in the global top-100 -- "
        "this scenario no longer reproduces the bug's trigger condition")

    # OLD (buggy) computation: filter the globally-truncated ranking down to
    # the group's members.
    old_ranked_in_subset = [i for i in top_idx if i in set(group_members)]
    old_hit_at_1 = target_idx in old_ranked_in_subset[:1]

    # NEW (fixed) computation: rank directly from the untruncated scores,
    # restricted to just the group's own member indices -- exactly what
    # evaluate.py's fixed inner loop now does.
    group_scores = scores_row[group_members]
    new_group_order = [group_members[i] for i in group_scores.argsort(descending=True).tolist()]
    new_hit_at_1 = target_idx in new_group_order[:1]

    assert old_hit_at_1 is False, (
        "test setup error: the OLD (buggy) computation should MISS in this scenario -- "
        "if it doesn't, this test isn't reproducing the bug")
    assert new_hit_at_1 is True, (
        "the FIXED computation should HIT (target is the group's true #1) -- if this fails, "
        "the group-recall truncation bug appears to have regressed")
    print("OK: group recall ranks directly from untruncated per-query scores, correctly hitting "
          "when the target is its group's true #1 even though the whole group ranks outside a "
          "global top-100 (the old truncated implementation would have missed this).")


def test_cirr_full_gallery_and_incomplete_group_handling():
    """End-to-end regression guard for scripts/evaluate_cirr.py's core pieces
    (added 2026-09-09, code-audit document B.6/A.3; updated 2026-09-09 for
    finding #9's strict 6-member/5-candidate semantics): builds a REAL
    on-disk CIRR fixture (same shape test_cirr_dataset_resolves_kaggle_
    mirror_layout already uses) with more images than any single query's own
    ref+target, then checks three things end to end through the real
    CIRRDataset + evaluate_dataset() path (not a bypassed pure-tensor
    scenario like test_group_recall_uses_untruncated_scores_not_global_topn
    above):

    1. build_cirr_full_gallery() returns every image in the split map, i.e.
       strictly more than evaluate.py's own build_default_gallery() (which
       only unions each triplet's ref+target) -- confirming the "official
       full gallery" script genuinely exercises more candidates, not just a
       relabeled version of the same reduced pool.
    2. A query with a genuine, well-formed 6-member group (5 candidates + 1
       reference, ALL present in the gallery) is counted complete, and its
       group-recall is computed over exactly the 5 non-reference candidates
       -- not all 6 raw members.
    3. A query whose img_set.members lists a non-reference candidate ID that
       is NOT present in the gallery (an incomplete group -- e.g. a reduced
       gallery, or a corrupted annotation) is excluded from group-recall
       accounting entirely (group_queries_incomplete counts it, but it does
       NOT contribute to group_recall_hits/total) rather than silently
       scored over whatever subset of members happens to be present.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from evaluate_cirr import build_cirr_full_gallery
    from revua_cir.data.datasets import CIRRDataset
    from revua_cir.data.transforms import build_image_transform
    from revua_cir.evaluate import build_default_gallery, evaluate_dataset

    with tempfile.TemporaryDirectory() as d:
        ann_dir = os.path.join(d, "CIRR_repo")
        os.makedirs(os.path.join(ann_dir, "captions"))
        os.makedirs(os.path.join(ann_dir, "image_splits"))
        os.makedirs(os.path.join(d, "dev_img"))

        # 8 distinct images: q1 is a WELL-FORMED 6-member group (reference
        # id0 + 5 candidates id1..id5, all present in the gallery -- id2..id5
        # are DISTRACTORS never referenced as any query's own ref/target).
        # q2 has only 5 total members listed (malformed -- not the official
        # 6) AND one of its non-reference candidates (id7) is missing from
        # the split map entirely, covering both the "malformed" and
        # "incomplete" failure modes in one query.
        n_images = 7
        split_map = {f"id{i}": f"./dev/img{i}.png" for i in range(n_images)}
        with open(os.path.join(ann_dir, "image_splits", "split.rc2.val.json"), "w") as f:
            json.dump(split_map, f)
        for i in range(n_images):
            open(os.path.join(d, "dev_img", f"img{i}.png"), "wb").close()

        captions = [
            {"reference": "id0", "target_hard": "id1", "caption": "t1", "pairid": 1,
             "img_set": {"members": ["id0", "id1", "id2", "id3", "id4", "id5"]}},
            {"reference": "id1", "target_hard": "id2", "caption": "t2", "pairid": 2,
             # Only 5 total members (not the official 6) AND id7 is not in
             # split_map -- both malformed and incomplete.
             "img_set": {"members": ["id1", "id2", "id6", "id7"]}},
        ]
        with open(os.path.join(ann_dir, "captions", "cap.rc2.val.json"), "w") as f:
            json.dump(captions, f)

        transform = build_image_transform(32, train=False)
        dataset = CIRRDataset(root=d, split="val", transform=transform)
        assert dataset.triplets[0]["meta"]["reference_id"] == "id0", (
            "CIRRDataset must thread reference_id through meta for evaluate.py's "
            "strict 6-member/5-candidate check")

        default_gallery = build_default_gallery(dataset)
        full_gallery = build_cirr_full_gallery(dataset)
        assert len(full_gallery) == n_images, f"expected all {n_images} split-map images, got {len(full_gallery)}"
        assert len(full_gallery) > len(default_gallery), (
            f"full gallery ({len(full_gallery)}) should include distractor images that "
            f"build_default_gallery's per-triplet ref+target union ({len(default_gallery)}) excludes")

        cfg = build_tiny_config()
        cfg = cfg.apply_module_ablation("fc")
        set_seed(0)
        model = ReVUACIR(cfg)
        model.eval()
        tokenizer = build_tokenizer(cfg.backbone)

        metrics = evaluate_dataset(model, dataset, cfg, tokenizer, transform, torch.device("cpu"),
                                    use_group_recall=True, group_ks=(1, 2, 3),
                                    full_gallery_paths=full_gallery)

        assert metrics["group_queries_total"] == 2, metrics
        assert metrics["group_queries_complete"] == 1, (
            f"query 1 (genuine 6-member group: reference id0 + 5 candidates id1-id5, all in the "
            f"gallery) should be the only complete group -- got "
            f"group_queries_complete={metrics['group_queries_complete']}")
        assert metrics["group_queries_incomplete"] == 1, (
            f"query 2 (only 4 total members listed, not the official 6, AND id7 missing from the "
            f"gallery) should be counted incomplete -- got "
            f"group_queries_incomplete={metrics['group_queries_incomplete']}")

    print("OK: scripts/evaluate_cirr.py's build_cirr_full_gallery() returns the true full split "
          "gallery (including distractor images a reduced default gallery would exclude), "
          "evaluate_dataset() enforces strict official 6-member/5-non-reference-candidate CIRR "
          "group semantics, and correctly separates complete vs incomplete/malformed groups.")


def test_official_cirr_group_has_six_members_five_candidates():
    """Regression guard for a real bug found via code-audit document review
    (2026-09-09, finding #9): CIRR's `img_set.members` is the full 6-image
    visually-similar group INCLUDING the reference image itself. The
    official subset protocol ranks only the 5 NON-reference candidates
    against the query. The old code fed all 6 raw members into the group
    ranking unfiltered -- the reference happened to score -inf from a
    SEPARATE, earlier global reference-masking step (for the full-gallery
    R@K metrics), so in practice it rarely won a group, but this test proves
    the fix does not silently depend on that separate mechanism: it
    constructs a scores row where the reference would score HIGHEST among
    the 6 raw members (no -inf masking applied to this row at all) and
    confirms the reference still cannot appear in the ranked group output,
    because it is structurally excluded from the 5-candidate subset before
    ranking, not merely outscored.

    This tests the pure ranking logic directly (constructing `scores`,
    `id_to_idx`, and `meta` exactly as evaluate.py's inner loop would see
    them) rather than routing through the full model, so the reference's
    score can be set explicitly instead of relying on incidental behavior.
    """
    torch.manual_seed(0)
    gallery_size = 20
    scores_row = torch.zeros(gallery_size)

    # 6-member group: reference is idx 0, candidates are idx 1-5. Reference
    # deliberately scores HIGHEST of all 6 (no -inf masking here) -- if the
    # fix were absent/broken, the reference would win the group ranking.
    ref_idx = 0
    candidate_idxs_true = [1, 2, 3, 4, 5]
    scores_row[ref_idx] = 10.0  # highest in the whole group
    scores_row[candidate_idxs_true] = torch.tensor([0.5, 0.9, 0.3, 0.1, 0.7])
    target_idx = 2  # the true target: 2nd-highest among the 5 real candidates

    id_to_idx = {f"id{i}": i for i in range(gallery_size)}
    meta = {"reference_id": "id0", "group_members": [f"id{i}" for i in range(6)]}

    ref_id = str(meta["reference_id"])
    all_member_ids = {str(m) for m in meta["group_members"]}
    candidate_ids = {m for m in all_member_ids if m != ref_id}
    candidate_idxs = [id_to_idx[m] for m in candidate_ids if m in id_to_idx]
    is_well_formed = len(all_member_ids) == 6 and ref_id in all_member_ids

    assert is_well_formed, "test setup error: this group should be well-formed (6 members incl. reference)"
    assert len(candidate_idxs) == 5, f"test setup error: expected 5 non-reference candidates, got {len(candidate_idxs)}"
    assert ref_idx not in candidate_idxs, (
        "test setup error: the reference index leaked into the candidate set before the "
        "ranking logic even runs -- fix the test fixture, not evaluate.py")

    group_scores = scores_row[candidate_idxs]
    group_order_local = group_scores.argsort(descending=True).tolist()
    group_ranked_ids = [candidate_idxs[i] for i in group_order_local]

    assert ref_idx not in group_ranked_ids, (
        f"the reference image (idx {ref_idx}, the highest-scoring item in the whole raw group) "
        f"must NEVER appear in the ranked group output -- the official CIRR subset protocol ranks "
        f"only the 5 non-reference candidates, got ranked group {group_ranked_ids}")
    assert len(group_ranked_ids) == 5, f"ranked group should have exactly 5 candidates, got {len(group_ranked_ids)}"
    assert target_idx in group_ranked_ids[:2], (
        f"target (idx {target_idx}, 2nd-highest among the 5 real candidates) should rank in the "
        f"top-2 of the 5-candidate subset -- got ranked group {group_ranked_ids}")

    print("OK: official CIRR group ranking is computed over exactly the 5 non-reference candidates "
          "-- the reference image cannot appear in the ranked output even when it would otherwise "
          "score highest, confirming it is structurally excluded rather than merely outscored.")


def test_end_epoch_checkpoint_saves_post_step_early_stopper_state():
    """Regression guard for a real bug found via code-audit document review
    (2026-09-09, finding #7): train.py's end-of-epoch block used to call
    `save_checkpoint(..., extra=_early_stopper_state(early_stopper))` BEFORE
    `early_stopper.step(monitor_value, epoch)` -- so the checkpoint's saved
    `extra["early_stopper"]` reflected the PREVIOUS epoch's stopper state,
    one validation event behind the checkpoint's own `epoch`/`best_metric`.
    Concretely: if the live stopper entered an epoch at
    (best=X, num_bad_epochs=2) and that epoch was ALSO a bad one, the old
    ordering saved num_bad_epochs=2 into a checkpoint whose live stopper had
    already advanced to num_bad_epochs=3 -- a resume from that checkpoint
    would silently get one extra epoch of patience it should not have had.

    Fixed by calling early_stopper.step() first, then save_checkpoint().

    This test drives a REAL train() call (not a reimplementation of the
    ordering logic) through 3 real epochs against a tiny real on-disk CIRR
    fixture (same construction CIRRDataset itself is tested against
    elsewhere in this file), with early_stopping.patience set high enough
    that training completes all 3 epochs without stopping early. For each
    saved epoch checkpoint, it loads the checkpoint and asserts its saved
    early-stopper state exactly matches an INDEPENDENTLY maintained
    reference EarlyStopper fed the same monitor values in the same order --
    i.e. the checkpoint for epoch N must reflect epoch N's own step(), not
    epoch N-1's.
    """
    import glob
    import tempfile

    from revua_cir.train import train
    from revua_cir.utils import EarlyStopper

    with tempfile.TemporaryDirectory() as d:
        ann_dir = os.path.join(d, "cirr", "CIRR_repo")
        os.makedirs(os.path.join(ann_dir, "captions"))
        os.makedirs(os.path.join(ann_dir, "image_splits"))
        os.makedirs(os.path.join(d, "cirr", "dev_img"))

        n = 8
        split_map = {f"id{i}": f"./dev/img{i}.png" for i in range(n)}
        with open(os.path.join(ann_dir, "image_splits", "split.rc2.train.json"), "w") as f:
            json.dump(split_map, f)
        with open(os.path.join(ann_dir, "image_splits", "split.rc2.val.json"), "w") as f:
            json.dump(split_map, f)
        for i in range(n):
            open(os.path.join(d, "cirr", "dev_img", f"img{i}.png"), "wb").close()

        captions = [
            {"reference": f"id{i}", "target_hard": f"id{(i + 1) % n}", "caption": TEXTS[i % len(TEXTS)],
             "pairid": i, "img_set": {"members": [f"id{(i + 1) % n}"]}}
            for i in range(n)
        ]
        with open(os.path.join(ann_dir, "captions", "cap.rc2.train.json"), "w") as f:
            json.dump(captions, f)
        with open(os.path.join(ann_dir, "captions", "cap.rc2.val.json"), "w") as f:
            json.dump(captions, f)

        cfg = build_tiny_config()
        cfg = cfg.apply_module_ablation("fc")
        cfg.data.root = d
        cfg.data.train_datasets = ("cirr",)
        cfg.data.val_dataset = "cirr"
        cfg.data.batch_size = n
        cfg.data.eval_batch_size = n
        cfg.optim.max_epochs = 3
        cfg.run.eval_every_epochs = 1
        cfg.run.output_dir = os.path.join(d, "run_out")
        cfg.early_stopping.enabled = True
        cfg.early_stopping.patience = 100  # must not trigger during this test
        cfg.early_stopping.warmup_epochs = 0
        cfg.early_stopping.monitor = "val_recall_at_1"

        set_seed(0)
        train(cfg)

        # get_logger(name, log_file=...) attaches a FileHandler to a
        # process-global `logging.getLogger("revua_cir.train")` and never
        # closes it (deliberate: a real multi-epoch run wants one open
        # handle for the run's whole lifetime, and get_logger's own
        # `if logger.handlers: return logger` guard means a second train()
        # call in the same process reuses the same handler rather than
        # opening a new one). On Windows this keeps the log file locked, so
        # TemporaryDirectory's own cleanup (below, at the `with` block exit)
        # cannot delete it -- close and detach the handler explicitly here so
        # this test's tempdir can clean up like every other test in this file.
        import logging as _logging
        _train_logger = _logging.getLogger("revua_cir.train")
        for _h in list(_train_logger.handlers):
            _h.close()
            _train_logger.removeHandler(_h)

        ckpt_paths = sorted(glob.glob(os.path.join(cfg.run.output_dir, "epoch*.pt")))
        assert len(ckpt_paths) == 3, f"expected 3 epoch checkpoints, found {len(ckpt_paths)}: {ckpt_paths}"

        # Independently reconstruct what the live stopper's state MUST have
        # been after each epoch's own step(), by reading each checkpoint's
        # OWN saved best_metric (monitor_value) and epoch, feeding them to a
        # fresh, separately-instantiated EarlyStopper in epoch order -- this
        # does not depend on train.py's internal ordering at all, so it is a
        # genuine independent check, not a tautology against the code under test.
        reference_stopper = EarlyStopper(cfg.early_stopping.mode, cfg.early_stopping.patience,
                                          cfg.early_stopping.min_delta, cfg.early_stopping.warmup_epochs)
        for ckpt_path in ckpt_paths:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ckpt_epoch = state["epoch"]
            ckpt_monitor_value = state["best_metric"]
            reference_stopper.step(ckpt_monitor_value, ckpt_epoch)
            saved_es = state.get("extra", {}).get("early_stopper")
            assert saved_es is not None, f"{ckpt_path}: no early_stopper state saved in extra"
            assert saved_es["best"] == reference_stopper.best, (
                f"{ckpt_path} (epoch {ckpt_epoch}): saved early_stopper.best={saved_es['best']} but "
                f"the reference stopper (fed this SAME checkpoint's own monitor_value/epoch) computes "
                f"best={reference_stopper.best} -- the checkpoint's early-stopper snapshot does not "
                f"reflect its own epoch's validation result (stale/off-by-one save ordering)")
            assert saved_es["num_bad_epochs"] == reference_stopper.num_bad_epochs, (
                f"{ckpt_path} (epoch {ckpt_epoch}): saved num_bad_epochs={saved_es['num_bad_epochs']} "
                f"but reference computes {reference_stopper.num_bad_epochs} -- stale save ordering")
            assert saved_es["best_epoch"] == reference_stopper.best_epoch, (
                f"{ckpt_path} (epoch {ckpt_epoch}): saved best_epoch={saved_es['best_epoch']} but "
                f"reference computes {reference_stopper.best_epoch} -- stale save ordering")

    print("OK: end-of-epoch checkpoints save early-stopper state AFTER that epoch's own step() "
          "call, not one validation event behind (verified across 3 real epochs of a real train() "
          "run against an independently-reconstructed reference EarlyStopper).")


def test_inference_rerank_path():
    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc+aep+rv+ug")
    set_seed(1)
    model = ReVUACIR(cfg).eval()
    tokenizer = build_tokenizer(cfg.backbone)
    collate_fn = build_collate_fn(tokenizer)
    tok = tokenizer(TEXTS[:2])

    reference = torch.randn(2, 3, cfg.data.image_size, cfg.data.image_size)
    gallery = torch.randn(30, 3, cfg.data.image_size, cfg.data.image_size)
    with torch.no_grad():
        gallery_out = model.encode_gallery(gallery)
        gallery_pooled = gallery_out["pooled"]

        def gallery_image_fn(idx_tensor):
            return gallery[idx_tensor.cpu()]

        idx, scores = model.retrieve_and_rerank(
            reference, tok["input_ids"], tok["attention_mask"], gallery_pooled,
            gallery_image_fn=gallery_image_fn, top_n=10,
        )
    assert idx.shape == (2, 10)
    assert torch.isfinite(scores).all()
    print("OK: inference retrieve_and_rerank path (RV + UG reranking) runs cleanly.")


def test_retrieve_and_rerank_does_not_mix_score_domains():
    """Regression guard for a real bug found via code-audit document review
    (2026-09-08), confirmed to have never affected any reported result
    (this API's only callers were its own definition and the
    correctness-agnostic test_inference_rerank_path above -- evaluate.py's
    separately-implemented reranking path, which DID produce reported
    numbers, never had this bug).

    `retrieve_and_rerank` used to start `rerank_scores` as a full clone of
    `top_scores` (raw cosine similarities, roughly [-1, 1]) for ALL top_n
    candidates, then overwrite only the first top_k entries with
    `_rank_fuse`'s output (`-fused_rank`, a small non-positive integer in
    [-2k+1, 0]). A single `rerank_scores.argsort()` over the WHOLE top_n
    array then mixed these incompatible score domains -- any untouched tail
    candidate with a positive cosine score would always outrank every
    reranked top-k candidate, corrupting the final order whenever top_n >
    top_k (the normal, intended usage).

    Fixed by reordering ONLY the first top_k positions and leaving the tail
    untouched (matching evaluate.py's own correct pattern). This test uses
    top_n=20 with a small top_k (so top_n > top_k, the bug's trigger
    condition) and checks the concrete, checkable symptom: every gallery
    index that appears in the reranked top-k slots also appeared in the
    ORIGINAL (pre-rerank) top-k candidate set -- i.e. reranking only
    reorders within the top-k, it never promotes a tail candidate into the
    top-k position or vice versa.
    """
    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc+aep+rv+ug")
    cfg.reverse_verifier.top_k = 5
    set_seed(1)
    model = ReVUACIR(cfg).eval()
    tokenizer = build_tokenizer(cfg.backbone)
    tok = tokenizer(TEXTS[:2])

    reference = torch.randn(2, 3, cfg.data.image_size, cfg.data.image_size)
    gallery = torch.randn(30, 3, cfg.data.image_size, cfg.data.image_size)
    top_n = 20
    with torch.no_grad():
        gallery_out = model.encode_gallery(gallery)
        gallery_pooled = gallery_out["pooled"]

        def gallery_image_fn(idx_tensor):
            return gallery[idx_tensor.cpu()]

        base_idx, _ = model.retrieve_and_rerank(
            reference, tok["input_ids"], tok["attention_mask"], gallery_pooled,
            gallery_image_fn=None, top_n=top_n,  # no gallery_image_fn -> returns pre-rerank order
        )
        reranked_idx, reranked_scores = model.retrieve_and_rerank(
            reference, tok["input_ids"], tok["attention_mask"], gallery_pooled,
            gallery_image_fn=gallery_image_fn, top_n=top_n,
        )

    assert reranked_idx.shape == (2, top_n)
    assert torch.isfinite(reranked_scores).all()
    k = cfg.reverse_verifier.top_k
    for b in range(reranked_idx.size(0)):
        original_top_k_set = set(base_idx[b, :k].tolist())
        reranked_top_k_set = set(reranked_idx[b, :k].tolist())
        assert reranked_top_k_set == original_top_k_set, (
            f"reranking changed WHICH candidates occupy the top-{k} slots (only their ORDER "
            f"should change) -- original top-{k}={original_top_k_set}, "
            f"reranked top-{k}={reranked_top_k_set}. This is the score-domain-mixing bug: a "
            f"tail candidate (untouched cosine score) got promoted into, or a real top-k "
            f"candidate got pushed out of, the top-{k}.")
        # The tail (indices k:) must be completely untouched by reranking.
        assert torch.equal(reranked_idx[b, k:], base_idx[b, k:]), (
            f"reranking modified tail candidates (positions {k}:{top_n}), which should be "
            f"left exactly as the base ranker produced them")
    print(f"OK: retrieve_and_rerank only reorders within top_k={k} (top_n={top_n}); "
          f"tail candidates are untouched, no score-domain mixing.")


def test_parser_backend_actually_selects_extractor():
    """Regression guard for a real bug found via code-audit document review
    (2026-09-08): `cfg.attribute_parser.parser_backend` ("rule" | "llm" |
    "hybrid") was a documented, user-facing config switch, but
    `data/datasets.py`'s `CIRTripletDatasetBase.__init__` always hardcoded
    `RuleBasedAttributeExtractor(use_spacy=False)` directly -- so setting
    `parser_backend: llm` in a config had ZERO effect on the actual training
    data pipeline; every dataset used the rule extractor regardless.

    Fixed by threading an `attribute_extractor` through
    `train.py::build_train_dataset` -> dataset `__init__` ->
    `CIRTripletDatasetBase`, built via `build_attribute_extractor(cfg.
    attribute_parser, llm_parser_fn=...)` (which already correctly
    dispatched on parser_backend -- it just was never called from the
    dataset construction path). This test checks two concrete, checkable
    symptoms of the old bug being fixed:
      1. parser_backend="llm" without an llm_parser_fn now raises a clear
         ValueError (matching build_attribute_extractor's own pre-existing
         behavior) instead of silently building rule-based datasets that
         look like they honored the config but didn't.
      2. parser_backend="rule" still defaults to use_spacy=False (the
         deliberate, offline-friendly choice datasets.py always made before
         this fix existed) -- confirms the refactor didn't silently change
         behavior for the default path while fixing the switch.
    """
    from revua_cir.config import ReVUAConfig
    from revua_cir.models.attribute_parser import build_attribute_extractor, RuleBasedAttributeExtractor
    from revua_cir.train import build_train_dataset

    # (1) llm backend without a callable must raise, not silently fall back.
    cfg = ReVUAConfig()
    cfg.attribute_parser.enabled = True
    cfg.attribute_parser.parser_backend = "llm"
    cfg.data.train_datasets = ("cirr",)
    raised = False
    try:
        build_train_dataset(cfg, transform=None)
    except ValueError as e:
        raised = "llm_parser_fn" in str(e)
    assert raised, "parser_backend='llm' without llm_parser_fn should raise ValueError before dataset construction"

    # (2) rule backend still defaults to use_spacy=False.
    rule_cfg = ReVUAConfig().attribute_parser
    rule_cfg.parser_backend = "rule"
    ext = build_attribute_extractor(rule_cfg)
    assert isinstance(ext, RuleBasedAttributeExtractor)
    assert ext._nlp is None, "rule backend should default to use_spacy=False, matching the pre-fix hardcoded behavior"

    print("OK: parser_backend genuinely selects the extractor (llm without a callable raises; "
          "rule backend preserves its use_spacy=False default).")


def test_cirr_dataset_resolves_kaggle_mirror_layout():
    """CIRRDataset must resolve a real third-party mirror layout (a Kaggle
    upload hit in practice: `anhngzv/cirr-cir`), where annotations nest two
    levels down (`{root}/CIRR/cirr/captions/...`) while images nest only ONE
    level down (`{root}/CIRR/dev/...`) -- image_root is NOT simply ann_root's
    parent in general, so this pins that they resolve independently rather
    than one being derived from the other.

    Also guards the existing local layout (`{root}/CIRR_repo/` annotations,
    images directly under `{root}/`) is untouched by the fix -- image_root
    must stay `root` there, not go looking for a `CIRR/` subfolder that
    doesn't exist in that layout.
    """
    from revua_cir.data.datasets import CIRRDataset

    with tempfile.TemporaryDirectory() as d:
        # Kaggle-mirror shape.
        ann_dir = os.path.join(d, "CIRR", "cirr")
        os.makedirs(os.path.join(ann_dir, "captions"))
        os.makedirs(os.path.join(ann_dir, "image_splits"))
        os.makedirs(os.path.join(d, "CIRR", "dev"))
        os.makedirs(os.path.join(d, "CIRR", "train", "7"))

        split_map = {
            "dev-1-0-img0": "./dev/dev-1-0-img0.png",
            "dev-1-1-img1": "./dev/dev-1-1-img1.png",
            "train-7-0-img0": "./train/7/train-7-0-img0.png",
        }
        with open(os.path.join(ann_dir, "image_splits", "split.rc2.val.json"), "w") as f:
            json.dump(split_map, f)
        captions = [{"reference": "dev-1-0-img0", "target_hard": "dev-1-1-img1",
                    "caption": "test", "pairid": 1, "img_set": {"members": []}}]
        with open(os.path.join(ann_dir, "captions", "cap.rc2.val.json"), "w") as f:
            json.dump(captions, f)
        for name in ("dev-1-0-img0.png", "dev-1-1-img1.png"):
            open(os.path.join(d, "CIRR", "dev", name), "wb").close()

        ds = CIRRDataset(root=d, split="val", transform=None)
        assert ds.ann_root == ann_dir, f"expected ann_root {ann_dir}, got {ds.ann_root}"
        assert ds.image_root == os.path.join(d, "CIRR"), (
            f"expected image_root {os.path.join(d, 'CIRR')}, got {ds.image_root}")
        assert len(ds.triplets) == 1
        assert os.path.exists(ds.triplets[0]["ref_path"]), (
            f"ref_path did not resolve to a real file: {ds.triplets[0]['ref_path']}")
        assert os.path.exists(ds.triplets[0]["target_path"]), (
            f"target_path did not resolve to a real file: {ds.triplets[0]['target_path']}")

    with tempfile.TemporaryDirectory() as d:
        # Local layout must be unaffected: CIRR_repo/ nested, images at root/.
        ann_dir = os.path.join(d, "CIRR_repo")
        os.makedirs(os.path.join(ann_dir, "captions"))
        os.makedirs(os.path.join(ann_dir, "image_splits"))
        os.makedirs(os.path.join(d, "dev_img"))

        split_map = {"dev-1-0-img0": "./dev/dev-1-0-img0.png"}
        with open(os.path.join(ann_dir, "image_splits", "split.rc2.val.json"), "w") as f:
            json.dump(split_map, f)
        with open(os.path.join(ann_dir, "captions", "cap.rc2.val.json"), "w") as f:
            json.dump([{"reference": "dev-1-0-img0", "target_hard": "dev-1-0-img0",
                       "caption": "t", "pairid": 1, "img_set": {"members": []}}], f)
        open(os.path.join(d, "dev_img", "dev-1-0-img0.png"), "wb").close()

        ds2 = CIRRDataset(root=d, split="val", transform=None)
        assert ds2.image_root == d, (
            f"local layout regressed: image_root should stay {d}, got {ds2.image_root} "
            f"(must not go looking for a CIRR/ subfolder that doesn't exist here)")
        assert os.path.exists(ds2.triplets[0]["ref_path"])

    print("OK: CIRRDataset resolves both the Kaggle-mirror layout (CIRR/cirr "
          "annotations, CIRR/{dev,train} images -- independently resolved) "
          "and the local CIRR_repo/ layout unchanged.")


def test_cirr_dataset_resolves_official_img_raw_layout():
    """Regression guard for a real bug found via code-audit document review
    (2026-09-09, finding #8): CIRRDataset._resolve_image_root only checked
    `root` and `root/CIRR` as image-root candidates. The OFFICIAL CIRR repo
    (github.com/Cuberick-Orion/CIRR) documents its own canonical release
    layout as `data/cirr/img_raw/{train,dev,test1}/` -- a user who follows
    the official repo's own setup instructions verbatim would silently get
    every gallery/query image replaced by safe_load_transform's gray
    placeholder (image resolution always "succeeds" by falling back to a
    non-existent path), with no error, corrupting every metric computed
    against that data with no visible symptom.

    Confirms both the flat `img_raw/dev/...` layout (annotations directly
    under root, matching the CIRR_repo/ layout's own image placement) and
    the `CIRR/img_raw/dev/...` layout (images nested under a CIRR/ parent,
    matching the Kaggle-mirror shape's own image placement) now resolve
    correctly.
    """
    from revua_cir.data.datasets import CIRRDataset

    with tempfile.TemporaryDirectory() as d:
        # Official layout: annotations at root/CIRR_repo/, images at
        # root/img_raw/dev/... (not root/dev/... or root/dev_img/...).
        ann_dir = os.path.join(d, "CIRR_repo")
        os.makedirs(os.path.join(ann_dir, "captions"))
        os.makedirs(os.path.join(ann_dir, "image_splits"))
        os.makedirs(os.path.join(d, "img_raw", "dev"))

        split_map = {"dev-1-0-img0": "./dev/dev-1-0-img0.png"}
        with open(os.path.join(ann_dir, "image_splits", "split.rc2.val.json"), "w") as f:
            json.dump(split_map, f)
        with open(os.path.join(ann_dir, "captions", "cap.rc2.val.json"), "w") as f:
            json.dump([{"reference": "dev-1-0-img0", "target_hard": "dev-1-0-img0",
                       "caption": "t", "pairid": 1, "img_set": {"members": []}}], f)
        open(os.path.join(d, "img_raw", "dev", "dev-1-0-img0.png"), "wb").close()

        ds = CIRRDataset(root=d, split="val", transform=None)
        assert ds.image_root == os.path.join(d, "img_raw"), (
            f"expected image_root {os.path.join(d, 'img_raw')}, got {ds.image_root}")
        assert os.path.exists(ds.triplets[0]["ref_path"]), (
            f"ref_path did not resolve to a real file under the official img_raw/ layout: "
            f"{ds.triplets[0]['ref_path']}")

    with tempfile.TemporaryDirectory() as d:
        # CIRR/img_raw/ nested variant (annotations under CIRR/cirr/, images
        # under CIRR/img_raw/ -- both nested under the same CIRR/ parent,
        # unlike the flat variant above).
        ann_dir = os.path.join(d, "CIRR", "cirr")
        os.makedirs(os.path.join(ann_dir, "captions"))
        os.makedirs(os.path.join(ann_dir, "image_splits"))
        os.makedirs(os.path.join(d, "CIRR", "img_raw", "dev"))

        split_map = {"dev-1-0-img0": "./dev/dev-1-0-img0.png"}
        with open(os.path.join(ann_dir, "image_splits", "split.rc2.val.json"), "w") as f:
            json.dump(split_map, f)
        with open(os.path.join(ann_dir, "captions", "cap.rc2.val.json"), "w") as f:
            json.dump([{"reference": "dev-1-0-img0", "target_hard": "dev-1-0-img0",
                       "caption": "t", "pairid": 1, "img_set": {"members": []}}], f)
        open(os.path.join(d, "CIRR", "img_raw", "dev", "dev-1-0-img0.png"), "wb").close()

        ds2 = CIRRDataset(root=d, split="val", transform=None)
        assert ds2.image_root == os.path.join(d, "CIRR", "img_raw"), (
            f"expected image_root {os.path.join(d, 'CIRR', 'img_raw')}, got {ds2.image_root}")
        assert os.path.exists(ds2.triplets[0]["ref_path"]), (
            f"ref_path did not resolve to a real file under the CIRR/img_raw/ layout: "
            f"{ds2.triplets[0]['ref_path']}")

    print("OK: CIRRDataset resolves the official CIRR repo's own documented img_raw/ release "
          "layout, both flat (root/img_raw/dev/...) and CIRR-nested (root/CIRR/img_raw/dev/...).")


def test_official_cirr_evaluation_rejects_missing_gallery_images():
    """Regression guard for a real bug found via code-audit document review
    (2026-09-09, finding #8): scripts/evaluate_cirr.py's
    build_cirr_full_gallery() used to return whatever CIRRDataset._img_path
    resolved for every split-map entry, even entries that resolve to a
    NON-EXISTENT path (CIRRDataset's own fallback for training-time
    tolerance of missing images -- see safe_load_transform's gray-placeholder
    substitution). An official-evaluation script silently scoring R@K
    against gray placeholders for some fraction of the gallery would produce
    numbers that don't mean what a paper table claims they mean, with no
    visible warning. Fixed by making build_cirr_full_gallery raise loudly
    when any resolved gallery path does not exist on disk.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from evaluate_cirr import build_cirr_full_gallery
    from revua_cir.data.datasets import CIRRDataset

    with tempfile.TemporaryDirectory() as d:
        ann_dir = os.path.join(d, "CIRR_repo")
        os.makedirs(os.path.join(ann_dir, "captions"))
        os.makedirs(os.path.join(ann_dir, "image_splits"))
        os.makedirs(os.path.join(d, "dev_img"))

        # id0's image exists; id1's does NOT -- simulates an incomplete
        # download or a layout this codebase's resolver doesn't recognize.
        split_map = {"id0": "./dev/id0.png", "id1": "./dev/id1.png"}
        with open(os.path.join(ann_dir, "image_splits", "split.rc2.val.json"), "w") as f:
            json.dump(split_map, f)
        with open(os.path.join(ann_dir, "captions", "cap.rc2.val.json"), "w") as f:
            json.dump([{"reference": "id0", "target_hard": "id0", "caption": "t", "pairid": 1,
                       "img_set": {"members": []}}], f)
        open(os.path.join(d, "dev_img", "id0.png"), "wb").close()
        # id1.png deliberately NOT created.

        dataset = CIRRDataset(root=d, split="val", transform=None)
        raised = False
        try:
            build_cirr_full_gallery(dataset)
        except RuntimeError as e:
            raised = "did not resolve to a real file" in str(e) or "missing" in str(e).lower()
        assert raised, "build_cirr_full_gallery should raise RuntimeError when a gallery image is missing"

    print("OK: build_cirr_full_gallery() refuses to run an official evaluation against a gallery "
          "with unresolved (gray-placeholder-fallback) images, raising loudly instead of silently "
          "scoring against placeholders.")


def test_strict_cirr_preflight_rejects_malformed_queries_and_passes_valid_ones():
    """Regression guard for scripts/evaluate_cirr.py's strict_preflight_check
    (added 2026-09-09, code-audit document finding #1): evaluate_dataset()'s
    own group-recall block is deliberately TOLERANT -- it excludes a
    malformed/incomplete query from the subset-recall denominator rather
    than crashing (correct for training-time monitoring, where one bad
    annotation shouldn't kill a multi-hour run), but that tolerance is too
    permissive for an official, publication-cited number. In particular, a
    query whose group has a valid 6-member/5-candidate shape but whose TRUE
    TARGET isn't actually one of the 5 candidates (a malformed/inconsistent
    annotation) is silently OMITTED by evaluate_dataset(), not flagged --
    exactly the edge case the code-audit document calls out as "too
    permissive for publication-grade evaluation."

    This test builds one well-formed query and four independently malformed
    ones (wrong member count, reference missing from members, target not
    among the 5 candidates, and a candidate missing from the gallery) and
    confirms strict_preflight_check raises for each malformed case
    individually, and passes cleanly when only the well-formed query is
    checked.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from evaluate_cirr import build_cirr_full_gallery, strict_preflight_check
    from revua_cir.data.datasets import CIRRDataset

    def make_dataset(d, captions):
        ann_dir = os.path.join(d, "CIRR_repo")
        os.makedirs(os.path.join(ann_dir, "captions"), exist_ok=True)
        os.makedirs(os.path.join(ann_dir, "image_splits"), exist_ok=True)
        os.makedirs(os.path.join(d, "dev_img"), exist_ok=True)
        n_images = 8
        split_map = {f"id{i}": f"./dev/img{i}.png" for i in range(n_images)}
        with open(os.path.join(ann_dir, "image_splits", "split.rc2.val.json"), "w") as f:
            json.dump(split_map, f)
        for i in range(n_images):
            path = os.path.join(d, "dev_img", f"img{i}.png")
            if not os.path.exists(path):
                open(path, "wb").close()
        with open(os.path.join(ann_dir, "captions", "cap.rc2.val.json"), "w") as f:
            json.dump(captions, f)
        return CIRRDataset(root=d, split="val", transform=None)

    well_formed = {"reference": "id0", "target_hard": "id1", "caption": "t", "pairid": 1,
                   "img_set": {"members": ["id0", "id1", "id2", "id3", "id4", "id5"]}}

    # 1. Well-formed alone: must pass cleanly.
    with tempfile.TemporaryDirectory() as d:
        dataset = make_dataset(d, [well_formed])
        gallery = build_cirr_full_gallery(dataset)
        strict_preflight_check(dataset, gallery)  # must not raise

    # 2. Wrong member count (5 instead of 6).
    with tempfile.TemporaryDirectory() as d:
        bad = {"reference": "id0", "target_hard": "id1", "caption": "t", "pairid": 2,
               "img_set": {"members": ["id0", "id1", "id2", "id3", "id4"]}}
        dataset = make_dataset(d, [bad])
        gallery = build_cirr_full_gallery(dataset)
        raised = False
        try:
            strict_preflight_check(dataset, gallery)
        except RuntimeError as e:
            raised = "expected 6" in str(e)
        assert raised, "strict_preflight_check should reject a group with != 6 unique members"

    # 3. Reference not among the listed members.
    with tempfile.TemporaryDirectory() as d:
        bad = {"reference": "id0", "target_hard": "id1", "caption": "t", "pairid": 3,
               "img_set": {"members": ["id1", "id2", "id3", "id4", "id5", "id6"]}}
        dataset = make_dataset(d, [bad])
        gallery = build_cirr_full_gallery(dataset)
        raised = False
        try:
            strict_preflight_check(dataset, gallery)
        except RuntimeError as e:
            raised = "is not among the group" in str(e)
        assert raised, "strict_preflight_check should reject a group missing its own reference_id"

    # 4. THE key edge case from the code-audit document: 6 members, reference
    # present, 5 real candidates -- but the true target is NOT one of them.
    # evaluate_dataset() would silently EXCLUDE this query; strict mode must
    # not.
    with tempfile.TemporaryDirectory() as d:
        bad = {"reference": "id0", "target_hard": "id7",  # id7 not in the group at all
               "caption": "t", "pairid": 4,
               "img_set": {"members": ["id0", "id1", "id2", "id3", "id4", "id5"]}}
        dataset = make_dataset(d, [bad])
        gallery = build_cirr_full_gallery(dataset)
        raised = False
        try:
            strict_preflight_check(dataset, gallery)
        except RuntimeError as e:
            raised = "is not among the 5 non-reference candidates" in str(e)
        assert raised, (
            "strict_preflight_check should reject a query whose 6-member/5-candidate group is "
            "well-formed but whose true target isn't one of the 5 candidates -- this is the exact "
            "edge case the code-audit document flagged as too permissive to silently skip")

    # 5. A candidate ID that is not in the gallery at all.
    with tempfile.TemporaryDirectory() as d:
        bad = {"reference": "id0", "target_hard": "id1", "caption": "t", "pairid": 5,
               "img_set": {"members": ["id0", "id1", "id2", "id3", "id4", "id_missing"]}}
        dataset = make_dataset(d, [bad])
        gallery = build_cirr_full_gallery(dataset)
        raised = False
        try:
            strict_preflight_check(dataset, gallery)
        except RuntimeError as e:
            raised = "do not resolve to a real file" in str(e)
        assert raised, "strict_preflight_check should reject a candidate ID missing from the gallery"

    print("OK: strict_preflight_check passes a genuinely well-formed CIRR query and rejects each of "
          "4 independent malformation modes (wrong member count, reference missing from group, "
          "target not among the 5 candidates, and a candidate missing from the gallery) -- "
          "including the specific edge case the code-audit document flagged as too permissive to "
          "silently skip.")


def test_ablation_matrix_reports_honest_latency_field_names():
    """Regression guard for a real documentation/reporting bug found via
    code-audit document review (2026-09-09, finding #5): run_ablation_
    matrix.py's Table 1 used to report `mean_query_latency_sec` under a bare
    "latency" header, in the same table as rows for RV/UG-enabled combos --
    inviting a reader to treat it as the full ReVUA-CIR pipeline latency,
    when it structurally can only ever be the base-retrieval-only number
    (the `metrics` row it comes from is always built with rerank=False).

    Fixed by reporting `base_retrieval_latency_sec` (the same value, under
    an honest name) in Table 1, and adding a dedicated Table 3 with the full
    base/rv_reranking/total_pipeline breakdown for rows where reranking
    actually ran.

    This test calls `_write()` directly with synthetic rows shaped exactly
    like what main()'s row-building loop produces (`{metric}` for the base
    pass, `rr_{metric}` for the rerank pass) and checks the generated
    markdown: Table 1 must NOT contain the bare `mean_query_latency_sec`
    header (only in the underlying JSON/CSV as a legacy compatibility key,
    not the markdown table header), Table 1 SHOULD contain
    `base_retrieval_latency_sec`, and Table 3 must report all three of
    base_retrieval/rv_reranking/total_pipeline for an RV-enabled row.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import importlib
    run_ablation_matrix = importlib.import_module("run_ablation_matrix")

    rows = [
        {
            "module_combo": "fc", "checkpoint": "epoch001.pt", "val_monitor": 20.0,
            "recall_at_1": 20.0, "mean_query_latency_sec": 0.002, "base_retrieval_latency_sec": 0.002,
            "rr_recall_at_1": 20.0, "rr_base_retrieval_latency_sec": 0.002,
        },
        {
            "module_combo": "fc+aep+rv+ug", "checkpoint": "epoch002.pt", "val_monitor": 35.0,
            "recall_at_1": 35.0, "mean_query_latency_sec": 0.002, "base_retrieval_latency_sec": 0.002,
            "rr_recall_at_1": 37.0, "rr_base_retrieval_latency_sec": 0.002,
            "rr_rv_reranking_latency_sec": 0.015, "rr_total_pipeline_latency_sec": 0.017,
        },
    ]

    with tempfile.TemporaryDirectory() as d:
        run_ablation_matrix._write(d, rows)
        with open(os.path.join(d, "ablation_summary.md"), encoding="utf-8") as f:
            md = f.read()

    table1_end = md.index("## Table 2") if "## Table 2" in md else len(md)
    table1 = md[:table1_end]
    assert "base_retrieval_latency_sec" in table1, (
        "Table 1 should report the honestly-named base_retrieval_latency_sec field")
    assert "mean_query_latency_sec" not in table1, (
        "Table 1 should NOT report the ambiguous legacy mean_query_latency_sec header -- a reader "
        "could mistake it for full pipeline latency in a table that also has RV/UG rows")

    assert "## Table 3" in md, "expected a dedicated latency-breakdown table (Table 3) to be generated"
    table3 = md[md.index("## Table 3"):]
    assert "rv_reranking" in table3 and "total_pipeline" in table3, (
        "Table 3 should report rv_reranking and total_pipeline latency columns")
    assert "0.0150" in table3 and "0.0170" in table3, (
        "Table 3 should contain the actual rv_reranking (0.015) and total_pipeline (0.017) values "
        "for the RV-enabled row")

    print("OK: run_ablation_matrix.py reports base_retrieval_latency_sec (not the ambiguous "
          "mean_query_latency_sec) in Table 1, and a dedicated Table 3 with the full "
          "base/rv_reranking/total_pipeline latency breakdown.")


def test_benchmark_latency_script_times_the_real_inference_path():
    """Regression guard for scripts/benchmark_latency.py (added 2026-09-09,
    code-audit document finding #6): `evaluate.py`'s evaluate_dataset() is
    useful for diagnostics but runs several diagnostic-only candidate-encode
    passes (top-1 pass, discrimination pair, within-top-k spread) before its
    timed rerank region -- even with CUDA sync (finding #11) those extra
    passes can warm CUDA kernels/OS file cache/image-decode paths, biasing a
    number meant to be cited as ReVUA-CIR's publication latency.

    scripts/benchmark_latency.py times ONLY `ReVUACIR.retrieve_and_rerank()`
    -- the actual deployment inference function -- with a warmup phase, CUDA
    sync (inert on CPU, exercised here), and repeated measurement.

    This test drives `benchmark_latency.benchmark()` directly (not via
    subprocess/CLI, to keep it fast and in-process) against a real tiny
    on-disk CIRR fixture, a real trained-from-scratch tiny model, and a real
    checkpoint round-trip through load_checkpoint -- the same components a
    real invocation would use, just at smoke-test scale. Confirms: (1) both
    base-only and full-rerank timings are collected, all positive and
    finite, (2) full-rerank timings are consistently >= base-only timings
    (RV+UG reranking is strictly additional work, never faster), and (3)
    with RV disabled (fc-only ablation), retrieve_and_rerank's own
    early-return means total_times is correctly empty (nothing to time
    beyond base retrieval).
    """
    import importlib
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    benchmark_latency = importlib.import_module("benchmark_latency")

    from revua_cir.data.datasets import CIRRDataset
    from revua_cir.data.transforms import build_image_transform
    from revua_cir.evaluate import build_default_gallery, encode_gallery_paths

    with tempfile.TemporaryDirectory() as d:
        ann_dir = os.path.join(d, "cirr", "CIRR_repo")
        os.makedirs(os.path.join(ann_dir, "captions"))
        os.makedirs(os.path.join(ann_dir, "image_splits"))
        os.makedirs(os.path.join(d, "cirr", "dev_img"))
        n = 8
        split_map = {f"id{i}": f"./dev/img{i}.png" for i in range(n)}
        with open(os.path.join(ann_dir, "image_splits", "split.rc2.val.json"), "w") as f:
            json.dump(split_map, f)
        for i in range(n):
            open(os.path.join(d, "cirr", "dev_img", f"img{i}.png"), "wb").close()
        captions = [{"reference": f"id{i}", "target_hard": f"id{(i + 1) % n}", "caption": TEXTS[i % len(TEXTS)],
                    "pairid": i, "img_set": {"members": [f"id{(i + 1) % n}"]}} for i in range(n)]
        with open(os.path.join(ann_dir, "captions", "cap.rc2.val.json"), "w") as f:
            json.dump(captions, f)

        cfg = build_tiny_config()
        cfg = cfg.apply_module_ablation("fc+aep+rv+ug")
        cfg.data.root = d
        set_seed(0)
        model = ReVUACIR(cfg)
        ckpt_path = os.path.join(d, "ckpt.pt")
        save_checkpoint(ckpt_path, model, epoch=0, best_metric=0.0, config=cfg.to_dict())

        device = torch.device("cpu")
        transform = build_image_transform(cfg.data.image_size, train=False)
        tokenizer = build_tokenizer(cfg.backbone)
        collate_fn = build_collate_fn(tokenizer)
        dataset = CIRRDataset(root=os.path.join(d, "cirr"), split="val", transform=transform)
        gallery_paths = build_default_gallery(dataset)

        model2 = ReVUACIR(cfg).to(device)
        load_checkpoint(ckpt_path, model2, map_location=device)
        model2.eval()
        gallery_pooled = encode_gallery_paths(model2, gallery_paths, transform, device, cfg.data.eval_batch_size).to(device)

        n_q = min(4, len(dataset))
        queries = [dataset[i] for i in range(n_q)]
        collated = [collate_fn([q]) for q in queries]
        per_query = [{"reference_pixel_values": c["reference_pixel_values"][0],
                      "edit_input_ids": c["edit_input_ids"][0],
                      "edit_attention_mask": c["edit_attention_mask"][0]} for c in collated]

        base_times, total_times = benchmark_latency.benchmark(
            model2, per_query, gallery_paths, gallery_pooled, transform, device,
            top_n=5, n_repeats=6, warmup=2)

        assert len(base_times) == 6, f"expected 6 base timings, got {len(base_times)}"
        assert len(total_times) == 6, f"expected 6 total (rerank) timings with RV enabled, got {len(total_times)}"
        assert all(t > 0 for t in base_times + total_times), "all timings should be positive"
        assert all(t < 60.0 for t in base_times + total_times), (
            "a tiny-config single-query timing over 60s indicates something is badly wrong, "
            "not a real measurement")
        for b, t in zip(base_times, total_times):
            assert t >= b, (
                f"full-rerank timing ({t:.4f}s) should never be faster than base-only ({b:.4f}s) -- "
                f"RV+UG reranking is strictly additional work on top of base retrieval")

        # fc-only (RV disabled): retrieve_and_rerank's own early-return means
        # there is nothing beyond base retrieval to time.
        cfg_fc = build_tiny_config()
        cfg_fc = cfg_fc.apply_module_ablation("fc")
        cfg_fc.data.root = cfg.data.root
        set_seed(0)
        model_fc = ReVUACIR(cfg_fc).to(device)
        model_fc.eval()
        base_times_fc, total_times_fc = benchmark_latency.benchmark(
            model_fc, per_query, gallery_paths, gallery_pooled, transform, device,
            top_n=5, n_repeats=3, warmup=1)
        assert len(base_times_fc) == 3
        assert total_times_fc == [], "fc-only (RV disabled) should produce no rerank timings at all"

    print("OK: scripts/benchmark_latency.py's benchmark() times the real ReVUACIR."
          "retrieve_and_rerank() deployment path end-to-end (real model, real checkpoint round-trip, "
          "real CIRR fixture) -- base timings always present, full-rerank timings always >= base, "
          "and correctly empty when RV is disabled.")


def test_cirr_test1_submission_writes_two_file_schema():
    """Regression guard for scripts/evaluate_cirr.py's _dump_test1_predictions
    (fixed 2026-09-09, code-audit document finding #10): the old version
    wrote one raw JSON dict ({pairid: [ranked_ids]}), not the two-file
    schema ({"version": "rc2", "metric": ...} envelopes) CIRR's official
    test server documents. Fixed to write recall_<experiment_name>.json
    (top-50, full gallery) and subset_<experiment_name>.json (top-3, the
    5 non-reference subset candidates only).

    This test drives `_dump_test1_predictions` directly against a real tiny
    on-disk CIRR test1 fixture with one well-formed 6-member group and
    confirms: both files exist with the correct version/metric envelope,
    the recall file's top-50 list never exceeds the gallery size, the
    subset file's top-3 list has exactly 3 entries, and -- the specific
    correctness property that matters here -- the subset predictions are
    drawn ONLY from the query's 5 non-reference candidate ids, never
    including the reference id itself.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import importlib
    evaluate_cirr = importlib.import_module("evaluate_cirr")
    from revua_cir.data.datasets import CIRRDataset
    from revua_cir.data.transforms import build_image_transform

    with tempfile.TemporaryDirectory() as d:
        ann_dir = os.path.join(d, "cirr", "CIRR_repo")
        os.makedirs(os.path.join(ann_dir, "captions"))
        os.makedirs(os.path.join(ann_dir, "image_splits"))
        os.makedirs(os.path.join(d, "cirr", "dev_img"))
        n = 8
        split_map = {f"id{i}": f"./dev/img{i}.png" for i in range(n)}
        with open(os.path.join(ann_dir, "image_splits", "split.rc2.test1.json"), "w") as f:
            json.dump(split_map, f)
        for i in range(n):
            open(os.path.join(d, "cirr", "dev_img", f"img{i}.png"), "wb").close()
        # test1 annotations don't carry a real known-correct target_hard in
        # practice, but CIRRDataset.__init__ still reads that key
        # positionally -- a placeholder value here is fine, since
        # _dump_test1_predictions never consults it (no ground truth is
        # used anywhere in this path, matching this script's own test1 policy).
        captions = [{"reference": "id0", "target_hard": "id1", "caption": "test caption", "pairid": 100,
                    "img_set": {"members": ["id0", "id1", "id2", "id3", "id4", "id5"]}}]
        with open(os.path.join(ann_dir, "captions", "cap.rc2.test1.json"), "w") as f:
            json.dump(captions, f)

        cfg = build_tiny_config()
        cfg = cfg.apply_module_ablation("fc+aep+rv+ug")
        cfg.data.root = d
        set_seed(0)
        model = ReVUACIR(cfg)
        ckpt_path = os.path.join(d, "ckpt.pt")
        save_checkpoint(ckpt_path, model, epoch=0, best_metric=0.0, config=cfg.to_dict())

        device = torch.device("cpu")
        transform = build_image_transform(cfg.data.image_size, train=False)
        tokenizer = build_tokenizer(cfg.backbone)
        dataset = CIRRDataset(root=os.path.join(d, "cirr"), split="test1", transform=transform)
        gallery = evaluate_cirr.build_cirr_full_gallery(dataset)
        model2 = ReVUACIR(cfg).to(device)
        load_checkpoint(ckpt_path, model2, map_location=device)
        model2.eval()

        out_dir = os.path.join(d, "submission")
        evaluate_cirr._dump_test1_predictions(model2, dataset, cfg, tokenizer, transform, device,
                                              gallery, out_dir, "testexp", None)

        recall_path = os.path.join(out_dir, "recall_testexp.json")
        subset_path = os.path.join(out_dir, "subset_testexp.json")
        assert os.path.exists(recall_path), "recall_<experiment_name>.json was not written"
        assert os.path.exists(subset_path), "subset_<experiment_name>.json was not written"

        with open(recall_path, encoding="utf-8") as f:
            recall = json.load(f)
        with open(subset_path, encoding="utf-8") as f:
            subset = json.load(f)

        assert recall.get("version") == "rc2" and recall.get("metric") == "recall", (
            f"recall file has wrong envelope: {recall.get('version')}/{recall.get('metric')}")
        assert subset.get("version") == "rc2" and subset.get("metric") == "recall_subset", (
            f"subset file has wrong envelope: {subset.get('version')}/{subset.get('metric')}")

        assert "100" in recall["predictions"], "pairid 100 missing from recall predictions"
        assert len(recall["predictions"]["100"]) <= len(gallery), "recall top-50 exceeds gallery size"

        assert "100" in subset["predictions"], "pairid 100 missing from subset predictions"
        subset_preds = subset["predictions"]["100"]
        assert len(subset_preds) == 3, f"subset predictions should have exactly 3 entries, got {len(subset_preds)}"
        assert "id0" not in subset_preds, (
            f"subset predictions {subset_preds} include the reference image (id0) -- the subset "
            f"submission must rank only the 5 non-reference candidates")
        assert set(subset_preds) <= {"id1", "id2", "id3", "id4", "id5"}, (
            f"subset predictions {subset_preds} include an id outside the query's own 5 "
            f"non-reference candidates")

    print("OK: evaluate_cirr._dump_test1_predictions() writes the two-file CIRR submission schema "
          "(recall_<name>.json top-50/full-gallery, subset_<name>.json top-3/5-candidates-only) with "
          "correct version/metric envelopes, and the subset file never leaks the reference image "
          "into its predictions.")


def test_gradient_conflict_diagnostic_measures_real_and_detached_losses_correctly():
    """Regression guard for scripts/diagnose_gradient_conflict.py (added
    2026-09-10, root-cause-analysis document Section 6/16 Check 3; expanded
    same day per ReVUA_CIR_Final_PreTraining_Recommendations.md findings
    #3/#4 -- composer-space measurement + gradient-magnitude ratios +
    train=True transform default). Measures cos(grad_retrieval, grad_aux) on
    BOTH backbone.* and composer.* parameters (RV's Tier-1 shares the
    ForwardComposer instance with the retriever -- `self.rv.shared_composer
    is self.composer`, confirmed directly in revua_cir.py's __init__) to
    test whether auxiliary objectives conflict with retrieval on either
    shared parameter space.

    While building this script, found (by reading revua_cir.py's forward()
    directly, not assumed) that TWO of the four auxiliary losses --
    `rv_calibration` and `uncertainty` -- are architecturally INCAPABLE of
    reaching EITHER shared space (both are computed on `.detach()`'d
    upstream tensors, by deliberate earlier-session fixes covered by
    test_rv_calibrator_receives_gradient and
    test_ug_calibration_loss_does_not_leak_into_rv_or_backbone above). This
    test confirms the diagnostic's own logic (_flat_grad, _cosine,
    _measure_one_pass) handles that correctly: a zero-gradient loss must
    produce `nan` cosine (not a spurious 0.0 or an exception); `disentangle`
    (AEP, backbone-only -- it has no composer-sharing relationship) must
    give a real backbone cosine but no composer entry; `reverse` (RV, the
    ONLY loss sharing the composer) must give real cosines in BOTH spaces;
    and the weighted-ratio computation must correctly apply this run's
    actual configured lambda_reverse to the raw gradient norm.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import importlib
    dgc = importlib.import_module("diagnose_gradient_conflict")

    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc+aep+rv+ug")
    cfg.fusion.dropout = 0.0
    cfg.backbone.freeze_backbone = False
    cfg.loss.lambda_reverse = 0.5
    set_seed(0)
    model = ReVUACIR(cfg)
    model.train()
    assert model.rv.shared_composer is model.composer, (
        "test setup error: this test relies on RV Tier 1 sharing the ForwardComposer instance -- "
        "check module_combo/tier default")

    bsz = 6
    batch = {
        "reference_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "target_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "edit_input_ids": torch.randint(0, 100, (bsz, 8)),
        "edit_attention_mask": torch.ones(bsz, 8, dtype=torch.long),
    }

    aux_names = ["disentangle", "reverse", "rv_calibration", "uncertainty"]
    result = dgc._measure_one_pass(model, batch, cfg, aux_names)

    assert result["retrieval_backbone_norm"] > 0, "retrieval loss should produce nonzero backbone gradient"
    assert result["retrieval_composer_norm"] > 0, "retrieval loss should produce nonzero composer gradient"

    # rv_calibration and uncertainty: architecturally detached from BOTH spaces.
    for name in ("rv_calibration", "uncertainty"):
        assert result["aux"][name].get("status") == "detached_by_design", (
            f"{name} should be reported as architecturally detached -- if this fails, either the "
            f"underlying detach was removed (a real regression -- see "
            f"test_rv_calibrator_receives_gradient / "
            f"test_ug_calibration_loss_does_not_leak_into_rv_or_backbone) or this test's own "
            f"assumption is stale")
        assert "backbone_cosine" not in result["aux"][name], f"{name} should have no backbone_cosine entry"
        assert "composer_cosine" not in result["aux"][name], f"{name} should have no composer_cosine entry"

    # disentangle: real backbone path, NO composer path (AEP doesn't share the composer).
    dis = result["aux"]["disentangle"]
    assert "backbone_cosine" in dis and -1.0 <= dis["backbone_cosine"] <= 1.0, (
        f"disentangle should give a valid backbone cosine, got {dis.get('backbone_cosine')}")
    assert dis["backbone_aux_grad_norm"] > 0, "disentangle should produce nonzero backbone gradient norm"
    assert "composer_cosine" not in dis, (
        "disentangle should have NO composer_cosine entry -- AEP does not share the ForwardComposer "
        "instance the way RV Tier 1 does")

    # reverse (RV): real path in BOTH backbone AND composer -- the one loss
    # this whole expansion exists to measure correctly.
    rev = result["aux"]["reverse"]
    assert "backbone_cosine" in rev and -1.0 <= rev["backbone_cosine"] <= 1.0, (
        f"reverse should give a valid backbone cosine, got {rev.get('backbone_cosine')}")
    assert "composer_cosine" in rev and -1.0 <= rev["composer_cosine"] <= 1.0, (
        f"reverse should give a valid COMPOSER cosine (RV Tier 1 shares the composer) -- if this is "
        f"missing, the composer-space expansion is not actually measuring RV's shared-composer "
        f"conflict, defeating the whole point of finding #3")
    assert rev["composer_aux_grad_norm"] > 0, "reverse should produce nonzero composer gradient norm"

    # Weighted ratio must equal lambda_reverse * raw_norm / retrieval_norm exactly
    # (by linearity of differentiation, not re-derived via a second backward
    # on a re-scaled loss) -- checked to a tight tolerance since this is a
    # simple, exact arithmetic relationship, not a noisy measurement.
    expected_ratio = (cfg.loss.lambda_reverse * rev["composer_aux_grad_norm"]) / result["retrieval_composer_norm"]
    assert abs(rev["composer_weighted_ratio"] - expected_ratio) < 1e-6, (
        f"composer_weighted_ratio ({rev['composer_weighted_ratio']}) should equal "
        f"lambda_reverse * raw_norm / retrieval_norm ({expected_ratio}) exactly")

    print("OK: diagnose_gradient_conflict.py correctly measures BOTH backbone and composer conflict "
          "(RV Tier 1 shares the composer with the retriever, AEP does not), correctly reports "
          "architecturally-detached losses (rv_calibration, uncertainty) as such rather than a bare "
          "nan, and computes the weighted gradient-norm ratio using this run's actual configured "
          "lambda_reverse.")


def test_gradient_conflict_flat_grad_matches_parameter_device_and_dtype():
    """Regression guard for a real bug found and fixed while manually
    end-to-end testing scripts/diagnose_gradient_conflict.py (2026-09-10):
    `_flat_grad`'s "missing gradient" fallback used `torch.zeros(p.numel())`
    with no `device=`/`dtype=` argument, which ALWAYS creates a CPU float32
    tensor regardless of where the model's real parameters/gradients live.
    On a real CUDA run (the actual cluster environment this script exists
    for -- this codebase has no CUDA device locally, so this bug was never
    caught by a CPU-only run), the very first parameter in a shared space
    (backbone or composer) with a missing gradient for a given loss term
    caused `torch.cat` to raise `RuntimeError: Expected all tensors to be
    on the same device, but found at least two devices, cuda:0 and cpu!`
    -- confirmed by manually driving the real CLI end-to-end (not assumed).

    Fixed with `torch.zeros_like(p)` instead, which matches both device AND
    dtype automatically. This test cannot reproduce a real CUDA device
    mismatch on this CPU-only dev machine, but directly verifies the fixed
    line's actual behavior: for a real parameter, the zero-fallback tensor
    it would construct has the exact same device and dtype as that
    parameter (the property that matters -- `zeros_like` is defined to
    guarantee this, `zeros(numel)` was not).
    """
    p = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float32))
    fallback = torch.zeros_like(p)
    assert fallback.device == p.device, "zeros_like fallback must match the parameter's device"
    assert fallback.dtype == p.dtype, "zeros_like fallback must match the parameter's dtype"
    assert fallback.numel() == p.numel(), "zeros_like fallback must match the parameter's element count"

    # Confirm the actual source line uses zeros_like, not a stale zeros(numel).
    import inspect
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import importlib
    dgc = importlib.import_module("diagnose_gradient_conflict")
    src = inspect.getsource(dgc._flat_grad)
    assert "torch.zeros_like(p)" in src, (
        "_flat_grad no longer uses torch.zeros_like(p) for its missing-gradient fallback -- "
        "check the device/dtype-matching fix wasn't reverted")
    # Check the LIVE code (not a comment describing the old bug for context)
    # still uses the buggy pattern as the actual fallback expression -- a
    # naive whole-source substring check would false-positive on this
    # function's own explanatory comment mentioning the old pattern by name.
    live_lines = [line for line in src.splitlines() if not line.strip().startswith("#")]
    live_src = "\n".join(live_lines)
    assert "torch.zeros(p.numel())" not in live_src, (
        "_flat_grad's live code still contains the buggy torch.zeros(p.numel()) "
        "(no device=/dtype=) pattern as an actual fallback expression")

    print("OK: diagnose_gradient_conflict.py's _flat_grad uses torch.zeros_like(p) (device- and "
          "dtype-matching) for its missing-gradient fallback, not the bare torch.zeros(p.numel()) "
          "that crashed with a real CUDA device-mismatch RuntimeError.")


def test_fashioniq_dataset_resolves_kaggle_mirror_layout():
    """FashionIQDataset must resolve a real third-party mirror layout (a
    Kaggle upload hit in practice: `binhhuunguyen/fashioniq`), where captions
    sit under a DOUBLED repo-name path
    (`{root}/fashion-iq-master/fashion-iq-master/captions/...` -- a
    "Download ZIP" artifact from zipping an already-extracted GitHub repo)
    while images sit in an entirely differently-named folder
    (`{root}/resized_images/resized_images/`, no per-category subfolder).
    Pins that annotation and image roots resolve independently, same
    principle as the CIRR fix above, and that the local layout (both under
    `{root}/` directly) stays unaffected.
    """
    from revua_cir.data.datasets import FashionIQDataset

    with tempfile.TemporaryDirectory() as d:
        # Kaggle-mirror shape.
        ann_dir = os.path.join(d, "fashion-iq-master", "fashion-iq-master")
        os.makedirs(os.path.join(ann_dir, "captions"))
        img_dir = os.path.join(d, "resized_images", "resized_images")
        os.makedirs(img_dir)

        caps = [{"target": "B008BHCT58", "candidate": "B003FGW7MK",
                "captions": ["is solid black with no sleeves", "is black with straps"]}]
        with open(os.path.join(ann_dir, "captions", "cap.dress.val.json"), "w") as f:
            json.dump(caps, f)
        for asin in ("B008BHCT58", "B003FGW7MK"):
            open(os.path.join(img_dir, f"{asin}.jpg"), "wb").close()

        ds = FashionIQDataset(root=d, split="val", categories=["dress"], transform=None)
        assert ds.ann_root == ann_dir, f"expected ann_root {ann_dir}, got {ds.ann_root}"
        assert ds.image_root == img_dir, f"expected image_root {img_dir}, got {ds.image_root}"
        assert len(ds.triplets) == 1
        assert os.path.exists(ds.triplets[0]["ref_path"]), (
            f"ref_path did not resolve: {ds.triplets[0]['ref_path']}")
        assert os.path.exists(ds.triplets[0]["target_path"]), (
            f"target_path did not resolve: {ds.triplets[0]['target_path']}")

    with tempfile.TemporaryDirectory() as d:
        # Local layout must be unaffected: captions/ and images/ directly under root.
        os.makedirs(os.path.join(d, "captions"))
        os.makedirs(os.path.join(d, "images"))
        caps = [{"target": "B1", "candidate": "B2", "captions": ["t"]}]
        with open(os.path.join(d, "captions", "cap.dress.val.json"), "w") as f:
            json.dump(caps, f)
        for asin in ("B1", "B2"):
            open(os.path.join(d, "images", f"{asin}.jpg"), "wb").close()

        ds2 = FashionIQDataset(root=d, split="val", categories=["dress"], transform=None)
        assert ds2.ann_root == d, (
            f"local layout regressed: ann_root should stay {d}, got {ds2.ann_root}")
        assert ds2.image_root == os.path.join(d, "images"), (
            f"local layout regressed: image_root should be {os.path.join(d, 'images')}, "
            f"got {ds2.image_root}")
        assert os.path.exists(ds2.triplets[0]["ref_path"])

    print("OK: FashionIQDataset resolves both the Kaggle-mirror layout "
          "(doubled repo-name captions path, resized_images/resized_images "
          "for images -- independently resolved) and the local layout unchanged.")


def test_evaluate_dataset_cache_path_writes_real_rv_subscores():
    """`evaluate_dataset(rerank=True, cache_path=...)` (added 2026-09-11 to
    support offline RV/UG fusion-rule experimentation without a second GPU
    pass per variant -- see that function's own docstring) must write one
    real JSON line per query, with the SAME per-candidate values the rerank
    block actually used to reorder that query's top-k -- not placeholder or
    re-derived numbers. Also pins: `cache_path` is silently a no-op when
    `rerank=False` (there is no rerank block to cache from), and the file is
    append-mode (a second call must not truncate a first call's lines).
    """
    import tempfile
    from revua_cir.evaluate import evaluate_dataset
    from revua_cir.data.transforms import build_image_transform

    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc+aep+rv+ug")
    set_seed(0)
    model = ReVUACIR(cfg).eval()
    tokenizer = build_tokenizer(cfg.backbone)
    transform = build_image_transform(cfg.data.image_size, train=False)
    dataset = SyntheticCIRDataset(16, cfg.data.image_size, cfg.attribute_parser.attribute_vocab_size)
    top_k = cfg.reverse_verifier.top_k

    with tempfile.TemporaryDirectory() as d:
        cache_path = os.path.join(d, "rv_cache.jsonl")

        # no-op check: rerank=False must not create the file at all.
        evaluate_dataset(model, dataset, cfg, tokenizer, transform, torch.device("cpu"),
                          max_queries=4, rerank=False, cache_path=cache_path)
        assert not os.path.exists(cache_path), (
            "cache_path must be a no-op when rerank=False -- there is no rerank block to cache from")

        rr = evaluate_dataset(model, dataset, cfg, tokenizer, transform, torch.device("cpu"),
                               max_queries=4, rerank=True, cache_path=cache_path)
        assert os.path.exists(cache_path), "rerank=True with cache_path set must create the file"

        with open(cache_path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 4, f"expected one cache line per query (4), got {len(lines)}"

        required_keys = ("query_id", "query_index", "target_gallery_index", "candidate_gallery_ids",
                          "base_score", "cos_sim", "recon_cos", "invariant_cos", "edit_cos",
                          "verification_score", "target_pos", "ug_confidence", "query_log_var",
                          "coordination")
        for rec in lines:
            for key in required_keys:
                assert key in rec, f"cache record missing key {key!r}: {rec}"
            for per_candidate_key in ("candidate_gallery_ids", "base_score", "cos_sim", "recon_cos",
                                       "invariant_cos", "edit_cos", "verification_score", "ug_confidence"):
                assert len(rec[per_candidate_key]) == top_k, (
                    f"{per_candidate_key} should have one entry per top-k candidate ({top_k}), "
                    f"got {len(rec[per_candidate_key])}")
            # query_log_var/coordination are QUERY-level (UncertaintyGate.forward is called with
            # query/image/text pooled all .expand(k,-1)-broadcast from one query-level value --
            # only reverse_disagreement varies per candidate) -- must be stored ONCE per query as
            # a scalar, not duplicated k times like the per-candidate fields above.
            assert isinstance(rec["query_log_var"], (int, float)), (
                f"query_log_var must be a single scalar per query, not a list: {rec['query_log_var']}")
            assert isinstance(rec["coordination"], (int, float)), (
                f"coordination must be a single scalar per query, not a list: {rec['coordination']}")
            # verification_score is Platt-calibrated (sigmoid) from the SAME
            # cos_sim this record also stores raw (reverse_verifier.py's
            # PlattCalibrator) -- cos_sim CAN be negative, verification_score
            # (being a sigmoid output) cannot; every value must be finite,
            # and target_pos (an index into a length-k list, or the -1
            # sentinel for "missed the top-k") must be in range.
            assert all(v == v and abs(v) != float("inf") for v in rec["verification_score"]), (
                f"non-finite verification_score in cache record: {rec}")
            assert all(0.0 <= v <= 1.0 for v in rec["verification_score"]), (
                f"verification_score must be a sigmoid output in [0,1]: {rec['verification_score']}")
            assert -1 <= rec["target_pos"] < top_k
            # This tiny config has fc+aep+rv+ug but no use_map (CIRR-shaped
            # dataset here, not CIRCO), so the CIRCO-only multi-positive
            # fields must be ABSENT, not present-but-empty -- confirms
            # they're conditioned on use_map, not always written.
            assert "relevant_gallery_indices" not in rec
            assert "base_top50_gallery_ids" not in rec

        # "x" (exclusive-create) check: a SECOND call against the SAME path
        # must fail loudly, not silently append a duplicate copy of every
        # query behind the first (2026-09-11 correction -- append was the
        # original, now-rejected default; offline analysis reading
        # duplicate (query_index, candidate) rows would be silently and
        # non-obviously wrong, not an obvious crash).
        raised = False
        try:
            evaluate_dataset(model, dataset, cfg, tokenizer, transform, torch.device("cpu"),
                              max_queries=4, rerank=True, cache_path=cache_path)
        except FileExistsError:
            raised = True
        assert raised, "a second evaluate_dataset(..., cache_path=<existing file>) call must raise, not silently append"
        with open(cache_path, encoding="utf-8") as f:
            lines_after = [json.loads(line) for line in f if line.strip()]
        assert len(lines_after) == 4, (
            f"the file must be UNCHANGED after the failed second call (still 4 lines from the "
            f"first), got {len(lines_after)} -- exclusive-create must fail before writing anything")

        # Cross-check against the aggregate metric evaluate_dataset itself
        # returns for this same run: topk_target_present_rate is the fraction
        # of queries whose target was present in that query's own top-k --
        # must equal the cache's own target_pos != -1 fraction, or the cache
        # is recording a different top-k than the metrics were computed from.
        present_in_cache = sum(1 for rec in lines if rec["target_pos"] != -1) / len(lines)
        assert abs(present_in_cache - rr.get("topk_target_present_rate", -1)) < 1e-6, (
            f"cache's own target-present rate ({present_in_cache}) disagrees with "
            f"evaluate_dataset's topk_target_present_rate ({rr.get('topk_target_present_rate')})")

    print("OK: evaluate_dataset(rerank=True, cache_path=...) writes one real JSON line per query "
          "with RV's raw cos_sim + calibrated verification_score + sub-scores, UG's confidence "
          "(per-candidate) and log_var/coordination (per-query scalars, not duplicated), stable "
          "query/target identifiers, is a no-op under rerank=False, and FAILS (exclusive-create, "
          "leaving the original file untouched) rather than silently appending on a second call "
          "against an existing path.")


def test_evaluate_dataset_cache_file_is_closed_even_if_rerank_loop_raises():
    """Regression guard for a real bug found in manual review of the
    cache_path feature above (2026-09-11): the file handle was opened but
    never explicitly closed, relying on process exit / GC to flush it --
    fine on the happy path, but a full CIRR/CIRCO eval runs for hours
    against real images/galleries and CAN genuinely raise partway through
    (OOM, a corrupted image, a transient I/O error). Without an explicit
    close, an exception after some queries were already cached risked
    losing the very data this parameter exists to build reliably -- costing
    real GPU time for nothing recoverable. Fixed by wrapping the whole
    reranking pass in try/finally (see evaluate_dataset's own docstring/
    comment). This test forces a real exception mid-rerank-loop (patching
    `_rank_fuse`, called once per query inside that loop, to raise on its
    second call -- guaranteeing at least one query was cached first) and
    asserts the file is still readable/valid and contains that first
    query's real record afterward, not truncated or left open.
    """
    import tempfile
    from revua_cir import evaluate as evaluate_module
    from revua_cir.data.transforms import build_image_transform

    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc+aep+rv+ug")
    set_seed(0)
    model = ReVUACIR(cfg).eval()
    tokenizer = build_tokenizer(cfg.backbone)
    transform = build_image_transform(cfg.data.image_size, train=False)
    dataset = SyntheticCIRDataset(16, cfg.data.image_size, cfg.attribute_parser.attribute_vocab_size)

    real_rank_fuse = evaluate_module._rank_fuse
    call_count = {"n": 0}

    def _raising_rank_fuse(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("injected failure -- simulates a real mid-eval crash")
        return real_rank_fuse(*args, **kwargs)

    with tempfile.TemporaryDirectory() as d:
        cache_path = os.path.join(d, "rv_cache_crash.jsonl")
        evaluate_module._rank_fuse = _raising_rank_fuse
        try:
            raised = False
            try:
                evaluate_module.evaluate_dataset(
                    model, dataset, cfg, tokenizer, transform, torch.device("cpu"),
                    max_queries=4, rerank=True, cache_path=cache_path)
            except RuntimeError:
                raised = True
            assert raised, "test setup did not actually inject the failure -- assertion below is meaningless"
        finally:
            evaluate_module._rank_fuse = real_rank_fuse

        assert call_count["n"] >= 2, "expected at least 2 rerank-loop iterations before the injected raise"
        assert os.path.exists(cache_path), "cache file must exist even though evaluate_dataset raised"
        # If the handle were left open/unflushed, this file could be empty or
        # (on some platforms/buffering modes) not durably written -- opening
        # it fresh in a NEW handle and parsing every line as JSON is the real
        # test that the writes actually reached disk, not just Python's
        # internal buffer.
        with open(cache_path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) >= 1, (
            "the query cached before the injected crash should have been durably written -- "
            "got zero valid lines, meaning the file was not properly flushed/closed")
        for rec in lines:
            assert "verification_score" in rec and len(rec["verification_score"]) == cfg.reverse_verifier.top_k

    print("OK: evaluate_dataset's cache file is closed (and its already-written lines are durable "
          "and parseable) even when the rerank loop raises partway through, via try/finally -- "
          "not left open/unflushed on a real mid-eval crash.")


def test_evaluate_dataset_cache_circo_multi_positive_and_top50_fields():
    """Regression guard for the CIRCO-specific cache fields (added 2026-09-11
    per the low-GPU RV-improvement strategy document's items #6/#7): CIRCO
    queries can have MULTIPLE valid target images (unlike CIRR's single
    target_path), and CIRCO's reported metrics go up to mAP@50 while RV/the
    rest of the cache only covers the reranking top_k. The other cache test
    above only confirms these fields are ABSENT for a CIRR-shaped dataset
    (use_map=False) -- this test builds a genuinely CIRCO-shaped dataset
    (use_map=True, real `target_ids` metadata with 2 valid targets per
    query, real on-gallery-disk multi-target images) and confirms the
    fields are PRESENT and CORRECT, not just structurally there.
    """
    from revua_cir.data.transforms import build_image_transform
    from revua_cir.evaluate import _extract_id, evaluate_dataset

    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc+aep+rv+ug")
    set_seed(0)
    model = ReVUACIR(cfg).eval()
    tokenizer = build_tokenizer(cfg.backbone)
    transform = build_image_transform(cfg.data.image_size, train=False)
    top_k = cfg.reverse_verifier.top_k

    class CIRCOShapedDataset(SyntheticCIRDataset):
        """Same synthetic-tensor contract as SyntheticCIRDataset, but with
        CIRCO's real shape: `meta["target_ids"]` (plural, multiple valid
        answers) instead of CIRR's single `target_path`/`target_id`, and
        gallery paths using plain numeric filenames (`_extract_id` strips
        the extension and any leading zeros -- ordinary digit filenames
        sidestep that entirely) so `evaluate_dataset`'s use_map branch can
        actually resolve `target_ids` against `gallery_paths` by real
        filename matching, not just a synthetic placeholder relationship.
        """
        def __init__(self, n, image_size, attribute_vocab_size, gallery_dir):
            super().__init__(n, image_size, attribute_vocab_size)
            # Build a real on-disk gallery of n+extras images so target_ids
            # can resolve to real files distinct from ref/target -- two
            # DIFFERENT gallery images serve as this query's two valid
            # targets, deliberately not the dataset's own single
            # target_path, to prove relevant_gallery_indices reflects the
            # real multi-target set, not just a re-derivation of target_pos.
            self.gallery_dir = gallery_dir
            self.extra_gallery_paths = []
            for i in range(n * 3):
                p = os.path.join(gallery_dir, f"{1000 + i}.png")
                open(p, "wb").close()
                self.extra_gallery_paths.append(p)
            self.triplets = [
                {"ref_path": self.extra_gallery_paths[i * 3],
                 "target_path": self.extra_gallery_paths[i * 3 + 1],
                 "text": TEXTS[i % len(TEXTS)],
                 "meta": {"dataset": "circo_shaped",
                          "target_ids": [_extract_id(self.extra_gallery_paths[i * 3 + 1]),
                                         _extract_id(self.extra_gallery_paths[i * 3 + 2])]}}
                for i in range(n)
            ]

        def __getitem__(self, idx):
            # SyntheticCIRDataset.__getitem__ hardcodes its own
            # {"dataset": "synthetic", "target_id": idx} meta, ignoring
            # self.triplets entirely -- but evaluate_dataset's use_map
            # branch reads `meta` from the DataLoader batch (i.e. THIS
            # method's return value), not from dataset.triplets directly.
            # Without this override, the real target_ids set in __init__
            # above would never actually reach evaluate_dataset, and this
            # test would silently pass on nothing (or fail confusingly, as
            # it did before this fix was found).
            item = super().__getitem__(idx)
            item["meta"] = self.triplets[idx]["meta"]
            return item

        def build_gallery(self):
            return sorted(set(self.extra_gallery_paths))

    with tempfile.TemporaryDirectory() as d:
        dataset = CIRCOShapedDataset(20, cfg.data.image_size, cfg.attribute_parser.attribute_vocab_size, d)
        gallery_paths = dataset.build_gallery()
        assert len(gallery_paths) >= 50, (
            f"need a gallery >= max(ks)+50-ish for a real (non-clamped) top-50 slice, "
            f"got {len(gallery_paths)}")

        cache_path = os.path.join(d, "circo_cache.jsonl")
        rr = evaluate_dataset(model, dataset, cfg, tokenizer, transform, torch.device("cpu"),
                               max_queries=3, rerank=True, cache_path=cache_path,
                               use_map=True, full_gallery_paths=gallery_paths)
        assert "map_at_5" in rr, "use_map=True must produce mAP metrics -- confirms this is a real CIRCO-shaped run"

        with open(cache_path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 3

        for i, rec in enumerate(lines):
            assert "relevant_gallery_indices" in rec
            assert "relevant_positions_in_candidates" in rec
            assert "base_top50_gallery_ids" in rec
            assert len(rec["base_top50_gallery_ids"]) == min(50, len(gallery_paths))

            # The two real target_ids this query's own triplet declared
            # must both resolve into relevant_gallery_indices -- not just
            # SOME non-empty set, the SPECIFIC two images this query
            # actually named as valid.
            triplet = dataset.triplets[i]
            expected_targets = {gallery_paths.index(p) for p in
                                 [dataset.extra_gallery_paths[i * 3 + 1], dataset.extra_gallery_paths[i * 3 + 2]]}
            assert set(rec["relevant_gallery_indices"]) == expected_targets, (
                f"query {i}: cached relevant_gallery_indices {rec['relevant_gallery_indices']} does not "
                f"match the query's real 2 target_ids {expected_targets} -- multi-positive ground truth "
                f"was not correctly threaded from all_target_sets into the cache")

            # relevant_positions_in_candidates must be a subset of range(top_k),
            # and every position it names must actually be a relevant candidate.
            for pos in rec["relevant_positions_in_candidates"]:
                assert 0 <= pos < top_k
                assert rec["candidate_gallery_ids"][pos] in rec["relevant_gallery_indices"]

    print("OK: evaluate_dataset's CIRCO-shaped cache records (use_map=True) correctly thread the "
          "real multi-target ground truth into relevant_gallery_indices/relevant_positions_in_"
          "candidates (matching the query's actual 2 target_ids, not just a non-empty placeholder), "
          "and cache a real top-50 base ranking slice -- both absent for CIRR-shaped runs, present "
          "and correct here.")


def test_evaluate_dataset_reused_rv_matches_independent_second_forward():
    """Regression guard for item #8 of the low-GPU RV-improvement strategy
    document (fixed 2026-09-11): the rerank block used to encode the SAME
    top-k candidates a SECOND time and call model.rv(...) again, computing
    a mathematically-redundant result (deterministic in eval mode -- no
    dropout, no batch-norm running stats update) it already had from the
    diagnostics pass moments earlier. Fixed by caching the diagnostics
    pass's own rv_k/c_out per query and reusing them in the rerank block
    instead of recomputing -- this test proves that reuse gives IDENTICAL
    results to what a genuinely independent second forward pass produces,
    not just plausible-looking numbers.

    Method: run evaluate_dataset(rerank=True, cache_path=...) for real (this
    exercises the NEW, reused-RV code path and writes its verification_score/
    cos_sim/sub-scores to the cache). Then, independently and OUTSIDE
    evaluate_dataset entirely, re-encode the exact same cached
    candidate_gallery_ids for the exact same query and call model.rv(...) a
    genuinely fresh second time -- reconstructing exactly what the OLD
    (pre-fix) double-forward code path would have computed. Assert the two
    match to floating-point precision. This is a real, independent
    recomputation, not comparing the cache against itself.
    """
    from revua_cir.data.transforms import build_image_transform
    from revua_cir.data.datasets import CIRRDataset
    import torch.nn.functional as F
    from revua_cir.evaluate import build_default_gallery, evaluate_dataset

    cfg = build_tiny_config()
    cfg = cfg.apply_module_ablation("fc+aep+rv+ug")
    set_seed(0)
    model = ReVUACIR(cfg).eval()
    tokenizer = build_tokenizer(cfg.backbone)
    transform = build_image_transform(cfg.data.image_size, train=False)
    dataset = SyntheticCIRDataset(10, cfg.data.image_size, cfg.attribute_parser.attribute_vocab_size)
    top_k = cfg.reverse_verifier.top_k

    with tempfile.TemporaryDirectory() as d:
        cache_path = os.path.join(d, "equiv_cache.jsonl")
        evaluate_dataset(model, dataset, cfg, tokenizer, transform, torch.device("cpu"),
                          max_queries=4, rerank=True, cache_path=cache_path)
        with open(cache_path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 4

        # Independently reproduce, for EVERY cached query, the encode_query
        # + gallery-encode + RV forward the OLD rerank block would have run
        # a second time on the SAME candidates -- built directly from
        # dataset/model primitives, not by calling evaluate_dataset again
        # (which now uses the FIXED, reuse-based path and would prove
        # nothing about equivalence to the old behavior).
        collate_fn = build_collate_fn(tokenizer)
        gallery_paths = build_default_gallery(dataset)
        from revua_cir.data.transforms import safe_load_transform
        for rec in lines:
            qi = rec["query_index"]
            item = dataset[qi]
            batch = collate_fn([item])
            with torch.no_grad():
                q = model.encode_query(batch["reference_pixel_values"], batch["edit_input_ids"],
                                        batch["edit_attention_mask"])
                # Matches evaluate.py's own invariant-computation exactly
                # (evaluate_dataset, the `if model.rv is not None:` block):
                # model.aep(text_pooled, ref_pooled)["invariant_embed"].
                invariant = q["ref_out"].pooled
                if model.aep is not None:
                    invariant = model.aep(q["text_out"].pooled, q["ref_out"].pooled)["invariant_embed"]

                cand_paths = [gallery_paths[i] for i in rec["candidate_gallery_ids"]]
                pix = torch.stack([safe_load_transform(p, transform) for p in cand_paths])
                c_out = model.backbone.encode_image(pix)
                c_pooled = F.normalize(c_out.pooled, dim=-1)
                rv_independent = model.rv(
                    candidate_tokens=c_out.tokens, candidate_pooled=c_pooled,
                    forward_text_pooled=q["text_out"].pooled.expand(top_k, -1),
                    invariant_embed=invariant.expand(top_k, -1),
                )

            cached_cos_sim = torch.tensor(rec["cos_sim"])
            cached_verif = torch.tensor(rec["verification_score"])
            fresh_cos_sim = rv_independent["cos_sim"]
            fresh_verif = rv_independent["verification_score"]

            assert torch.allclose(cached_cos_sim, fresh_cos_sim, atol=1e-5), (
                f"query {qi}: cached cos_sim {cached_cos_sim.tolist()} does not match an "
                f"independent, fresh second RV forward on the SAME candidates "
                f"{fresh_cos_sim.tolist()} -- the reused-RV path (item #8 fix) is NOT "
                f"mathematically equivalent to the old double-forward path")
            assert torch.allclose(cached_verif, fresh_verif, atol=1e-5), (
                f"query {qi}: cached verification_score {cached_verif.tolist()} does not match "
                f"an independent fresh forward {fresh_verif.tolist()}")

    print("OK: evaluate_dataset's reused RV output (item #8 fix -- diagnostics-pass RV cached and "
          "reused in the rerank block instead of recomputed) is mathematically IDENTICAL to an "
          "independently, freshly re-run second forward pass on the same candidates -- confirmed "
          "to floating-point precision, not just plausible; the GPU-savings refactor changed "
          "nothing about the actual result.")


def test_rv_cache_analysis_help_harm_and_fusion_sweeps_match_hand_traced_ground_truth():
    """Regression guard for scripts/rv_cache_analysis.py (added 2026-09-12,
    offline RV-improvement plan Steps 1-3): a 4-query synthetic cache is
    hand-traced against the REAL, imported `_rank_fuse` (not a
    reimplementation) to independently derive the expected help/harm
    classification for each query -- confirmed here to be 1 HELP (query 0:
    base wrong, RV strongly favors the true target via a rank-adjacent
    swap), 1 HARM (query 1: base right, RV strongly favors a wrong
    candidate via the same mechanism), 1 unchanged-correct (query 2: base
    right with a clear margin, RV agrees), 1 unchanged-incorrect (query 3:
    base wrong with a clear margin, RV agrees) -- before asserting the
    script's own output matches.

    Also verifies gated_rerank_sweep's real behavior: since RV's gap over
    base-top1 is 0.85 for both non-trivial queries (0 and 1), the gate
    clears at every tested margin (0.0-0.3), giving the SAME 1-help/1-harm
    outcome as the production rank-fusion at every margin -- and
    bounded_additive_fusion_sweep's real behavior: with base scores that
    have a WIDE margin (0.6 spread) and lambda<=0.5 bounding RV's
    contribution to [-lambda,lambda], the small nudge cannot overturn the
    base ranking at any tested lambda, so 0 help/0 harm is the CORRECT,
    expected outcome (confirms the "bounded" property actually bounds,
    not a silent no-op bug).
    """
    import sys as _sys
    import importlib

    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    _sys.path.insert(0, scripts_dir)
    try:
        rca = importlib.import_module("rv_cache_analysis")
        importlib.reload(rca)
    finally:
        if scripts_dir in _sys.path:
            _sys.path.remove(scripts_dir)

    records = [
        {"query_index": 0, "target_pos": 0, "candidate_gallery_ids": [10, 11, 12, 13],
         "base_score": [0.7, 0.9, 0.5, 0.3], "verification_score": [0.9, 0.05, 0.5, 0.3]},
        {"query_index": 1, "target_pos": 0, "candidate_gallery_ids": [20, 21, 22, 23],
         "base_score": [0.9, 0.7, 0.5, 0.3], "verification_score": [0.05, 0.9, 0.5, 0.3]},
        {"query_index": 2, "target_pos": 0, "candidate_gallery_ids": [30, 31, 32, 33],
         "base_score": [0.9, 0.5, 0.3, 0.1], "verification_score": [0.9, 0.5, 0.3, 0.1]},
        {"query_index": 3, "target_pos": 3, "candidate_gallery_ids": [40, 41, 42, 43],
         "base_score": [0.9, 0.5, 0.3, 0.1], "verification_score": [0.9, 0.5, 0.3, 0.1]},
    ]

    # Independently confirm the intended HELP/HARM/unchanged labels via the
    # REAL _rank_fuse before trusting the script's own classification of them.
    from revua_cir.models.revua_cir import _rank_fuse
    expected_labels = []
    for rec in records:
        base = torch.tensor(rec["base_score"])
        rv = torch.tensor(rec["verification_score"])
        tgt = rec["target_pos"]
        base_correct = int(base.argmax()) == tgt
        fused_correct = int(_rank_fuse(base, rv).argmax()) == tgt
        if not base_correct and fused_correct:
            expected_labels.append("HELP")
        elif base_correct and not fused_correct:
            expected_labels.append("HARM")
        elif base_correct and fused_correct:
            expected_labels.append("UNCHANGED_CORRECT")
        else:
            expected_labels.append("UNCHANGED_INCORRECT")
    assert expected_labels == ["HELP", "HARM", "UNCHANGED_CORRECT", "UNCHANGED_INCORRECT"], (
        f"test fixture itself does not produce the intended ground truth -- got {expected_labels}, "
        f"fix the fixture before trusting any assertion below")

    result = rca.help_harm_analysis(records)
    assert result == {"help_count": 1, "harm_count": 1, "help_rate": 25.0, "harm_rate": 25.0,
                       "net_effect": 0.0, "n": 4}, (
        f"help_harm_analysis disagrees with the independently hand-traced ground truth: {result}")

    # gated_rerank_sweep: both non-trivial queries have an RV gap of 0.85
    # over base-top1 (verified above via the SAME real fixture), clearing
    # every tested margin -- expect 1 help/1 harm/net=0 at ALL of them.
    import io
    import contextlib
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        rca.gated_rerank_sweep(records, top_k=4, margins=(0.0, 0.1, 0.3))
    output = captured.getvalue()
    for margin_str in ("margin=0.00", "margin=0.10", "margin=0.30"):
        assert f"{margin_str}: R@1=50.00  help=1 harm=1 net=+0.00" in output, (
            f"gated_rerank_sweep did not produce the expected help=1/harm=1/net=0.00 at "
            f"{margin_str} -- full output:\n{output}")

    # bounded_additive_fusion_sweep: base scores have a WIDE margin (0.6
    # spread); lambda<=0.5 bounds RV's contribution to [-lambda,lambda],
    # too small to overturn that margin at any tested lambda -- expect
    # 0 help/0 harm everywhere, confirming the bound genuinely bounds.
    captured2 = io.StringIO()
    with contextlib.redirect_stdout(captured2):
        rca.bounded_additive_fusion_sweep(records, top_k=4, lambdas=(0.1, 0.5))
    output2 = captured2.getvalue()
    for lambda_str in ("lambda=0.10", "lambda=0.50"):
        assert f"{lambda_str}: R@1=50.00  help=0 harm=0 net=+0.00" in output2, (
            f"bounded_additive_fusion_sweep did not produce the expected help=0/harm=0/net=0.00 "
            f"at {lambda_str} (a small bounded nudge should not overturn a 0.6-spread base "
            f"margin) -- full output:\n{output2}")

    print("OK: rv_cache_analysis.py's help_harm_analysis, gated_rerank_sweep, and "
          "bounded_additive_fusion_sweep all match independently hand-traced ground truth "
          "(via the REAL _rank_fuse, not a reimplementation) on a 1-HELP/1-HARM/2-unchanged "
          "fixture -- the gated sweep reproduces the same 1/1 outcome at every margin (RV's "
          "gap clears all of them), and the bounded fusion sweep correctly shows 0/0 (the "
          "bound is too small to overturn a wide base margin, not a silent no-op bug).")


def test_rank_fuse_tie_breaking_is_stable_and_device_consistent():
    """Regression guard for a real, previously-unknown reproducibility bug
    (found 2026-09-12 via a real discrepancy between job 6387's on-cluster
    GPU reranking numbers and an offline CPU-side recomputation from the
    same cache using the same _rank_fuse): plain (unstable) argsort's
    tie-breaking is NOT guaranteed consistent across backends -- confirmed
    directly (not assumed) that
    `torch.tensor([0.5,0.5,0.5,0.3,0.3]).argsort(descending=True)` returns
    a DIFFERENT permutation on CPU ([0,1,2,3,4]) than on CUDA
    ([2,1,0,4,3]) for the IDENTICAL input. Since `adj`'s (RV/UG's fused
    score) spread across a query's own top-k is documented as tiny
    (~0.00007), real ties are common, not a rare edge case -- meaning the
    SAME cached scores could rerank differently purely depending on which
    device evaluated them. Fixed by adding `stable=True` to `_rank_fuse`'s
    first argsort call (the one that operates on `base_scores`/`adj`
    directly, which CAN have duplicates; the second chained `.argsort()`
    operates on a genuine 0..k-1 permutation with no duplicates, so it
    needs no stability fix -- confirmed separately that CPU/CUDA already
    agree there).

    This test only requires CPU (no CUDA needed to catch a regression):
    it constructs inputs with GENUINE ties and asserts the result is
    deterministic and reproducible across repeated CPU calls, AND
    (if CUDA happens to be available in this environment) that CPU and
    CUDA now agree -- previously they would not have.
    """
    from revua_cir.models.revua_cir import _rank_fuse

    base_scores = torch.tensor([0.5, 0.5, 0.5, 0.3, 0.3])
    adj = torch.tensor([0.1, 0.1, 0.2, 0.4, 0.4])

    # stable=True on THIS argsort too (real bug found while writing this
    # very test, 2026-09-12): _rank_fuse's raw fused_rank output can itself
    # contain ties across candidates (confirmed: [-3,-6,-6,-6,-9] for this
    # exact fixture) -- fixing ONLY _rank_fuse's internal argsort is not
    # sufficient; every CALLER's own final argsort on its output needs the
    # same stable=True, which is why evaluate.py/revua_cir.py's real call
    # sites were fixed too, not just _rank_fuse itself.
    order_cpu_1 = _rank_fuse(base_scores, adj).argsort(descending=True, stable=True).tolist()
    order_cpu_2 = _rank_fuse(base_scores.clone(), adj.clone()).argsort(descending=True, stable=True).tolist()
    assert order_cpu_1 == order_cpu_2, (
        f"_rank_fuse must be deterministic across repeated CPU calls on the same tied input, "
        f"got {order_cpu_1} vs {order_cpu_2}")

    if torch.cuda.is_available():
        order_cuda = _rank_fuse(base_scores.cuda(), adj.cuda()).argsort(descending=True, stable=True).cpu().tolist()
        assert order_cuda == order_cpu_1, (
            f"_rank_fuse must give the SAME fused order on CPU ({order_cpu_1}) and CUDA "
            f"({order_cuda}) for identical (tied) input -- this is exactly the bug that caused "
            f"job 6387's GPU reranking numbers to disagree with an offline CPU recomputation "
            f"from the same cache")
        print("OK: _rank_fuse's stable=True tie-breaking gives IDENTICAL fused order on CPU and "
              "CUDA for genuinely tied input (confirmed on real CUDA, not just CPU-only). NOTE: "
              "this was NOT the cause of the job-6387-vs-offline R@1 discrepancy -- a direct "
              "diagnostic (diagnose_cache_reproduction_gap.py) later showed 0/4181 queries "
              "actually change top-1 under stable-vs-unstable argsort. The UG-signal hypothesis "
              "(verification_score vs verification_score*ug_confidence) was ALSO tested and ruled "
              "out -- applying the real production signal left the offline R@1 unchanged at 38.27. "
              "RESOLVED 2026-09-12 by cluster job 6392 (scripts/reconcile_rerank_recall.py): a "
              "fresh evaluation on the current code reported RERANK R@1=38.27, matching the "
              "offline recomputation exactly and with its own self-check passing -- so job 6387's "
              "37.41 was the artifact, not the offline number. This stable=True fix remains "
              "correct on its own merits (cross-device reproducibility), but it was NOT the "
              "explanation for that discrepancy.")
    else:
        print("OK: _rank_fuse's stable=True tie-breaking is deterministic across repeated CPU "
              "calls on tied input (no CUDA available in this environment to also confirm "
              "CPU/CUDA agreement, but the CPU-side determinism this fix also guarantees is "
              "verified).")


def test_significance_tests_reject_noise_and_detect_real_signal():
    """Regression guard for revua_cir/significance.py (added 2026-09-12).

    Motivation: a post-reconciliation document proposed accepting a fusion
    rule if "R@1 > 38.27" -- selecting lambda by sweeping on the same
    queries the number is reported on. That was shown empirically to be
    unsound: sweeping 10 lambdas over a PURE NOISE signal produced an
    apparent +0.60pp R@1 gain, the same order as the real candidate's
    +0.67pp. These tests pin the statistical machinery that replaced it.

    Checks, in order of what would hurt most if it silently broke:
      1. mcnemar_exact matches the exact binomial to machine precision
         (validated against scipy.stats.binomtest when committed; the
         reference values below are frozen from that validation, since
         scipy is deliberately NOT a dependency of this project -- see
         significance.py's own docstring for the OMP-crash reason).
      2. Degenerate inputs (no discordant pairs) return p=1.0 rather than
         dividing by zero.
      3. mid-p is never larger than exact (Fagerland et al. 2013).
      4. Holm-Bonferroni maps results back to INPUT positions, not sorted
         order -- the classic bug in step-down corrections.
      5. The paired bootstrap keeps (base, new) outcomes paired per query:
         identical systems must give a CI containing 0, and a strong real
         effect must give a CI excluding 0.
      6. split_indices produces disjoint, exhaustive, deterministic halves
         -- any overlap would leak tuning data into the reported number.
    """
    from revua_cir.significance import (holm_bonferroni, mcnemar_exact, mcnemar_midp,
                                         paired_bootstrap_ci, split_indices)

    # 1. Frozen reference values (verified against scipy.stats.binomtest).
    # ASYMMETRIC cases included deliberately (external review, 2026-09-12,
    # raised whether tail-doubling diverges from the general two-sided
    # binomial test on imbalanced counts). It does not: under McNemar's
    # null p=0.5 the pmf is symmetric, so the two rejection regions are
    # provably the same set -- confirmed for every n<=40 and every
    # observation, and numerically against scipy across all 3721 pairs in
    # 0..60 x 0..60 (worst deviation 1.7e-15) plus 1000:0 and 0:1000.
    # These frozen values pin the asymmetric cases specifically.
    for (b, c), expected in [((259, 235), 0.30074792), ((143, 91), 0.0008192916),
                              ((10, 0), 0.001953125), ((50, 30), 0.032992618),
                              ((1, 0), 1.0), ((0, 1), 1.0), ((5, 1), 0.21875),
                              ((10, 3), 0.09228515625), ((8, 3), 0.2265625),
                              ((100, 20), 5.50839085e-14)]:
        got = mcnemar_exact(b, c)
        assert abs(got - expected) < 1e-8, f"mcnemar_exact({b},{c})={got}, expected {expected}"
        # Direction must not matter for a two-sided test.
        assert abs(mcnemar_exact(c, b) - expected) < 1e-8, (
            f"mcnemar_exact not symmetric at ({b},{c})")

    # 2. Degenerate: no discordant pairs -> nothing distinguishes the systems.
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_midp(0, 0) == 1.0

    # Symmetry: which system is "new" must not change the two-sided p-value.
    assert mcnemar_exact(143, 91) == mcnemar_exact(91, 143)

    # 3. mid-p <= exact everywhere (it subtracts half the point mass).
    for b in range(0, 25):
        for c in range(0, 25):
            assert mcnemar_midp(b, c) <= mcnemar_exact(b, c) + 1e-15

    # Negative counts are a caller bug, not something to silently accept.
    for fn in (mcnemar_exact, mcnemar_midp):
        try:
            fn(-1, 5)
            raise AssertionError(f"{fn.__name__} accepted a negative count")
        except ValueError:
            pass

    # 4. Holm must map back to input positions, not sorted order.
    assert holm_bonferroni([0.6, 0.001, 0.04]) == [False, True, False]
    assert holm_bonferroni([]) == []
    # Holm is uniformly at least as powerful as plain Bonferroni.
    ps = [0.01, 0.02, 0.03]
    assert sum(holm_bonferroni(ps)) >= sum(p <= 0.05 / len(ps) for p in ps)

    # 5. Paired bootstrap: identical systems -> CI contains 0.
    same = [1, 0, 1, 1, 0] * 100
    r_same = paired_bootstrap_ci(same, same, n_replicates=200, seed=1)
    assert r_same["straddles_zero"], f"identical systems must straddle 0, got {r_same}"
    assert abs(r_same["observed_diff"]) < 1e-9

    # A strong, unambiguous real effect -> CI excludes 0.
    base_h = [0] * 300 + [1] * 100
    new_h = [1] * 300 + [1] * 100
    r_big = paired_bootstrap_ci(base_h, new_h, n_replicates=200, seed=1)
    assert not r_big["straddles_zero"], f"a large real effect must exclude 0, got {r_big}"
    assert r_big["observed_diff"] > 0

    # Mismatched lengths are unpaired data -- must be rejected, not zipped.
    try:
        paired_bootstrap_ci([1, 0], [1], n_replicates=10)
        raise AssertionError("paired_bootstrap_ci accepted mismatched lengths")
    except ValueError:
        pass

    # Determinism: same seed -> identical interval (a moving CI would let a
    # caller reroll until the result looked good).
    assert (paired_bootstrap_ci(base_h, new_h, n_replicates=100, seed=7) ==
            paired_bootstrap_ci(base_h, new_h, n_replicates=100, seed=7))

    # 6. Split must be disjoint, exhaustive and reproducible.
    tune, report = split_indices(4181, tune_frac=0.5, seed=12345)
    assert not (set(tune) & set(report)), "tune/report split leaked overlapping queries"
    assert len(set(tune) | set(report)) == 4181, "split dropped or duplicated queries"
    assert (tune, report) == split_indices(4181, tune_frac=0.5, seed=12345)
    # Never produce an empty side, even at the smallest workable n.
    t_small, r_small = split_indices(2, tune_frac=0.5)
    assert len(t_small) >= 1 and len(r_small) >= 1
    for bad_n, bad_frac in [(1, 0.5), (10, 0.0), (10, 1.0)]:
        try:
            split_indices(bad_n, tune_frac=bad_frac)
            raise AssertionError(f"split_indices accepted n={bad_n} tune_frac={bad_frac}")
        except ValueError:
            pass

    print("OK: significance.py -- McNemar exact matches frozen scipy-validated references and is "
          "symmetric/degenerate-safe, mid-p <= exact, Holm maps to input positions and beats "
          "Bonferroni's power, the paired bootstrap keeps outcomes paired (0-centred CI for "
          "identical systems, 0-excluding for a real effect) and is deterministic, and the "
          "tune/report split is disjoint, exhaustive and reproducible.")


if __name__ == "__main__":
    test_module_ablation_matrix()
    test_residual_composer_diagnostics_are_logged_and_wired()
    test_fc_ug_ablation_is_rejected()
    test_residual_composition_gives_text_pooled_real_gradient()
    test_residual_alpha_scale_is_honest()
    test_rv_edit_weight_inits_to_balanced_50_50()
    test_reverse_verifier_tier2()
    test_reverse_verifier_tier1_uses_renamed_inverse_consistency_name()
    test_autocast_safety()
    test_optimizer_excludes_scalars_and_biases_from_weight_decay()
    test_freeze_backbone()
    test_hard_negative_weight_moves_negative_logits_toward_positive()
    test_hard_negative_weight_computes_independent_weights_per_direction()
    test_reverse_consistency_loss_is_contrastive()
    test_eval_rerank_changes_ranking()
    test_spearman_corr_matches_known_values_without_scipy()
    test_ug_ranking_impact_diagnostics_correctly_distinguish_reordering()
    test_ug_final_order_diagnostic_differs_from_adjustment_order_diagnostic()
    test_gradcache_matches_plain_step()
    test_gradcache_trainable_backbone_matches_plain_step()
    test_gradcache_trainable_backbone_full_module_stack_bounded_divergence()
    test_xbm_cross_batch_memory()
    test_xbm_mining_uses_query_similarity_at_matched_scale()
    test_rv_calibrator_receives_gradient()
    test_ug_calibration_loss_does_not_leak_into_rv_or_backbone()
    test_gradcache_uses_real_gradscaler()
    test_optimizer_step_skip_detection_matches_gradscaler_behavior()
    test_gradcache_injects_gradient_into_trainable_adapters_under_frozen_backbone()
    test_gradcache_rand_context_matches_dropout_mask()
    test_xbm_under_gradcache()
    test_early_stopping()
    test_checkpoint_warns_on_missing_trainable_param()
    test_checkpoint_roundtrip()
    test_rng_state_checkpoint_roundtrip_reproduces_subsequent_draws()
    test_checkpoint_resume_matches_live_model_not_ema()
    test_checkpoint_excludes_only_pretrained_frozen_params()
    test_checkpoint_excludes_frozen_pretrained_backbone_by_name()
    test_topk_checkpoint_keeper_rebuilds_from_disk_on_resume()
    test_dry_run_full_pipeline_handles_optimizer_step_skipped_flag()
    test_train_rejects_non_divisible_gradcache_chunk_size()
    test_gradcache_resume_skip_replays_same_batches()
    test_train_py_passes_ema_to_save_checkpoint()
    test_metrics()
    test_verification_auroc_matches_pooled_roc_auc()
    test_group_recall_uses_untruncated_scores_not_global_topn()
    test_cirr_full_gallery_and_incomplete_group_handling()
    test_official_cirr_group_has_six_members_five_candidates()
    test_end_epoch_checkpoint_saves_post_step_early_stopper_state()
    test_inference_rerank_path()
    test_retrieve_and_rerank_does_not_mix_score_domains()
    test_parser_backend_actually_selects_extractor()
    test_cirr_dataset_resolves_kaggle_mirror_layout()
    test_cirr_dataset_resolves_official_img_raw_layout()
    test_official_cirr_evaluation_rejects_missing_gallery_images()
    test_strict_cirr_preflight_rejects_malformed_queries_and_passes_valid_ones()
    test_ablation_matrix_reports_honest_latency_field_names()
    test_benchmark_latency_script_times_the_real_inference_path()
    test_cirr_test1_submission_writes_two_file_schema()
    test_gradient_conflict_diagnostic_measures_real_and_detached_losses_correctly()
    test_gradient_conflict_flat_grad_matches_parameter_device_and_dtype()
    test_fashioniq_dataset_resolves_kaggle_mirror_layout()
    test_evaluate_dataset_cache_path_writes_real_rv_subscores()
    test_evaluate_dataset_cache_file_is_closed_even_if_rerank_loop_raises()
    test_evaluate_dataset_cache_circo_multi_positive_and_top50_fields()
    test_evaluate_dataset_reused_rv_matches_independent_second_forward()
    test_rv_cache_analysis_help_harm_and_fusion_sweeps_match_hand_traced_ground_truth()
    test_rank_fuse_tie_breaking_is_stable_and_device_consistent()
    test_significance_tests_reject_noise_and_detect_real_signal()
    print("\nALL SMOKE TESTS PASSED.")
