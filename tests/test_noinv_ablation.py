"""Tests for the retrained no-invariant ablation (review item B6.4.3).

The change is small -- one config flag, one branch in the verifier score --
but it will be paid for with a full L40S training run, so each property the
experiment depends on is tested directly against the real modules:

  * Every existing combo, and the published fc+aep+rv run in particular,
    computes exactly the score it always did.
  * The new combo computes the SAME renormalised score that Section 5.5's
    post-hoc ablation computes, so the two results are comparable.
  * The invariant term truly carries no weight in the new combo.
  * It trains: plain and gradcache steps give finite gradients everywhere
    except the one head that is meant to be unused.
  * It starts from the same initialisation as fc+aep+rv, so the only
    difference between the two runs is the removed term.
  * The ablation harness does not start this paid run unless asked.

Runs on CPU in seconds with the tiny backbone from test_smoke.
"""
from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, _HERE)

from revua_cir.config import ReVUAConfig                    # noqa: E402
from revua_cir.models.revua_cir import ReVUACIR             # noqa: E402
from revua_cir.train import build_optimizer                 # noqa: E402
from revua_cir.utils import set_seed                        # noqa: E402
from test_smoke import SyntheticCIRDataset, build_tiny_config  # noqa: E402

LADDER = ("fc", "fc+aep", "fc+aep+rv", "fc+aep+rv+ug")
NOINV = "fc+aep+rv+noinv"
INV_HEAD = ("rv.candidate_decomp.invariant_head.weight",
            "rv.candidate_decomp.invariant_head.bias")


def _model(combo, seed=0):
    cfg = build_tiny_config().apply_module_ablation(combo)
    set_seed(seed)
    model = ReVUACIR(cfg)
    model.eval()
    return model, cfg


def _rv_inputs(model, cfg, bsz=6, seed=1):
    """Real candidate tokens from the tiny backbone, plus query-side vectors."""
    g = torch.Generator().manual_seed(seed)
    size = cfg.data.image_size
    img = torch.randn(bsz, 3, size, size, generator=g)
    with torch.no_grad():
        enc = model.backbone.encode_image(img)
    pooled = torch.nn.functional.normalize(enc.pooled, dim=-1)
    d = pooled.shape[-1]
    text = torch.nn.functional.normalize(torch.randn(bsz, d, generator=g), dim=-1)
    inv = torch.nn.functional.normalize(torch.randn(bsz, d, generator=g), dim=-1)
    return dict(candidate_tokens=enc.tokens, candidate_pooled=pooled,
                forward_text_pooled=text, invariant_embed=inv)


# ---------------------------------------------------------------- config --
def test_default_and_every_ladder_combo_keep_the_invariant_term():
    assert ReVUAConfig().reverse_verifier.use_invariant_term is True
    base = build_tiny_config()
    for combo in LADDER:
        cfg = base.apply_module_ablation(combo)
        assert cfg.reverse_verifier.use_invariant_term is True, combo


def test_noinv_combo_matches_fc_aep_rv_except_the_term():
    base = build_tiny_config()
    ref = base.apply_module_ablation("fc+aep+rv")
    new = base.apply_module_ablation(NOINV)
    assert new.reverse_verifier.use_invariant_term is False
    # Everything else identical to the reported fc+aep+rv configuration.
    ref_d, new_d = ref.to_dict(), new.to_dict()
    new_d["reverse_verifier"]["use_invariant_term"] = True
    assert ref_d == new_d, "noinv must differ from fc+aep+rv in one field only"
    assert new.attribute_parser.enabled and new.reverse_verifier.enabled
    assert not new.uncertainty_gate.enabled
    assert new.loss.lambda_uncertainty == 0.0


def test_apply_module_ablation_does_not_mutate_the_base_config():
    base = build_tiny_config()
    base.apply_module_ablation(NOINV)
    assert base.reverse_verifier.use_invariant_term is True


def test_harness_default_does_not_include_the_paid_branch():
    import run_ablation_matrix
    assert run_ablation_matrix.MODULE_COMBOS == list(LADDER)
    assert NOINV in ReVUAConfig.VALID_MODULE_ABLATIONS


# ----------------------------------------------------------------- score --
def test_full_score_is_unchanged_for_the_reported_combo():
    model, cfg = _model("fc+aep+rv")
    with torch.no_grad():
        out = model.rv(**_rv_inputs(model, cfg))
        w = torch.sigmoid(model.rv.edit_weight)
    expected = ((1.0 - w) * (0.5 * out["recon_cos"] + 0.5 * out["invariant_cos"])
                + w * out["edit_cos"])
    assert torch.allclose(out["cos_sim"], expected, atol=1e-7)


def test_noinv_score_equals_the_post_hoc_definition_in_section_5_5():
    from rv_component_ablation import component_score

    model, cfg = _model(NOINV)
    # Move w away from 0.5 so a wrong weighting cannot pass by coincidence.
    with torch.no_grad():
        model.rv.edit_weight.fill_(0.9)
        out = model.rv(**_rv_inputs(model, cfg))
        w = float(torch.sigmoid(model.rv.edit_weight))
    rec = {k: out[k].tolist() for k in ("recon_cos", "invariant_cos", "edit_cos")}
    post_hoc = component_score(rec, w, use_recon=True, use_invariant=False,
                               use_edit=True)
    assert torch.allclose(out["cos_sim"], torch.tensor(post_hoc,
                                                       dtype=out["cos_sim"].dtype),
                          atol=1e-6)


def test_noinv_score_does_not_depend_on_the_invariant_term():
    model, cfg = _model(NOINV)
    inputs = _rv_inputs(model, cfg)
    with torch.no_grad():
        before = model.rv(**inputs)
        model.rv.candidate_decomp.invariant_head.weight.add_(
            torch.randn_like(model.rv.candidate_decomp.invariant_head.weight))
        after = model.rv(**inputs)
    assert not torch.allclose(before["invariant_cos"], after["invariant_cos"]), \
        "perturbation must actually move invariant_cos for this test to mean anything"
    assert torch.allclose(before["cos_sim"], after["cos_sim"], atol=1e-7)


def test_the_same_perturbation_does_move_the_full_score():
    # Control for the test above: in the reported combo the term matters.
    model, cfg = _model("fc+aep+rv")
    inputs = _rv_inputs(model, cfg)
    with torch.no_grad():
        before = model.rv(**inputs)["cos_sim"]
        model.rv.candidate_decomp.invariant_head.weight.add_(
            torch.randn_like(model.rv.candidate_decomp.invariant_head.weight))
        after = model.rv(**inputs)["cos_sim"]
    assert not torch.allclose(before, after)


def test_noinv_score_is_finite_at_extreme_edit_weights():
    model, cfg = _model(NOINV)
    inputs = _rv_inputs(model, cfg)
    for logit in (-30.0, 0.0, 30.0):
        with torch.no_grad():
            model.rv.edit_weight.fill_(logit)
            out = model.rv(**inputs)
        assert torch.isfinite(out["cos_sim"]).all(), logit
        assert (out["cos_sim"].abs() <= 1.0 + 1e-6).all(), logit


# ------------------------------------------------------------- training --
def test_noinv_trains_with_only_the_invariant_head_gradless():
    # test_smoke.run_forward_backward requires EVERY parameter to receive a
    # gradient, which this combo deliberately violates for one head, so the
    # check is done here with that head excluded explicitly.
    model, cfg = _model(NOINV)
    model.train()
    from revua_cir.data.collate import build_collate_fn
    from revua_cir.data.tokenizer import build_tokenizer
    ds = SyntheticCIRDataset(8, cfg.data.image_size,
                             cfg.attribute_parser.attribute_vocab_size)
    batch = build_collate_fn(build_tokenizer(cfg.backbone))([ds[i] for i in range(8)])
    loss = model(batch)["losses"]["total"]
    assert torch.isfinite(loss)
    loss.backward()
    missing = {n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None}
    may_be_missing = {"rv.calibrator.scale", "rv.calibrator.bias"}
    assert set(INV_HEAD) <= missing, "invariant head should get no gradient"
    assert missing - set(INV_HEAD) <= may_be_missing, \
        "unexpected gradless params: %s" % sorted(missing - set(INV_HEAD))
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n
    opt = build_optimizer(model, cfg)
    opt.step()                                    # None grads must not crash it


def test_noinv_gradcache_step_with_trainable_backbone():
    from revua_cir.gradcache import gradcache_step

    cfg = build_tiny_config().apply_module_ablation(NOINV)
    cfg.optim.amp = False
    cfg.fusion.dropout = 0.0
    cfg.optim.gradcache_chunk_size = 2
    cfg.backbone.freeze_backbone = False
    set_seed(0)
    model = ReVUACIR(cfg)
    model.eval()

    bsz, chunk = 8, 2
    batch = {
        "reference_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "target_pixel_values": torch.randn(bsz, 3, cfg.data.image_size, cfg.data.image_size),
        "edit_input_ids": torch.randint(0, 100, (bsz, 8)),
        "edit_attention_mask": torch.ones(bsz, 8, dtype=torch.long),
        "attribute_targets": None,
        "meta": [{} for _ in range(bsz)],
    }
    chunks = [{k: (v[i:i + chunk] if torch.is_tensor(v) or isinstance(v, list) else v)
               for k, v in batch.items()} for i in range(0, bsz, chunk)]

    class _NoStep:
        def __init__(self, opt):
            self.opt = opt

        def zero_grad(self, **kw):
            self.opt.zero_grad(**kw)

        def step(self):
            pass

    losses = gradcache_step(model, chunks, _NoStep(build_optimizer(model, cfg)), cfg, None)
    assert torch.isfinite(losses["total"]), losses["total"]
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    for name in INV_HEAD:
        assert grads[name] is None, name
    assert grads["rv.edit_weight"] is not None, "edit weight must still learn"
    for n, g in grads.items():
        if g is not None:
            assert torch.isfinite(g).all(), n


# ------------------------------------------------------- initialisation --
def test_noinv_starts_from_the_same_initialisation_as_fc_aep_rv():
    """Point 1 of the run checklist: same starting weights.

    The invariant head is still constructed in the new combo precisely so
    that parameter creation consumes the RNG identically; dropping it would
    shift the initialisation of every module built after it.
    """
    ref, _ = _model("fc+aep+rv", seed=42)
    new, _ = _model(NOINV, seed=42)
    a, b = ref.state_dict(), new.state_dict()
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), "initialisation differs at %s" % k


if __name__ == "__main__":
    import inspect
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and inspect.isfunction(f)]
    for n, f in fns:
        f()
        print("ok  %s" % n)
    print("\n%d tests passed" % len(fns))
