"""
Central configuration for ReVUA-CIR.

Everything that Section 4 / Section 8 of the paper framework calls out as an
ablation switch (module on/off, verifier tier, loss weights, backbone choice,
K for candidate re-verification, ...) is a field here so that a single YAML
file (see configs/default.yaml) or a single Optuna trial can toggle it
without touching model code.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict

import yaml


# --------------------------------------------------------------------------- #
# Backbone / Forward Composer
# --------------------------------------------------------------------------- #
@dataclass
class BackboneConfig:
    # "clip" (openai/transformers CLIP), "open_clip" (LAION checkpoints incl.
    # DFN/SigLIP-style ones), or "tiny" (randomly-initialized fallback used
    # for offline unit tests / CI so the whole pipeline is exercised without
    # network access to a model hub).
    kind: str = "clip"
    pretrained_name: str = "openai/clip-vit-large-patch14"
    open_clip_name: str = "ViT-B-32"
    open_clip_pretrained: str = "laion2b_s34b_b79k"
    freeze_backbone: bool = False
    freeze_layers: int = 0            # freeze the first N transformer blocks
    embed_dim: int = 768              # projection dim used by every module downstream
    tiny_vocab_size: int = 30522
    tiny_image_size: int = 32
    tiny_patch_size: int = 8
    tiny_width: int = 128
    tiny_layers: int = 2
    tiny_heads: int = 4


@dataclass
class FusionConfig:
    """Forward Composer: Cross-Attentive Gated Fusion (CAGF).

    Latest-generation fusion recipe combining three ideas that each showed
    gains over plain concatenation/addition fusion in the 2023-2026 CIR
    literature reviewed in the paper framework:
      * bidirectional multi-head cross-attention between image patch/token
        embeddings and text token embeddings (query-aware visual grounding,
        as in task-oriented CLIP composition / CoLLM-style fusion),
      * a learned scalar-per-dimension convex gate that blends the
        cross-attended representation with the raw reference-image embedding
        (Combiner-style residual gating, generalized to a full vector gate),
      * a final residual MLP projection head with GELU + LayerNorm.
    """
    num_heads: int = 8
    num_layers: int = 2
    ffn_mult: int = 4
    dropout: float = 0.1
    gate_hidden_mult: int = 2
    use_learned_temperature: bool = True
    init_temperature: float = 0.07
    # Residual composition (fixed 2026-09-08, opt-in via this flag so the
    # prior CAGF-only path stays available for a controlled ablation): when
    # True, the final query is q = norm(q0 + alpha * CAGF_output), where q0
    # = norm(norm(image_pooled) + norm(text_pooled)) is CLIP's own pretrained
    # additive composition and alpha is a learned scalar initialized near 0
    # -- training starts from CLIP's already-aligned geometry and only
    # gradually learns a CAGF correction, instead of replacing it outright.
    # Motivated by a real bug found via code audit: the pre-fix CAGF path
    # computed `fused = fused + text_pooled * 0.0`, silently discarding
    # CLIP's pretrained text embedding entirely (zero forward contribution,
    # zero gradient) -- see fusion.py's ForwardComposer docstring for the
    # full history. use_residual_composition=False reproduces that CAGF-only
    # path exactly (with the dead `* 0.0` term removed, a no-op change) for
    # anyone deliberately comparing against the old behavior.
    use_residual_composition: bool = True
    # NOT 0.0 (confirmed via direct test 2026-09-08): at alpha=0 exactly,
    # d(query)/d(theta_CAGF) = alpha * d(CAGF_output)/d(theta_CAGF) = 0 --
    # the entire CAGF branch (cross-attention, gating, out_proj) receives
    # ZERO gradient at the first step and stays untrained until alpha itself
    # moves away from 0 via its own (nonzero) gradient. 0.05 keeps q0
    # dominant (preserving pretrained CLIP geometry, the whole point of this
    # fix) while giving CAGF's own parameters a real, immediate gradient
    # signal from step 0 instead of a cold start.
    residual_alpha_init: float = 0.05


# --------------------------------------------------------------------------- #
# Attribute Edit Parser
# --------------------------------------------------------------------------- #
@dataclass
class AttributeParserConfig:
    enabled: bool = True
    # "rule" = spaCy dependency-parse rule extraction of {attribute, old, new}
    # triples (fully offline, no external API); "llm" = delegate parsing to a
    # pluggable callable (e.g. an OpenAI/local-LLM prompt) supplied at
    # runtime via AttributeEditParser(llm_parser_fn=...); "hybrid" tries the
    # LLM parser and falls back to the rule-based parser on failure/timeout.
    parser_backend: str = "rule"
    max_attributes: int = 8
    attribute_vocab_size: int = 512
    hidden_dim: int = 512


# --------------------------------------------------------------------------- #
# Reverse Verifier
# --------------------------------------------------------------------------- #
@dataclass
class ReverseVerifierConfig:
    enabled: bool = True
    # "tier1_inverse_consistency" (renamed from "tier1_vqascore", 2026-09-09,
    #   novelty-review finding #7 -- the old name invited readers/reviewers
    #   to assume this implements actual VQAScore [Lin et al.], a generative
    #   image-text likelihood method; it does not -- see reverse_verifier.py's
    #   own module docstring for what Tier 1 actually computes: a shared-
    #   composer cosine-similarity reconstruction check, not a VQA call):
    #   cheap, frozen-encoder consistency score (Section 4.3, Tier 1) - no
    #   extra training needed.
    # "tier2_reconstruction": learned lightweight reconstruction head that
    #   predicts an "expected reference feature" from (candidate, reverse
    #   instruction) and matches it to the true reference embedding
    #   (Section 4.3, Tier 2) - trained jointly, more compute.
    tier: str = "tier1_inverse_consistency"
    top_k: int = 10                    # how many forward candidates get re-verified
    hidden_dim: int = 512
    tier2_num_layers: int = 2
    # Cross-Batch Memory (Wang et al., CVPR 2020, arXiv:1912.06798) for the
    # negative-mining pool used by reverse_consistency_loss. Semi-hard
    # in-batch mining (the default when this is disabled) was found to
    # collapse at full training convergence -- topk_target_rv_argmax_rate
    # settled at ~14-16%, barely above the 10% chance baseline for top_k=10
    # (scripts/verify_semihard_negative.sbatch, 2026-09-07) -- because the
    # in-batch argmax/semi-hard selector is entangled with the SAME
    # embedding space it is selecting over: as that space converges, the
    # candidate pool it draws from (the current batch's ~B-1 other targets)
    # is both tiny and drawn from the same non-stationary distribution being
    # optimized. XBM decouples the negative SOURCE from the live optimizer
    # step by drawing from a FIFO queue of recently-computed (not
    # instantaneous) embeddings -- justified by the paper's own "slow drift"
    # measurement (Section 3.2): embedding drift stabilizes within ~3K
    # iterations, so embeddings cached up to ~1K steps ago remain valid
    # negatives. use_xbm=False (default) preserves the exact existing
    # semi-hard in-batch behavior; this is strictly opt-in.
    use_xbm: bool = False
    xbm_queue_size: int = 1024          # entries; paper sizes as a dataset fraction, not a batch multiple -- see gradcache.py-adjacent notes
    xbm_warmup_steps: int = 1000        # paper's own warm-up before drawing from the queue (their Algorithm 1)
    # Whether the candidate-invariant term enters the verification score
    # (added 2026-10-11 for the retrained "no-invariant" ablation, review
    # item B6.4.3). True reproduces every reported run exactly. False drops
    # the term and renormalises the two that remain in their trained ratio,
    # (0.5(1-w)*recon + w*edit) / (0.5(1-w) + w) -- the same definition
    # scripts/rv_component_ablation.py applies post hoc, so the retrained
    # and post-hoc "no-invariant" results measure the same score. Set only
    # through apply_module_ablation("fc+aep+rv+noinv").
    use_invariant_term: bool = True


# --------------------------------------------------------------------------- #
# Uncertainty Gate
# --------------------------------------------------------------------------- #
@dataclass
class UncertaintyGateConfig:
    enabled: bool = True
    variance_hidden_dim: int = 256
    neighborhood_k: int = 10                  # for the neighborhood-deviation-style regularizer
    coordination_hidden_dim: int = 256
    calibration_temperature_init: float = 1.0
    fuse_reverse_disagreement: bool = True    # feed RV disagreement into UG (Section 4.5 connective mechanism)


# --------------------------------------------------------------------------- #
# Loss weights (the connective joint loss of Section 4.5)
# L = L_retrieval + lambda1 * L_disentangle + lambda2 * L_reverse_consistency
#     + lambda3 * L_uncertainty_calibration
# --------------------------------------------------------------------------- #
@dataclass
class LossConfig:
    lambda_disentangle: float = 0.3
    lambda_reverse: float = 0.5
    lambda_uncertainty: float = 0.2
    label_smoothing: float = 0.0
    hard_negative_weight: float = 0.0   # >0 enables in-batch hard-negative up-weighting
    # Margin for the Reverse Verifier's in-batch negative hinge. The verifier is
    # trained on positives *and* wrong candidates so its score discriminates
    # between them -- required for inference-time reranking to do anything.
    reverse_negative_margin: float = 0.2


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    root: str = "./datasets"
    train_datasets: tuple = ("fashioniq", "cirr")
    val_dataset: str = "cirr"
    image_size: int = 224
    num_workers: int = 8
    batch_size: int = 128
    eval_batch_size: int = 256
    noise_injection_rate: float = 0.0   # ablation #6: synthetic label-noise fraction


# --------------------------------------------------------------------------- #
# Optimization / training loop
# --------------------------------------------------------------------------- #
@dataclass
class OptimConfig:
    lr: float = 2e-5
    backbone_lr_mult: float = 0.1       # discriminative LR: backbone trains slower than new heads
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-6
    warmup_ratio: float = 0.06
    scheduler: str = "cosine"           # "cosine" | "linear" | "constant"
    max_epochs: int = 30
    grad_clip_norm: float = 5.0
    amp: bool = True
    ema_decay: float = 0.0              # 0 disables EMA of weights
    # Gradient caching (arXiv:2101.06983). With in-batch InfoNCE the batch size
    # *is* the negative count, and CIR baselines train with 511-1023 negatives
    # vs 63 at batch_size 64. Enabling this decouples negatives from memory:
    # measured bs=1024 peaks at 3.43 GB on an 8 GB card (less than a plain bs=64
    # step) at roughly half the throughput. See revua_cir/gradcache.py.
    use_gradcache: bool = False
    gradcache_chunk_size: int = 32


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    # R@1, not R@10: RV/UG reranking only permutes the existing top-k
    # candidate set, so R@10 is provably near-invariant to it (see
    # default.yaml's own comment on this) -- monitoring R@10 makes model
    # selection blind to the modules the paper is actually about. Every
    # shipped config already overrides this explicitly to val_recall_at_1;
    # this default previously disagreed with that, a landmine for any future
    # config that omits the override (fixed 2026-09-08).
    monitor: str = "val_recall_at_1"    # metric name from metrics.py to track
    mode: str = "max"
    patience: int = 5
    min_delta: float = 1e-4
    warmup_epochs: int = 1              # do not allow stopping before this many epochs


@dataclass
class RunConfig:
    seed: int = 42
    output_dir: str = "./runs/revua_cir"
    log_every: int = 50
    eval_every_epochs: int = 1
    save_top_k: int = 3
    device: str = "cuda"
    deterministic: bool = False
    # Mid-epoch checkpoint every N steps (0 disables). Needed on Kaggle: a
    # session can end (12h cap, or an interrupted connection) at any point, and
    # the existing end-of-epoch-only checkpoint meant losing up to a full
    # epoch's compute -- 22-34 min per epoch measured with gradcache at
    # batch_size 1024. These are unranked "latest" saves distinct from the
    # top-k *validated* checkpoints TopKCheckpointKeeper tracks, and only the
    # single latest one is kept (see train.py) so they do not multiply disk use.
    checkpoint_every_n_steps: int = 10


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #
@dataclass
class ReVUAConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    attribute_parser: AttributeParserConfig = field(default_factory=AttributeParserConfig)
    reverse_verifier: ReverseVerifierConfig = field(default_factory=ReverseVerifierConfig)
    uncertainty_gate: UncertaintyGateConfig = field(default_factory=UncertaintyGateConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # ---- module-ablation convenience switches (Section 8, Ablation #1) ---- #
    # The only module combos this project reports (2026-09-08, per explicit
    # user decision on the UG-only-ablation finding): a strictly progressive
    # ladder. "fc+ug" (RV disabled, UG enabled) is deliberately EXCLUDED --
    # UncertaintyGate.forward's fuse_reverse_disagreement signal falls back
    # to a constant zero tensor when RV is absent (see uncertainty_gate.py),
    # making part of UG's own fusion layer structurally dead input in that
    # combo, and UG would receive only positive calibration supervision
    # (reverse_disagreement never varies with candidate correctness without
    # RV) -- a weak, misleading "uncertainty ablation" that doesn't actually
    # isolate UG's contribution. Chosen fix: keep UG dependent on RV (do NOT
    # add a parallel independent negative-mining mechanism just to make
    # fc+ug runnable) and enforce the valid ladder here, at the single
    # chokepoint every ablation combo passes through, rather than leaving
    # this as a documentation-only convention a future run could ignore.
    #
    # "fc+aep+rv+noinv" (added 2026-10-11, review item B6.4.3) is a branch
    # off the ladder, not a rung: fc+aep+rv with the candidate-invariant term
    # removed from the verification score, retrained from scratch. It exists
    # to test whether that term's post-hoc harm (Section 5.5) is also a
    # training-time effect. UG stays off, matching the reported fc+aep+rv.
    VALID_MODULE_ABLATIONS = ("fc", "fc+aep", "fc+aep+rv", "fc+aep+rv+ug",
                              "fc+aep+rv+noinv")

    def apply_module_ablation(self, modules: str) -> "ReVUAConfig":
        """modules must be one of ReVUAConfig.VALID_MODULE_ABLATIONS -- the
        strictly progressive fc -> +aep -> +rv -> +ug ladder. Any other
        combo (in particular "fc+ug", RV disabled with UG enabled) raises,
        rather than silently constructing a degenerate ablation -- see
        VALID_MODULE_ABLATIONS' own comment for why fc+ug specifically is
        excluded."""
        if modules not in self.VALID_MODULE_ABLATIONS:
            raise ValueError(
                f"module_ablation={modules!r} is not one of the supported combos "
                f"{self.VALID_MODULE_ABLATIONS}. In particular, 'fc+ug' (RV disabled, UG "
                f"enabled) is deliberately unsupported: UG's fuse_reverse_disagreement signal "
                f"falls back to a constant zero without RV, making it a structurally degenerate "
                f"ablation, not a genuine test of UG's standalone contribution.")
        cfg = copy.deepcopy(self)
        wanted = set(modules.split("+"))
        cfg.attribute_parser.enabled = "aep" in wanted
        cfg.reverse_verifier.enabled = "rv" in wanted
        cfg.reverse_verifier.use_invariant_term = "noinv" not in wanted
        cfg.uncertainty_gate.enabled = "ug" in wanted
        if not cfg.attribute_parser.enabled:
            cfg.loss.lambda_disentangle = 0.0
        if not cfg.reverse_verifier.enabled:
            cfg.loss.lambda_reverse = 0.0
        if not cfg.uncertainty_gate.enabled:
            cfg.loss.lambda_uncertainty = 0.0
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "ReVUAConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "ReVUAConfig":
        cfg = cls()
        for section_name, section_cls in [
            ("backbone", BackboneConfig), ("fusion", FusionConfig),
            ("attribute_parser", AttributeParserConfig), ("reverse_verifier", ReverseVerifierConfig),
            ("uncertainty_gate", UncertaintyGateConfig), ("loss", LossConfig),
            ("data", DataConfig), ("optim", OptimConfig),
            ("early_stopping", EarlyStoppingConfig), ("run", RunConfig),
        ]:
            if section_name in raw and raw[section_name]:
                current = getattr(cfg, section_name)
                merged = {**asdict(current), **raw[section_name]}
                setattr(cfg, section_name, section_cls(**merged))
        return cfg

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
