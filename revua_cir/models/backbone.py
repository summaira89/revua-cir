"""
Vision-language backbone abstraction.

Section 4.1 (Forward Composer) calls for "a CLIP or BLIP-2 backbone". This
module exposes a single interface, `BackboneOutput`, so the rest of the
codebase (fusion, attribute parser, reverse verifier) never needs to know
which concrete backbone is loaded. Three backends are supported:

  * "clip"      - HuggingFace `transformers` CLIP (default, matches the
                   paper framework's primary recommendation and every
                   mandatory baseline in Section 7).
  * "open_clip" - LAION `open_clip` checkpoints (drop-in alternative,
                   used by the backbone ablation in Section 8, #4).
  * "tiny"      - a small randomly-initialized ViT + Transformer text
                   encoder with the exact same interface. This exists so
                   the full pipeline (fusion -> AEP -> RV -> UG -> losses
                   -> metrics -> HPO) can be unit-tested and smoke-tested
                   without network access to a model hub or a GPU -- swap
                   back to "clip"/"open_clip" for real experiments by
                   changing one config field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from revua_cir.config import BackboneConfig


@dataclass
class BackboneOutput:
    pooled: torch.Tensor          # (B, D) sentence/image-level embedding, D = embed_dim
    tokens: torch.Tensor          # (B, N, D) token/patch-level embeddings, D = embed_dim
    mask: Optional[torch.Tensor] = None  # (B, N) attention mask, 1 = valid token (text only; None for images)


class _TinyPatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, width: int):
        super().__init__()
        self.proj = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size)
        n_patches = (image_size // patch_size) ** 2
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, n_patches + 1, width) * 0.02)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)          # (B, N, width)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        return x


class _TinyTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, mlp_mult: int = 4):
        super().__init__()
        block = nn.TransformerEncoderLayer(d_model=width, nhead=heads, dim_feedforward=width * mlp_mult,
                                            dropout=0.0, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.ln = nn.LayerNorm(width)

    def forward(self, x, key_padding_mask=None):
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.ln(x)


class TinyBackbone(nn.Module):
    """Offline fallback backbone -- random weights, same I/O contract as real CLIP."""

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        width = cfg.tiny_width
        self.patch_embed = _TinyPatchEmbed(cfg.tiny_image_size, cfg.tiny_patch_size, width)
        self.vision_encoder = _TinyTransformer(width, cfg.tiny_layers, cfg.tiny_heads)
        self.token_embed = nn.Embedding(cfg.tiny_vocab_size, width, padding_idx=0)
        self.text_pos = nn.Parameter(torch.randn(1, 512, width) * 0.02)
        self.text_encoder = _TinyTransformer(width, cfg.tiny_layers, cfg.tiny_heads)
        self.image_proj = nn.Linear(width, cfg.embed_dim)
        self.text_proj = nn.Linear(width, cfg.embed_dim)

        # Mirror the real backbones' freezing semantics, so the offline test path
        # can actually exercise `freeze_backbone` -- the HF-CLIP freeze bug hid
        # precisely because the tiny backbone ignored this flag entirely.
        # Trunks freeze; the projections stay trainable, matching how the CLIP
        # path keeps its adapters in-graph.
        self.frozen = cfg.freeze_backbone
        if self.frozen:
            for m in (self.patch_embed, self.vision_encoder, self.token_embed, self.text_encoder):
                for p in m.parameters():
                    p.requires_grad_(False)
            self.text_pos.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            for m in (self.patch_embed, self.vision_encoder, self.token_embed, self.text_encoder):
                m.eval()
        return self

    def encode_image(self, pixel_values: torch.Tensor) -> BackboneOutput:
        with torch.set_grad_enabled(not self.frozen and torch.is_grad_enabled()):
            x = self.patch_embed(pixel_values)
            x = self.vision_encoder(x)
        if self.frozen:
            x = x.detach()
        x = self.image_proj(x)
        return BackboneOutput(pooled=x[:, 0], tokens=x[:, 1:], mask=None)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> BackboneOutput:
        key_padding_mask = attention_mask == 0
        with torch.set_grad_enabled(not self.frozen and torch.is_grad_enabled()):
            x = self.token_embed(input_ids) + self.text_pos[:, : input_ids.size(1)]
            x = self.text_encoder(x, key_padding_mask=key_padding_mask)
        if self.frozen:
            x = x.detach()
        x = self.text_proj(x)
        # pooled = first non-pad token (CLS-equivalent at position 0, assumed always valid)
        pooled = x[:, 0]
        return BackboneOutput(pooled=pooled, tokens=x, mask=attention_mask)


class HFClipBackbone(nn.Module):
    """`transformers` CLIP backbone -- the default, matches Section 7 baselines."""

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        from transformers import CLIPModel

        self.cfg = cfg
        self.clip = CLIPModel.from_pretrained(cfg.pretrained_name)
        # Token streams come out of the encoders at `hidden_size`; pooled vectors come
        # out of CLIP's contrastive heads at `projection_dim` (these differ -- e.g.
        # ViT-L/14 is 1024 hidden / 768 projection). Both have to land on embed_dim,
        # so each needs its own adapter sized from the dim it actually receives.
        vis_dim = self.clip.config.vision_config.hidden_size
        txt_dim = self.clip.config.text_config.hidden_size
        proj_dim = self.clip.config.projection_dim
        self.image_out_proj = self._adapter(vis_dim, cfg.embed_dim)
        self.text_out_proj = self._adapter(txt_dim, cfg.embed_dim)
        self.image_pooled_proj = self._adapter(proj_dim, cfg.embed_dim)
        self.text_pooled_proj = self._adapter(proj_dim, cfg.embed_dim)

        self.frozen = cfg.freeze_backbone
        if cfg.freeze_backbone:
            for p in self.clip.parameters():
                p.requires_grad_(False)
        elif cfg.freeze_layers > 0:
            for name, p in self.clip.vision_model.encoder.layers[: cfg.freeze_layers].named_parameters():
                p.requires_grad_(False)
            for name, p in self.clip.text_model.encoder.layers[: cfg.freeze_layers].named_parameters():
                p.requires_grad_(False)

    @staticmethod
    def _adapter(in_dim: int, out_dim: int) -> nn.Module:
        return nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def train(self, mode: bool = True):
        """Keep a frozen backbone in eval mode so its dropout/LayerNorm stats do
        not drift while only the adapters and the four modules train."""
        super().train(mode)
        if self.frozen:
            self.clip.eval()
        return self

    def encode_image(self, pixel_values: torch.Tensor) -> BackboneOutput:
        # A frozen backbone must run under no_grad, not merely requires_grad=False:
        # the latter still builds the full ViT activation graph for backward, which
        # is what actually exhausts an 8 GB card (bs 32 OOMs without this, and
        # fits in ~4.2 GB with it). The adapters below stay in-graph.
        with torch.set_grad_enabled(not self.frozen and torch.is_grad_enabled()):
            out = self.clip.vision_model(pixel_values=pixel_values)
        hidden, pooled_raw = out.last_hidden_state, out.pooler_output
        if self.frozen:
            hidden, pooled_raw = hidden.detach(), pooled_raw.detach()
            with torch.no_grad():
                projected = self.clip.visual_projection(pooled_raw)
        else:
            projected = self.clip.visual_projection(pooled_raw)
        tokens = self.image_out_proj(hidden)   # (B, 1+N, D) incl. CLS at index 0
        pooled = self.image_pooled_proj(projected)
        return BackboneOutput(pooled=pooled, tokens=tokens[:, 1:], mask=None)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> BackboneOutput:
        with torch.set_grad_enabled(not self.frozen and torch.is_grad_enabled()):
            out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden, pooled_raw = out.last_hidden_state, out.pooler_output
        if self.frozen:
            hidden, pooled_raw = hidden.detach(), pooled_raw.detach()
            with torch.no_grad():
                projected = self.clip.text_projection(pooled_raw)
        else:
            projected = self.clip.text_projection(pooled_raw)
        tokens = self.text_out_proj(hidden)
        pooled = self.text_pooled_proj(projected)
        return BackboneOutput(pooled=pooled, tokens=tokens, mask=attention_mask)


class OpenClipBackbone(nn.Module):
    """`open_clip` backbone -- used for the backbone-choice ablation (Section 8, #4)."""

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        import open_clip

        self.cfg = cfg
        self.model, _, _ = open_clip.create_model_and_transforms(cfg.open_clip_name, pretrained=cfg.open_clip_pretrained)
        vis_dim = self.model.visual.output_dim
        txt_dim = self.model.text_projection.shape[1] if hasattr(self.model, "text_projection") else vis_dim
        self.image_out_proj = nn.Identity() if vis_dim == cfg.embed_dim else nn.Linear(vis_dim, cfg.embed_dim)
        self.text_out_proj = nn.Identity() if txt_dim == cfg.embed_dim else nn.Linear(txt_dim, cfg.embed_dim)
        if cfg.freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def encode_image(self, pixel_values: torch.Tensor) -> BackboneOutput:
        feats = self.model.encode_image(pixel_values, normalize=False)
        pooled = self.image_out_proj(feats)
        # open_clip does not expose patch tokens uniformly across archs -> broadcast pooled as a 1-token sequence
        return BackboneOutput(pooled=pooled, tokens=pooled.unsqueeze(1), mask=None)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> BackboneOutput:
        feats = self.model.encode_text(input_ids, normalize=False)
        pooled = self.text_out_proj(feats)
        return BackboneOutput(pooled=pooled, tokens=pooled.unsqueeze(1), mask=torch.ones(pooled.size(0), 1, device=pooled.device))


def build_backbone(cfg: BackboneConfig) -> nn.Module:
    if cfg.kind == "clip":
        return HFClipBackbone(cfg)
    if cfg.kind == "open_clip":
        return OpenClipBackbone(cfg)
    if cfg.kind == "tiny":
        return TinyBackbone(cfg)
    raise ValueError(f"Unknown backbone kind: {cfg.kind}")
