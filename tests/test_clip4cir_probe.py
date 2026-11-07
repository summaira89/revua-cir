"""clip4cir_probe.py (amendment §24): helper contracts on CPU, including the weight-rounding paths."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import clip4cir_probe as cp  # noqa: E402
import revision_followups3 as r3  # noqa: E402

RNG = np.random.default_rng(24)


def test_query_ranks_match_rank_of_and_subset_rank():
    for _ in range(200):
        s = np.round(RNG.uniform(size=40), 2)                       # coarse values force ties
        ex, tgt, *others = [int(x) for x in RNG.choice(40, 6, replace=False)]
        members = [tgt] + others
        q = cp.query_ranks(s, ex, tgt, members)
        assert q["rank"] == r3.rank_of(s, tgt, ex)
        mem_order = [m for m in r3.top_k_excluding(s, ex, 39) if m in members]
        assert q["subset_rank"] == mem_order.index(tgt)


def test_query_ranks_aborts_when_target_not_member():
    with pytest.raises(SystemExit):
        cp.query_ranks(np.arange(5.0), 0, 1, [2, 3])


def test_cirr_metrics_and_gate():
    ranks = [{"rank": 0, "subset_rank": 0}, {"rank": 7, "subset_rank": 2}, {"rank": 60, "subset_rank": 4}]
    m = cp.cirr_metrics(ranks, [0, 1, 2])
    assert m["R@1"] == pytest.approx(100 / 3) and m["R@10"] == pytest.approx(200 / 3) and m["R@50"] == pytest.approx(200 / 3)
    assert m["Rs@1"] == pytest.approx(100 / 3) and m["Rs@3"] == pytest.approx(200 / 3)
    pub = {"R@1": 33.5, "Rs@1": 33.0}
    assert cp.gate(m, pub, 0.5)["pass"] is True
    assert cp.gate(m, {"R@1": 34.0}, 0.5)["pass"] is False


def test_check_shapes_accepts_released_dims_and_rejects_others():
    import torch
    clip_sd = {"visual.attnpool.positional_embedding": torch.zeros(82, 2560), "text_projection": torch.zeros(640, 640)}
    comb = {"text_projection_layer.weight": torch.zeros(2560, 640), "combiner_layer.weight": torch.zeros(5120, 5120),
            "output_layer.weight": torch.zeros(640, 5120)}
    cp.check_shapes(clip_sd, comb, cp.CONFIGS["RN50x4_fullft"])
    with pytest.raises(SystemExit):
        cp.check_shapes(clip_sd, comb, cp.CONFIGS["RN50_fullft"])


def _tiny_clip_sd():
    import torch
    from clip.model import CLIP
    torch.manual_seed(0)
    m = CLIP(16, 64, (1, 1, 1, 1), 8, None, 77, 49408, 64, 1, 1)
    sd = m.state_dict()
    for k, v in sd.items():                                         # values not representable in fp16
        if v.is_floating_point():
            sd[k] = v + 1e-4 * torch.rand_like(v) + 1e-7
    return sd


def test_released_path_rounds_exactly_like_clip_load_and_exact_path_does_not():
    import torch
    from clip.model import build_model
    sd = _tiny_clip_sd()
    rel = cp.build_clip(sd, "released").state_dict()
    exa = cp.build_clip(sd, "exact_fp32").state_dict()
    # clip.load on CUDA: fp16-converted model, load_state_dict, then .float()
    ref = build_model(dict(sd))
    ref.load_state_dict(sd)
    ref = ref.float().state_dict()
    for k, v in sd.items():
        if not v.is_floating_point():
            continue
        assert rel[k].dtype == torch.float32 and exa[k].dtype == torch.float32
        assert torch.equal(exa[k], v)
        assert torch.equal(rel[k], ref[k])
    w = "visual.layer1.0.conv1.weight"
    assert torch.equal(rel[w], sd[w].half().float()) and not torch.equal(rel[w], sd[w])
    assert torch.equal(rel["token_embedding.weight"], sd["token_embedding.weight"])   # embeddings are not rounded
    with pytest.raises(ValueError):
        cp.build_clip(sd, "fp16")
