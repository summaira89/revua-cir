"""Tests for scripts/b64_analysis.py on synthetic data. Runs without pytest."""
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))

import b64_analysis as A  # noqa: E402
from fc_vs_bounded_paired import PairingError  # noqa: E402


def test_holm_adjusted_matches_hand_values():
    adj = A.holm_adjusted([0.01, 0.04])
    assert abs(adj[0] - 0.02) < 1e-12 and abs(adj[1] - 0.04) < 1e-12
    adj = A.holm_adjusted([0.04, 0.01])           # order-independent
    assert abs(adj[0] - 0.04) < 1e-12 and abs(adj[1] - 0.02) < 1e-12
    assert A.holm_adjusted([0.9, 0.8]) == [1.0, 1.0]  # 2*0.8 capped, carried
    adj = A.holm_adjusted([0.03, 0.04])               # step-down monotonicity:
    assert abs(adj[0] - 0.06) < 1e-12 and abs(adj[1] - 0.06) < 1e-12  # 0.04 -> 0.06


def _rec(i, w, noinv, rng, tgi):
    rc = [rng.uniform(-1, 1) for _ in range(5)]
    inv = [rng.uniform(-1, 1) for _ in range(5)]
    ed = [rng.uniform(-1, 1) for _ in range(5)]
    if noinv:
        a = 0.5 * (1 - w)
        cos = [(a * r + w * e) / (a + w) for r, e in zip(rc, ed)]
    else:
        cos = [(1 - w) * (0.5 * r + 0.5 * v) + w * e for r, v, e in zip(rc, inv, ed)]
    return {"query_index": i, "candidate_gallery_ids": list(range(5)),
            "base_score": [rng.random() for _ in range(5)],
            "verification_score": [rng.random() for _ in range(5)],
            "target_pos": rng.randrange(-1, 5), "target_gallery_index": tgi,
            "cos_sim": cos, "recon_cos": rc, "invariant_cos": inv, "edit_cos": ed}


def test_recover_noinv_w_is_exact_on_noinv_scores():
    rng = random.Random(1)
    recs = [_rec(i, 0.61, True, rng, i) for i in range(40)]
    w, sd, n, resid = A.recover_noinv_w(recs)
    assert abs(w - 0.61) < 1e-9 and sd < 1e-9 and n > 0 and resid < 1e-9


def test_full_formula_does_not_fit_a_noinv_cache():
    # Distinguishes the two code paths, so a cache made by the wrong model shows.
    rng = random.Random(2)
    recs = [_rec(i, 0.61, True, rng, i) for i in range(40)]
    assert A.full_formula_residual(recs, 0.61) > 1e-3
    full = [_rec(i, 0.49, False, rng, i) for i in range(40)]
    assert A.full_formula_residual(full, 0.49) < 1e-12


def test_load_hits_defaults_valid_from_targets():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"hit_at_1": [1, 0], "targets": [3, -1]}, fh)
    try:
        hits, targets, valid, _ = A.load_hits(fh.name)
        assert valid == [1, 0]
    finally:
        os.unlink(fh.name)


def test_load_hits_rejects_ragged_input():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"hit_at_1": [1, 0, 1], "targets": [3, 4]}, fh)
    try:
        try:
            A.load_hits(fh.name)
            assert False, "should have raised"
        except PairingError:
            pass
    finally:
        os.unlink(fh.name)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok  " + f.__name__)
    print("\n%d tests passed" % len(fns))
