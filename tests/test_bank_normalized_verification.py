"""Tests for scripts/bank_normalized_verification.py. Runs without pytest.

Each external-audit finding (17 October 2026) has a test that fails on the
defect: CSLS coefficient, non-finite inputs, row misalignment, K/exclusion
edge cases, FashionIQ category completeness, matched calibrated arms, Holm."""
import json
import math
import os
import random
import shutil
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import bank_normalized_verification as B  # noqa: E402

D, K_CAND, W, CA, CB = 16, 6, 0.4, 0.9, -0.07


def _unit(rng, *shape):
    x = rng.standard_normal(shape + (D,)).astype(np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


CK_SHA = "a" * 64


def _build(dirpath, name, n, seed, dataset="cirr", category=None, ckpt=CK_SHA):
    """Cache + vectors + manifest whose cached terms are exactly the vectors'
    cosines, mixed with W and calibrated with (CA, CB)."""
    rng = np.random.default_rng(seed)
    pyr = random.Random(seed)
    arr = {"e_inv": _unit(rng, n), "t_hat": _unit(rng, n), "q_comp": _unit(rng, n),
           "rec": _unit(rng, n, K_CAND), "cinv": _unit(rng, n, K_CAND),
           "cedit": _unit(rng, n, K_CAND), "c_pool": _unit(rng, n, K_CAND)}
    cand = np.arange(n * K_CAND, dtype=np.int64).reshape(n, K_CAND)
    cache = os.path.join(dirpath, name + ".jsonl")
    with open(cache, "w", encoding="utf-8") as fh:
        for q in range(n):
            rc = (arr["rec"][q] @ arr["e_inv"][q]).astype(float)
            ic = (arr["cinv"][q] @ arr["e_inv"][q]).astype(float)
            ec = (arr["cedit"][q] @ arr["t_hat"][q]).astype(float)
            cs = (1 - W) * 0.5 * (rc + ic) + W * ec
            fh.write(json.dumps({
                "query_id": q, "candidate_gallery_ids": cand[q].tolist(),
                "base_score": (arr["c_pool"][q] @ arr["q_comp"][q]).astype(float).tolist(),
                "recon_cos": rc.tolist(), "invariant_cos": ic.tolist(), "edit_cos": ec.tolist(),
                "cos_sim": cs.tolist(),
                "verification_score": [B.calibrate(x, CA, CB) for x in cs],
                "target_pos": pyr.choice([0, 1, 2, 3, -1])}) + "\n")
    npz = os.path.join(dirpath, name + ".npz")
    np.savez(npz, query_id=np.arange(n, dtype=np.int64), candidate_ids=cand, **arr)
    _manifest(npz, cache, n, dataset, category, ckpt)
    return cache, npz


def _manifest(npz, cache, n, dataset, category, ckpt):
    man = {"schema": B.SCHEMA, "dataset": dataset, "split": "val", "category": category,
           "n_queries": n, "k": K_CAND, "dim": D, "cache_sha256": B.sha256_file(cache),
           "npz_sha256": B.sha256_file(npz), "checkpoint_sha256": ckpt,
           "config_sha256": "c" * 64, "module_combo": "fc+aep+rv",
           "mixture_w": W, "calibrator_scale": CA, "calibrator_bias": CB}
    with open(npz[:-4] + ".manifest.json", "w", encoding="utf-8") as fh:
        json.dump(man, fh)


def _expect(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return
    raise AssertionError("expected %s" % exc.__name__)


class _Tmp:
    def __enter__(self):
        self.d = tempfile.mkdtemp()
        return self.d

    def __exit__(self, *exc):
        shutil.rmtree(self.d)


def test_valid_pair_loads_and_deployed_score_is_reproduced():
    with _Tmp() as d:
        c, z = _build(d, "a", 40, 0)
        recs, rels, _, man = B.load_pair(c, z)
        dep, par = B.deployed_scores(recs, rels, list(range(40)), man)
        assert par["fused_permutation_differs"] == 0
        for i in range(40):
            assert max(abs(x - y) for x, y in zip(dep[i], recs[i]["verification_score"])) < 1e-9


def test_misaligned_rows_are_rejected():
    """Audit B2: reversed e_inv rows with an honest manifest must fail."""
    with _Tmp() as d:
        c, z = _build(d, "a", 40, 1)
        with np.load(z) as f:
            arr = {k: f[k] for k in f.files}
        arr["e_inv"] = arr["e_inv"][::-1].copy()
        np.savez(z, **arr)
        _manifest(z, c, 40, "cirr", None, CK_SHA)
        _expect(B.ProtocolError, B.load_pair, c, z)


def _shift_cache(cache, npz, n, field, delta, keep_consistent=True):
    """Add delta to record 0, candidate 0 of `field` in the cache. With
    keep_consistent a shifted term also updates that record's cos_sim and
    verification_score, so only the vector-side check can object. The
    manifest is rewritten (new cache digest)."""
    with open(cache, encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh]
    r = recs[0]
    r[field][0] += delta
    if keep_consistent and field in B.TERMS:
        cs = (1 - W) * 0.5 * (r["recon_cos"][0] + r["invariant_cos"][0]) + W * r["edit_cos"][0]
        r["cos_sim"][0], r["verification_score"][0] = cs, B.calibrate(cs, CA, CB)
    with open(cache, "w", encoding="utf-8") as fh:
        for rec in recs:
            fh.write(json.dumps(rec) + "\n")
    _manifest(npz, cache, n, "cirr", None, CK_SHA)


def test_vector_tolerance_is_per_field():
    """Audit 18 Oct, items 3/4: each cached quantity is held to its OWN
    vector-side tolerance (deterministic deviations at 0.9x and 1.1x of each
    limit), and the relaxed base-score limit cannot let a verifier-term error
    of the same size through."""
    cases = [(f, m * tol, m < 1.0) for f, tol in B.VECTOR_TOL.items()
             if f != "verification_score" for m in (0.9, 1.1)]
    cases += [("base_score", 5e-4, True), ("recon_cos", 5e-4, False),
              ("invariant_cos", 5e-4, False), ("edit_cos", 5e-4, False)]
    for field, delta, accepted in cases:
        with _Tmp() as d:
            c, z = _build(d, "a", 20, 11)
            _shift_cache(c, z, 20, field, delta)
            if accepted:
                B.load_pair(c, z)
            else:
                _expect(B.ProtocolError, B.load_pair, c, z)


def test_cache_inconsistency_is_held_to_tol_cache():
    """Audit 18 Oct, item 4: the cache-internal mixture and calibration checks
    stay at TOL_CACHE, independent of the vector-side policy."""
    for field in ("cos_sim", "verification_score"):
        for m, accepted in ((0.5, True), (2.0, False)):
            with _Tmp() as d:
                c, z = _build(d, "a", 20, 12)
                _shift_cache(c, z, 20, field, m * B.TOL_CACHE, keep_consistent=False)
                if accepted:
                    B.load_pair(c, z)
                else:
                    _expect(B.ProtocolError, B.load_pair, c, z)


def test_misalignment_is_rejected_after_a_tolerated_shift():
    """A tolerated base-score shift does not open the door to misaligned rows."""
    with _Tmp() as d:
        c, z = _build(d, "a", 40, 9)
        _shift_cache(c, z, 40, "base_score", 5e-4)
        B.load_pair(c, z)                                  # accepted
        with np.load(z) as f:
            arr = {k: f[k] for k in f.files}
        arr["e_inv"] = arr["e_inv"][::-1].copy()           # now misaligned
        np.savez(z, **arr)
        _manifest(z, c, 40, "cirr", None, CK_SHA)
        _expect(B.ProtocolError, B.load_pair, c, z)


def test_provenance_records_the_tolerance_policy():
    with _Tmp() as d:
        c, z = _build(d, "a", 20, 13)
        man = B.load_pair(c, z)[3]
        man["tolerance_policy"] = dict(B.VECTOR_TOL)
        prov = B.provenance("cirr", [man], 10)
        assert prov["vector_tolerance"] == B.VECTOR_TOL
        assert prov["cache_tolerance"] == B.TOL_CACHE
        assert prov["inputs"][0]["dump_tolerance"] == B.VECTOR_TOL
        man.pop("tolerance_policy")
        man["tolerance"] = 1e-4                            # the pre-policy CIRR manifest
        assert B.provenance("cirr", [man], 10)["inputs"][0]["dump_tolerance"] == 1e-4


def test_nonfinite_vectors_are_rejected():
    """Audit B1: NaN must not pass validation."""
    with _Tmp() as d:
        c, z = _build(d, "a", 30, 2)
        with np.load(z) as f:
            arr = {k: f[k] for k in f.files}
        arr["rec"][3, 1, 0] = np.nan
        np.savez(z, **arr)
        _manifest(z, c, 30, "cirr", None, CK_SHA)
        _expect(B.ProtocolError, B.load_pair, c, z)


def test_manifest_metadata_is_required():
    """Audit 4/2: missing or unsupported config/module metadata must fail,
    whether it is missing in one manifest or in all of them."""
    with _Tmp() as d:
        c, z = _build(d, "a", 30, 7)
        man_path = z[:-4] + ".manifest.json"
        with open(man_path) as fh:
            good = json.load(fh)
        for broken in ({k: v for k, v in good.items() if k != "config_sha256"},
                       {k: v for k, v in good.items() if k != "module_combo"},
                       dict(good, module_combo="fc+aep"),
                       dict(good, config_sha256="not-a-digest"),
                       dict(good, dim=True)):
            with open(man_path, "w") as fh:
                json.dump(broken, fh)
            _expect(B.ProtocolError, B.load_pair, c, z)


def test_cache_or_vector_tampering_is_rejected():
    with _Tmp() as d:
        c, z = _build(d, "a", 30, 3)
        with open(c, "a", encoding="utf-8") as fh:
            fh.write("\n")
        _expect(B.ProtocolError, B.load_pair, c, z)


def test_topk_means_mixed_exclusion_and_all_mean():
    """Audit B6: rows with exclude -1 keep the full K."""
    a = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    bank = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    m = B.topk_means(a, bank, (3,), exclude=np.array([0, -1]))
    assert abs(m[3][0] - (0.6 + 0.0 - 1.0) / 3) < 1e-6          # col 0 dropped
    assert abs(m[3][1] - (1.0 + 0.6 + 0.0) / 3) < 1e-6          # nothing dropped
    assert abs(m["all"][0] - (0.0 - 1.0 + 0.6) / 3) < 1e-6
    assert abs(m["all"][1] - (1.0 + 0.0 - 1.0 + 0.6) / 4) < 1e-6


def test_topk_means_validates_arguments():
    a = np.ones((2, 2), dtype=np.float32)
    bank = np.ones((3, 2), dtype=np.float32)
    _expect(B.ProtocolError, B.topk_means, a, bank, (0,))
    _expect(B.ProtocolError, B.topk_means, a, bank, (2.0,))
    _expect(B.ProtocolError, B.topk_means, a, bank, (4,))
    _expect(B.ProtocolError, B.topk_means, a, bank, (3,), np.array([0, -1]))
    _expect(B.ProtocolError, B.topk_means, a, np.ones((3, 3), dtype=np.float32), (1,))


def test_csls_coefficient_is_half():
    """Audit section 4: the primary correction is s - 0.5 h."""
    assert B.CSLS_COEF == 0.5
    with _Tmp() as d:
        c, z = _build(d, "a", 80, 4)
        recs, _, vec, man = B.load_pair(c, z)
        idx = [5, 7]
        bank = {"e_inv": vec["e_inv"][20:80], "t_hat": vec["t_hat"][20:80]}
        pen = B.term_penalties(vec, idx, bank)
        got = B.corrected_scores(recs, idx, pen, man)
        r, j, h = recs[5], 2, {t: pen[t][B.K_PRIMARY][0][2] for t in B.TERMS}
        s = {t: r[t][j] - 0.5 * h[t] for t in B.TERMS}
        mix = (1 - W) * 0.5 * (s["recon_cos"] + s["invariant_cos"]) + W * s["edit_cos"]
        assert abs(got[0][j] - B.calibrate(mix, CA, CB)) < 1e-12


def test_auroc_values_and_degenerate_input():
    assert B.auroc([3.0], [1.0, 2.0]) == 1.0
    assert B.auroc([1.0], [2.0, 3.0]) == 0.0
    assert B.auroc([2.0], [2.0]) == 0.5
    _expect(B.ProtocolError, B.auroc, [], [1.0])


def test_analyse_cirr_runs_and_is_internally_consistent():
    with _Tmp() as d:
        c, z = _build(d, "a", 240, 5)
        res = B.analyse_cirr(c, z, n_rep=20, expected_anchor=None)
        assert res["coef"] == 0.5 and "csls" in res["fusion"] and "P1" in res
        assert res["fusion"]["deployed"]["r1"] == res["anchor_r1"] or \
            abs(res["fusion"]["deployed"]["r1"] - res["anchor_r1"]) < 0.005
        json.dumps(res, allow_nan=False)


def test_fiq_requires_each_category_exactly_once():
    with _Tmp() as d:
        loaded = []
        for cat in ("dress", "dress", "toptee"):
            c, z = _build(d, cat + str(len(loaded)), 20, len(loaded), "fashioniq", cat)
            loaded.append(B.load_pair(c, z))
        _expect(B.ProtocolError, B._validate_fiq_set, loaded)


def test_analyse_fiq_end_to_end_with_small_categories():
    sizes = {"dress": 140, "shirt": 134, "toptee": 130}
    saved = dict(B.FIQ_EXPECTED)
    B.FIQ_EXPECTED.clear()
    B.FIQ_EXPECTED.update(sizes)
    try:
        with _Tmp() as d:
            cc, cz = _build(d, "cirr", 320, 11)
            pairs = [_build(d, cat, n, 20 + i, "fashioniq", cat)
                     for i, (cat, n) in enumerate(sizes.items())]
            res = B.analyse_fiq(cc, cz, pairs, n_rep=20)
            assert res["n_eval"] == sum(n - n // 2 for n in sizes.values())
            assert set(res["auroc"]) == {"deployed", "csls_target", "csls_source",
                                         "csls_transductive"}
            assert "P2" in res and set(res["per_category"]) == set(sizes)
            json.dumps(res, allow_nan=False)
    finally:
        B.FIQ_EXPECTED.clear()
        B.FIQ_EXPECTED.update(saved)


def test_bank_subsets_never_touch_the_evaluation_half():
    """Audit 3/8: sensitivity varies only the bank; evaluation stays fixed."""
    bank, evaluation = B.split_indices(500, B.SPLIT_SEED)
    subsets = B.bank_subsets(bank)
    assert set(subsets) == set(B.EXTRA_BANK_SEEDS)
    for pick in subsets.values():
        assert set(pick) <= set(bank) and not set(pick) & set(evaluation)
        assert len(pick) == int(0.8 * len(bank)) and len(set(pick)) == len(pick)
    assert subsets == B.bank_subsets(bank)                     # deterministic
    assert len({tuple(v) for v in subsets.values()}) == len(subsets)


def test_spearman_handles_ties_and_constants():
    """Audit A: tied exposure must not yield a manufactured correlation."""
    rho, status = B.spearman([2, 2, 2, 2], [0.1, 0.2, 0.3, 0.4])
    assert rho is None and status == "one variable is constant"
    rho, _ = B.spearman([1, 1, 2], [1.0, 2.0, 3.0])        # average ranks 1.5, 1.5, 3
    assert abs(rho - 0.8660254037844386) < 1e-12
    recs = [{"candidate_gallery_ids": [1, 2, 3]}, {"candidate_gallery_ids": [1, 2, 3]}]
    pen = {t: {B.K_PRIMARY: np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])} for t in B.TERMS}
    out = B.exposure_diagnostic(recs, [0, 1], pen)
    assert out["spearman_exposure_vs_mean_penalty"] is None
    json.dumps(out, allow_nan=False)


def test_calibrate_is_stable_for_large_logits():
    """Audit E: a finite but large logit must not overflow."""
    assert B.calibrate(-1.0, 1000.0, 0.0) == 0.0 or B.calibrate(-1.0, 1000.0, 0.0) < 1e-300
    assert B.calibrate(1.0, 1000.0, 0.0) == 1.0
    assert abs(B.calibrate(0.3, 2.0, -0.1) + B.calibrate(-0.3, 2.0, 0.1) - 1.0) < 1e-15


def test_deployed_control_parity():
    """Amendment 4: the deployed control is the recomputed score; it must
    match the cache to TOL_PARITY and keep every fused Hit@1/Hit@5. A
    float32-storage tie may reorder candidates (counted, accepted); a real
    discrepancy, even one leaving hit@1 unchanged, is still caught (audit C)."""
    man = {"k": 3, "mixture_w": 0.5, "calibrator_scale": 1.0, "calibrator_bias": 0.0}
    terms = {"recon_cos": [0.1, 0.2, 0.3], "invariant_cos": [0.1, 0.2, 0.3],
             "edit_cos": [0.1, 0.2, 0.3]}
    exact = [B.calibrate(x, 1.0, 0.0) for x in (0.1, 0.2, 0.3)]
    rec = dict(terms, base_score=[0.9, 0.5, 0.49], verification_score=exact)
    dep, par = B.deployed_scores([rec], [{0}], [0], man)
    assert dep == [exact] and par["verifier_order_differs"] == 0 and par["max_abs_diff_vs_cache"] == 0
    # a real discrepancy (lower ranks swapped, top-1 unchanged) is caught
    swapped = [exact[1], exact[0], exact[2]]
    assert B._order(swapped)[0] == B._order(exact)[0] and B._order(swapped) != B._order(exact)
    _expect(B.ProtocolError, B.deployed_scores,
            [dict(rec, verification_score=swapped)], [{0}], [0], man)
    # a float32 storage tie: two recomputed scores 2e-8 apart stored equal
    tie_terms = {"recon_cos": [0.2, 0.2 + 4e-8, 0.3], "invariant_cos": [0.2, 0.2, 0.3],
                 "edit_cos": [0.2, 0.2, 0.3]}
    rt = [B.calibrate((1 - 0.5) * 0.5 * (r + i) + 0.5 * e, 1.0, 0.0) for r, i, e in
          zip(tie_terms["recon_cos"], tie_terms["invariant_cos"], tie_terms["edit_cos"])]
    stored = [rt[0], rt[0], rt[2]]                         # cache rounded the pair together
    tied = dict(tie_terms, base_score=[0.1, 0.1, 0.9], verification_score=stored)
    dep, par = B.deployed_scores([tied], [{2}], [0], man)
    assert par["verifier_order_differs"] == 1 and par["max_abs_diff_vs_cache"] < B.TOL_PARITY
    # a sub-tolerance difference that changes Hit@1 is still rejected
    # recomputed: candidates 0 and 1 tie (index order puts 0 first, a hit);
    # cached: candidate 1 higher by 5e-7 (< TOL_PARITY) and first, a miss
    near = {"recon_cos": [0.2, 0.2, 0.3], "invariant_cos": [0.2, 0.2, 0.3],
            "edit_cos": [0.2, 0.2, 0.3], "base_score": [0.5, 0.5, 0.1],
            "verification_score": [exact[1], exact[1] + 5e-7, exact[2]]}
    _expect(B.ProtocolError, B.deployed_scores, [near], [{0}], [0], man)
    # a change of Hit@5 alone (top-1 unchanged) is rejected too: candidates 4
    # and 5 tie when recomputed (4 stays inside the top 5); cached, 5 is
    # higher by 5e-7 and pushes the relevant candidate 4 out
    man6 = dict(man, k=6)
    v = [0.6, 0.5, 0.4, 0.3, 0.2, 0.2]
    cal = [B.calibrate(x, 1.0, 0.0) for x in v]
    five = {"recon_cos": v, "invariant_cos": v, "edit_cos": v, "base_score": v,
            "verification_score": cal[:5] + [cal[5] + 5e-7]}
    _expect(B.ProtocolError, B.deployed_scores, [five], [{4}], [0], man6)


def test_fiq_rejects_mixed_checkpoints():
    """Audit B: categories must share one checkpoint digest."""
    with _Tmp() as d:
        loaded = []
        for i, cat in enumerate(("dress", "shirt", "toptee")):
            ck = CK_SHA if i < 2 else "b" * 64
            c, z = _build(d, cat, 20, i, "fashioniq", cat, ckpt=ck)
            loaded.append(B.load_pair(c, z))
        saved = dict(B.FIQ_EXPECTED)
        B.FIQ_EXPECTED.clear()
        B.FIQ_EXPECTED.update({"dress": 20, "shirt": 20, "toptee": 20})
        try:
            _expect(B.ProtocolError, B._validate_fiq_set, loaded)
        finally:
            B.FIQ_EXPECTED.clear()
            B.FIQ_EXPECTED.update(saved)


DEP = {d: "d" * 64 for d in B.DEPENDENCIES}
CIRR_IN = {"dataset": "cirr", "category": None, "cache_sha256": "1" * 64, "npz_sha256": "2" * 64}
FIQ_IN = [{"dataset": "fashioniq", "category": c, "cache_sha256": str(i + 3) * 64,
           "npz_sha256": str(i + 6) * 64} for i, c in enumerate(("dress", "shirt", "toptee"))]
PROV = {"result_schema": B.RESULT_SCHEMA, "script_sha256": "s", "dependency_sha256": DEP,
        "checkpoint_sha256": CK_SHA, "split_seed": 12345, "K": 10, "coef": 0.5, "lambda": 1.0,
        "n_bootstrap": 20, "vector_tolerance": dict(B.VECTOR_TOL), "cache_tolerance": B.TOL_CACHE}
GOOD = {"diff": 0.01, "ci_low": 0.0, "ci_high": 0.02, "p_two_sided": 0.04, "metric": "m"}


def _results(d, p1=None, p2=None, cirr_inputs=None, fiq_inputs=None):
    """Write a compatible CIRR/FashionIQ result pair; overrides break one aspect."""
    c, f = os.path.join(d, "c.json"), os.path.join(d, "f.json")
    with open(c, "w") as fh:
        json.dump({"provenance": dict(PROV, mode="cirr",
                                      inputs=[CIRR_IN] if cirr_inputs is None else cirr_inputs),
                   "P1": GOOD if p1 is None else p1}, fh)
    with open(f, "w") as fh:
        json.dump({"provenance": dict(PROV, mode="fiq",
                                      inputs=[CIRR_IN] + FIQ_IN if fiq_inputs is None else fiq_inputs),
                   "P2": GOOD if p2 is None else p2}, fh)
    return c, f


def test_decide_rejects_mismatched_or_malformed_inputs():
    """Audit 3/1: the CIRR source must be the same cache and vectors in both."""
    other_cache = dict(CIRR_IN, cache_sha256="9" * 64)
    other_npz = dict(CIRR_IN, npz_sha256="9" * 64)
    cases = [dict(fiq_inputs=[other_cache] + FIQ_IN),          # different source cache
             dict(fiq_inputs=[other_npz] + FIQ_IN),            # different source vectors
             dict(cirr_inputs=[]),                             # missing inputs
             dict(fiq_inputs=[CIRR_IN, CIRR_IN] + FIQ_IN),     # duplicated source entry
             dict(fiq_inputs=[CIRR_IN] + FIQ_IN[:2])]          # a category missing
    with _Tmp() as d:
        for kw in cases:
            c, f = _results(d, **kw)
            _expect(B.ProtocolError, B.decide, c, f)


def test_decide_rejects_invalid_statistics():
    """Audit 3/2: invalid p-values or intervals must not yield decisions."""
    bad = [dict(GOOD, p_two_sided=-0.2), dict(GOOD, p_two_sided=1.5),
           dict(GOOD, p_two_sided=float("nan")), dict(GOOD, p_two_sided=True),
           {k: v for k, v in GOOD.items() if k != "ci_low"},
           dict(GOOD, ci_low=0.03, ci_high=0.01)]
    with _Tmp() as d:
        for t in bad:
            c, f = _results(d, p1=t)
            _expect(B.ProtocolError, B.decide, c, f)


def test_decide_rejects_incompatible_results():
    """Settings, including the tolerance policy the inputs were accepted
    under, must agree between the two results."""
    for key, value in (("K", 20), ("vector_tolerance", dict(B.VECTOR_TOL, base_score=1e-2)),
                       ("cache_tolerance", 1e-3)):
        with _Tmp() as d:
            c, f = _results(d)
            B.decide(c, f)                                 # compatible as written
            with open(f) as fh:
                res = json.load(fh)
            res["provenance"][key] = value
            with open(f, "w") as fh:
                json.dump(res, fh)
            _expect(B.ProtocolError, B.decide, c, f)


def test_decide_reports_missing_or_malformed_inputs_cleanly():
    """A missing or unreadable result file is an ERROR line and exit 1, not a traceback."""
    import contextlib
    import io
    with _Tmp() as d:
        c, f = _results(d)
        with open(f, "w") as fh:
            fh.write("{not json")
        for args in (["decide", os.path.join(d, "absent.json"), f], ["decide", c, f]):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                assert B.main(args) == 1
            assert err.getvalue().startswith("ERROR:"), err.getvalue()


def test_decide_applies_holm():
    with _Tmp() as d:
        c, f = _results(d, p1=dict(GOOD, p_two_sided=0.01), p2=dict(GOOD, p_two_sided=0.04))
        out = B.decide(c, f)
        assert math.isclose(out["P1"]["p_holm"], 0.02) and out["P1"]["rejected"]
        assert math.isclose(out["P2"]["p_holm"], 0.04) and out["P2"]["rejected"]
        # step-down monotonicity: p = 0.03, 0.04 -> adjusted 0.06, 0.06 (not 0.04)
        c, f = _results(d, p1=dict(GOOD, p_two_sided=0.03), p2=dict(GOOD, p_two_sided=0.04))
        out = B.decide(c, f)
        assert math.isclose(out["P1"]["p_holm"], 0.06) and not out["P1"]["rejected"]
        assert math.isclose(out["P2"]["p_holm"], 0.06) and not out["P2"]["rejected"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok  " + f.__name__)
    print("\n%d tests passed" % len(fns))
