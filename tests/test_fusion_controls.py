"""Regression tests for fusion_controls.

The claims this script is meant to support are comparative, so the dangerous
failure is a rule implemented slightly differently from the deployed one, or
a control that destroys more than it should. Both are targeted below:
`test_rank_rule_matches_hand_traced_rank_fuse_formula` pins the rank rule
against a hand-traced copy of the formula in revua_cir.py (not an import of
the production model), and the shuffle tests check that the within-window
permutation preserves each query's score multiset exactly.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

try:
    import pytest
except ImportError:  # pragma: no cover - cluster venv without pytest
    class _Approx:
        def __init__(self, expected, abs=None, rel=None):
            self.expected, self.abs, self.rel = expected, abs, rel

        def __eq__(self, other):
            tol = self.abs if self.abs is not None else max(
                1e-12, (self.rel if self.rel is not None else 1e-6)
                * abs(self.expected))
            return abs(other - self.expected) <= tol

    class Skipped(Exception):
        """Named like pytest's own skip outcome, which the runner below matches."""

    class _RaisesCtx:
        def __init__(self, exc, match=None):
            self.exc, self.match = exc, match

        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            if et is None:
                raise AssertionError("DID NOT RAISE %s" % (self.exc,))
            if not issubclass(et, self.exc):
                return False
            if self.match is not None:
                import re as _re
                if not _re.search(self.match, str(ev)):
                    raise AssertionError("pattern %r not in %r"
                                         % (self.match, str(ev)))
            return True

    class _PytestShim:
        Skipped = Skipped

        @staticmethod
        def approx(expected, abs=None, rel=None):
            return _Approx(expected, abs=abs, rel=rel)

        @staticmethod
        def raises(exc, match=None):
            return _RaisesCtx(exc, match=match)

        @staticmethod
        def skip(reason=""):
            raise Skipped(reason)

    pytest = _PytestShim()  # type: ignore[assignment]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import fusion_controls as fc  # noqa: E402


def make_record(qid, base, rv, target_pos=None, relevant=None):
    n = len(base)
    rec = {"query_id": qid,
           "candidate_gallery_ids": list(range(qid * 100, qid * 100 + n)),
           "base_score": list(base),
           "verification_score": list(rv)}
    if target_pos is not None:
        rec["target_pos"] = target_pos
    if relevant is not None:
        rec["relevant_positions_in_candidates"] = list(relevant)
    return rec


def write_cache(records):
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8")
    for r in records:
        fh.write(json.dumps(r) + "\n")
    fh.close()
    return fh.name


# ---------------------------------------------------------------------------
# the rank rule must match production exactly
# ---------------------------------------------------------------------------

def test_rank_rule_matches_hand_traced_rank_fuse_formula():
    """Hand-traced against revua_cir.py's `_rank_fuse`:

        base_rank = argsort(desc).argsort()      # 0 = best
        adj_rank  = argsort(desc).argsort()      # 0 = best
        fused     = -(2*base_rank + adj_rank)

    base [0.9, 0.5, 0.7] -> ranks [0, 2, 1]
    rv   [0.1, 0.9, 0.5] -> ranks [2, 0, 1]
    fused = -[2*0+2, 2*2+0, 2*1+1] = [-2, -4, -3]
    """
    got = fc.rule_rank([0.9, 0.5, 0.7], [0.1, 0.9, 0.5], lam=0.0)
    assert got == [pytest.approx(-2.0), pytest.approx(-4.0),
                   pytest.approx(-3.0)]


def test_rank_rule_ignores_lambda():
    a = fc.rule_rank([0.9, 0.5], [0.1, 0.9], lam=0.0)
    b = fc.rule_rank([0.9, 0.5], [0.1, 0.9], lam=7.5)
    assert a == b


def test_ranks_desc_is_zero_based_best_first():
    assert fc._ranks_desc([0.9, 0.5, 0.7]) == [0, 2, 1]
    assert fc._ranks_desc([0.1, 0.1, 0.1]) == [0, 1, 2]  # ties by index


# ---------------------------------------------------------------------------
# the other rules
# ---------------------------------------------------------------------------

def test_base_rule_returns_base_unchanged():
    b = [0.9, 0.5, 0.7]
    assert fc.rule_base(b, [0.1, 0.2, 0.3], lam=3.0) == b


def test_bounded_at_lambda_zero_equals_zscored_base():
    b, r = [0.9, 0.5, 0.7], [0.1, 0.9, 0.5]
    assert fc.rule_bounded(b, r, 0.0) == [pytest.approx(v)
                                          for v in fc._zscore(b)]


def test_bounded_contribution_is_within_plus_minus_lambda():
    """The property the manuscript's corrected bound rests on."""
    b, r = [0.9, 0.5, 0.7, 0.2], [0.1, 0.9, 0.5, 0.3]
    lam = 2.0
    zb = fc._zscore(b)
    fused = fc.rule_bounded(b, r, lam)
    for i in range(len(b)):
        assert abs(fused[i] - zb[i]) < lam


def test_linear_has_no_bound():
    """The contrast with bounded: a large z can exceed lambda."""
    b = [0.9] + [0.5] * 9
    r = [9.0] + [0.0] * 9          # one extreme verification score
    lam = 1.0
    zb = fc._zscore(b)
    lin = fc.rule_linear(b, r, lam)
    assert abs(lin[0] - zb[0]) > lam


def test_zscore_matches_the_other_scripts():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                    "scripts"))
    import multipositive_analysis as mp
    xs = [0.63, 0.60, 0.59, 0.55]
    assert fc._zscore(xs) == [pytest.approx(v) for v in mp._zscore(xs)]


def test_zscore_constant_input_is_zeros():
    for v in fc._zscore([0.5, 0.5, 0.5]):
        assert math.isfinite(v) and v == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

def test_within_window_shuffle_preserves_each_querys_multiset():
    """The defining property of the primary control: the distribution is
    untouched, only the pairing changes."""
    recs = [make_record(i, [0.9, 0.5, 0.7, 0.2],
                        [0.11, 0.22, 0.33, 0.44], target_pos=0)
            for i in range(20)]
    out = fc.shuffled_within_window(recs, seed=7)
    for a, b in zip(recs, out):
        assert sorted(a["verification_score"]) == \
            sorted(b["verification_score"])


def test_within_window_shuffle_actually_changes_some_order():
    recs = [make_record(i, [0.9, 0.5, 0.7, 0.2],
                        [0.11, 0.22, 0.33, 0.44], target_pos=0)
            for i in range(40)]
    out = fc.shuffled_within_window(recs, seed=7)
    changed = sum(1 for a, b in zip(recs, out)
                  if a["verification_score"] != b["verification_score"])
    assert changed > 0, "shuffle left every window untouched"


def test_within_window_shuffle_does_not_touch_base_scores():
    recs = [make_record(i, [0.9, 0.5, 0.7], [0.1, 0.2, 0.3], target_pos=1)
            for i in range(10)]
    out = fc.shuffled_within_window(recs, seed=3)
    for a, b in zip(recs, out):
        assert a["base_score"] == b["base_score"]
        assert a["candidate_gallery_ids"] == b["candidate_gallery_ids"]
        assert a.get("target_pos") == b.get("target_pos")


def test_shuffle_is_deterministic_for_a_seed():
    recs = [make_record(i, [0.9, 0.5, 0.7], [0.1, 0.2, 0.3], target_pos=0)
            for i in range(10)]
    a = fc.shuffled_within_window(recs, seed=11)
    b = fc.shuffled_within_window(recs, seed=11)
    assert [r["verification_score"] for r in a] == \
        [r["verification_score"] for r in b]


def test_shuffle_does_not_mutate_the_input():
    recs = [make_record(0, [0.9, 0.5, 0.7], [0.1, 0.2, 0.3], target_pos=0)]
    before = list(recs[0]["verification_score"])
    fc.shuffled_within_window(recs, seed=5)
    assert recs[0]["verification_score"] == before


def test_noise_control_stays_in_the_observed_range():
    recs = [make_record(i, [0.9, 0.5], [0.40, 0.60], target_pos=0)
            for i in range(30)]
    out = fc.noise_verifier(recs, seed=9)
    allv = [s for r in out for s in r["verification_score"]]
    assert min(allv) >= 0.40 - 1e-9 and max(allv) <= 0.60 + 1e-9


# ---------------------------------------------------------------------------
# the split and tuning
# ---------------------------------------------------------------------------

def test_split_partitions_exactly_and_is_deterministic():
    tune, rep = fc.split_indices(4181)
    assert len(tune) == 2090 and len(rep) == 2091
    assert set(tune) & set(rep) == set()
    assert sorted(tune + rep) == list(range(4181))
    assert (tune, rep) == fc.split_indices(4181)


def test_parameterless_rules_get_lambda_zero():
    recs = [make_record(i, [0.9, 0.5], [0.1, 0.2], target_pos=0)
            for i in range(10)]
    rel = [{0} for _ in recs]
    lam = fc.tune_lambda(recs, rel, fc.rule_rank, False, list(range(10)))
    assert lam == 0.0


def test_tuning_picks_the_lambda_that_maximises_tune_hits():
    """Constructed so a large lambda is needed: the verifier knows the
    answer and the base does not."""
    recs = []
    for i in range(40):
        recs.append(make_record(i, [0.90, 0.89, 0.88], [0.1, 0.9, 0.2],
                                target_pos=1))
    rel = [{1} for _ in recs]
    idx = list(range(40))
    lam = fc.tune_lambda(recs, rel, fc.rule_bounded, True, idx)
    assert lam > 0.0
    hits, _ = fc.evaluate(recs, rel, fc.rule_bounded, lam, idx)
    assert hits == 40


def test_tuning_ties_break_toward_smaller_lambda():
    """When several lambdas tie, the conservative choice leans less on the
    verifier. Constructed so the verifier is uninformative, making every
    lambda equivalent."""
    recs = [make_record(i, [0.9, 0.5, 0.2], [0.3, 0.3, 0.3], target_pos=0)
            for i in range(20)]
    rel = [{0} for _ in recs]
    lam = fc.tune_lambda(recs, rel, fc.rule_bounded, True, list(range(20)))
    assert lam == fc.LAMBDA_GRID[0]


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def test_paired_bootstrap_identical_arms_gives_zero():
    a = [1, 0, 1, 1, 0] * 20
    r = fc.paired_bootstrap(a, list(a), n_rep=200)
    assert r["delta_pp"] == pytest.approx(0.0)
    assert r["ci_low"] == pytest.approx(0.0)
    assert r["ci_high"] == pytest.approx(0.0)


def test_paired_bootstrap_detects_a_real_difference():
    a = [1] * 100 + [0] * 20
    b = [0] * 100 + [0] * 20
    r = fc.paired_bootstrap(a, b, n_rep=500)
    assert r["delta_pp"] > 0 and r["ci_low"] > 0 and r["p_two_sided"] < 0.05


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="differ in length"):
        fc.paired_bootstrap([1, 0], [1], n_rep=10)


def test_mcnemar_counts_discordant_pairs_only():
    a = [1, 1, 0, 0, 1]
    b = [0, 1, 1, 0, 1]
    r = fc.mcnemar_exact(a, b)
    assert r["fixed"] == 1      # a=1,b=0 at index 0
    assert r["broken"] == 1     # a=0,b=1 at index 2


def test_mcnemar_identical_arms_is_p_one():
    a = [1, 0, 1, 1]
    r = fc.mcnemar_exact(a, list(a))
    assert r["fixed"] == 0 and r["broken"] == 0
    assert r["p_exact"] == pytest.approx(1.0)


def test_mcnemar_matches_a_hand_computed_case():
    """n01=5, n10=0, n=5 -> two-sided exact p = 2 * (1/2)^5 = 0.0625."""
    a = [1] * 5 + [1, 0]
    b = [0] * 5 + [1, 0]
    r = fc.mcnemar_exact(a, b)
    assert r["fixed"] == 5 and r["broken"] == 0
    assert r["p_exact"] == pytest.approx(0.0625)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_analyse_runs_and_reports_every_rule_and_variant():
    recs = [make_record(i, [0.90, 0.70, 0.50], [0.2, 0.9, 0.1],
                        target_pos=1 if i % 3 == 0 else 0)
            for i in range(60)]
    path = write_cache(recs)
    try:
        res = fc.analyse(path, n_rep=200)
        for v in ("real", "shuffled", "noise"):
            for r in ("base", "bounded", "linear", "rank"):
                assert r in res["results"][v]
        assert "bounded vs rank" in res["contrasts"]
        assert "bounded vs linear" in res["contrasts"]
    finally:
        os.unlink(path)


def test_control_contrast_uses_the_reported_median_permutation():
    """The paired contrast for a control must be computed on the same
    permutation whose R@1 the table reports (the median one). Using an
    arbitrary draw made the two columns disagree (36.44 vs -0.33)."""
    import random as _r
    rng = _r.Random(3)
    recs = [make_record(i, sorted([rng.random() for _ in range(6)], reverse=True),
                        [rng.random() for _ in range(6)], target_pos=rng.randrange(6))
            for i in range(120)]
    path = write_cache(recs)
    try:
        res = fc.analyse(path, n_rep=100)
        base = res["results"]["real"]["base"]["hit1_pct"]
        for rname in ("bounded", "rank"):
            for v in ("shuffled", "noise"):
                med = res["results"][v][rname]["hit1_pct"]
                c = res["control_contrasts"]["%s / %s vs base" % (v, rname)]
                assert abs(c["delta_pp"] - (med - base)) < 1e-9, (v, rname)
    finally:
        os.unlink(path)


def test_control_delta_range_is_consistent():
    import random as _r
    rng = _r.Random(5)
    recs = [make_record(i, sorted([rng.random() for _ in range(6)], reverse=True),
                        [rng.random() for _ in range(6)], target_pos=rng.randrange(6))
            for i in range(90)]
    path = write_cache(recs)
    try:
        res = fc.analyse(path, n_rep=50)
        base = res["results"]["real"]["base"]["hit1_pct"]
        for v in ("shuffled", "noise"):
            for r in ("bounded", "rank"):
                d = res["results"][v][r]
                assert d["delta_min"] <= d["delta_median"] <= d["delta_max"]
                assert abs(d["delta_median"] - (d["hit1_pct"] - base)) < 1e-9
    finally:
        os.unlink(path)


def _raw_cache(lines):
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    fh.write("\n".join(lines) + "\n")
    fh.close()
    return fh.name


def test_validator_rejects_malformed_positions_and_records():
    import json as _json
    good = {"candidate_gallery_ids": [7, 8, 9], "base_score": [0.9, 0.8, 0.7],
            "verification_score": [0.5, 0.6, 0.4], "target_pos": 1}
    bad_records = [
        dict(good, target_pos=0.5),                          # fractional position
        dict(good, target_pos=True),                         # bool equals 1 in Python
        dict(good, candidate_gallery_ids=[7, 8.0, 9]),       # non-integer id
        dict(good, candidate_gallery_ids=[], base_score=[], verification_score=[]),
    ]
    for rec in bad_records:
        path = _raw_cache([_json.dumps(rec)])
        try:
            with pytest.raises(fc.CacheError):
                fc.load_records(path)
        finally:
            os.unlink(path)
    path = _raw_cache(["[1, 2, 3]"])                         # not an object
    try:
        with pytest.raises(fc.CacheError):
            fc.load_records(path)
    finally:
        os.unlink(path)
    nulled = dict(good, relevant_positions_in_candidates=None)
    path = _raw_cache([_json.dumps(nulled)])                  # null multi-positive field
    try:
        with pytest.raises(fc.CacheError):
            fc.load_records(path)
    finally:
        os.unlink(path)
    multi = dict(good, relevant_positions_in_candidates=[1, 1])
    path = _raw_cache([_json.dumps(multi)])                   # duplicate positives
    try:
        with pytest.raises(fc.CacheError):
            fc.load_records(path)
    finally:
        os.unlink(path)


def test_mcnemar_rejects_unequal_or_nonbinary_arms():
    with pytest.raises(ValueError):
        fc.mcnemar_exact([1, 1], [0])
    with pytest.raises(ValueError):
        fc.mcnemar_exact([1, 2], [0, 1])


def test_base_rule_is_identical_across_all_three_variants():
    """The controls alter only R, so the base rule's score cannot move. If
    it does, a control is corrupting something it should not touch."""
    recs = [make_record(i, [0.90, 0.70, 0.50], [0.2, 0.9, 0.1],
                        target_pos=i % 3)
            for i in range(60)]
    path = write_cache(recs)
    try:
        res = fc.analyse(path, n_rep=100)
        vals = {v: res["results"][v]["base"]["hit1_pct"]
                for v in ("real", "shuffled", "noise")}
        assert len(set(round(x, 9) for x in vals.values())) == 1, vals
    finally:
        os.unlink(path)


def test_report_runs(capsys):
    recs = [make_record(i, [0.9, 0.7, 0.5], [0.2, 0.9, 0.1], target_pos=1)
            for i in range(30)]
    path = write_cache(recs)
    try:
        res = fc.analyse(path, n_rep=100)
        fc.report(res)
        out = capsys.readouterr().out
        assert "FUSION-RULE CONTROLS" in out
        assert "CONTROL CONTRASTS" in out
    finally:
        os.unlink(path)


def test_main_rejects_a_bad_cache(capsys):
    path = write_cache([make_record(0, [0.9, 0.5], [0.1, 0.2],
                                    target_pos=42)])
    try:
        assert fc.main([path]) == 1
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# real cache
# ---------------------------------------------------------------------------

def _find_circo():
    for p in [os.path.join(os.path.dirname(__file__), "..", "..",
                           "rv_score_cache_circo_fc_aep_rv.jsonl"),
              os.path.join(os.path.dirname(__file__), "..",
                           "rv_score_cache_circo_fc_aep_rv.jsonl")]:
        if os.path.exists(p):
            return p
    return None


def test_real_cache_base_rule_reproduces_the_known_rate():
    """Anchors this script to the rest of the project: the base rule over
    the whole CIRCO cache must give the published 31.818%."""
    path = _find_circo()
    if path is None:
        pytest.skip("CIRCO cache not present")
    recs, multi = fc.load_records(path)
    rel = [fc.relevant_set(r, multi) for r in recs]
    hits, _ = fc.evaluate(recs, rel, fc.rule_base, 0.0, list(range(len(recs))))
    assert 100.0 * hits / len(recs) == pytest.approx(31.818, abs=0.01)


def test_real_cache_shuffle_preserves_every_distribution():
    path = _find_circo()
    if path is None:
        pytest.skip("CIRCO cache not present")
    recs, _ = fc.load_records(path)
    out = fc.shuffled_within_window(recs, seed=1)
    for a, b in zip(recs, out):
        assert sorted(a["verification_score"]) == \
            sorted(b["verification_score"])


if __name__ == "__main__":
    import traceback
    fns = [(n, o) for n, o in sorted(globals().items())
           if n.startswith("test_") and callable(o)]
    failed = skipped = 0
    for name, fn in fns:
        try:
            if fn.__code__.co_argcount:
                print("  [skip] %s (needs a pytest fixture)" % name)
                skipped += 1
                continue
            fn()
            print("  [ok] %s" % name)
        except Exception as exc:  # noqa: BLE001
            if exc.__class__.__name__ == "Skipped":
                print("  [skip] %s (%s)" % (name, exc))
                skipped += 1
                continue
            failed += 1
            print("  [FAIL] %s: %s" % (name, exc))
            traceback.print_exc()
    print()
    if failed:
        print("%d TEST(S) FAILED." % failed)
        raise SystemExit(1)
    print("ALL FUSION-CONTROL TESTS PASSED (%d run, %d skipped)." % (len(fns) - skipped, skipped))
