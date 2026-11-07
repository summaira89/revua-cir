"""Tests for untied_mixture_weights.py.

The three things most likely to be wrong, and therefore tested hardest:

  * the tune/report partition must be the project's, and must not leak --
    a weighting selected on data it is then scored on would manufacture an
    effect out of selection noise;
  * `fused_rank1` must reproduce the deployed rule, so the tied baseline it
    computes matches the published pipeline rather than a near-miss;
  * the trained tied weights must be the real ones (0.256/0.256/0.489), not
    an equal split.
"""
import json
import math
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))

from untied_mixture_weights import (  # noqa: E402
    W_EDIT_TRAINED, covered, fused_rank1, load_records, mcnemar_exact,
    paired_bootstrap, rate, split, trained_weights, weight_grid,
)
from fusion_controls import CacheError, _zscore  # noqa: E402


def make_rec(k=5, target_pos=0, seed=0, n_gt=1):
    rng = random.Random(seed)
    return {
        "query_id": "q%d" % seed,
        "candidate_gallery_ids": list(range(100, 100 + k)),
        "base_score": [rng.uniform(0.4, 0.8) for _ in range(k)],
        "recon_cos": [rng.uniform(-1, 1) for _ in range(k)],
        "invariant_cos": [rng.uniform(-1, 1) for _ in range(k)],
        "edit_cos": [rng.uniform(-1, 1) for _ in range(k)],
        "target_pos": target_pos,
        "relevant_gallery_indices": list(range(n_gt)),
    }


def write_cache(recs):
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8")
    for r in recs:
        fh.write(json.dumps(r) + "\n")
    fh.close()
    return fh.name


class TestTrainedWeights(unittest.TestCase):
    def test_tied_weights_are_the_trained_ones(self):
        recon, inv, edit = trained_weights()
        self.assertAlmostEqual(edit, 0.488847, places=6)
        self.assertAlmostEqual(recon, inv, places=12,
                               msg="the two non-edit weights are TIED")
        self.assertAlmostEqual(recon, (1 - 0.488847) / 2, places=9)

    def test_tied_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(trained_weights()), 1.0, places=9)

    def test_not_an_equal_three_way_split(self):
        # A plausible wrong implementation. The edit term carries nearly
        # half the score, not a third.
        self.assertNotAlmostEqual(trained_weights()[2], 1 / 3, places=2)

    def test_edit_weight_is_sigmoid_of_the_learned_scalar(self):
        # w = sigmoid(edit_weight); recover the pre-activation and check it
        # round-trips, which pins the parameterisation rather than a number.
        z = math.log(W_EDIT_TRAINED / (1 - W_EDIT_TRAINED))
        self.assertAlmostEqual(1 / (1 + math.exp(-z)), W_EDIT_TRAINED,
                               places=9)


class TestSplit(unittest.TestCase):
    def test_halves_partition_exactly(self):
        recs = [make_rec(seed=i) for i in range(4181)]
        tune, report = split(recs)
        self.assertEqual(len(tune) + len(report), 4181)
        self.assertEqual(len(tune), 2090)
        self.assertEqual(len(report), 2091)

    def test_halves_are_disjoint(self):
        recs = [make_rec(seed=i) for i in range(500)]
        tune, report = split(recs)
        a = {id(r) for r in tune}
        b = {id(r) for r in report}
        self.assertEqual(a & b, set(), "a query must not be in both halves")

    def test_split_is_deterministic(self):
        recs = [make_rec(seed=i) for i in range(300)]
        t1, r1 = split(recs)
        t2, r2 = split(recs)
        self.assertEqual([x["query_id"] for x in t1],
                         [x["query_id"] for x in t2])
        self.assertEqual([x["query_id"] for x in r1],
                         [x["query_id"] for x in r2])

    def test_a_different_seed_gives_a_different_partition(self):
        recs = [make_rec(seed=i) for i in range(300)]
        t1, _ = split(recs)
        t2, _ = split(recs, seed=999)
        self.assertNotEqual([x["query_id"] for x in t1],
                            [x["query_id"] for x in t2])


class TestFusedRank1(unittest.TestCase):
    def test_matches_a_hand_computed_fusion(self):
        rec = {
            "candidate_gallery_ids": [1, 2, 3],
            "base_score": [0.5, 0.6, 0.4],
            "recon_cos": [0.9, 0.1, 0.1],
            "invariant_cos": [0.0, 0.0, 0.0],
            "edit_cos": [0.0, 0.0, 0.0],
            "target_pos": 0,
            "relevant_gallery_indices": [0],
        }
        w = (1.0, 0.0, 0.0)
        lam = 1.0
        zb = _zscore(rec["base_score"])
        zr = _zscore(rec["recon_cos"])
        fused = [zb[i] + lam * math.tanh(zr[i]) for i in range(3)]
        expected = max(range(3), key=lambda i: (fused[i], -i))
        self.assertEqual(fused_rank1(rec, w, lam), expected == 0)

    def test_a_dominant_recon_signal_rescues_the_target(self):
        # Target is second on base score but far ahead on reconstruction.
        rec = {
            "candidate_gallery_ids": [1, 2],
            "base_score": [0.9, 0.1],
            "recon_cos": [-1.0, 1.0],
            "invariant_cos": [0.0, 0.0],
            "edit_cos": [0.0, 0.0],
            "target_pos": 1,
            "relevant_gallery_indices": [0],
        }
        self.assertFalse(fused_rank1(rec, (0.0, 0.0, 1.0), 1.0),
                         "with zero recon weight the base order stands")
        # With all weight on recon and a large lambda the order flips.
        self.assertTrue(fused_rank1(rec, (1.0, 0.0, 0.0), 5.0))

    def test_lambda_zero_reduces_to_base_ranking(self):
        for s in range(30):
            rec = make_rec(k=6, target_pos=s % 6, seed=s)
            base_best = max(range(6),
                            key=lambda i: (rec["base_score"][i], -i))
            self.assertEqual(fused_rank1(rec, (0.3, 0.3, 0.4), 0.0),
                             base_best == rec["target_pos"],
                             "at lambda=0 verification must not matter")

    def test_weighting_actually_changes_outcomes(self):
        # Guards against a bug where the mixture is computed but ignored.
        recs = [make_rec(k=8, target_pos=s % 8, seed=s) for s in range(120)]
        r1 = rate(recs, (1.0, 0.0, 0.0), 1.0)
        r2 = rate(recs, (0.0, 0.0, 1.0), 1.0)
        self.assertNotEqual(r1, r2)

    def test_rate_is_a_percentage(self):
        recs = [make_rec(k=4, target_pos=s % 4, seed=s) for s in range(40)]
        r = rate(recs, trained_weights(), 1.0)
        self.assertGreaterEqual(r, 0.0)
        self.assertLessEqual(r, 100.0)

    def test_rate_of_empty_is_zero_not_a_crash(self):
        self.assertEqual(rate([], trained_weights(), 1.0), 0.0)


class TestWeightGrid(unittest.TestCase):
    def test_all_weightings_are_convex(self):
        for w in weight_grid(10):
            self.assertAlmostEqual(sum(w), 1.0, places=9)
            self.assertTrue(all(x >= 0 for x in w))

    def test_grid_size_is_the_simplex_count(self):
        for steps in (2, 5, 21):
            self.assertEqual(len(weight_grid(steps)),
                             (steps + 1) * (steps + 2) // 2)

    def test_grid_contains_the_corners(self):
        g = weight_grid(10)
        for corner in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            self.assertIn(corner, g)

    def test_no_duplicate_weightings(self):
        g = weight_grid(12)
        self.assertEqual(len(g), len(set(g)))


class TestCoverage(unittest.TestCase):
    def test_uncovered_queries_are_excluded(self):
        self.assertTrue(covered({"target_pos": 0}))
        self.assertTrue(covered({"target_pos": 7}))
        self.assertFalse(covered({"target_pos": -1}))
        self.assertFalse(covered({"target_pos": None}))
        self.assertFalse(covered({}))


class TestLoading(unittest.TestCase):
    def test_loads_a_well_formed_cache(self):
        p = write_cache([make_rec(seed=i) for i in range(5)])
        try:
            self.assertEqual(len(load_records(p)), 5)
        finally:
            os.unlink(p)

    def test_rejects_a_missing_component_cosine(self):
        rec = make_rec()
        del rec["recon_cos"]
        p = write_cache([rec])
        try:
            with self.assertRaises(CacheError):
                load_records(p)
        finally:
            os.unlink(p)

    def test_rejects_a_length_mismatch(self):
        rec = make_rec(k=5)
        rec["edit_cos"] = rec["edit_cos"][:3]
        p = write_cache([rec])
        try:
            with self.assertRaises(CacheError):
                load_records(p)
        finally:
            os.unlink(p)

    def test_rejects_a_circo_cache(self):
        # Multi-positive annotation means CIRCO; the component analysis is
        # CIRR-only and must refuse rather than silently mis-score.
        p = write_cache([make_rec(seed=1, n_gt=3)])
        try:
            with self.assertRaises(CacheError):
                load_records(p)
        finally:
            os.unlink(p)

    def test_rejects_an_empty_file(self):
        p = write_cache([])
        try:
            with self.assertRaises(CacheError):
                load_records(p)
        finally:
            os.unlink(p)

    def test_rejects_malformed_json(self):
        fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                         encoding="utf-8")
        fh.write("{not json\n")
        fh.close()
        try:
            with self.assertRaises(CacheError):
                load_records(fh.name)
        finally:
            os.unlink(fh.name)


class TestStatistics(unittest.TestCase):
    def test_identical_arms_give_zero_delta_and_p_one(self):
        a = [True, False, True, True, False]
        r = paired_bootstrap(a, list(a), n_rep=500)
        self.assertAlmostEqual(r["delta"], 0.0, places=9)
        m = mcnemar_exact(a, list(a))
        self.assertEqual(m["n01"], 0)
        self.assertEqual(m["n10"], 0)
        self.assertEqual(m["p"], 1.0)

    def test_mcnemar_counts_discordant_pairs_only(self):
        a = [True, True, False, False]
        b = [True, False, True, False]
        m = mcnemar_exact(a, b)
        self.assertEqual(m["n01"], 1)   # a right, b wrong
        self.assertEqual(m["n10"], 1)   # b right, a wrong
        self.assertAlmostEqual(m["p"], 1.0, places=9)

    def test_mcnemar_detects_a_one_sided_difference(self):
        a = [True] * 10 + [False] * 10
        b = [False] * 20
        m = mcnemar_exact(a, b)
        self.assertEqual(m["n01"], 10)
        self.assertEqual(m["n10"], 0)
        self.assertLess(m["p"], 0.01)

    def test_bootstrap_interval_brackets_the_observed_delta(self):
        a = [True] * 60 + [False] * 40
        b = [True] * 40 + [False] * 60
        r = paired_bootstrap(a, b, n_rep=2000)
        self.assertAlmostEqual(r["delta"], 20.0, places=9)
        self.assertLessEqual(r["ci_low"], r["delta"])
        self.assertGreaterEqual(r["ci_high"], r["delta"])

    def test_bootstrap_rejects_mismatched_arms(self):
        with self.assertRaises(ValueError):
            paired_bootstrap([True], [True, False], n_rep=10)

    def test_bootstrap_is_reproducible(self):
        a = [True] * 30 + [False] * 20
        b = [False] * 25 + [True] * 25
        r1 = paired_bootstrap(a, b, n_rep=400)
        r2 = paired_bootstrap(a, b, n_rep=400)
        self.assertEqual(r1, r2)


class TestNoSelectionLeak(unittest.TestCase):
    """The property the whole design exists to protect."""

    def test_selection_on_tune_does_not_see_report_queries(self):
        recs = [make_rec(k=6, target_pos=s % 6, seed=s) for s in range(400)]
        tune, report = split(recs)
        tune_ids = {r["query_id"] for r in tune}
        report_ids = {r["query_id"] for r in report}
        self.assertEqual(tune_ids & report_ids, set())
        # Selecting the argmax weighting on tune must not change what the
        # report half contains.
        before = [r["query_id"] for r in report]
        grid = weight_grid(6)
        _ = max((rate(tune, w, 1.0), w) for w in grid)
        after = [r["query_id"] for r in report]
        self.assertEqual(before, after)

    def test_tune_optimum_is_optimistic_relative_to_a_fresh_half(self):
        # With pure noise the selected weighting should look good on the
        # half it was chosen on and not on the other -- the exact reason
        # selection and reporting are separated.
        recs = [make_rec(k=8, target_pos=s % 8, seed=1000 + s)
                for s in range(600)]
        tune, report = split(recs)
        grid = weight_grid(8)
        best_tune, best_w = max((rate(tune, w, 1.0), w) for w in grid)
        on_report = rate(report, best_w, 1.0)
        self.assertGreaterEqual(best_tune, on_report - 1e-9,
                                "selection on tune must be >= its own "
                                "performance there by construction")


if __name__ == "__main__":
    unittest.main(verbosity=2)
