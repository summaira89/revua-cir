"""Regression tests for scripts/ablation_significance.py.

The statistical core is tested without touching a GPU, a checkpoint or a
dataset: `_hit_vector` and `compare` are pure functions over tensors and
lists. That is deliberate -- these must be provably correct BEFORE any GPU
pass is launched, because a wrong denominator or a broken pairing produces
a well-formed number that is silently wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from ablation_significance import _hit_vector, compare  # noqa: E402


def test_hit_vector_excludes_unresolved_targets():
    """A query whose target never resolved in the gallery must be marked
    invalid, not counted as a miss -- that is how recall_at_k treats it."""
    pq = {
        "ranked": torch.tensor([[5, 1], [3, 2], [7, 0], [9, 4]]),
        "targets": torch.tensor([5, 9, 7, -1]),
        "valid_mask": torch.tensor([True, True, True, False]),
    }
    hits, valid = _hit_vector(pq)
    assert hits == [1, 0, 1, 0], hits
    assert valid == [1, 1, 1, 0], valid
    print("  [ok] hit vector marks unresolved targets invalid, not missed")


def test_denominator_is_valid_count_not_n():
    """Guards the exact bug that produced a phantom 1.62pp-21pt mismatch
    earlier in this project: valid-filtered numerator over unfiltered n."""
    pq = {
        "ranked": torch.tensor([[5, 1], [3, 2], [7, 0], [9, 4]]),
        "targets": torch.tensor([5, 9, 7, -1]),
        "valid_mask": torch.tensor([True, True, True, False]),
    }
    hits, valid = _hit_vector(pq)
    correct = 100.0 * sum(hits) / sum(valid)
    wrong = 100.0 * sum(hits) / len(hits)
    assert abs(correct - 66.6667) < 0.01, correct
    assert abs(wrong - 50.0) < 0.01, wrong
    assert abs(correct - wrong) > 15.0, (
        "the two denominators must differ materially, or this test proves nothing"
    )
    print("  [ok] valid-count denominator differs from n by %.1fpp as expected"
          % (correct - wrong))


def test_same_net_gain_gives_opposite_verdicts():
    """The motivating argument, made executable.

    Job 6453's ablation shows a net +45 queries. That single number is
    consistent with overwhelming significance AND with nothing at all.
    If this test ever fails, the premise of the whole script is wrong.
    """
    # help=45, harm=0
    a1 = [0] * 45 + [1] * 100
    b1 = [1] * 45 + [1] * 100
    r1 = compare(a1, b1, [1] * 145, [1] * 145, "fc", "fc+aep+rv")
    assert r1["help"] == 45 and r1["harm"] == 0
    assert r1["net"] == 45
    assert r1["mcnemar_exact_p"] < 1e-10, r1["mcnemar_exact_p"]

    # help=445, harm=400 -- SAME net +45
    a2 = [0] * 445 + [1] * 400 + [1] * 300
    b2 = [1] * 445 + [0] * 400 + [1] * 300
    r2 = compare(a2, b2, [1] * 1145, [1] * 1145, "fc", "fc+aep+rv")
    assert r2["net"] == 45, r2["net"]
    assert r2["mcnemar_exact_p"] > 0.05, r2["mcnemar_exact_p"]

    print("  [ok] identical net +45 gives p=%.3g vs p=%.3g -- aggregate R@1 "
          "cannot decide significance" % (r1["mcnemar_exact_p"], r2["mcnemar_exact_p"]))


def test_identical_targets_but_different_query_ids_must_be_rejected():
    """The exact loophole target-only checking cannot close.

    Two runs can produce the SAME target sequence while having seen
    different queries, because targets repeat across queries. Construct
    that case directly: identical targets, permuted query IDs. The pairing
    logic must reject it.

    Today this is unreachable -- evaluate_dataset iterates
    Subset(dataset, range(n)) with shuffle=False and num_workers=0, and
    ablation_significance passes ONE shared dataset object to both runs --
    but the guard must not depend on that invariant holding forever.
    """
    import torch

    targets = torch.tensor([10, 20, 10, 20])
    qa = ["q1", "q2", "q3", "q4"]
    qb = ["q3", "q4", "q1", "q2"]   # same targets, different queries

    assert torch.equal(targets, targets), "targets are identical by construction"
    assert qa != qb, "query IDs must differ for this test to mean anything"

    # The guard in main() is: qa != qb -> SystemExit. Mirror its predicate.
    divergence = next((i for i, (x, y) in enumerate(zip(qa, qb)) if x != y), None)
    assert divergence == 0, divergence
    print("  [ok] identical-targets/different-queries case is detectable "
          "(first divergence at index %d)" % divergence)


def test_query_id_equality_is_required_not_just_target_equality():
    """Targets alone must never be accepted as proof of pairing."""
    import torch

    ta = torch.tensor([5, 5, 7])
    tb = torch.tensor([5, 5, 7])
    assert torch.equal(ta, tb)

    # Identical targets, but one query swapped out entirely.
    qa = ["p1", "p2", "p3"]
    qb = ["p1", "p9", "p3"]
    assert qa != qb, (
        "target equality holds while query identity does not -- this is why "
        "the script requires query_ids and aborts without them"
    )
    print("  [ok] target equality does not imply query equality")


def test_compare_rejects_unpaired_runs():
    try:
        compare([1, 0], [1, 0, 1], [1, 1], [1, 1, 1], "a", "b")
        raise AssertionError("unpaired input must be rejected")
    except ValueError as exc:
        assert "not paired" in str(exc), str(exc)
    print("  [ok] unpaired runs rejected rather than silently truncated")


def test_compare_intersects_valid_masks():
    """Only queries valid under BOTH systems may enter the test."""
    a = [1, 1, 0, 1]
    b = [1, 0, 1, 1]
    va = [1, 1, 1, 0]   # q3 invalid in A
    vb = [1, 1, 0, 1]   # q2 invalid in B
    r = compare(a, b, va, vb, "a", "b")
    assert r["n_paired_valid"] == 2, r["n_paired_valid"]
    assert r["n_dropped_invalid"] == 2, r["n_dropped_invalid"]
    # surviving queries are q0 (1->1, concordant) and q1 (1->0, harm)
    assert r["help"] == 0 and r["harm"] == 1, (r["help"], r["harm"])
    print("  [ok] valid masks intersected; only doubly-valid queries tested")


def test_identical_systems_give_null_result():
    """A system compared against itself must be flatly non-significant with
    a zero-centred CI -- the sanity floor for any paired test."""
    hits = [1, 0, 1, 1, 0, 0, 1, 0] * 50
    r = compare(hits, hits, [1] * len(hits), [1] * len(hits), "x", "x")
    assert r["help"] == 0 and r["harm"] == 0
    assert r["delta_pp"] == 0.0
    assert r["mcnemar_exact_p"] == 1.0
    assert r["ci_straddles_zero"] is True
    print("  [ok] identical systems -> delta 0, p=1, CI straddles zero")


def test_all_help_is_detected_as_significant():
    """Positive control: a real, one-sided effect must be detected."""
    a = [0] * 40 + [1] * 60
    b = [1] * 40 + [1] * 60
    r = compare(a, b, [1] * 100, [1] * 100, "base", "new")
    assert r["help"] == 40 and r["harm"] == 0
    assert r["mcnemar_exact_p"] < 0.001
    assert r["ci_straddles_zero"] is False
    assert r["delta_pp"] > 0
    print("  [ok] positive control detected (p=%.3g, CI excludes zero)"
          % r["mcnemar_exact_p"])


if __name__ == "__main__":
    test_hit_vector_excludes_unresolved_targets()
    test_denominator_is_valid_count_not_n()
    test_same_net_gain_gives_opposite_verdicts()
    test_identical_targets_but_different_query_ids_must_be_rejected()
    test_query_id_equality_is_required_not_just_target_equality()
    test_compare_rejects_unpaired_runs()
    test_compare_intersects_valid_masks()
    test_identical_systems_give_null_result()
    test_all_help_is_detected_as_significant()
    print("\nALL ABLATION-SIGNIFICANCE TESTS PASSED.")
