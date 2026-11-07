"""revision_followups3.py (amendment §22): helper contracts on synthetic inputs."""
import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import revision_followups3 as r3  # noqa: E402

RNG = np.random.default_rng(11)


def test_top_k_excludes_and_breaks_ties_to_lower_row():
    s = np.array([0.5, 0.9, 0.9, 0.1, 0.95])
    assert r3.top_k_excluding(s, exclude=4, k=3) == [1, 2, 0]
    assert 4 not in r3.top_k_excluding(s, exclude=4, k=5)


def test_rank_of_agrees_with_top_k_excluding():
    for _ in range(200):
        s = np.round(RNG.uniform(size=30), 2)          # coarse values force ties
        ex, tgt = RNG.choice(30, 2, replace=False)
        order = r3.top_k_excluding(s, int(ex), 29)
        assert r3.rank_of(s, int(tgt), int(ex)) == order.index(int(tgt))


def test_window_hits_uncovered_target_never_hits():
    h = r3.window_hits([3, 4, 5], target=9, b=[0.9, 0.5, 0.1], t=[0.1, 0.5, 0.9])
    assert h == {"B": 0, "T": 0, "BT": 0, "covered": 0}
    h = r3.window_hits([3, 4, 5], target=5, b=[0.9, 0.5, 0.1], t=[0.1, 0.5, 0.9])
    assert h["covered"] == 1 and h["B"] == 0 and h["T"] == 1


def test_subset_rank_ties_to_lower_position():
    assert r3.subset_rank([0.2, 0.9, 0.9, 0.1, 0.0], 2) == 1
    assert r3.subset_rank([0.2, 0.9, 0.9, 0.1, 0.0], 1) == 0


def test_per_query_utility_weights():
    assert r3.per_query_utility([1, 0, 1, 0], [0, 1, 1, 0]) == [1.0, -2.0, 0.0, 0.0]


def _saved_and_rec():
    fw = {"u": 0.1, "r1": 50.0, "rescues": 2, "breaks": 0, "accepted": 3}
    saved = {"per_fold": {f: {"policies": {"FW": dict(fw)}, "control": "SW", "sw_lambda": 0.25}
                          for f in ("fit_tune", "fit_report")}}
    hits, base = [1, 1, 0, 0], [0, 1, 0, 0]
    rec = {"folds": {f: {"FW": dict(fw), "control": "SW", "sw_lambda": 0.25} for f in ("fit_tune", "fit_report")},
           "fw_hits": hits, "fw_accept": [True, False, False, False], "control_hits": base, "base": base}
    return saved, rec


def test_gate_parity_passes_and_each_failure_aborts(monkeypatch):
    saved, rec = _saved_and_rec()
    monkeypatch.setitem(r3.PUBLISHED, "fw_pooled_u", 0.25)
    monkeypatch.setitem(r3.PUBLISHED, "control_pooled_u", 0.0)
    assert r3.check_gate_parity(rec, copy.deepcopy(rec), saved)["deterministic"] is True
    bad = copy.deepcopy(rec)
    bad["folds"]["fit_tune"]["FW"]["breaks"] = 1                      # one fold quantity differs
    with pytest.raises(SystemExit):
        r3.check_gate_parity(bad, bad, saved)
    bad = copy.deepcopy(rec)
    bad["folds"]["fit_report"]["control"] = "R"                       # control selection differs
    with pytest.raises(SystemExit):
        r3.check_gate_parity(bad, bad, saved)
    again = copy.deepcopy(rec)
    again["fw_accept"] = [False, True, False, False]                  # non-deterministic decisions
    with pytest.raises(SystemExit):
        r3.check_gate_parity(rec, again, saved)
    monkeypatch.setitem(r3.PUBLISHED, "fw_pooled_u", 0.2)             # pooled utility differs
    with pytest.raises(SystemExit):
        r3.check_gate_parity(rec, copy.deepcopy(rec), saved)


def test_subset_members_drops_reference_and_checks_alignment():
    names = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}
    cap = [{"reference": "a", "target_hard": "c", "img_set": {"members": ["a", "b", "c", "d", "e", "f"]}}]
    recs = [{}]
    assert r3.subset_members(cap, recs, names, np.array([0]), np.array([2])) == [[1, 2, 3, 4, 5]]
    with pytest.raises(SystemExit):
        r3.subset_members(cap, recs, names, np.array([1]), np.array([2]))      # reference row mismatch
    with pytest.raises(SystemExit):
        r3.subset_members(cap, [{}, {}], names, np.array([0]), np.array([2]))  # length mismatch
