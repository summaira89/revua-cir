"""revision_followups8.py (amendment §28): helper contracts."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import revision_followups8 as r8  # noqa: E402


def test_counts():
    assert r8.counts([1, 0, 1, 0], [0, 1, 1, 0]) == {"fixed": 1, "broken": 1}


def test_clustered_mean_equals_paired_delta(monkeypatch):
    monkeypatch.setattr(r8, "N_REP", 300)
    new, ref = [1, 1, 0, 1, 0, 1], [0, 1, 0, 0, 1, 1]
    c = r8.clustered(new, ref, [0, 0, 1, 1, 2, 2])
    assert c["diff"] == pytest.approx(100.0 * (sum(new) - sum(ref)) / len(new)) and c["n_clusters"] == 3


def test_contrast_requires_locked_counts(monkeypatch):
    monkeypatch.setattr(r8, "N_REP", 300)
    new, ref = [1, 1, 0, 1, 0, 1], [0, 1, 0, 0, 1, 1]
    want = {"fixed": 2, "broken": 1, "ci_low": -10.0, "ci_high": 40.0}
    out = r8.contrast("x", new, ref, [0, 1, 2, 3, 4, 5], want)
    assert out["fixed"] == 2 and out["query_bootstrap_ci"] == [-10.0, 40.0] and out["ci_width_ratio"] > 0
    with pytest.raises(SystemExit):
        r8.contrast("x", new, ref, [0, 1, 2, 3, 4, 5], dict(want, broken=2))
