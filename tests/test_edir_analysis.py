"""edir_analysis.py: the bootstrap reproduces bbnv_policy's exactly; decisions,
TOST and the fixed sequence follow §2/§3/§15; end to end on a synthetic chain."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import bbnv_policy as pol  # noqa: E402
import edir_analysis as ea  # noqa: E402

RNG = np.random.default_rng(3)
CL = np.repeat(np.arange(400), 2)[:700]          # 700 queries in 350 groups


def test_bootstrap_matches_clustered_bootstrap_exactly():
    d = RNG.normal(0.01, 0.2, len(CL))
    ref = pol.clustered_bootstrap(d, CL, 10000)
    reps = ea.cluster_replicates(d, CL)
    assert ea.interval(reps, 0.95) == [ref["ci_low"], ref["ci_high"]]
    r = ea.significance(d, CL, alpha=0.05)
    assert r["estimate"] == ref["diff"] and r["p_two_sided"] == ref["p_two_sided"] and r["ci"] == [ref["ci_low"], ref["ci_high"]]


def test_registered_level_is_97_5():
    d = RNG.normal(0.0, 0.2, len(CL))
    r = ea.significance(d, CL)
    assert r["ci_level"] == 0.975
    lo95, hi95 = ea.interval(ea.cluster_replicates(d, CL), 0.95)
    assert r["ci"][0] <= lo95 and r["ci"][1] >= hi95                # wider than 95%


@pytest.mark.parametrize("shift,ben,adv", [(0.2, True, False), (-0.2, False, True), (0.0, False, False)])
def test_decisions(shift, ben, adv):
    d = RNG.normal(0, 0.05, len(CL)) + shift
    d = d - d.mean() + shift
    r = ea.significance(d, CL)
    assert (r["beneficial"], r["adverse"]) == (ben, adv)


def _res(lo, hi, ben=False, adv=False):
    return {"ci_tost": [lo, hi], "ci_tost_level": 0.95, "beneficial": ben, "adverse": adv}


def test_tost():
    wide = ea.tost(_res(-0.02, 0.02))
    assert wide["equivalent"] is False and wide["status"] == "equivalence not established"
    assert wide["precision_diagnostic"]["adequate"] is False
    ok = ea.tost(_res(-0.004, 0.003))
    assert ok["equivalent"] is True and ok["status"] == "equivalence established" and ok["ci_level"] == 0.95
    off = ea.tost(_res(0.006, 0.0115, ben=True))                # narrow, but crosses the margin
    assert off["equivalent"] is False and off["precision_diagnostic"]["adequate"] is True
    assert off["superiority_test"] == "significant positive difference"
    imprecise = ea.tost(_res(-0.008, 0.008))                    # half-width > 0.006, still inside the bounds
    assert imprecise["equivalent"] is True and imprecise["precision_diagnostic"]["adequate"] is False
    assert ea.tost(_res(-0.01, 0.002))["equivalent"] is False   # the bound itself is not inside (strict)


def test_tost_interval_is_95_at_alpha_a():
    d = RNG.normal(0.0, 0.2, len(CL))
    r = ea.significance(d, CL)
    assert r["ci_tost_level"] == 0.95
    assert r["ci_tost"] == ea.interval(ea.cluster_replicates(d, CL), 0.95)


def test_fixed_sequence():
    B, N, A = {"beneficial": True, "adverse": False}, {"beneficial": False, "adverse": False}, \
        {"beneficial": False, "adverse": True}
    s = ea.sequence([B, N, B])
    assert s[0].startswith("CONFIRMED") and "stops" in s[1] and s[2].startswith("descriptive")
    s = ea.sequence([A, B, B])
    assert s[0].startswith("adverse") and all(x.startswith("descriptive") for x in s[1:])
    assert all(x.startswith("CONFIRMED") for x in ea.sequence([B, B, B]))


def test_q1_contrast_is_mean_over_seed_pairs():
    au = {(a, s): [0.5 + 0.1 * s + (0.2 if a == "F+I" else 0.0), 0.5] for a in ("F+I", "F") for s in (0, 1, 2)}
    assert np.allclose(ea.q1_contrast(au, "F+I", "F"), [0.2, 0.0])
