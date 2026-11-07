"""Tests for the BBNV code: scripts/bbnv_core.py, bbnv_partition.py,
bbnv_oracle.py, bbnv_provenance.py and bbnv_manifest.py. Runs without pytest.

The first eight tests are the invariants of v3.1 §6, in order; the rest cover
the v3.1 §8 corrections (fixed edit weight, cyclic seed pairs, split
disjointness, negative-pool sufficiency), the input contract, one regression
test per finding of the Phase A source audit (19 Oct 2026), and the
linked-group partition of amendment §10 with its CIRR manifest."""
import json
import os
import sys

# see bbnv_oracle.py: keep numpy's MKL off OpenMP so torch's runtime is the only one
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import bbnv_analyze as ana  # noqa: E402
import bbnv_core as core  # noqa: E402
import bbnv_extract as ext  # noqa: E402
import bbnv_manifest as man  # noqa: E402
import bbnv_oracle as oracle_mod  # noqa: E402
import bbnv_partition as part  # noqa: E402
import bbnv_policy as pol  # noqa: E402
import bbnv_provenance as prov  # noqa: E402
import bbnv_score as scr  # noqa: E402
import bbnv_train as trn  # noqa: E402
import cirr_split_windows as csw  # noqa: E402
import cirr_test as ctest  # noqa: E402
import bbnv_transfer as tfr  # noqa: E402

RNG = np.random.default_rng(20261019)


def _unit(n, d, rng=RNG):
    x = rng.standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _expect(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return
    raise AssertionError("expected %s" % exc.__name__)


def _random_forward_tables(n_ref, n_cand, rng):
    logits = rng.normal(size=(n_ref, n_cand)) * 2.0
    return core.log_softmax(logits, axis=1)


# ---------------------------------------------------------- invariant 1 --
def test_inv1_mixture_is_normalised_and_pi_are_masses():
    """Every distribution sums to one on its support, and the component-
    normalised mixture equals sum_m pi_m q_m with pi as masses."""
    hyp = _unit(5, 6).reshape(1, 5, 6)
    log_pi = core.log_softmax(RNG.normal(size=(1, 5)), axis=1)
    support = _unit(40, 6)
    lp = core.mixture_log_prob_batch(hyp, log_pi, np.array([7.0]), support)[0]
    assert abs(core.logsumexp(lp)) < 1e-12
    comps = np.exp(core.log_softmax(7.0 * hyp[0] @ support.T, axis=1))       # (M, N)
    manual = np.exp(log_pi[0]) @ comps
    assert np.max(np.abs(np.exp(lp) - manual)) < 1e-12
    rep = core.mixture_masses_report(log_pi[0], np.log(comps))
    assert abs(sum(rep["masses"]) - 1.0) < 1e-12 and rep["mean_pairwise_js_divergence"] > 0


# ---------------------------------------------------------- invariant 2 --
def test_inv2_query_reference_has_positive_prior_also_when_in_bank():
    ids = ["b%d" % i for i in range(6)]
    emb = _unit(6, 4)
    outside = core.build_universe(ids, emb, ["q"], _unit(1, 4))
    assert len(outside.ids) == 7 and outside.index("q") == 6
    inside = core.build_universe(ids, emb, ["b3"], emb[[3]])
    assert len(inside.ids) == 6 and inside.index("b3") == 3              # not added twice
    for u in (outside, inside):
        rho = core.log_prior_uniform(len(u.ids))
        assert np.all(np.isfinite(rho)) and abs(core.logsumexp(rho)) < 1e-12
    _expect(core.ContractError, core.build_universe, ids, emb, ["b3"], _unit(1, 4))  # same id, other vector


# ---------------------------------------------------------- invariant 3 --
def test_inv3_induced_inverse_equals_direct_bayes_on_enumerated_joint():
    orc = oracle_mod.Oracle(oracle_mod.OracleSpec())
    for v in range(orc.spec.n_attribute):
        t = orc.tables(v)
        for row in (0, 5, len(t["refs"]) - 1):
            ref_item = int(t["refs"][row])
            win, _ = orc.window(ref_item, v)
            fq = core.forward_query_scores(t["log_f"], t["log_rho"], row, win)
            direct = t["log_inv"][win, row]                     # log p(r | c, t) by enumeration
            assert np.max(np.abs(fq["ie_induced"] - direct)) < 1e-12
            assert np.max(np.abs(fq["pmi_f"] - t["pmi"][row, win])) < 1e-12


# ---------------------------------------------------------- invariant 4 --
def test_inv4_corrected_forward_and_induced_inverse_rank_identically():
    rng = np.random.default_rng(4)
    for _ in range(20):
        log_f = _random_forward_tables(65, 80, rng)
        log_rho = core.log_softmax(rng.normal(size=65))          # non-uniform prior too
        win = rng.choice(80, size=10, replace=False)
        fq = core.forward_query_scores(log_f, log_rho, 64, win)
        assert np.array_equal(np.argsort(-fq["pmi_f"], kind="stable"),
                              np.argsort(-fq["ie_induced"], kind="stable"))
        lab = np.zeros(10, bool)
        lab[0] = True
        assert core.pairwise_auroc(fq["pmi_f"], lab) == core.pairwise_auroc(fq["ie_induced"], lab)
    # uniform extension of a bank-only marginal: log q_I^F = -log(1 + R e^{-s})
    R = 64
    log_f = _random_forward_tables(R + 1, 50, rng)
    win = np.arange(50)
    s = log_f[R] - core.logsumexp(np.full(R, -np.log(R))[:, None] + log_f[:R], axis=0)
    fq = core.forward_query_scores(log_f, core.log_prior_uniform(R + 1), R, win)
    assert np.max(np.abs(fq["ie_induced"] + np.log1p(R * np.exp(-s)))) < 1e-12


# ---------------------------------------------------------- invariant 5 --
def test_inv5_chunked_marginal_equals_unchunked():
    rng = np.random.default_rng(5)
    log_f = _random_forward_tables(300, 120, rng)
    log_rho = core.log_softmax(rng.normal(size=300))
    win = rng.choice(120, size=10, replace=False)
    full = core.forward_query_scores(log_f, log_rho, 7, win)["log_marginal"]
    ids = tuple("c%d" % j for j in range(120))
    for chunk in (1, 7, 64, 1000):
        got = core.chunked_log_marginal(lambda a, b: (log_f[a:b], ids), log_rho, win, chunk, ids)
        assert np.max(np.abs(got - full)) < 1e-12, chunk


# ---------------------------------------------------------- invariant 6 --
def test_inv6_permutation_duplicates_and_in_bank_reference():
    rng = np.random.default_rng(6)
    log_f = _random_forward_tables(30, 40, rng)
    log_rho = core.log_prior_uniform(30)
    win = np.array([3, 9, 17, 22, 31], dtype=np.int64)
    perm = np.array([4, 0, 3, 1, 2])
    a = core.forward_query_scores(log_f, log_rho, 2, win)
    b = core.forward_query_scores(log_f, log_rho, 2, win[perm])
    for key in a:
        assert np.allclose(a[key][perm], b[key], rtol=0, atol=1e-15), key
    log_i = core.log_softmax(rng.normal(size=(5, 30)), axis=1)
    assert np.allclose(core.inverse_query_scores(log_i, 2)[perm],
                       core.inverse_query_scores(log_i[perm], 2), rtol=0, atol=0)
    _expect(core.ContractError, core.forward_query_scores, log_f, log_rho, 2, [3, 9, 3])
    _expect(core.ContractError, core.build_universe, ["x", "y", "x"], _unit(3, 4))


# ---------------------------------------------------------- invariant 7 --
def test_inv7_delta_is_exactly_the_difference_of_the_supplied_features():
    rng = np.random.default_rng(7)
    learned = rng.normal(size=10) + 3.0                         # non-zero means on purpose
    induced = rng.normal(size=10) - 5.0
    feats = core.gate_window_features(learned, induced, rng.normal(size=10))
    assert np.array_equal(feats["delta"], feats["c_ie_learned"] - feats["c_ie_induced"])
    assert abs(feats["c_ie_learned"].mean()) < 1e-12 and abs(feats["c_ie_induced"].mean()) < 1e-12


# ---------------------------------------------------------- invariant 8 --
def test_inv8_hypothesis_outcomes_are_recorded_not_asserted():
    """A deliberately weak run (few steps, misspecified forward head) must
    complete and return its outcomes as data, whatever they are."""
    spec = oracle_mod.OracleSpec(n_identity=2, n_attribute=3, n_hub=1)
    orc = oracle_mod.Oracle(spec)
    rng = np.random.default_rng(8)
    train = oracle_mod.sample_triplets(orc, 300, rng)
    test = oracle_mod.sample_triplets(orc, 50, rng)
    res = oracle_mod.run_q1_and_representation(
        orc, train, test, oracle_mod.TrainConfig(hidden=2, steps=3, batch=32),
        oracle_mod.TrainConfig(hidden=4, steps=3, batch=32), seeds=(0, 1, 2))
    assert set(res["q1"]) == set(oracle_mod.ARM_NAMES)
    assert res["induced_ranks_like_corrected_forward"] is True        # invariant 4 on learned heads
    assert res["_predictions"] and all("query" in row for row in res["_predictions"])
    for pair, cmp in res["paired_contrasts"].items():                 # query-paired, shared mask
        assert cmp["F+I_vs_F"]["n_shared"] > 0, pair
    json.dumps(res, allow_nan=False)


# ------------------------------------------------------- v3.1 §8 rules --
def test_integrate_keeps_edit_weight_fixed_across_one_and_two_head_arms():
    """V_H averages heads: duplicating the same head must not change the
    verification-to-edit balance (a sum would double it)."""
    rng = np.random.default_rng(9)
    s, edit = rng.normal(size=10), rng.normal(size=10)
    one = core.integrate([s], edit, gamma=1.0)
    two = core.integrate([s, s.copy()], edit, gamma=1.0)
    assert np.allclose(one, two, rtol=0, atol=1e-15)
    assert np.allclose(one, core.zscore(s) + core.zscore(edit), rtol=0, atol=1e-15)
    _expect(core.ContractError, core.integrate, [], edit, 1.0)
    _expect(core.ContractError, core.integrate, [s[:5]], edit, 1.0)


def test_zscore_floor_zeroes_only_the_degenerate_component():
    rng = np.random.default_rng(10)
    s, edit = rng.normal(size=10), np.full(10, 0.3)             # edit constant in this window
    v = core.integrate([s], edit, gamma=2.0)
    assert np.all(np.isfinite(v)) and np.allclose(v, core.zscore(s), rtol=0, atol=1e-15)
    assert np.array_equal(core.zscore(np.full(4, 1.0)), np.zeros(4))


def test_seed_pairs_are_the_fixed_cyclic_pairs():
    assert core.SEED_PAIRS == ((0, 1), (1, 2), (2, 0)) and core.DEPLOYED_SEED == 0
    core.validate_seed_pairs()
    _expect(core.ContractError, core.validate_seed_pairs, ((0, 1), (0, 2), (1, 2)))


def _triplets():
    """Synthetic triplets in which images reappear across roles."""
    imgs = ["img%03d" % i for i in range(400)]
    rng = np.random.default_rng(11)
    return [{"reference": imgs[a], "target": imgs[b]}
            for a, b in rng.integers(0, 400, size=(3000, 2)) if a != b]


def _grouped_triplets(n_groups=300, seed=5):
    """Synthetic triplets in many small linked groups of 2-6 images (ids
    gNNN_i), each with 1-12 triplets, so every partition receives groups."""
    rng = np.random.default_rng(seed)
    trips = []
    for g in range(n_groups):
        imgs = ["g%03d_%d" % (g, i) for i in range(int(rng.integers(2, 7)))]
        for i in range(1, len(imgs)):                       # a chain keeps the group connected
            trips.append({"reference": imgs[i - 1], "target": imgs[i]})
        for _ in range(int(rng.integers(0, 14 - len(imgs)))):
            a, b = rng.choice(len(imgs), size=2, replace=False)
            trips.append({"reference": imgs[a], "target": imgs[b]})
    return trips


def _error_message(fn, *a):
    try:
        fn(*a)
    except part.PartitionError as exc:
        return str(exc)
    raise AssertionError("expected PartitionError")


def test_split_is_role_disjoint_and_fully_accounted():
    trips = _grouped_triplets()
    alloc = part.allocate_images(trips, part.known_positives(trips))
    kept, counts = part.split_triplets(trips, alloc)
    assert counts["total"] == len(trips) == sum(counts["kept_" + n] for n in part.PARTITION_NAMES)
    imgs = part.images_by_partition(kept)
    for i, a in enumerate(part.PARTITION_NAMES):
        for b in part.PARTITION_NAMES[i + 1:]:
            assert not (imgs[a] & imgs[b]), (a, b)
    for name, trips_n in kept.items():                      # both roles in the declared partition
        for t in trips_n:
            assert alloc[t["reference"]] == name == alloc[t["target"]]
    bad = {"head": [{"reference": "x1", "target": "x2"}], "dev_select": [{"reference": "x2", "target": "x3"}]}
    _expect(part.PartitionError, part.assert_role_disjoint, bad)
    # a map that splits a triplet is refused, never silently discarded
    t0 = trips[0]
    wrong = dict(alloc)
    wrong[t0["target"]] = next(n for n in part.PARTITION_NAMES if n != alloc[t0["reference"]])
    assert "crosses" in _error_message(part.split_triplets, trips, wrong)
    missing = {k: v for k, v in alloc.items() if k != t0["target"]}
    assert "unallocated" in _error_message(part.split_triplets, trips, missing)


def test_linked_groups_are_components_keyed_by_their_smallest_id():
    trips = [{"reference": "c", "target": "b"}, {"reference": "b", "target": "a"},
             {"reference": "z", "target": "y"}]
    groups = part.linked_groups(part.known_positives(trips), ["lonely"])
    assert groups == {"a": "a", "b": "a", "c": "a", "y": "y", "z": "y", "lonely": "lonely"}
    assert part.linked_groups(part.known_positives(trips[::-1]), ["lonely"]) == groups
    for edge in ({"reference": "a", "target": "z"}, {"reference": "z", "target": "a"}):   # either orientation
        assert part.linked_groups(part.known_positives([edge])) == {"a": "a", "z": "a"}
    rng = np.random.default_rng(8)                          # independent BFS oracle on random graphs
    for _ in range(40):
        ids = ["i%02d" % k for k in rng.permutation(30)]
        edges = [{"reference": ids[a], "target": ids[b]}
                 for a, b in rng.integers(0, 30, size=(int(rng.integers(5, 40)), 2)) if a != b]
        adj = {}
        for e in edges:
            adj.setdefault(e["reference"], set()).add(e["target"])
            adj.setdefault(e["target"], set()).add(e["reference"])
        expect = {}
        for start in adj:
            if start in expect:
                continue
            comp, todo = {start}, [start]
            while todo:
                for nb in adj[todo.pop()] - comp:
                    comp.add(nb)
                    todo.append(nb)
            expect.update({i: min(comp) for i in comp})
        assert part.linked_groups(part.known_positives(edges)) == expect
        assert part.linked_groups(part.known_positives(edges[::-1])) == expect


def test_group_allocation_keeps_every_triplet_and_meets_the_shares():
    trips = _grouped_triplets()
    pos = part.known_positives(trips)
    alloc = part.allocate_images(trips, pos)
    _, counts = part.split_triplets(trips, alloc)
    sizes = {}
    for t in trips:
        sizes[t["reference"][:4]] = sizes.get(t["reference"][:4], 0) + 1
    bound = 3 * max(sizes.values())         # greedy: overshoot < 1 group, undershoot < 3 groups
    for name, lo, hi in part.PARTITIONS:
        assert abs(counts["kept_" + name] - (hi - lo) / 100.0 * len(trips)) <= bound, name
    shuffled = [trips[i] for i in np.random.default_rng(1).permutation(len(trips))]
    assert part.allocate_images(shuffled, part.known_positives(shuffled)) == alloc   # deterministic
    # an extra positive pair merges two groups that would otherwise be apart
    a = next(t["reference"] for t in trips if alloc[t["reference"]] == "head")
    b = next(t["reference"] for t in trips if alloc[t["reference"]] == "dev_calibrate")
    linked = {k: set(v) for k, v in pos.items()}
    linked[a].add(b)
    linked[b].add(a)
    merged = part.allocate_images(trips, linked)
    assert merged[a] == merged[b]
    # images in no triplet keep the per-image rule; positives must contain every triplet pair
    assert (part.allocate_images(trips, pos, ["only_negative_7"])["only_negative_7"]
            == part.bucket_partition("only_negative_7"))
    gap = {k: set(v) for k, v in pos.items()}
    gap[trips[0]["reference"]].discard(trips[0]["target"])
    assert "lack the triplet pair" in _error_message(part.allocate_images, trips, gap)


def test_partition_ranges_tile_and_dev_is_split_in_two():
    edges = [(lo, hi) for _, lo, hi in part.PARTITIONS]
    assert edges[0][0] == 0 and edges[-1][1] == 100
    assert all(a[1] == b[0] for a, b in zip(edges, edges[1:]))
    assert part.PARTITION_NAMES == ("head", "negbank", "dev_select", "dev_calibrate")


def test_negative_pool_masks_positives_and_refuses_shortfall():
    pool = ["n%d" % i for i in range(20)]
    got = part.sample_negatives(pool, {"n0", "n1"}, 18, np.random.default_rng(0))
    assert len(set(got)) == 18 and not ({"n0", "n1"} & set(got))
    again = part.sample_negatives(pool, {"n0", "n1"}, 18, np.random.default_rng(0))
    assert got == again                                      # seeded, deterministic
    _expect(part.PartitionError, part.sample_negatives, pool, {"n0", "n1", "n2"}, 18,
            np.random.default_rng(0))
    _expect(part.PartitionError, part.sample_negatives, pool + ["n3"], set(), 5,
            np.random.default_rng(0))


# ------------------------------------------------------ contract + parity --
def test_zero_marginal_mass_makes_pmi_undefined():
    log_f = np.full((3, 4), -np.log(3.0))
    log_f[:, 3] = -np.inf                                    # candidate 3 has no mass anywhere
    _expect(core.ContractError, core.forward_query_scores, log_f, core.log_prior_uniform(3), 0, [0, 3])


def test_pairwise_auroc_half_credit_for_ties_and_undefined_cases():
    assert core.pairwise_auroc(np.array([1.0, 1.0, 0.0]), np.array([True, False, False])) == 0.75
    assert core.pairwise_auroc(np.array([1.0, 2.0]), np.array([True, True])) is None
    assert core.macro_auroc([0.5, None, 1.0]) == (0.75, 2)


def test_contract_rejects_bad_inputs():
    good = _unit(3, 4).reshape(1, 3, 4)
    lp = core.log_softmax(np.zeros((1, 3)), axis=1)
    sup = _unit(10, 4)
    _expect(core.ContractError, core.mixture_log_prob_batch, good * 2, lp, np.array([5.0]), sup)
    _expect(core.ContractError, core.mixture_log_prob_batch, good, lp + 1.0, np.array([5.0]), sup)
    _expect(core.ContractError, core.mixture_log_prob_batch, good, lp, np.array([0.5]), sup)
    _expect(core.ContractError, core.mixture_log_prob_batch, good, lp, np.array([5.0]), sup[:, :3])


def test_bayes_check_passes_on_the_oracle():
    res = oracle_mod.bayes_check(oracle_mod.Oracle(oracle_mod.OracleSpec()))
    assert res["max_bayes_error"] < 1e-12 and res["max_normalisation_error"] < 1e-12


def test_torch_mixture_matches_the_numpy_core():
    import torch
    hyp = _unit(12, 6).reshape(3, 4, 6)
    log_pi = core.log_softmax(RNG.normal(size=(3, 4)), axis=1)
    kappa = np.array([2.0, 10.0, 60.0])
    sup = _unit(25, 6)
    ref = core.mixture_log_prob_batch(hyp, log_pi, kappa, sup)
    got = oracle_mod.torch_mixture_log_prob(torch.tensor(hyp), torch.tensor(log_pi),
                                            torch.tensor(kappa), torch.tensor(sup)).numpy()
    assert np.max(np.abs(ref - got)) < 1e-12


def test_width_matched_formula_agrees_with_the_modules():
    in_dim, out_dim = 24, 16
    wide = oracle_mod.width_matched_hidden(in_dim, out_dim, 64, 8)
    m8 = oracle_mod.param_count(oracle_mod.build_head(in_dim, out_dim, 64, 8, 0))
    m1 = oracle_mod.param_count(oracle_mod.build_head(in_dim, out_dim, wide, 1, 0))
    assert abs(m1 - m8) <= 0.05 * m8, (m1, m8)


# ------------------------------------------- Phase A source audit (19 Oct) --
def test_audit1_prior_and_table_ranks_are_enforced():
    """Finding 1: a (2,1) prior used to broadcast into a (2,2) 'PMI' with an
    induced log-probability above 0. Ranks are now checked before use."""
    F = np.log([[.8, .2], [.3, .7]])
    for bad in (np.zeros((2, 1)), np.zeros((1, 2)), np.zeros((2, 1, 1))):
        _expect(core.ContractError, core._check_prior, bad, 2)
        _expect(core.ContractError, core.forward_query_scores, F, bad, 0, [0, 1])
    for bad_table in (F[0], F[None]):
        _expect(core.ContractError, core.forward_query_scores, bad_table, np.log([.5, .5]), 0, [0, 1])
    for bad_ref in (0.0, True, np.float64(1.0), 2, -1):
        _expect(core.ContractError, core.forward_query_scores, F, np.log([.5, .5]), bad_ref, [0, 1])
    rng = np.random.default_rng(101)
    for _ in range(50):                                    # induced log-prob <= 0 on valid input
        log_f = _random_forward_tables(40, 30, rng)
        fq = core.forward_query_scores(log_f, core.log_softmax(rng.normal(size=40)), 39,
                                       rng.choice(30, size=8, replace=False))
        assert np.max(fq["ie_induced"]) <= 1e-9
        assert all(v.shape == (8,) for v in fq.values())


def test_audit2_window_indices_must_be_genuine_integers():
    """Finding 2: floats and booleans used to be coerced into valid positions."""
    for bad in ([0.9, 1.9], [0.0, 1.0], ["0", "1"], [False, True], np.array([False, True]),
                np.array([0.0, 1.0]), [0, 5], [np.nan, 1]):
        _expect(core.ContractError, core._window, bad, 2)
    assert core._window([np.int64(0), 1], 2).tolist() == [0, 1]
    assert core._window(np.array([1, 0], dtype=np.int32), 2).tolist() == [1, 0]


def test_audit3_chunked_marginal_enforces_the_full_contract():
    """Finding 3: zero-prior rows, zero marginals, changed column counts and
    reordered candidate universes are rejected, as in the full path."""
    F = np.log([[.8, .2], [.3, .7]])
    ids = ("a", "b")
    rows = lambda a, b: (F[a:b], ids)                       # noqa: E731
    with np.errstate(divide="ignore"):
        zero_prior = np.log([1.0, 0.0])
    _expect(core.ContractError, core.chunked_log_marginal, rows, zero_prior, [0, 1], 1, ids)
    _expect(core.ContractError, core.forward_query_scores, F, zero_prior, 0, [0, 1])
    Z = np.array([[0.0, -np.inf], [0.0, -np.inf]])          # candidate b has no mass anywhere
    _expect(core.ContractError, core.chunked_log_marginal, lambda a, b: (Z[a:b], ids),
            np.log([.5, .5]), [0, 1], 1, ids)
    _expect(core.ContractError, core.forward_query_scores, Z, np.log([.5, .5]), 0, [0, 1])
    grow = {0: (np.log([[.5, .5]]), ("a", "b")), 1: (np.log([[.25, .25, .5]]), ("a", "b", "c"))}
    _expect(core.ContractError, core.chunked_log_marginal, lambda a, b: grow[a],
            np.log([.5, .5]), [0, 1], 1, ("a", "b"))
    swap = {0: (F[0:1], ("a", "b")), 1: (F[1:2], ("b", "a"))}   # same size, reordered
    _expect(core.ContractError, core.chunked_log_marginal, lambda a, b: swap[a],
            np.log([.5, .5]), [0, 1], 1, ("a", "b"))


def test_audit4_auroc_labels_and_macro_values_are_validated():
    """Finding 4: a NaN label used to become True and score AUROC 1.0."""
    for bad in (np.array([np.nan, 0.0]), np.array([2, 0]), np.array([-1, 1]), np.array(["1", "0"])):
        _expect(core.ContractError, core.pairwise_auroc, np.array([1.0, 0.0]), bad)
    assert core.pairwise_auroc(np.array([1.0, 0.0]), np.array([1, 0])) == 1.0
    for bad in ([1.5], [-0.2], [np.nan], [True]):
        _expect(core.ContractError, core.macro_auroc, bad)
    res = core.paired_macro_auroc({"q1": 0.8, "q2": None, "q3": 0.6}, {"q1": 0.5, "q2": 0.9, "q3": None})
    assert res["n_shared"] == 1 and abs(res["difference"] - 0.3) < 1e-12
    _expect(core.ContractError, core.paired_macro_auroc, {"q1": 0.5}, {"q2": 0.5})


def test_audit5_window_transforms_reject_batches_and_bad_floors():
    """Finding 5: a (queries, candidates) array used to be standardised globally."""
    _expect(core.ContractError, core.integrate, [np.array([[0.0, 1.0], [10.0, 11.0]])],
            np.zeros((2, 2)), 1.0)
    _expect(core.ContractError, core.center, [])
    for bad_floor in (0, 0.0, -1e-6, float("nan"), float("inf"), True):
        _expect(core.ContractError, core.zscore, [1.0, 1.0], bad_floor)


def test_audit6_mixture_report_requires_matching_components():
    """Finding 6: two masses with one component distribution used to pass."""
    _expect(core.ContractError, core.mixture_masses_report, np.log([.5, .5]), np.log([[.2, .8]]))
    rep = core.mixture_masses_report(np.log([.5, .5]), np.log([[.2, .8], [.2, .8]]))
    assert rep["mean_pairwise_js_divergence"] == 0.0         # identical components: 0 divergence


def test_lambda_interpolation_spans_raw_and_pmi():
    """v3.1 §9: s_lambda = log q_F - lambda log p_C = (1-lambda) raw + lambda PMI."""
    rng = np.random.default_rng(102)
    log_f = _random_forward_tables(20, 15, rng)
    fq = core.forward_query_scores(log_f, core.log_prior_uniform(20), 19, list(range(6)))
    raw, marg, pmi = fq["log_f_raw"], fq["log_marginal"], fq["pmi_f"]
    assert np.array_equal(core.forward_lambda_score(raw, marg, 0.0), raw)
    assert np.allclose(core.forward_lambda_score(raw, marg, 1.0), pmi, rtol=0, atol=1e-15)
    assert np.allclose(core.forward_lambda_score(raw, marg, 0.25), 0.75 * raw + 0.25 * pmi,
                       rtol=0, atol=1e-14)
    for bad in (-0.1, 1.1, float("nan"), True, "0.5"):
        _expect(core.ContractError, core.forward_lambda_score, raw, marg, bad)


def test_audit10_canonical_image_ids_and_known_positives():
    """Finding 10: one id representation at the boundary, and every known
    positive partner of either image masked from the negatives."""
    for bad in (5, b"x", None, "", " img1", "img1 "):
        _expect(part.PartitionError, part.canonical_id, bad)
    trips = _triplets()
    pos = part.known_positives(trips)
    a, b = trips[0]["reference"], trips[0]["target"]
    assert b in pos[a] and a in pos[b]                     # both directions
    third = next(t for t in trips if a in (t["reference"], t["target"]) and b not in (
        t["reference"], t["target"]))
    pool = sorted({t["reference"] for t in trips} | {t["target"] for t in trips})
    negs = part.negatives_for(third, pool, pos, 50, np.random.default_rng(3))
    blocked = {third["reference"], third["target"]} | pos[third["reference"]] | pos[third["target"]]
    assert b in blocked and not (set(negs) & blocked)


def test_audit10_oracle_rejects_bad_directions_seeds_and_configs():
    spec = oracle_mod.OracleSpec(n_identity=2, n_attribute=3, n_hub=1)
    orc = oracle_mod.Oracle(spec)
    data = oracle_mod.sample_triplets(orc, 20, np.random.default_rng(4))
    cfg = oracle_mod.TrainConfig(hidden=2, steps=1, batch=4)
    for fn in (lambda: oracle_mod.train_head(orc, data, "Forward", cfg, 0),
               lambda: oracle_mod.true_nll(orc, data, "reverse"),
               lambda: oracle_mod.run_q1_and_representation(orc, data, data, cfg, cfg, seeds=(0, 1)),
               lambda: oracle_mod.train_head(orc, data, "forward",
                                             oracle_mod.TrainConfig(hidden=0), 0),
               lambda: oracle_mod.train_head(orc, data[:0], "forward", cfg, 0)):
        _expect(ValueError, fn)


def test_audit9_atomic_write_leaves_no_partial_and_hashes_match():
    import tempfile
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "out.json")
        sha = prov.atomic_write_json(path, {"a": [1, 2.5], "b": None})
        assert not os.path.exists(path + ".partial") and sha == prov.sha256_file(path)
        with open(path, encoding="utf-8") as fh:
            assert json.load(fh) == {"a": [1, 2.5], "b": None}
        rows_path = os.path.join(d, "rows.jsonl")
        prov.atomic_write_jsonl(rows_path, [{"q": 1}, {"q": 2}])
        assert not os.path.exists(rows_path + ".partial")
        _expect(ValueError, prov.atomic_write_json, path, {"x": float("nan")})
        assert not os.path.exists(path + ".partial")                # a failed write leaves nothing
        with open(path, encoding="utf-8") as fh:                   # the old file is intact
            assert json.load(fh) == {"a": [1, 2.5], "b": None}
        _expect(ValueError, prov.atomic_write_jsonl, rows_path, [{"q": float("inf")}])
        assert not os.path.exists(rows_path + ".partial")
        a64 = np.arange(3, dtype=np.int64)                          # dtype is part of the digest
        assert prov.sha256_array(a64) != prov.sha256_array(a64.astype(np.int32))
        assert prov.sha256_array(a64) == prov.sha256_array(a64.copy())
    finally:
        import shutil
        shutil.rmtree(d)


def test_failed_write_after_the_temporary_file_leaves_nothing():
    """A failure AFTER the .partial file was written and closed (here: the
    final replace) removes it, and the previous file survives intact."""
    import shutil
    import tempfile
    d = tempfile.mkdtemp()
    real_replace = prov.os.replace

    def failing_replace(src, dst):
        raise OSError("simulated failure at replace")
    try:
        path = os.path.join(d, "out.json")
        prov.atomic_write_json(path, {"v": 1})
        prov.os.replace = failing_replace
        _expect(OSError, prov.atomic_write_json, path, {"v": 2})
        _expect(OSError, prov.atomic_write_jsonl, os.path.join(d, "rows.jsonl"), [{"q": 1}])
        prov.os.replace = real_replace
        assert sorted(os.listdir(d)) == ["out.json"]              # no .partial anywhere
        with open(path, encoding="utf-8") as fh:
            assert json.load(fh) == {"v": 1}
    finally:
        prov.os.replace = real_replace
        shutil.rmtree(d)


def test_run_directory_is_fresh_complete_and_consistent():
    """Predictions are written first and results.json last, into a directory
    that must be new or empty; verify_run catches tampering and stray files."""
    import shutil
    import tempfile
    d = tempfile.mkdtemp()
    try:
        out = os.path.join(d, "run1")
        prov.write_run(out, {"manifest": {"x": 1}}, {"predictions.jsonl": [{"q": 1}, {"q": 2}]})
        res = prov.verify_run(out)
        assert set(res["output_files"]) == {"predictions.jsonl"}
        _expect(IOError, prov.write_run, out, {"manifest": {}}, {})          # never reuse a run dir
        _expect(IOError, prov.prepare_fresh_dir, out)
        with open(os.path.join(out, "predictions.jsonl"), "a", encoding="utf-8") as fh:
            fh.write('{"q": 3}\n')                                          # tampered companion
        _expect(IOError, prov.verify_run, out)
        out2 = os.path.join(d, "run2")
        prov.write_run(out2, {"m": 1}, {"predictions.jsonl": [{"q": 1}]})
        open(os.path.join(out2, "stray.txt"), "w").close()
        _expect(IOError, prov.verify_run, out2)                              # unexpected file
        out3 = os.path.join(d, "run3")
        os.makedirs(out3)
        _expect(IOError, prov.verify_run, out3)                              # no results.json
        _expect(ValueError, prov.write_run, os.path.join(d, "run4"), {}, {"results.json": []})
    finally:
        shutil.rmtree(d)


def test_paired_auroc_validates_every_value_before_masking():
    """A control value is validated even when its method value is undefined."""
    _expect(core.ContractError, core.paired_macro_auroc, {"q1": None, "q2": 0.5},
            {"q1": 7.0, "q2": 0.4})
    _expect(core.ContractError, core.paired_macro_auroc, {"q1": float("nan"), "q2": 0.5},
            {"q1": None, "q2": 0.4})
    ok = core.paired_macro_auroc({"q1": None, "q2": 0.5}, {"q1": 0.9, "q2": 0.4})
    assert ok["n_shared"] == 1 and abs(ok["difference"] - 0.1) < 1e-12


# ------------------------------------------------ CIRR manifest (§10) --
def _cirr_rows():
    """CIRR-format rows over images p0..p5 and q0..q3."""
    def row(pid, ref, tgt, soft):
        return {"pairid": pid, "reference": ref, "target_hard": tgt, "target_soft": soft,
                "caption": "make it red", "img_set": {}}
    return [row(0, "p0", "p1", {"p1": 1.0, "p2": 0.5}),
            row(1, "p1", "p3", {"p3": 0.5, "p4": 0.2, "q3": -1.0, "p1": -1.0}),   # hard target 0.5; ref -1
            row(2, "q0", "q1", {"q0": 1.0}),                 # reference labelled 1, hard target absent
            row(3, "q1", "q2", {"q2": 1.0, "p5": 1.0})]


def _expect_manifest_error(msg, fn, *a):
    try:
        fn(*a)
    except man.ManifestError as exc:
        assert msg in str(exc), (msg, str(exc))
        return
    raise AssertionError("accepted: " + msg)


def test_manifest_parses_cirr_rows_and_defines_positives():
    images = {"p%d" % i for i in range(6)} | {"q%d" % i for i in range(4)}
    trips, soft, anomalies = man.parse_captions(_cirr_rows(), images)
    assert [t["pairid"] for t in trips] == [0, 1, 2, 3]
    # partial matches are positives; -1, the reference itself and the hard target are not extra pairs
    assert sorted(soft) == [("p0", "p2"), ("p1", "p4"), ("q1", "p5")]
    assert anomalies == [{"pairid": 1, "kind": "reference_soft_label", "label": "-1.0"},
                         {"pairid": 1, "kind": "hard_target_soft_label", "label": "0.5"},
                         {"pairid": 2, "kind": "reference_soft_label", "label": "1.0"},
                         {"pairid": 2, "kind": "hard_target_soft_label", "label": "absent"}]
    assert man._anomaly_counts(anomalies) == {"hard_target_soft_label": {"0.5": 1, "absent": 1},
                                              "reference_soft_label": {"-1.0": 1, "1.0": 1}}
    assert trips[0]["caption"] == "make it red"
    pos = man.build_positives(trips, soft)
    assert "p5" in pos["q1"] and "q1" in pos["p5"] and "q3" not in pos["p1"]
    for change, msg in ((lambda r: r[1]["target_soft"].update(p4=0.7), "outside"),
                        (lambda r: r[0].update(reference="zz"), "not in the split map"),
                        (lambda r: r[3].update(pairid=0), "unique"),
                        (lambda r: r[2].update(caption="  "), "empty caption")):
        rows = _cirr_rows()
        change(rows)
        _expect_manifest_error(msg, man.parse_captions, rows, images)


def test_manifest_build_keeps_all_triplets_and_refuses_a_small_pool():
    split = {"%s%d" % (c, i): "./train/%s%d.png" % (c, i) for c, n in (("p", 6), ("q", 4)) for i in range(n)}
    split.update({"n%03d" % i: "./train/n%03d.png" % i for i in range(300)})
    rows = _cirr_rows() + [{"pairid": 10 + i, "reference": "n%03d" % (2 * i),
                            "target_hard": "n%03d" % (2 * i + 1), "target_soft": {},
                            "caption": "x", "img_set": {}} for i in range(40)]
    old = man.N_NEGATIVES
    try:
        man.N_NEGATIVES = 1
        _expect_manifest_error("keeps no triplets", man.build, _cirr_rows(), split)   # 2 groups, 3 partitions
        summary, comp = man.build(rows, split)
        trip_rows, image_rows = comp["triplets.jsonl"], comp["images.jsonl"]
        assert sorted(comp) == sorted(man.COMPANIONS)
        assert summary["triplet_counts"]["total"] == len(trip_rows) == len(rows)
        assert [t["caption"] for t in trip_rows] == [r["caption"] for r in rows]
        part_of = {r["image_id"]: r["partition"] for r in image_rows}
        assert len(part_of) == len(split)
        assert all(part_of[t["reference"]] == part_of[t["target"]] == t["partition"] for t in trip_rows)
        assert part_of["p5"] == part_of["q1"]               # the soft positive links two groups
        per = summary["linked_groups"]["per_partition"]
        assert sum(v["groups_with_triplets"] for v in per.values()) == summary["linked_groups"]["with_triplets"]
        man.N_NEGATIVES = summary["negative_pool"]["size"] + 1
        _expect_manifest_error("too small", man.build, rows, split)
    finally:
        man.N_NEGATIVES = old


def test_written_manifest_is_semantically_validated():
    """Integrity (hashes) is not semantics: a correctly hashed manifest whose
    target image sits in another partition must be rejected."""
    import shutil
    import tempfile
    split = {"%s%d" % (c, i): "./train/%s%d.png" % (c, i) for c, n in (("p", 6), ("q", 4)) for i in range(n)}
    split.update({"n%03d" % i: "./train/n%03d.png" % i for i in range(300)})
    rows = _cirr_rows() + [{"pairid": 10 + i, "reference": "n%03d" % (2 * i),
                            "target_hard": "n%03d" % (2 * i + 1), "target_soft": {},
                            "caption": "x", "img_set": {}} for i in range(40)]
    old = man.N_NEGATIVES
    d = tempfile.mkdtemp()
    try:
        man.N_NEGATIVES = 1
        summary, comp = man.build(rows, split)
        good = os.path.join(d, "good")
        prov.write_run(good, summary, comp)
        man.validate_written_manifest(good, rows, split)
        tampered = {k: [dict(r) for r in v] for k, v in comp.items()}
        tgt = tampered["triplets.jsonl"][0]["target"]
        img = next(r for r in tampered["images.jsonl"] if r["image_id"] == tgt)
        img["partition"] = next(n for n in part.PARTITION_NAMES if n != img["partition"])
        bad = os.path.join(d, "bad")
        prov.write_run(bad, summary, tampered)              # correctly hashed, semantically wrong
        prov.verify_run(bad)
        _expect_manifest_error("crosses", man.validate_written_manifest, bad, rows, split)
        tampered = {k: [dict(r) for r in v] for k, v in comp.items()}
        tampered["images.jsonl"][0]["group"] = "zzz"
        bad2 = os.path.join(d, "bad2")
        prov.write_run(bad2, summary, tampered)
        _expect_manifest_error("group key", man.validate_written_manifest, bad2, rows, split)
        tampered = {k: [dict(r) for r in v] for k, v in comp.items()}
        tampered["positives.jsonl"].pop()
        bad3 = os.path.join(d, "bad3")
        prov.write_run(bad3, summary, tampered)
        _expect_manifest_error("positives", man.validate_written_manifest, bad3, rows, split)
        # publication: out-dir receives results.json only after validation has passed
        out = os.path.join(d, "published")
        os.makedirs(out)
        wrong_source = [dict(r) for r in rows]
        wrong_source[0]["caption"] = "a different caption"
        _expect_manifest_error("does not match", man.publish_validated, out, summary, comp, wrong_source, split)
        assert os.listdir(out) == [] and os.path.isdir(man.staging_dir(out))   # nothing published
        out2 = os.path.join(d, "published2")
        os.makedirs(out2)
        man.publish_validated(out2, summary, comp, rows, split)
        assert not os.path.exists(man.staging_dir(out2))
        man.validate_written_manifest(out2, rows, split)
    finally:
        man.N_NEGATIVES = old
        shutil.rmtree(d)


# ------------------------------------------- Phase B extraction (CPU parts) --
def _unit32(n, d, seed):
    x = np.random.default_rng(seed).standard_normal((n, d))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _dump(n=5, k=3, d=8, g=10, seed=1):
    """A valid verifier-vector dump over a g-image gallery, with its manifest."""
    gallery = _unit32(g, d, seed)
    rng = np.random.default_rng(seed)
    cand = np.stack([rng.choice(g, size=k, replace=False) for _ in range(n)]).astype(np.int64)
    vec = {"query_id": np.arange(n, dtype=np.int64), "candidate_ids": cand,
           "t_hat": _unit32(n, d, seed + 1), "c_pool": gallery[cand]}
    vman = {"schema": ext.VECTOR_SCHEMA, "dataset": "cirr", "split": "val", "category": None,
            "module_combo": "fc+aep+rv", "n_queries": n, "gallery_size": g, "k": k, "dim": d}
    return gallery, vec, vman


def test_extract_reproduction_compares_every_encoded_occurrence():
    gallery, vec, _ = _dump()
    cand, t_hat, c_pool = vec["candidate_ids"], vec["t_hat"], vec["c_pool"]
    rows = list(range(len(gallery)))
    errs = ext.reproduction_errors(t_hat, gallery, rows, t_hat, c_pool, cand)
    assert errs["n_text_compared"] == 5 and errs["n_image_occurrences"] == cand.size
    ext.check_reproduction(errs)
    drifted = gallery.copy()
    drifted[int(cand[2, 1])] = _unit32(1, 8, 9)[0]
    try:
        ext.check_reproduction(ext.reproduction_errors(t_hat, drifted, rows, t_hat, c_pool, cand))
    except ext.ExtractError as exc:
        assert "image_max" in str(exc)
    else:
        raise AssertionError("a drifted image vector was accepted")
    text = t_hat.copy()
    text[4] = _unit32(1, 8, 4)[0]
    _expect(ext.ExtractError, ext.check_reproduction, ext.reproduction_errors(text, gallery, rows, t_hat, c_pool, cand))
    # a true cosine, not a bare dot product: a long, rotated old vector is not "identical"
    u = c_pool[0, 0].astype(np.float64)
    w = _unit32(1, 8, 11)[0].astype(np.float64)
    w = (w - (w @ u) * u) / np.linalg.norm(w - (w @ u) * u)          # orthogonal to u
    long_old = c_pool.copy()
    long_old[0, 0] = (2 * u + 2 * w).astype(np.float32)      # dot with u is 2 (clips to 1); cosine 0.707
    _expect(ext.ExtractError, ext.check_reproduction,
            ext.reproduction_errors(t_hat, gallery, rows, t_hat, long_old, cand))
    _expect(ext.ExtractError, ext.check_reproduction, dict(errs, image_max_1_minus_cos=float("nan")))
    _expect(ext.ExtractError, ext.check_reproduction, dict(errs, n_image_occurrences=0))


def test_extract_probe_always_covers_the_probed_queries():
    """Audit finding 1: a gallery prefix and a query prefix may not overlap;
    the probe instead encodes the probed queries' own cached candidates."""
    gallery, vec, _ = _dump(n=4, k=2, g=6)
    cand = np.array([[4, 5], [5, 3], [0, 1], [2, 4]], dtype=np.int64)
    c_pool = gallery[cand]
    _expect(ext.ExtractError, ext.reproduction_errors, vec["t_hat"][:2], gallery[:2], [0, 1],
            vec["t_hat"], c_pool, cand)                            # the old prefix design: no overlap
    rows = ext.probe_image_rows(cand, 2, 3)
    assert rows == [4, 5, 3]
    errs = ext.reproduction_errors(vec["t_hat"][:2], gallery[rows], rows, vec["t_hat"], c_pool, cand)
    assert errs["n_image_occurrences"] == 4 and errs["n_distinct_images"] == 3
    ext.check_reproduction(errs)
    assert ext.probe_image_rows(cand, 4, 2) == [4, 5]              # capped at one batch
    _expect(ext.ExtractError, ext.reproduction_errors, vec["t_hat"][:2], gallery[[4, 4]], [4, 4],
            vec["t_hat"], c_pool, cand)                            # an encoded row named twice


def test_extract_dump_contract_is_validated_before_use():
    _, vec, vman = _dump()
    ext.validate_dump(vec, vman, 5, 10, "cirr", None, "fc+aep+rv")

    def broken(**changes):
        v = {k: a.copy() for k, a in vec.items()}
        m = dict(vman)
        for key, val in changes.items():
            (v if key in v else m)[key] = val
        return v, m
    cand = vec["candidate_ids"]
    first = cand == cand[0, 0]
    cases = (broken(candidate_ids=np.where(first, -1, cand)),                       # negative id
             broken(candidate_ids=np.where(first, 10, cand)),                       # outside the gallery
             broken(candidate_ids=cand.astype(bool)),                               # boolean ids
             broken(candidate_ids=cand.astype(np.float64)),                         # float ids
             broken(candidate_ids=np.column_stack([cand[:, :2], cand[:, :1]])),     # a repeated candidate
             broken(t_hat=vec["t_hat"][:4]),                                        # shape mismatch
             broken(c_pool=vec["c_pool"][:, :2]),                                   # shape mismatch
             broken(c_pool=2 * vec["c_pool"]),                                      # not unit
             broken(t_hat=vec["t_hat"].astype(np.float64)),                         # wrong dtype
             broken(query_id=vec["query_id"][::-1].copy()),                         # wrong order
             broken(category="dress"), broken(split="test"), broken(module_combo="fc"),
             broken(dim=16), broken(k=4))
    for v, m in cases:
        _expect(ext.ExtractError, ext.validate_dump, v, m, 5, 10, "cirr", None, "fc+aep+rv")
    _expect(ext.ExtractError, ext.validate_dump, vec, vman, 5, 9, "cirr", None, "fc+aep+rv")


def test_extract_train_alignment_and_strict_inputs():
    rows = [{"pairid": 0, "reference": "a", "target": "b", "caption": "red"},
            {"pairid": 1, "reference": "b", "target": "c", "caption": "blue"}]
    ds = [{"text": "red", "meta": {"pairid": 0, "reference_id": "a", "target_id": "b"}},
          {"text": "blue", "meta": {"pairid": 1, "reference_id": "b", "target_id": "c"}}]
    idx = ext.check_train_alignment(rows, ds, ["c", "a", "b"])
    assert idx["pairid"].tolist() == [0, 1] and idx["ref_row"].tolist() == [1, 2] and idx["tgt_row"].tolist() == [2, 0]
    for key, val in (("text", "green"), ("meta", {"pairid": 1, "reference_id": "a", "target_id": "c"})):
        bad = [dict(d) for d in ds]
        bad[1][key] = val
        _expect(ext.ExtractError, ext.check_train_alignment, rows, bad, ["c", "a", "b"])
    _expect(ext.ExtractError, ext.check_train_alignment, rows[:1], ds, ["c", "a", "b"])
    assert ext.rows_of(["x", "z"], {"x": 0, "y": 1, "z": 2}).tolist() == [0, 2]
    _expect(ext.ExtractError, ext.rows_of, ["q"], {"x": 0})
    _expect(ext.ExtractError, ext.require_files, [__file__, __file__ + ".missing"])
    ext.check_unit_rows("ok", _unit32(3, 4, 5), 2)
    _expect(ext.ExtractError, ext.check_unit_rows, "f64", _unit32(3, 4, 5).astype(np.float64), 2)
    _expect(ext.ExtractError, ext.check_unit_rows, "scaled", 2 * _unit32(3, 4, 5), 2)


def test_extract_corrupt_image_raises_through_the_real_loader():
    """Audit finding 4: strictness is a property of the actual load path, not
    of a file-existence check."""
    import shutil
    import tempfile
    from PIL import Image
    d = tempfile.mkdtemp()
    try:
        good, corrupt, truncated = (os.path.join(d, n) for n in ("good.png", "corrupt.png", "truncated.png"))
        Image.new("RGB", (64, 64), (10, 20, 30)).save(good)
        with open(corrupt, "wb") as fh:
            fh.write(b"not an image at all")
        with open(good, "rb") as fh:
            data = fh.read()
        with open(truncated, "wb") as fh:
            fh.write(data[: len(data) // 2])
        files = ext.ImageFiles([good, corrupt, truncated], np.asarray)
        assert files[0].shape == (64, 64, 3)
        for i in (1, 2):
            try:
                files[i]
            except OSError:
                pass
            else:
                raise AssertionError("an unreadable image (%d) was loaded" % i)
    finally:
        shutil.rmtree(d)


def test_extract_resume_reuses_only_a_verified_run_of_this_target():
    import shutil
    import tempfile
    d = tempfile.mkdtemp()
    try:
        out = os.path.join(d, "t")
        assert not ext.completed_run(out, "cirr_val", None, "abc")            # absent
        os.makedirs(out)
        assert not ext.completed_run(out, "cirr_val", None, "abc")            # empty
        ext.publish_npz_run(out, {"x": _unit32(2, 4, 1)},
                            {"schema": ext.SCHEMA, "target": "cirr_val", "category": None,
                             "manifest": {"checkpoint_sha256": "abc"}})
        assert ext.completed_run(out, "cirr_val", None, "abc")
        _expect(ext.ExtractError, ext.completed_run, out, "cirr_val", None, "other")   # other checkpoint
        _expect(ext.ExtractError, ext.completed_run, out, "fiq_val", "dress", "abc")   # other target
        with open(os.path.join(out, ext.FEATURES_NAME), "ab") as fh:
            fh.write(b"x")                                                     # tampered
        _expect(ext.ExtractError, ext.completed_run, out, "cirr_val", None, "abc")
        os.makedirs(man.staging_dir(os.path.join(d, "u")))
        _expect(ext.ExtractError, ext.completed_run, os.path.join(d, "u"), "cirr_val", None, "abc")
    finally:
        shutil.rmtree(d)


def test_extract_refuses_a_manifest_that_is_not_the_approved_allocation():
    import shutil
    import tempfile
    split = {"%s%d" % (c, i): "./train/%s%d.png" % (c, i) for c, n in (("p", 6), ("q", 4)) for i in range(n)}
    split.update({"n%03d" % i: "./train/n%03d.png" % i for i in range(300)})
    rows = _cirr_rows() + [{"pairid": 10 + i, "reference": "n%03d" % (2 * i),
                            "target_hard": "n%03d" % (2 * i + 1), "target_soft": {},
                            "caption": "x", "img_set": {}} for i in range(40)]
    old_n, old_sha = man.N_NEGATIVES, ext.APPROVED_IMAGES_SHA256
    d = tempfile.mkdtemp()
    try:
        cap, spl = os.path.join(d, "cap.json"), os.path.join(d, "split.json")
        with open(cap, "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
        with open(spl, "w", encoding="utf-8") as fh:
            json.dump(split, fh)
        man.N_NEGATIVES = 1
        summary, comp = man.build(rows, split)
        summary["manifest"] = {"inputs_sha256": {"captions": prov.sha256_file(cap), "splits": prov.sha256_file(spl)}}
        out = os.path.join(d, "m")
        os.makedirs(out)
        man.publish_validated(out, summary, comp, rows, split)
        try:                                               # valid, but not the pinned allocation
            ext.check_manifest(out, cap, spl)
        except ext.ExtractError as exc:
            assert "approved allocation" in str(exc)
        else:
            raise AssertionError("an unapproved allocation was accepted")
        ext.APPROVED_IMAGES_SHA256 = prov.sha256_file(os.path.join(out, "images.jsonl"))
        got = ext.check_manifest(out, cap, spl)
        assert len(got["triplets"]) == len(rows) and len(got["images"]) == len(split)
        with open(spl, "a", encoding="utf-8") as fh:        # same content, other bytes: other inputs
            fh.write("\n")
        try:
            ext.check_manifest(out, cap, spl)
        except ext.ExtractError as exc:
            assert "other caption/split files" in str(exc)
        else:
            raise AssertionError("a manifest built from other inputs was accepted")
    finally:
        man.N_NEGATIVES, ext.APPROVED_IMAGES_SHA256 = old_n, old_sha
        shutil.rmtree(d)


def test_extract_publication_is_reloaded_verified_and_atomic():
    import shutil
    import tempfile
    d = tempfile.mkdtemp()
    try:
        arrays = {"image_emb": _unit32(4, 8, 6), "image_ids": np.asarray(["a", "b", "c", "d"]),
                  "ref_row": np.arange(4, dtype=np.int64)}
        out = os.path.join(d, "f")
        os.makedirs(out)
        ext.publish_npz_run(out, arrays, {"schema": ext.SCHEMA})
        assert not os.path.exists(man.staging_dir(out))
        res = prov.verify_run(out)
        assert res["schema"] == ext.SCHEMA and list(res["output_files"]) == [ext.FEATURES_NAME]
        with np.load(os.path.join(out, ext.FEATURES_NAME), allow_pickle=False) as z:
            assert all(np.array_equal(z[k], v) and z[k].dtype == v.dtype for k, v in arrays.items())
        out2 = os.path.join(d, "g")
        os.makedirs(out2)
        bad = dict(arrays, meta=np.array([{"x": 1}], dtype=object))   # needs pickle to re-load
        try:
            ext.publish_npz_run(out2, bad, {"schema": ext.SCHEMA})
        except ValueError:
            pass
        else:
            raise AssertionError("an array that cannot be re-loaded safely was published")
        assert os.listdir(out2) == []                      # nothing published
    finally:
        shutil.rmtree(d)


# --------------------------------------------- Phase C head training (§11) --
def _train_feat(seed=3):
    """Synthetic cirr_train-shaped features: 12 images (4 per partition used),
    6 triplets, unit 768-d rows."""
    parts = np.array(["head"] * 4 + ["dev_select"] * 4 + ["negbank"] * 4)
    return {"image_ids": np.array(["i%02d" % i for i in range(12)]), "partition": parts,
            "image_emb": _unit32(12, 768, seed), "text_emb": _unit32(6, 768, seed + 1),
            "ref_row": np.array([0, 1, 2, 4, 5, 6]), "tgt_row": np.array([1, 2, 3, 5, 6, 7]),
            "pairid": np.arange(6)}


def test_train_negative_sampling_is_uniform_without_replacement_and_seeded():
    cols = trn.sample_negative_columns(2000, 6, 3, np.random.default_rng(0))
    assert cols.shape == (2000, 3) and all(len(set(r)) == 3 for r in cols.tolist())
    freq = np.bincount(cols.ravel(), minlength=6) / cols.size
    assert np.abs(freq - 1 / 6).max() < 0.02                       # uniform over the pool
    again = trn.sample_negative_columns(2000, 6, 3, np.random.default_rng(0))
    assert np.array_equal(cols, again)
    _expect(trn.TrainError, trn.sample_negative_columns, 2, 3, 4, np.random.default_rng(0))


def test_train_loss_is_the_m1_mixture_over_the_candidate_set():
    import torch
    rng = np.random.default_rng(5)
    hyp, pos, pool = _unit32(4, 768, 6), _unit32(4, 768, 7), _unit32(9, 768, 8)
    cols = trn.sample_negative_columns(4, 9, 5, rng)
    got = trn.candidate_nll(torch.from_numpy(hyp), torch.from_numpy(pos), torch.from_numpy(pool),
                            torch.from_numpy(cols), 20.0).numpy()
    for i in range(4):
        support = np.concatenate([pos[i:i + 1], pool[cols[i]]]).astype(np.float64)
        lq = core.mixture_log_prob(hyp[i:i + 1].astype(np.float64), np.zeros(1), 20.0, support)
        assert abs(got[i] - (-lq[0])) < 1e-4, (got[i], -lq[0])


def test_train_split_directions_and_positive_pool_guard():
    feat = _train_feat()
    _expect(trn.TrainError, trn.split_examples, feat, {})                  # pool of 4 < 1,023 negatives
    old = trn.N_NEG
    trn.N_NEG = 3
    try:
        _check_split_and_directions(feat)
    finally:
        trn.N_NEG = old


def _check_split_and_directions(feat):
    split = trn.split_examples(feat, {})
    assert split["head"].tolist() == [0, 1, 2] and split["dev_select"].tolist() == [3, 4, 5]
    assert split["pool"].tolist() == [8, 9, 10, 11]
    x_f, p_f = trn.direction_arrays(feat, split["head"], "forward")
    x_i, p_i = trn.direction_arrays(feat, split["head"], "inverse")
    assert np.array_equal(x_f[:, :768], feat["image_emb"][[0, 1, 2]])       # forward conditions on r
    assert np.array_equal(p_f, feat["image_emb"][[1, 2, 3]])                # and predicts the target
    assert np.array_equal(x_i[:, :768], feat["image_emb"][[1, 2, 3]])       # inverse conditions on c
    assert np.array_equal(p_i, feat["image_emb"][[0, 1, 2]])                # and predicts the reference
    assert np.array_equal(x_f[:, 768:], feat["text_emb"][[0, 1, 2]])
    _expect(trn.TrainError, trn.direction_arrays, feat, split["head"], "sideways")
    _expect(trn.TrainError, trn.split_examples, feat, {"i00": {"i09"}})     # a positive in the pool
    crossing = dict(feat, tgt_row=np.array([1, 2, 4, 5, 6, 7]))
    _expect(trn.TrainError, trn.split_examples, crossing, {})


def test_train_selection_and_determinism():
    runs = {5.0: {"best_dev_nll": 2.0}, 10.0: {"best_dev_nll": 1.5}, 20.0: {"best_dev_nll": 1.5}}
    assert trn.select(runs) == 10.0                                          # ties -> smaller kappa
    feat = _train_feat()
    old = trn.N_NEG
    try:
        trn.N_NEG = 3
        split = trn.split_examples(feat, {})
        data = {"train": trn.direction_arrays(feat, split["head"], "forward"),
                "dev": trn.direction_arrays(feat, split["dev_select"], "forward"),
                "pool_emb": feat["image_emb"][split["pool"]],
                "dev_cols": trn.sample_negative_columns(3, 4, 3, np.random.default_rng(1))}
        a = trn.train_one("forward", 0, 10.0, data, 3)
        b = trn.train_one("forward", 0, 10.0, data, 3)
        c = trn.train_one("inverse", 0, 10.0, data, 3)
    finally:
        trn.N_NEG = old
    assert a["dev_nll"] == b["dev_nll"] and a["train_nll"] == b["train_nll"]           # deterministic
    assert all(np.array_equal(a["state"][k].numpy(), b["state"][k].numpy()) for k in a["state"])
    assert a["best_dev_nll"] == min(a["dev_nll"]) and a["dev_nll"][a["best_epoch"] - 1] == a["best_dev_nll"]
    assert not np.array_equal(a["state"]["hyp.weight"].numpy(), c["state"]["hyp.weight"].numpy())   # own init


# ------------------------------------------------ Phase D scoring (§12) --
def test_score_banks_are_disjoint_seeded_and_from_negbank():
    ids = np.array(["n%02d" % i for i in range(20)])
    part = np.array(["negbank"] * 12 + ["head"] * 8)
    b = scr.make_banks(ids, part, seed=7, size=5)
    assert len(b["B_R"]) == len(b["B_C"]) == 5 and not set(b["B_R"]) & set(b["B_C"])
    assert set(b["B_R"]) | set(b["B_C"]) <= set(range(12))
    again = scr.make_banks(ids[::-1].copy(), part[::-1].copy(), seed=7, size=5)   # row order does not matter
    assert ids[::-1][again["B_R"]].tolist() == ids[b["B_R"]].tolist()
    _expect(scr.ScoreError, scr.make_banks, ids, part, 7, 7)                     # 12 < 2 x 7


def test_score_hard_negative_window_excludes_reference_and_positives():
    emb = _unit32(8, 4, 2).astype(np.float64)
    rows = np.arange(10, 18)
    q = emb[3] + 0.01 * emb[5]
    win = scr.hard_negative_window(q, rows, emb, 17, {13, 17, 15}, k=4)
    assert win[0] == 17 and len(set(win.tolist())) == 4
    assert not {13, 15} & set(win[1:].tolist())
    eligible = [r for r in rows.tolist() if r not in {13, 15, 17}]
    top = sorted(eligible, key=lambda r: -float(emb[r - 10] @ q))[:3]
    assert win[1:].tolist() == top
    _expect(scr.ScoreError, scr.hard_negative_window, q, rows, emb, 17, set(range(10, 16)), 4)


def test_score_forward_window_matches_direct_computation():
    """R_q = B_R ∪ {r}, uniform prior, C_q = B_C ∪ W: the marginal, raw score
    and PMI recomputed by hand from the head's own hypotheses."""
    feat = _unit32(14, 768, 21).astype(np.float64)
    text = _unit32(1, 768, 22)[0].astype(np.float64)
    head = trn.build_head(0)
    head.eval()
    bank_r, bank_c, window, ref = np.array([0, 1, 2]), np.array([3, 4, 5, 6]), np.array([7, 8, 4]), 9
    got = scr.forward_window(head, 20.0, feat, text, ref, bank_r, bank_c, window)
    r_ids, c_ids = [0, 1, 2, 9], [3, 4, 5, 6, 7, 8]          # window image 4 is already in B_C
    hyp = scr.hypotheses(head, feat[r_ids], text[None, :])[:, 0]
    logits = 20.0 * hyp @ feat[c_ids].T
    log_f = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    cols = [c_ids.index(w) for w in (7, 8, 4)]
    marg = np.log(np.exp(log_f[:, cols]).mean(0))
    assert np.allclose(got["log_f_raw"], log_f[3, cols], atol=1e-10)
    assert np.allclose(got["log_marginal"], marg, atol=1e-10)
    assert np.allclose(got["pmi_f"], log_f[3, cols] - marg, atol=1e-10)
    in_bank = scr.forward_window(head, 20.0, feat, text, 1, bank_r, bank_c, window)   # reference already in B_R
    hyp_b = scr.hypotheses(head, feat[[0, 1, 2]], text[None, :])[:, 0]
    lb = 20.0 * hyp_b @ feat[c_ids].T
    lfb = lb - np.log(np.exp(lb).sum(1, keepdims=True))
    assert np.allclose(in_bank["log_marginal"], np.log(np.exp(lfb[:, cols]).mean(0)), atol=1e-10)


def test_score_inverse_window_uses_the_forward_reference_universe():
    feat = _unit32(14, 768, 31).astype(np.float64)
    text = _unit32(1, 768, 32)[0].astype(np.float64)
    head = trn.build_head(1000)
    head.eval()
    bank_r, window = np.array([0, 1, 2]), np.array([7, 8, 4])
    for ref, r_ids in ((9, [0, 1, 2, 9]), (1, [0, 1, 2])):               # reference outside / inside B_R
        got = scr.inverse_window(head, 40.0, feat, text, ref, bank_r, window)
        hyp = scr.hypotheses(head, feat[window], text[None, :])[:, 0]
        lg = 40.0 * hyp @ feat[r_ids].T
        log_i = lg - np.log(np.exp(lg).sum(1, keepdims=True))
        assert np.allclose(got, log_i[:, r_ids.index(ref)], atol=1e-10)
        assert scr.reference_universe(bank_r, ref) == r_ids


def _val_fixture(d, n_q=3, n_img=12):
    """A verified validation-feature run, its cache and vector-dump manifest."""
    feats = os.path.join(d, "val")
    os.makedirs(feats)
    tgt = np.array([2, 5, 7])
    ext.publish_npz_run(feats, {"image_emb": _unit32(n_img, 4, 1), "text_emb": _unit32(n_q, 4, 2),
                                "ref_row": np.array([0, 1, 3]), "tgt_row": tgt},
                        {"schema": ext.SCHEMA, "target": "cirr_val", "category": None,
                         "vectors_sha256": "v" * 64, "manifest": {"checkpoint_sha256": "c" * 64}})
    cache = os.path.join(d, "cache.jsonl")
    with open(cache, "w", encoding="utf-8") as fh:
        for q in range(n_q):
            cand = [int(tgt[q])] + [i for i in range(n_img) if i != tgt[q]][:9]
            fh.write(json.dumps({"query_id": q, "candidate_gallery_ids": cand, "target_gallery_index": int(tgt[q]),
                                 "target_pos": 0, "base_score": [0.5] * 10, "verification_score": [0.5] * 10,
                                 "edit_cos": [0.1] * 10}) + "\n")
    vman = os.path.join(d, "vec.manifest.json")
    with open(vman, "w", encoding="utf-8") as fh:
        json.dump({"checkpoint_sha256": "c" * 64, "npz_sha256": "v" * 64, "cache_sha256": prov.sha256_file(cache)}, fh)
    return feats, cache, vman


def test_score_validation_inputs_are_bound_together():
    import shutil
    import tempfile
    d = tempfile.mkdtemp()
    try:
        feats, cache, vman = _val_fixture(d)
        got = scr.load_validation(feats, cache, vman, "c" * 64)
        assert len(got["recs"]) == 3
        _expect(scr.ScoreError, scr.load_validation, feats, cache, vman, "x" * 64)      # other checkpoint
        lines = open(cache, encoding="utf-8").read().splitlines()
        rec = json.loads(lines[1])

        def variant(name, change):
            r = dict(rec)
            change(r)
            path = os.path.join(d, name)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join([lines[0], json.dumps(r), lines[2]]) + "\n")
            m = os.path.join(d, name + ".manifest.json")
            with open(m, "w", encoding="utf-8") as fh:
                json.dump({"checkpoint_sha256": "c" * 64, "npz_sha256": "v" * 64,
                           "cache_sha256": prov.sha256_file(path)}, fh)
            return path, m
        assert rec["candidate_gallery_ids"].index(6) == 6
        for name, change in (("tgt", lambda r: r.update(target_gallery_index=6, target_pos=6)),   # consistent cache,
                                                                                                # wrong for the dataset
                             ("pos", lambda r: r.update(target_pos=3)),
                             ("edit", lambda r: r.update(edit_cos=[0.1] * 9)),
                             ("order", lambda r: r.update(query_id=2))):
            path, m = variant(name, change)
            _expect(scr.ScoreError, scr.load_validation, feats, path, m, "c" * 64)
        path, _ = variant("unrecorded", lambda r: r.update(base_score=[0.4] * 10))
        _expect(scr.ScoreError, scr.load_validation, feats, path, vman, "c" * 64)       # not the dumped cache
    finally:
        shutil.rmtree(d)


def test_score_lambda_selection_takes_the_best_mean_and_the_smaller_tie():
    assert scr.select_lambda({0.0: [0.6, 0.62], 0.5: [0.7, 0.7], 1.0: [0.65, 0.7]}) == 0.5
    assert scr.select_lambda({0.0: [0.5], 0.25: [0.7], 0.75: [0.7]}) == 0.25


def test_score_heads_load_only_from_a_full_run_on_these_features():
    import shutil
    import tempfile
    d = tempfile.mkdtemp()
    try:
        feats = os.path.join(d, "feats")
        os.makedirs(feats)
        ext.publish_npz_run(feats, {"x": _unit32(2, 4, 1)}, {"schema": ext.SCHEMA})
        arrays, sel = {}, {}
        for direction in trn.DIRECTIONS:
            for s in trn.SEEDS:
                key = "%s_s%d" % (direction, s)
                for name, v in trn.build_head(s).state_dict().items():
                    arrays["%s/%s" % (key, name)] = v.numpy()
                arrays[key + "/kappa"] = np.asarray(40.0, dtype=np.float32)
                sel[key] = {"selected_kappa": 40.0}
        fsha = prov.sha256_file(os.path.join(feats, prov.RESULTS_NAME))

        def heads_run(name, smoke=False, features_sha=fsha, selections=sel):
            out = os.path.join(d, name)
            os.makedirs(out)
            ext.publish_npz_run(out, arrays, {"schema": trn.SCHEMA, "smoke": smoke, "selections": selections,
                                              "inputs": {"features_results_sha256": features_sha}},
                                name=trn.HEADS_NAME)
            return out
        heads = scr.load_heads(heads_run("ok"), feats)
        assert sorted(heads) == sorted(sel) and all(h["kappa"] == 40.0 for h in heads.values())
        _expect(scr.ScoreError, scr.load_heads, heads_run("smoke", smoke=True), feats)
        _expect(scr.ScoreError, scr.load_heads, heads_run("other", features_sha="0" * 64), feats)
        bad_sel = dict(sel, forward_s0={"selected_kappa": 80.0})
        _expect(scr.ScoreError, scr.load_heads, heads_run("kappa", selections=bad_sel), feats)
    finally:
        shutil.rmtree(d)


# ------------------------------------------------ Phase D analysis (§14) --
def _window_row(seed, q=0, split="tune", tp=0):
    rng = np.random.default_rng(seed)
    heads = {}
    for s in trn.SEEDS:
        heads["F%d" % s] = {k: rng.normal(size=10).tolist() for k in ("raw", "marginal", "pmi", "induced")}
        heads["I%d" % s] = {"ie": rng.normal(size=10).tolist()}
    return {"query": q, "split": split, "target_pos": tp, "window": list(range(10)),
            "base_score": rng.normal(size=10).tolist(), "edit_cos": rng.normal(size=10).tolist(),
            "verification_score": rng.normal(size=10).tolist(), "heads": heads}


def test_analysis_arms_follow_the_declared_components():
    assert ana.arm_components("F+F'", 2) == ["F2", "F0"] and ana.arm_components("I+I'", 0) == ["I0", "I1"]
    assert ana.arm_components("F+I", 1) == ["F1", "I1"] and ana.arm_components("Ind", 2) == ["Ind2"]
    _expect(ana.AnalysisError, ana.arm_components, "F+X", 0)
    row = _window_row(3)
    sig = ana.signals(row, 0.25)
    f0 = row["heads"]["F0"]
    assert np.allclose(sig["F0"], np.asarray(f0["raw"]) - 0.25 * np.asarray(f0["marginal"]))
    v = ana.arm_score(sig, "F+I", 0, row["edit_cos"])
    manual = 0.5 * (core.zscore(sig["F0"]) + core.zscore(sig["I0"])) + core.zscore(np.asarray(row["edit_cos"]))
    assert np.allclose(v, manual)


def test_analysis_utility_and_folds():
    view = {"changed": True, "hit_fused": 1, "hit_base": 0}
    assert ana.query_utility(view, True) == 1.0 and ana.query_utility(view, False) == 0.0
    assert ana.query_utility(dict(view, hit_fused=0, hit_base=1), True) == -2.0
    assert ana.query_utility(dict(view, hit_fused=0), True) == 0.0
    assert ana.query_utility(dict(view, changed=False), True) == 0.0
    labels = [1] * 12 + [0] * 8
    folds = ana.stratified_folds(labels, 5, 7)
    assert sorted(i for f in folds for i in f) == list(range(20))
    assert all(sum(labels[i] for i in f) in (2, 3) and len(f) == 4 for f in folds)
    assert folds == ana.stratified_folds(labels, 5, 7)


def test_analysis_weighted_logistic_is_the_penalised_optimum_and_respects_weights():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + 0.5 * rng.normal(size=200) > 0).astype(float)
    w = np.where(y == 1, 1.0, 2.0)
    predict, params = ana.fit_weighted_logistic(X, y, w, 1.0)
    beta = np.asarray(params["beta"])
    Z = np.hstack([np.ones((200, 1)), (X - params["mean"]) / params["sd"]])
    p = predict(X)
    R = np.eye(4)
    R[0, 0] = 0.0
    assert np.abs(Z.T @ (w * (p - y)) + R @ beta).max() < 1e-8         # stationarity of the weighted objective
    p_unw = ana.fit_weighted_logistic(X, y, np.ones(200), 1.0)[0](X)
    assert p.mean() < p_unw.mean()                                      # breaks weigh double: fewer acceptances


def test_analysis_gate_views_features_and_labels():
    row = _window_row(5, tp=3)
    v = ana.gate_views([row], 0.0, 0)[0]
    f = ana.fused_top(row["base_score"], row["verification_score"])
    b = core_top = max(range(10), key=lambda i: (row["base_score"][i], -i))
    sig = ana.signals(row, 0.0)
    delta = sig["I0"] - sig["Ind0"]
    assert np.isclose(v["feat"]["Delta_c"], delta[f] - delta.mean()) and np.isclose(v["feat"]["Delta_m"], delta[f] - delta[b])
    same = sig["I0"] - sig["I1"]
    assert np.isclose(v["feat"]["Same_c"], same[f] - same.mean())
    expected = (None if f == core_top else 1 if f == 3 else 0 if b == 3 else None)
    assert v["changed"] == (f != b) and v["label"] == expected
    assert ana.feature_names("delta")[-4:] == ["Delta_c", "Delta_m", "S_c", "S_m"]
    _expect(ana.AnalysisError, ana.feature_names, "oracle")


def test_analysis_gates_select_C_and_the_strongest_control_on_tune_only():
    rng = np.random.default_rng(9)

    def views(n):
        out = []
        for _ in range(n):
            x = rng.normal()
            changed = rng.random() < 0.6
            rescue = changed and (x + 0.3 * rng.normal() > 0)
            brk = changed and not rescue and rng.random() < 0.8
            feat = {k: rng.normal() for g in ana.GATES for k in ana.feature_names(g)}
            feat["Delta_c"] = x
            out.append({"changed": changed, "hit_fused": int(rescue), "hit_base": int(brk), "feat": feat,
                        "label": (1 if rescue else 0 if brk else None) if changed else None})
        return out
    tune, report = views(400), views(400)
    names = ana.feature_names("delta")
    C, cv = ana.select_C(tune, names)
    best = max(cv.values())
    assert C == min(c for c in ana.C_GRID if cv[str(c)] == best)
    old = ana.cv_utility
    try:
        ana.cv_utility = lambda *a, **k: 0.25                            # every C ties
        assert ana.select_C(tune, names)[0] == min(ana.C_GRID)           # ties go to the smaller C
    finally:
        ana.cv_utility = old
    # the gate is exactly the 1:2 weighted fit with the 0.5 rule
    accept, params = ana._fit_gate(tune, names, 1.0)
    lab = [v for v in tune if v["label"] is not None]
    y = np.asarray([v["label"] for v in lab], dtype=float)
    predict, ref = ana.fit_weighted_logistic(ana._matrix(lab, names), y, np.where(y == 1, 1.0, 2.0), 1.0)
    assert np.allclose(params["beta"], ref["beta"])
    assert accept(report) == [bool(p > 0.5) for p in predict(ana._matrix(report, names))]
    res = ana.run_gates(tune, report, 200)
    assert res["strongest_control"] in ana.CONTROLS
    assert res["gates"]["delta"]["mean_utility"] > 0                      # the informative signal helps
    assert "minus_strongest_control" in res["gates"]["delta"]
    assert "minus_strongest_control" not in res["gates"][res["strongest_control"]]
    for name, g in res["gates"].items():
        assert g["rescues"] - 2 * g["breaks"] == round(g["mean_utility"] * len(report))


# ----------------------------------------- frozen-gate transfer (§16) --
def test_transfer_frozen_gate_reproduces_the_fitted_gate_without_refitting():
    rng = np.random.default_rng(12)
    names = ana.feature_names("forward_only")
    views = []
    for _ in range(300):
        feat = {k: rng.normal() for k in names}
        lab = int(feat["F_c"] + 0.5 * rng.normal() > 0)
        views.append({"changed": True, "hit_fused": lab, "hit_base": 1 - lab, "feat": feat, "label": lab})
    accept, params = ana._fit_gate(views, names, 1.0)
    assert tfr.frozen_gate(params, names)(views) == accept(views)
    shifted = dict(params, mean=[m + 1.0 for m in params["mean"]])          # the stored means are used
    assert tfr.frozen_gate(shifted, names)(views) != accept(views)
    _expect(tfr.TransferError, tfr.frozen_gate, dict(params, beta=params["beta"][:-1]), names)
    _expect(tfr.TransferError, tfr.frozen_gate, dict(params, sd=[0.0] * len(names)), names)


def test_transfer_summary_reports_the_ceiling_against_the_comparator():
    def v(changed, hf, hb):
        return {"changed": changed, "hit_fused": hf, "hit_base": hb}
    views = [v(True, 1, 0), v(True, 1, 0), v(True, 0, 1), v(True, 0, 0), v(False, 1, 1), v(False, 0, 0)]
    acc = {"forward_only": [True, True, False, True, False, False],      # 2 rescues kept, no break: u = 2/6
           "base_only": [True, False, True, False, False, False]}        # 1 rescue, 1 break: u = -1/6
    s = tfr.summarise(views, acc, 200)
    assert s["rescues_available"] == 2 and s["breaks_available"] == 1 and s["changed"] == 4
    assert abs(s["u_oracle"] - 2 / 6) < 1e-12 and abs(s["u_ungated_fusion"] - 0.0) < 1e-12
    assert abs(s["gates"]["base_only"]["mean_utility_vs_base"] + 1 / 6) < 1e-12
    assert abs(s["gates"]["forward_only"]["mean_utility_vs_base"] - 2 / 6) < 1e-12
    assert abs(s["oracle_ceiling_vs_control"] - 3 / 6) < 1e-12      # oracle minus a NEGATIVE control: larger
    assert abs(s["paired_difference"]["diff"] - 3 / 6) < 1e-12
    assert s["gates"]["forward_only"]["acceptance_rate_of_changed"] == 0.75


# ---------------------------------------- cross-fitted policies (§17) --
def test_policy_matched_acceptance_uses_scores_only():
    scores = [0.9, 0.1, 0.5, 0.7, 0.3]
    changed = [True, True, False, True, True]
    assert pol.matched_acceptance(scores, changed, 2, True) == [True, False, False, True, False]
    assert pol.matched_acceptance(scores, changed, 2, False) == [False, True, False, False, True]
    assert pol.matched_acceptance([0.5] * 5, changed, 2, True) == [True, True, False, False, False]   # ties by index
    _expect(pol.PolicyError, pol.matched_acceptance, scores, changed, 5, True)


def test_policy_utility_and_outcome():
    assert pol.utility([1, 0, 1, 0], [0, 1, 1, 0]) == (1 - 2 + 0 + 0) / 4
    view = {"changed": True, "hit_fused": 1, "hit_base": 0}
    assert pol.outcome(view, True) == 1 and pol.outcome(view, False) == 0
    assert pol.outcome(dict(view, changed=False), True) == 0
    assert pol.threshold_record(float("-inf")) == "-inf" and pol.threshold_record(0.25) == 0.25


def test_policy_bootstrap_resamples_whole_clusters():
    diff = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    one = pol.clustered_bootstrap(diff, [0, 0, 1, 2, 3, 4], 2000)
    assert abs(one["diff"] - 2 / 6) < 1e-12 and one["n_clusters"] == 5
    ind = pol.clustered_bootstrap(diff, [0, 1, 2, 3, 4, 5], 2000)
    assert one["ci_high"] - one["ci_low"] > ind["ci_high"] - ind["ci_low"]     # correlated pair widens the CI
    flat = pol.clustered_bootstrap([0.5] * 6, [0, 0, 1, 1, 2, 2], 500)
    assert flat["ci_low"] == flat["ci_high"] == 0.5


def test_policy_simple_control_order_is_conservative_first():
    assert pol.SIMPLE == ("R", "BM", "SW", "A")
    fit_u = {"R": 0.0, "BM": 0.0, "SW": 0.0, "A": -0.01}
    assert next(k for k in pol.SIMPLE if fit_u[k] == max(fit_u.values())) == "R"


# ----------------------------------------------- CIRR test submission (§18) --
def test_cirr_submission_format_is_enforced():
    pairids, known = [11, 12], {"a", "b", "c"}
    good = {"version": "rc2", "metric": "recall", "11": ["a", "b"], "12": ["c"]}
    ctest.validate_submission(good, pairids, known)
    for bad in (dict(good, version="v0"), dict(good, metric="recall_subset"),
                {k: v for k, v in good.items() if k != "12"},                 # a pair missing
                dict(good, extra=["a"]),                                       # an extra key
                dict(good, **{"11": ["a", "a"]}),                              # a repeated id
                dict(good, **{"11": ["zz"]}),                                  # an unknown id
                dict(good, **{"11": [1]}),                                     # not a string
                dict(good, **{"11": []}),                                      # empty
                dict(good, **{"11": ["a"] * 51})):                             # too many
        _expect(ctest.TestError, ctest.validate_submission, bad, pairids, known)
    assert ctest.compact(good) == json.dumps(good, separators=(",", ":"))


def test_cirr_rankings_follow_the_frozen_systems():
    base, ver = [0.9, 0.8, 0.1], [0.0, 1.0, 0.0]
    order = ctest.fused_order(base, ver, 1.0)
    assert order[0] == 1                                       # the verifier lifts position 1
    assert ctest.system_ranking([7, 8, 9, 3, 4], order) == [8, 7, 9, 3, 4]
    assert ctest.recall_at([[1, 2], [3, 4]], np.array([2, 5]), ks=(1, 2)) == {"R@1": 0.0, "R@2": 50.0}
    rows = [_window_row(21, q=q, tp=-1) for q in range(4)]
    for r in rows:
        r["base_score"] = sorted(r["base_score"], reverse=True)               # window in base order
    arr = {"top50": np.array([list(range(q * 20, q * 20 + 12)) for q in range(4)])}
    names = ana.feature_names("forward_only")
    accept_all = {"beta": [10.0] + [0.0] * len(names), "mean": [0.0] * len(names), "sd": [1.0] * len(names)}
    reject_all = dict(accept_all, beta=[-10.0] + [0.0] * len(names))
    views = ana.gate_views(rows, 0.0, 0)
    for t3, expect_t3 in (({"choice": "SW", "sw_lambda": 0.25}, True), ({"choice": "R"}, False),
                          ({"choice": "A"}, False)):
        for t4, acc in ((accept_all, True), (reject_all, False)):
            freeze = {"lambda_star": 0.0, "gate_seed": 0, "T3": t3,
                      "T4": {"params": t4, "features": names}}
            rk = ctest.rankings(freeze, arr, rows, 10)
            assert ("T3" in rk) == expect_t3
            for q, v in enumerate(views):
                assert rk["T1"][q] == arr["top50"][q].tolist()
                assert rk["T4"][q] == (rk["T2"][q] if (acc and v["changed"]) else rk["T1"][q])
    bm_inf = {"lambda_star": 0.0, "gate_seed": 0, "T3": {"choice": "BM", "bm_threshold": "-inf"},
              "T4": {"params": accept_all, "features": names}}
    _expect(ctest.TestError, ctest.rankings, bm_inf, arr, rows, 10)


def test_split_windows_must_reproduce_the_cache():
    cand = np.array([[3, 1, 2], [0, 4, 5]])
    fields = {f: np.full((2, 3), 0.5, dtype=np.float32) for f in csw.FIELDS}
    base = np.full((2, 3), 0.7, dtype=np.float32)
    recs = [{"candidate_gallery_ids": c, "base_score": [0.7] * 3, **{f: [0.5] * 3 for f in csw.FIELDS}}
            for c in ([3, 1, 2], [0, 4, 5])]
    worst = csw.check_against_cache(cand, fields, base, recs)
    assert set(csw.VECTOR_TOL) <= set(worst)
    _expect(csw.WindowError, csw.check_against_cache, cand[:, ::-1], fields, base, recs)        # other order
    drift = dict(fields, edit_cos=fields["edit_cos"] + np.float32(2e-4))
    _expect(csw.WindowError, csw.check_against_cache, cand, drift, base, recs)
    _expect(csw.WindowError, csw.check_against_cache, cand[:1], fields, base, recs)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok  " + f.__name__)
    print("\n%d tests passed" % len(fns))
