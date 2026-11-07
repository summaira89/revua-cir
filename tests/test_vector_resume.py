"""Tests for scripts/preflight_vectors.py and scripts/vector_repeatability.py
(audit of 18 October 2026, items 1, 2 and 5). Runs without pytest."""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, _HERE)

import bank_normalized_verification as B  # noqa: E402
import dump_verifier_vectors as DV  # noqa: E402
import preflight_vectors as P  # noqa: E402
import vector_repeatability as R  # noqa: E402
from test_bank_normalized_verification import _Tmp, _build, _expect  # noqa: E402

COMBO = "fc+aep+rv"
FIQ_N = {"dress": 20, "shirt": 22, "toptee": 21}


def _set_manifest(npz, **fields):
    path = npz[:-4] + ".manifest.json"
    with open(path, encoding="utf-8") as fh:
        man = json.load(fh)
    man.update(fields)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(man, fh)


def _remove_dump(npz):
    os.remove(npz)
    os.remove(npz[:-4] + ".manifest.json")


class _Run:
    """A config, a checkpoint, a valid CIRR dump of this run, and three
    FashionIQ caches with their dump locations (dress complete, others clean)."""

    def __enter__(self):
        self.d = tempfile.mkdtemp()
        self.config = os.path.join(self.d, "cfg.yaml")
        self.ckpt = os.path.join(self.d, "w.pt")
        for path, text in ((self.config, "cfg"), (self.ckpt, "weights")):
            with open(path, "w") as fh:
                fh.write(text)
        self.ck_sha = DV.sha256_file(self.ckpt)
        self.cfg_sha = DV.sha256_file(self.config)
        self.cirr = _build(self.d, "cirr", 30, 1, ckpt=self.ck_sha)
        _set_manifest(self.cirr[1], config_sha256=self.cfg_sha)
        self.fiq = []
        for i, (cat, n) in enumerate(FIQ_N.items()):
            cache, npz = _build(self.d, cat, n, 10 + i, "fashioniq", cat, ckpt=self.ck_sha)
            _set_manifest(npz, config_sha256=self.cfg_sha)
            if cat != "dress":
                _remove_dump(npz)
            self.fiq.append((cache, npz))
        self.saved = dict(B.FIQ_EXPECTED)
        B.FIQ_EXPECTED.clear()
        B.FIQ_EXPECTED.update(FIQ_N)         # the same dict object preflight imported
        return self

    def check(self):
        return P.check(self.config, self.ckpt, COMBO, self.cirr, self.fiq)

    def __exit__(self, *exc):
        B.FIQ_EXPECTED.clear()
        B.FIQ_EXPECTED.update(self.saved)
        shutil.rmtree(self.d)


def test_preflight_reports_resume_state():
    with _Run() as run:
        assert run.check() == {"dress": True, "shirt": False, "toptee": False}


def test_preflight_rejects_a_cirr_dump_of_another_run():
    for field, value in (("checkpoint_sha256", "f" * 64), ("config_sha256", "0" * 64),
                         ("module_combo", "fc+aep"), ("n_queries", 29)):
        with _Run() as run:
            _set_manifest(run.cirr[1], **{field: value})
            _expect(ValueError, run.check)
    with _Run() as run:
        _remove_dump(run.cirr[1])
        _expect(ValueError, run.check)


def test_preflight_runs_the_analysis_loader_on_cirr():
    """Identity and digest can be right while the rows are misaligned; only
    load_pair (what the analysis runs) sees that."""
    with _Run() as run:
        npz = run.cirr[1]
        with np.load(npz) as f:
            arr = {k: f[k] for k in f.files}
        arr["e_inv"] = arr["e_inv"][::-1].copy()
        np.savez(npz, **arr)
        _set_manifest(npz, npz_sha256=DV.sha256_file(npz))
        _expect(B.ProtocolError, run.check)


def test_preflight_rejects_bad_fashioniq_states():
    with _Run() as run:                       # partial leftover from an interrupted run
        open(DV.artifact_paths(run.fiq[1][1])["npz_partial"], "w").close()
        _expect(ValueError, run.check)
    with _Run() as run:                       # a complete dress dump of another cache
        _set_manifest(run.fiq[0][1], cache_sha256="1" * 64)
        _expect(ValueError, run.check)
    with _Run() as run:                       # a cache with the wrong query count
        B.FIQ_EXPECTED["toptee"] = 99
        _expect(ValueError, run.check)


# ------------------------------------------------------------ repeatability --
N_PROBE = 60


def _probes(d, perturb=None, batch=(1, 32), n=N_PROBE):
    """Two probe dumps of one cache; `perturb` replaces vectors of probe B."""
    cache, npz_a = _build(d, "a", n, 5, "fashioniq", "dress")
    npz_b = os.path.join(d, "b.npz")
    with np.load(npz_a) as f:
        arr = {k: f[k] for k in f.files}
    if perturb:
        perturb(arr)
    np.savez(npz_b, **arr)
    shutil.copy(npz_a[:-4] + ".manifest.json", npz_b[:-4] + ".manifest.json")
    for npz, bs in zip((npz_a, npz_b), batch):
        _set_manifest(npz, npz_sha256=DV.sha256_file(npz),
                      runtime={"query_batch_size": bs, "max_candidates_per_pass": bs * 6})
    return cache, npz_a, npz_b


def test_identical_probes_have_zero_differences():
    with _Tmp() as d:
        res = R.compare(*_probes(d))
        assert all(v["max_abs"] == 0.0 for v in res["vectors"].values())
        assert all(x == 0.0 for t in res["penalty_max_abs"].values() for x in t.values())
        assert res["corrected_score_max_abs"] == 0.0
        assert all(v == 0 for v in res["flips"].values())
        json.dumps(res, allow_nan=False)


def test_perturbed_probe_differences_and_flips_are_counted():
    """Counts must equal a direct query-by-query recount, not be assumed zero."""
    def scramble(arr):
        rng = np.random.default_rng(99)
        x = rng.standard_normal(arr["rec"].shape).astype(np.float32)
        arr["rec"] = x / np.linalg.norm(x, axis=-1, keepdims=True)
    with _Tmp() as d:
        cache, a, b = _probes(d, scramble)
        res = R.compare(cache, a, b)
        assert res["vectors"]["rec"]["max_abs"] > 0.1
        assert res["vectors"]["e_inv"]["max_abs"] == 0.0
        assert res["penalty_max_abs"]["recon_cos"]["10"] > 0.0
        assert res["penalty_max_abs"]["edit_cos"]["10"] == 0.0
        recs, multi = B.load_records(cache)
        rels = [B.relevant_set(r, multi) for r in recs]
        idx = list(range(N_PROBE))
        scores = []
        for npz in (a, b):
            with np.load(npz) as f:
                vec = {k: f[k].astype(np.float32) for k in R.VECTORS}
            pen = B.term_penalties(vec, idx, {k: vec[k] for k in ("e_inv", "t_hat")},
                                   self_col=idx)
            scores.append(B.corrected_scores(recs, idx, pen, {
                "mixture_w": 0.4, "calibrator_scale": 0.9, "calibrator_bias": -0.07}))
        fused = [[B._order(B.r_bounded(recs[i]["base_score"], s[i], 1.0)) for i in idx]
                 for s in scores]
        want = {"verifier_order": sum(B._order(scores[0][i]) != B._order(scores[1][i])
                                      for i in idx),
                "fused_permutation": sum(fused[0][i] != fused[1][i] for i in idx),
                "fused_top1": sum(fused[0][i][0] != fused[1][i][0] for i in idx),
                "r1_hit": sum(x != y for x, y in zip(B.hits_at(recs, rels, idx, scores[0]),
                                                      B.hits_at(recs, rels, idx, scores[1])))}
        assert res["flips"] == want, (res["flips"], want)
        assert want["verifier_order"] > 0 and want["fused_top1"] > 0, want
        assert res["residual_vs_cache"][0]["recon_cos"] < 1e-5
        assert res["residual_vs_cache"][1]["recon_cos"] > 0.1


def test_repeatability_refuses_an_uninformative_or_mismatched_pair():
    with _Tmp() as d:                                     # nothing varied
        _expect(B.ProtocolError, R.compare, *_probes(d, batch=(32, 32)))
    with _Tmp() as d:                                     # different checkpoints
        cache, a, b = _probes(d)
        _set_manifest(b, checkpoint_sha256="f" * 64)
        _expect(B.ProtocolError, R.compare, cache, a, b)
    with _Tmp() as d:                                     # bank smaller than K=50
        _expect(B.ProtocolError, R.compare, *_probes(d, n=50))


def test_parity_diagnostic_counts_reordered_queries():
    """diagnose_parity.py: 0 on a clean cache; a planted swap is found."""
    import diagnose_parity as DP
    with _Tmp() as d:
        cache, npz = _build(d, "a", 30, 3)
        assert DP.diagnose(cache, npz) == 0
        with open(cache, encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh]
        v = recs[5]["verification_score"]
        top, second = sorted(range(len(v)), key=lambda j: -v[j])[:2]
        v[top], v[second] = v[second] - 1e-9, v[top]
        with open(cache, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        assert DP.diagnose(cache, npz) == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok  " + f.__name__)
    print("\n%d tests passed" % len(fns))
