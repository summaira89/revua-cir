"""Tests for the torch-free helpers of scripts/dump_verifier_vectors.py."""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import dump_verifier_vectors as DV  # noqa: E402


def _write(lines):
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for line in lines:
        fh.write(line + "\n")
    fh.close()
    return fh.name


def _rec(q, k=3, **over):
    r = {"query_id": q, "candidate_gallery_ids": list(range(k)), "base_score": [0.5] * k,
         "verification_score": [0.6] * k, "cos_sim": [0.1] * k, "recon_cos": [0.1] * k,
         "invariant_cos": [0.1] * k, "edit_cos": [0.1] * k}
    r.update(over)
    return r


def _expect_value_error(path):
    try:
        DV.load_cache(path)
    except ValueError:
        return
    finally:
        os.unlink(path)
    raise AssertionError("expected ValueError")


def test_load_cache_accepts_well_formed_records():
    p = _write([json.dumps(_rec(0)), json.dumps(_rec(1))])
    try:
        assert len(DV.load_cache(p)) == 2
    finally:
        os.unlink(p)


def test_load_cache_rejects_short_field():
    _expect_value_error(_write([json.dumps(_rec(0, recon_cos=[0.1, 0.1]))]))


def test_load_cache_rejects_nonfinite_value():
    _expect_value_error(_write([json.dumps(_rec(0, edit_cos=[0.1, float("nan"), 0.1]))]))


def test_load_cache_rejects_empty_file():
    _expect_value_error(_write([]))


def test_load_cache_rejects_non_object_line():
    _expect_value_error(_write(["[1, 2, 3]"]))


def test_output_state_covers_every_interrupted_publication():
    """Audit 3/8: clean, partial leftovers, npz-without-manifest, orphan
    manifest and complete output each give the right verdict."""
    d = tempfile.mkdtemp()
    try:
        paths = DV.artifact_paths(os.path.join(d, "v.npz"))
        assert DV.output_state_error(paths) is None

        def touch(*keys):
            for k in keys:
                open(paths[k], "w").close()

        def clear():
            for k in paths:
                if os.path.exists(paths[k]):
                    os.remove(paths[k])
        for state, expect in ((("npz_partial",), "interrupted run"),
                              (("manifest_partial",), "interrupted run"),
                              (("npz",), "has no manifest"),
                              (("manifest",), "orphan manifest"),
                              (("npz", "manifest"), "refusing to overwrite")):
            touch(*state)
            msg = DV.output_state_error(paths)
            assert msg and expect in msg, (state, msg)
            clear()
    finally:
        shutil.rmtree(d)          # removes leftovers too, so a failed assert is not masked


def test_numpy_sigmoid_is_warning_free_at_extreme_logits():
    """Audit 3/8: no overflow warning even for enormous calibrator slopes."""
    import warnings
    rng = np.random.default_rng(1)

    def unit(*shape):
        x = rng.standard_normal(shape + (4,))
        return x / np.linalg.norm(x, axis=-1, keepdims=True)
    arr = {"e_inv": unit(2), "t_hat": unit(2), "q_comp": unit(2), "rec": unit(2, 3),
           "cinv": unit(2, 3), "cedit": unit(2, 3), "c_pool": unit(2, 3)}
    recs = [{"recon_cos": [0.0] * 3, "invariant_cos": [0.0] * 3, "edit_cos": [0.0] * 3,
             "base_score": [0.0] * 3, "verification_score": [0.5] * 3} for _ in range(2)]
    # overflow, invalid and divide are errors; underflow of exp(-|z|) to exactly
    # 0 is the correct saturated result and stays allowed
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            DV.reproduction_errors(arr, recs, 0.4, 1e6, 0.0)


def test_reproduction_errors_zero_on_exact_and_raise_on_nan():
    rng = np.random.default_rng(0)

    def unit(*shape):
        x = rng.standard_normal(shape + (8,))
        return x / np.linalg.norm(x, axis=-1, keepdims=True)
    arr = {"e_inv": unit(4), "t_hat": unit(4), "q_comp": unit(4), "rec": unit(4, 3),
           "cinv": unit(4, 3), "cedit": unit(4, 3), "c_pool": unit(4, 3)}
    w, a, b = 0.4, 0.9, -0.07
    recs = []
    for q in range(4):
        rc, ic = arr["rec"][q] @ arr["e_inv"][q], arr["cinv"][q] @ arr["e_inv"][q]
        ec = arr["cedit"][q] @ arr["t_hat"][q]
        mix = (1 - w) * 0.5 * (rc + ic) + w * ec
        recs.append({"recon_cos": rc.tolist(), "invariant_cos": ic.tolist(),
                     "edit_cos": ec.tolist(),
                     "base_score": (arr["c_pool"][q] @ arr["q_comp"][q]).tolist(),
                     "verification_score": (1 / (1 + np.exp(-(a * mix + b)))).tolist()})
    worst = DV.reproduction_errors(arr, recs, w, a, b)
    assert max(worst.values()) < 1e-12
    arr["rec"][0, 0, 0] = np.nan
    try:
        DV.reproduction_errors(arr, recs, w, a, b)
    except ValueError:
        return
    raise AssertionError("NaN must raise, not be silently ignored")


def _exact_case(n, k, dim, seed):
    """float32 unit vectors and cache records holding their float64 cosines,
    mixture and calibrated score exactly."""
    rng = np.random.default_rng(seed)

    def unit(*shape):
        x = rng.standard_normal(shape + (dim,)).astype(np.float32)
        return x / np.linalg.norm(x, axis=-1, keepdims=True)
    arr = {"e_inv": unit(n), "t_hat": unit(n), "q_comp": unit(n), "rec": unit(n, k),
           "cinv": unit(n, k), "cedit": unit(n, k), "c_pool": unit(n, k)}
    w, a, b = 0.4, 0.9, -0.07
    f = {key: v.astype(np.float64) for key, v in arr.items()}
    recs = []
    for q in range(n):
        rc, ic = f["rec"][q] @ f["e_inv"][q], f["cinv"][q] @ f["e_inv"][q]
        ec = f["cedit"][q] @ f["t_hat"][q]
        mix = (1 - w) * 0.5 * (rc + ic) + w * ec
        recs.append({"recon_cos": rc.tolist(), "invariant_cos": ic.tolist(),
                     "edit_cos": ec.tolist(),
                     "base_score": (f["c_pool"][q] @ f["q_comp"][q]).tolist(),
                     "verification_score": (1 / (1 + np.exp(-(a * mix + b)))).tolist()})
    return arr, recs, (w, a, b)


def test_reproduction_is_evaluated_in_float64():
    """Audit 18 Oct, item 7: the check must add no float32 rounding of its own.
    With cached values that are the exact float64 cosines of float32 vectors,
    float64 evaluation reproduces them to ~1e-15; float32 arithmetic over
    d = 768 would leave ~1e-7."""
    arr, recs, (w, a, b) = _exact_case(6, 5, 768, 3)
    worst = DV.reproduction_errors(arr, recs, w, a, b)
    assert max(worst.values()) < 1e-12, worst


def test_reproduction_on_saved_and_reloaded_arrays():
    """The publication path re-checks the arrays after np.savez/np.load."""
    arr, recs, (w, a, b) = _exact_case(5, 4, 64, 4)
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "v.npz")
        np.savez(path, **arr)
        with np.load(path) as z:
            loaded = {name: z[name] for name in arr}
        assert all(v.dtype == np.float32 for v in loaded.values())
        assert DV.reproduction_errors(loaded, recs, w, a, b) == DV.reproduction_errors(
            arr, recs, w, a, b)
        assert DV.tolerance_violations(DV.reproduction_errors(loaded, recs, w, a, b)) == []
    finally:
        shutil.rmtree(d)


def test_tolerance_violations_per_field():
    """Audit 18 Oct, items 3/4: every field against its own limit; the relaxed
    base-score limit never applies to a verifier term; a missing field is an
    error, not a pass."""
    zero = {f: 0.0 for f in DV.VECTOR_TOL}
    assert DV.tolerance_violations(dict(zero)) == []
    assert DV.tolerance_violations(dict(DV.VECTOR_TOL)) == []          # at the limit
    for f, tol in DV.VECTOR_TOL.items():
        assert DV.tolerance_violations(dict(zero, **{f: 0.9 * tol})) == []
        bad = DV.tolerance_violations(dict(zero, **{f: 1.1 * tol}))
        assert len(bad) == 1 and bad[0].startswith(f + " "), bad
    assert DV.tolerance_violations(dict(zero, base_score=5e-4)) == []
    for f in ("recon_cos", "invariant_cos", "edit_cos", "verification_score"):
        assert DV.tolerance_violations(dict(zero, **{f: 5e-4})), f
    try:
        DV.tolerance_violations({"base_score": 0.0})
    except ValueError:
        return
    raise AssertionError("a missing field must raise")


def _complete_dump(d, cache_text="{}\n", config_text="cfg", ckpt_sha="e" * 64):
    """A published dump (vector file + manifest) and the identity of its run."""
    cache, config = os.path.join(d, "c.jsonl"), os.path.join(d, "cfg.yaml")
    with open(cache, "w") as fh:
        fh.write(cache_text)
    with open(config, "w") as fh:
        fh.write(config_text)
    paths = DV.artifact_paths(os.path.join(d, "v.npz"))
    np.savez(paths["npz"], x=np.zeros(3, dtype=np.float32))
    identity = DV.run_identity("fashioniq", "dress", 7, "fc+aep+rv", cache, config, ckpt_sha)
    man = dict(identity, schema=DV.SCHEMA, npz_sha256=DV.sha256_file(paths["npz"]))
    with open(paths["manifest"], "w") as fh:
        json.dump(man, fh)
    return paths, identity, cache


def test_complete_output_requires_this_runs_identity():
    """Audit 18 Oct, item 5: a resume may skip only a verified dump of THIS
    run; a clean location is 'to dump'; anything else is an error."""
    d = tempfile.mkdtemp()
    try:
        paths, identity, cache = _complete_dump(d)
        assert DV.complete_output(paths, identity) is True
        for key, value in (("checkpoint_sha256", "f" * 64), ("config_sha256", "0" * 64),
                           ("cache_sha256", "1" * 64), ("module_combo", "fc"),
                           ("category", "shirt"), ("n_queries", 6), ("dataset", "cirr")):
            try:
                DV.complete_output(paths, dict(identity, **{key: value}))
            except ValueError as exc:
                assert "different run" in str(exc) and key in str(exc), exc
            else:
                raise AssertionError("a dump of another run (%s) was accepted" % key)
        np.savez(paths["npz"], x=np.ones(3, dtype=np.float32))           # tampered vectors
        try:
            DV.complete_output(paths, identity)
        except ValueError as exc:
            assert "does not verify" in str(exc), exc
        else:
            raise AssertionError("a tampered vector file was accepted")
        for key in ("npz", "manifest"):
            os.remove(paths[key])
        assert DV.complete_output(paths, identity) is False               # clean
        open(paths["npz_partial"], "w").close()
        try:
            DV.complete_output(paths, identity)
        except ValueError as exc:
            assert "interrupted run" in str(exc), exc
        else:
            raise AssertionError("partial leftovers were accepted")
    finally:
        shutil.rmtree(d)


def test_skip_if_complete_is_refused_for_a_probe():
    """Refused for that reason, before anything else is read."""
    import contextlib
    import io
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            DV.main(["--config", "c", "--checkpoint", "k", "--module-combo", "fc+aep+rv",
                     "--dataset", "fashioniq", "--category", "dress", "--cache", "x.jsonl",
                     "--out", "p.npz", "--max-queries", "8", "--skip-if-complete"])
    except SystemExit as exc:
        assert exc.code == 2 and "never skipped" in err.getvalue(), err.getvalue()
        return
    raise AssertionError("a probe must never be skipped")


def test_missing_checkpoint_is_a_usage_error():
    """A missing input is reported cleanly (exit 2) before anything is hashed."""
    import contextlib
    import io
    d = tempfile.mkdtemp()
    try:
        cfg = os.path.join(d, "cfg.yaml")
        open(cfg, "w").close()
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                DV.main(["--config", cfg, "--checkpoint", os.path.join(d, "none.pt"),
                         "--module-combo", "fc+aep+rv", "--dataset", "cirr",
                         "--cache", os.path.join(d, "c.jsonl"), "--out", os.path.join(d, "v.npz")])
        except SystemExit as exc:
            assert exc.code == 2 and "--checkpoint" in err.getvalue(), err.getvalue()
            return
        raise AssertionError("a missing checkpoint must be a usage error")
    finally:
        shutil.rmtree(d)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok  " + f.__name__)
    print("\n%d tests passed" % len(fns))
