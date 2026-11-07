"""Bank-normalised (CSLS) inverse-consistency verification. CPU only.

Method
------
Each verification term is a cosine between a candidate-side vector a_qi and a
query-side target b_q (recon: FC(candidate, t_rev) vs the AEP invariant
target; invariant: the candidate's invariant part vs the same target; edit:
the candidate's editable part vs the edit text). Cross-domain similarity
local scaling (Conneau et al., ICLR 2018, Eq. 6) ranks by
    2 cos(a, b) - r_T(b) - r_S(a);
for a fixed query r_T(b) is constant across its candidates, so the ranking
score is, up to a positive factor,
    s'_qi = cos(a_qi, b_q) - 0.5 * h(a_qi),
    h(a)  = mean of the K largest cos(a, b_q') over a bank of OTHER queries'
            query-side targets (query-derived vectors, no relevance labels).
The corrected mixture uses the checkpoint's mixture weight and passes through
the checkpoint's own calibrator, sigmoid(scale * s + bias), exactly as the
deployed score does; fusion is the published bounded rule at lambda = 1.

Primary analyses (pre-registered; plan and dated addendum in
BNV_Study_Plan_2026-10-17.md), two-sided, Holm at 0.05 via `decide`:
  P1  CIRR validation, report half, bank = tune-half queries, K = 10:
      bounded-fusion R@1, CSLS-corrected vs the deployed verifier.
  P2  FashionIQ validation, inductive: per category the queries are split
      once (seed 12345) into a bank half and an evaluation half; within-window
      verifier AUROC (target vs the window's other candidates, averaged over
      covered evaluation queries), CSLS-corrected vs deployed, pooled.
Everything else is descriptive: s - h, full-bank mean centring, one corrected
term at a time, K in {5, 20, 50}, base-score-only normalisation, the
transductive leave-one-out FashionIQ bank, a size-matched CIRR source bank,
an 80% bank-subset sensitivity (five seeded subsets of the bank half, the
evaluation half held fixed; smaller banks than the primary, so not
uncertainty across full-size banks), R@1/R@5, rescues/breaks, URisk/TRisk,
and a slot-weighted
candidate-exposure diagnostic.

Usage
-----
  python scripts/bank_normalized_verification.py cirr  CIRR_CACHE CIRR_NPZ --json ../bnv_cirr.json
  python scripts/bank_normalized_verification.py fiq   CIRR_CACHE CIRR_NPZ \\
      --fiq DRESS_CACHE DRESS_NPZ SHIRT_CACHE SHIRT_NPZ TOPTEE_CACHE TOPTEE_NPZ --json ../bnv_fiq.json
  python scripts/bank_normalized_verification.py decide ../bnv_cirr.json ../bnv_fiq.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apply_frozen_policy import CIRR_REPORT_ANCHORS, _order, hit_at  # noqa: E402
from fusion_controls import CacheError, load_records, mcnemar_exact, relevant_set, split_indices  # noqa: E402
from fusion_suite import r_bounded, urisk_trisk  # noqa: E402
from gate_suite import paired_mean_bootstrap  # noqa: E402
from dump_verifier_vectors import VECTOR_TOL  # noqa: E402

SCHEMA = "revua-verifier-vectors/2"
RESULT_SCHEMA = "revua-bnv-result/1"
TERMS = ("recon_cos", "invariant_cos", "edit_cos")
VEC = {"recon_cos": ("rec", "e_inv"), "invariant_cos": ("cinv", "e_inv"),
       "edit_cos": ("cedit", "t_hat")}
K_PRIMARY, K_SENS = 10, (5, 20, 50)
CSLS_COEF = 0.5          # 2s - h  ==  s - 0.5 h for a fixed query
LAMBDA = 1.0
SPLIT_SEED = 12345
EXTRA_BANK_SEEDS = (1, 2, 3, 4, 5)
FIQ_EXPECTED = {"dress": 2017, "shirt": 2038, "toptee": 1961}
SUPPORTED_MODULE_COMBOS = ("fc+aep+rv",)   # the verifier this study corrects
# Vector reproduction uses the dumper's per-field policy VECTOR_TOL (one
# definition, imported). It is an acceptance threshold, not a bound on the
# corrected scores: the deployed baseline uses cached scores; numerical
# sensitivity of the vector-derived corrections is assessed separately
# (vector_repeatability.py). The cache's own internal consistency (mixture
# and calibration of cached values, no GPU recomputation) is checked at:
TOL_CACHE = 1e-4
# Deployed-control parity: the recomputed deployed score must equal the cached
# one to float32 storage resolution (the float32 step at scores in [0.5, 1) is
# 6e-8; the largest observed difference over CIRR and FashionIQ is 9.6e-8).
TOL_PARITY = 1e-6
N_REP = 10000


class ProtocolError(ValueError):
    pass


def _finite(x, what: str):
    if not np.all(np.isfinite(np.asarray(x, dtype=np.float64))):
        raise ProtocolError("non-finite values in %s" % what)


def calibrate(s: float, a: float, b: float) -> float:
    """sigmoid(a * s + b), evaluated sign-stably so no finite logit overflows."""
    z = a * s + b
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


# ------------------------------------------------------------------ data ----
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pair(cache_path: str, npz_path: str):
    """Records, relevance sets, vectors and manifest, cross-validated:
    manifest identity (schema, cache hash, npz hash), shapes, finiteness,
    candidate ids, the vectors reproducing every cached term, and the cached
    mixture and calibrated score matching the checkpoint's parameters."""
    if not npz_path.endswith(".npz"):
        raise ProtocolError("vector file must end in .npz")
    man_path = npz_path[:-4] + ".manifest.json"
    if not os.path.exists(man_path):
        raise ProtocolError("no manifest beside %s" % npz_path)
    with open(man_path, encoding="utf-8") as fh:
        man = json.load(fh)
    if man.get("schema") != SCHEMA:
        raise ProtocolError("%s: schema %r, expected %r" % (man_path, man.get("schema"), SCHEMA))
    if man["cache_sha256"] != sha256_file(cache_path):
        raise ProtocolError("%s was not built from %s" % (npz_path, cache_path))
    if man["npz_sha256"] != sha256_file(npz_path):
        raise ProtocolError("%s does not match its manifest" % npz_path)
    if man.get("split") != "val":
        raise ProtocolError("%s: split %r, expected 'val'" % (man_path, man.get("split")))
    if not _is_digest(man.get("checkpoint_sha256")):
        raise ProtocolError("%s: missing or malformed checkpoint_sha256" % man_path)
    # Every dump of one study is made with the SAME config file (one CONFIG in
    # dump_vectors.sbatch), so its raw digest is required here and compared
    # across dumps; a missing digest can never count as "compatible".
    if not _is_digest(man.get("config_sha256")):
        raise ProtocolError("%s: missing or malformed config_sha256" % man_path)
    if man.get("module_combo") not in SUPPORTED_MODULE_COMBOS:
        raise ProtocolError("%s: module_combo %r is not one of %s"
                            % (man_path, man.get("module_combo"), SUPPORTED_MODULE_COMBOS))
    for key in ("n_queries", "k", "dim"):
        v = man.get(key)
        if not (isinstance(v, int) and not isinstance(v, bool) and v > 0):
            raise ProtocolError("%s: %s must be a positive integer" % (man_path, key))
    recs, multi = load_records(cache_path)
    n, k, d = man["n_queries"], man["k"], man["dim"]
    if len(recs) != n:
        raise ProtocolError("cache has %d records, manifest %d" % (len(recs), n))
    with np.load(npz_path) as z:
        vec = {name: z[name].astype(np.float32)
               for name in ("e_inv", "t_hat", "q_comp", "rec", "cinv", "cedit", "c_pool")}
        cand, qid = z["candidate_ids"], z["query_id"]
    for name, v in vec.items():
        want = (n, d) if name in ("e_inv", "t_hat", "q_comp") else (n, k, d)
        if v.shape != want:
            raise ProtocolError("%s has shape %s, expected %s" % (name, v.shape, want))
        _finite(v, name)
    if not np.array_equal(qid, np.arange(n)):
        raise ProtocolError("query ids are not 0..n-1")
    for i, r in enumerate(recs):
        if r.get("query_id") != i or list(cand[i]) != r["candidate_gallery_ids"]:
            raise ProtocolError("vector row %d does not match cache record %d" % (i, i))
        for f in ("cos_sim", "verification_score", "base_score") + TERMS:
            if len(r.get(f, [])) != k:
                raise ProtocolError("record %d: %s has the wrong length" % (i, f))
            _finite(r[f], "record %d %s" % (i, f))
    for term, (a_name, b_name) in list(VEC.items()) + [("base_score", ("c_pool", "q_comp"))]:
        got = np.einsum("nkd,nd->nk", vec[a_name].astype(np.float64),
                        vec[b_name].astype(np.float64))
        err = np.abs(got - np.array([r[term] for r in recs], dtype=np.float64)).max()
        if err > VECTOR_TOL[term]:
            raise ProtocolError("vectors do not reproduce cached %s (%.3g > %.3g)"
                                % (term, err, VECTOR_TOL[term]))
    w, a, b = man["mixture_w"], man["calibrator_scale"], man["calibrator_bias"]
    _finite([w, a, b], "manifest parameters")
    if not 0.0 < w < 1.0 or a <= 0.0:
        raise ProtocolError("implausible mixture weight %r or calibrator slope %r" % (w, a))
    for i, r in enumerate(recs):
        mix = [(1 - w) * 0.5 * (rc + ic) + w * ec for rc, ic, ec in
               zip(r["recon_cos"], r["invariant_cos"], r["edit_cos"])]
        if max(abs(x - y) for x, y in zip(mix, r["cos_sim"])) > TOL_CACHE:
            raise ProtocolError("record %d: cos_sim is not the checkpoint's mixture" % i)
        if max(abs(calibrate(x, a, b) - y)
               for x, y in zip(r["cos_sim"], r["verification_score"])) > TOL_CACHE:
            raise ProtocolError("record %d: verification_score is not calibrate(cos_sim)" % i)
    rels = [relevant_set(r, multi) for r in recs]
    return recs, rels, vec, man


# ------------------------------------------------------------- penalties ----
def topk_means(a: np.ndarray, bank: np.ndarray, ks: Sequence[int],
               exclude: Optional[np.ndarray] = None, chunk: int = 2048) -> Dict:
    """For each row of `a` (m, d): the mean of its K largest cosines with the
    bank (B, d) for every K in ks, plus "all" = mean over the whole bank.
    exclude[j] >= 0 drops that bank column for row j; -1 drops nothing. One
    top-max(ks) selection per chunk serves every K; every row must keep at
    least max(ks) bank entries after exclusion."""
    if a.ndim != 2 or bank.ndim != 2 or a.shape[1] != bank.shape[1]:
        raise ProtocolError("incompatible shapes %s and %s" % (a.shape, bank.shape))
    if not ks or any(not isinstance(K, int) or K < 1 for K in ks):
        raise ProtocolError("K must be a positive integer")
    if chunk < 1:
        raise ProtocolError("chunk must be positive")
    kmax, B = max(ks), bank.shape[0]
    if exclude is not None and (exclude.shape != (a.shape[0],)
                                or np.any(exclude >= B) or np.any(exclude < -1)):
        raise ProtocolError("exclude must be one bank index or -1 per row")
    if B - (1 if exclude is not None and np.any(exclude >= 0) else 0) < kmax:
        raise ProtocolError("bank of %d is too small for K=%d" % (B, kmax))
    out = {K: np.empty(a.shape[0]) for K in ks}
    out["all"] = np.empty(a.shape[0])
    for s in range(0, a.shape[0], chunk):
        sims = (a[s:s + chunk] @ bank.T).astype(np.float64)
        n_avail = np.full(sims.shape[0], B, dtype=np.float64)
        total = sims.sum(axis=1)
        if exclude is not None:
            ex = exclude[s:s + chunk]
            rows = np.nonzero(ex >= 0)[0]
            total[rows] -= sims[rows, ex[rows]]
            sims[rows, ex[rows]] = -np.inf
            n_avail[rows] -= 1
        out["all"][s:s + chunk] = total / n_avail
        top = -np.sort(-np.partition(sims, -kmax, axis=1)[:, -kmax:], axis=1)
        for K in ks:
            out[K][s:s + chunk] = top[:, :K].mean(axis=1)
    for key, v in out.items():
        _finite(v, "penalty %s" % key)
    return out


def term_penalties(vec, idx: Sequence[int], bank: Dict[str, np.ndarray],
                   self_col: Optional[Sequence[int]] = None) -> Dict[str, Dict]:
    """Per term: {K or 'all': (len(idx), k) penalty array}. self_col[n] is the
    bank column of query idx[n] itself (excluded), when the bank contains it."""
    k = vec["rec"].shape[1]
    ex = None if self_col is None else np.repeat(np.asarray(self_col, dtype=np.int64), k)
    out = {}
    for term in TERMS:
        a_name, b_name = VEC[term]
        a = vec[a_name][list(idx)].reshape(-1, vec[a_name].shape[-1])
        m = topk_means(a, bank[b_name], (K_PRIMARY,) + K_SENS, ex)
        out[term] = {key: v.reshape(len(idx), k) for key, v in m.items()}
    return out


# ---------------------------------------------------------------- scores ----
def corrected_scores(recs, idx, pen, man, coef: float = CSLS_COEF, key=K_PRIMARY,
                     correct: Sequence[str] = TERMS) -> List[List[float]]:
    """Calibrated verification scores with the chosen terms corrected by
    coef * penalty[key]; coef = 0 or correct = () gives the deployed score."""
    w, a, b = man["mixture_w"], man["calibrator_scale"], man["calibrator_bias"]
    out = []
    for n, i in enumerate(idx):
        r = recs[i]
        t = {term: [r[term][j] - (coef * pen[term][key][n][j] if term in correct else 0.0)
                    for j in range(len(r[term]))] for term in TERMS}
        out.append([calibrate((1 - w) * 0.5 * (rc + ic) + w * ec, a, b)
                    for rc, ic, ec in zip(t["recon_cos"], t["invariant_cos"], t["edit_cos"])])
    return out


def deployed_scores(recs, rels, idx, man) -> Tuple[List[List[float]], dict]:
    """The deployed control: the verification score recomputed from the
    cached terms through the SAME code path as every corrected arm (zero
    correction), so the arms differ only by the correction itself
    (Amendment 4). It must reproduce the cached deployed score: every value
    within TOL_PARITY, and identical bounded-fusion Hit@1 and Hit@5 for every
    query. Orderings can still differ where float32 storage tied or nearly
    tied cached scores; those queries are counted, not hidden.
    Returns (scores, parity report)."""
    zero = {t: {K_PRIMARY: np.zeros((len(idx), man["k"]))} for t in TERMS}
    recomputed = corrected_scores(recs, idx, zero, man)
    worst, n_ver, n_fused, hit_changed = 0.0, 0, 0, []
    for n, i in enumerate(idx):
        cached, base = recs[i]["verification_score"], recs[i]["base_score"]
        worst = max(worst, max(abs(x - y) for x, y in zip(recomputed[n], cached)))
        fc = _order(r_bounded(base, cached, LAMBDA))
        fr = _order(r_bounded(base, recomputed[n], LAMBDA))
        n_ver += _order(recomputed[n]) != _order(cached)
        n_fused += fc != fr
        if any(hit_at(fc, rels[i], at) != hit_at(fr, rels[i], at) for at in (1, 5)):
            hit_changed.append(i)
    if worst > TOL_PARITY:
        raise ProtocolError("the recomputed deployed score differs from the cache by %.3g > %.3g"
                            % (worst, TOL_PARITY))
    if hit_changed:
        raise ProtocolError("the recomputed deployed score changes Hit@1 or Hit@5 for queries %s"
                            % hit_changed[:10])
    return recomputed, {"n_queries": len(idx), "max_abs_diff_vs_cache": worst,
                        "verifier_order_differs": n_ver, "fused_permutation_differs": n_fused}


# --------------------------------------------------------------- metrics ----
def hits_at(recs, rels, idx, ver, base=None, at: int = 1) -> List[int]:
    """Hit@at of bounded fusion (base, ver); base defaults to the cached score."""
    out = []
    for n, i in enumerate(idx):
        b = recs[i]["base_score"] if base is None else base[n]
        out.append(hit_at(_order(r_bounded(b, ver[n], LAMBDA)), rels[i], at))
    return out


def base_hits(recs, rels, idx, at: int = 1) -> List[int]:
    return [hit_at(_order(recs[i]["base_score"]), rels[i], at) for i in idx]


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Mann-Whitney AUROC, ties counted half."""
    if not pos or not neg:
        raise ProtocolError("AUROC needs positives and negatives")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        rank_sum += (i + j + 1) / 2.0 * sum(1 for t in allv[i:j] if t[1] == 1)
        i = j
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def window_aucs(rels, idx, ver) -> Tuple[List[int], List[float]]:
    """Within-window AUROC for every covered query (some but not all
    candidates relevant): (query indices, AUROCs)."""
    qs, vals = [], []
    for n, i in enumerate(idx):
        rel, s = rels[i], ver[n]
        if rel and len(rel) < len(s):
            qs.append(i)
            vals.append(auroc([s[j] for j in rel], [s[j] for j in range(len(s)) if j not in rel]))
    return qs, vals


def fusion_summary(hits1, hits5, base1) -> dict:
    mc = mcnemar_exact(hits1, base1)
    return {"r1": 100.0 * sum(hits1) / len(hits1), "r5": 100.0 * sum(hits5) / len(hits5),
            "rescued": mc["fixed"], "broken": mc["broken"], "p_vs_base": mc["p_exact"],
            "risk1": urisk_trisk(hits1, base1, 1.0)}


def contrast(a: Sequence[float], b: Sequence[float], n_rep: int) -> dict:
    if n_rep < 1:
        raise ProtocolError("bootstrap needs at least one replicate")
    if len(a) != len(b) or not a:
        raise ProtocolError("contrast arms must be non-empty and of equal length")
    _finite(list(a) + list(b), "contrast arms")
    return paired_mean_bootstrap(list(a), list(b), n_rep)


def average_ranks(x: Sequence[float]) -> np.ndarray:
    """1-based ranks with ties given their average rank (as Spearman requires)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> Tuple[Optional[float], str]:
    """Tie-corrected Spearman correlation; (None, reason) when undefined."""
    if len(x) != len(y) or len(x) < 2:
        return None, "fewer than two paired observations"
    rx, ry = average_ranks(x), average_ranks(y)
    if np.all(rx == rx[0]) or np.all(ry == ry[0]):
        return None, "one variable is constant"
    return float(np.corrcoef(rx, ry)[0, 1]), "ok"


def exposure_diagnostic(recs, idx, pen) -> dict:
    """Does the CSLS penalty track how often a candidate appears across the
    evaluated windows (its exposure)? Descriptive and slot-weighted: a gallery
    image appearing in several windows contributes one observation per slot,
    so this is not an independent candidate-level test."""
    count: Dict[int, int] = {}
    for i in idx:
        for c in recs[i]["candidate_gallery_ids"]:
            count[c] = count.get(c, 0) + 1
    xs, ys = [], []
    for n, i in enumerate(idx):
        for j, c in enumerate(recs[i]["candidate_gallery_ids"]):
            xs.append(count[c])
            ys.append(float(np.mean([pen[t][K_PRIMARY][n][j] for t in TERMS])))
    rho, status = spearman(xs, ys)
    return {"spearman_exposure_vs_mean_penalty": rho, "spearman_status": status,
            "weighting": "per candidate slot", "n_candidate_slots": len(xs),
            "n_distinct_candidates": len(count), "max_exposure": int(max(xs))}


# ------------------------------------------------------------ provenance ----
PROVENANCE_KEYS = ("result_schema", "script_sha256", "dependency_sha256",
                   "checkpoint_sha256", "split_seed", "K", "coef", "lambda", "n_bootstrap",
                   "vector_tolerance", "cache_tolerance")
DEPENDENCIES = ("apply_frozen_policy.py", "fusion_controls.py", "fusion_suite.py",
                "gate_suite.py", "dump_verifier_vectors.py")
HEX = set("0123456789abcdef")


def provenance(mode: str, manifests, n_rep: int) -> dict:
    """What a result was computed from and under which settings; `decide`
    refuses to combine results whose provenance disagrees."""
    cks = {m["checkpoint_sha256"] for m in manifests}
    if len(cks) != 1:
        raise ProtocolError("inputs come from different checkpoints")
    here = os.path.dirname(os.path.abspath(__file__))
    return {"result_schema": RESULT_SCHEMA, "mode": mode,
            "script_sha256": sha256_file(os.path.abspath(__file__)),
            "dependency_sha256": {d: sha256_file(os.path.join(here, d)) for d in DEPENDENCIES},
            "checkpoint_sha256": cks.pop(), "split_seed": SPLIT_SEED, "K": K_PRIMARY,
            "coef": CSLS_COEF, "lambda": LAMBDA, "n_bootstrap": n_rep,
            "vector_tolerance": dict(VECTOR_TOL), "cache_tolerance": TOL_CACHE,
            # What each dump was accepted under when written (the CIRR dump of
            # job 7517 predates the per-field policy and records one scalar).
            "inputs": [{"dataset": m["dataset"], "category": m["category"],
                        "cache_sha256": m["cache_sha256"], "npz_sha256": m["npz_sha256"],
                        "dump_tolerance": m.get("tolerance_policy", m.get("tolerance")),
                        "dump_worst_error": m.get("worst_reproduction_error")}
                       for m in manifests]}


def _is_digest(x) -> bool:
    return isinstance(x, str) and len(x) == 64 and set(x) <= HEX


def _source_input(prov: dict, label: str) -> dict:
    """The single CIRR input of a result's provenance, validated."""
    inputs = prov.get("inputs")
    if not isinstance(inputs, list) or not inputs or not all(isinstance(i, dict) for i in inputs):
        raise ProtocolError("%s result has no valid input list" % label)
    for i in inputs:
        if not (_is_digest(i.get("cache_sha256")) and _is_digest(i.get("npz_sha256"))):
            raise ProtocolError("%s result has an input without valid digests" % label)
    cirr = [i for i in inputs if i.get("dataset") == "cirr"]
    if len(cirr) != 1:
        raise ProtocolError("%s result must have exactly one CIRR input, has %d" % (label, len(cirr)))
    return cirr[0]


def _valid_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _validated_test(t, label: str) -> dict:
    """A primary-test record with finite statistics, p in [0, 1], ci_low <= ci_high."""
    if not isinstance(t, dict):
        raise ProtocolError("%s is missing" % label)
    for f in ("diff", "ci_low", "ci_high", "p_two_sided"):
        if not _valid_number(t.get(f)):
            raise ProtocolError("%s.%s is missing or not a finite number" % (label, f))
    if not 0.0 <= t["p_two_sided"] <= 1.0:
        raise ProtocolError("%s.p_two_sided %r is outside [0, 1]" % (label, t["p_two_sided"]))
    if t["ci_low"] > t["ci_high"]:
        raise ProtocolError("%s has a reversed interval" % label)
    if not isinstance(t.get("metric"), str):
        raise ProtocolError("%s.metric is missing" % label)
    return t


# -------------------------------------------------------------- analyses ----
PUBLISHED_ANCHOR = CIRR_REPORT_ANCHORS[("bounded_frozen", "fusion")]


def analyse_cirr(cache, npz, n_rep: int = N_REP,
                 expected_anchor: Optional[float] = PUBLISHED_ANCHOR) -> dict:
    """expected_anchor: the published report-half R@1 the deployed fusion must
    reproduce (None only for synthetic test data)."""
    recs, rels, vec, man = load_pair(cache, npz)
    if man["dataset"] != "cirr":
        raise ProtocolError("expected a CIRR manifest")
    tune, report = split_indices(len(recs), SPLIT_SEED)
    k = man["k"]
    base1, base5 = base_hits(recs, rels, report), base_hits(recs, rels, report, 5)

    # Parity first: the deployed control is recomputed through the corrected
    # arms' code path and must reproduce the cached deployed scores and every
    # fused Hit@1/Hit@5; its fusion must then reproduce the published
    # report-half anchor.
    dep, parity = deployed_scores(recs, rels, report, man)
    dep1 = hits_at(recs, rels, report, dep)
    anchor = round(100.0 * sum(dep1) / len(dep1), 2)
    if expected_anchor is not None and anchor != expected_anchor:
        raise ProtocolError("deployed fusion gives %.2f, not the published anchor" % anchor)

    bank = {name: vec[name][tune] for name in ("e_inv", "t_hat")}
    pen = term_penalties(vec, report, bank)
    variants = {"csls": corrected_scores(recs, report, pen, man),
                "full_subtraction": corrected_scores(recs, report, pen, man, coef=1.0),
                "mean_centring": corrected_scores(recs, report, pen, man, key="all")}
    for K in K_SENS:
        variants["csls_K%d" % K] = corrected_scores(recs, report, pen, man, key=K)
    for t in TERMS:
        variants["csls_only_" + t] = corrected_scores(recs, report, pen, man, correct=(t,))

    # base-score-only normalisation, deployed verifier unchanged
    bpen = topk_means(vec["c_pool"][report].reshape(-1, vec["c_pool"].shape[-1]),
                      vec["q_comp"][tune], (K_PRIMARY,))[K_PRIMARY].reshape(len(report), k)
    base_norm = [[recs[i]["base_score"][j] - CSLS_COEF * bpen[n][j] for j in range(k)]
                 for n, i in enumerate(report)]

    res = {"provenance": provenance("cirr", [man], n_rep),
           "n_report": len(report), "K": K_PRIMARY, "coef": CSLS_COEF,
           "mixture_w": man["mixture_w"], "anchor_r1": anchor, "deployed_parity": parity,
           "fusion": {}, "auroc": {}}
    res["fusion"]["base"] = fusion_summary(base1, base5, base1)
    res["fusion"]["deployed"] = fusion_summary(dep1, hits_at(recs, rels, report, dep, at=5), base1)
    for name, ver in variants.items():
        res["fusion"][name] = fusion_summary(hits_at(recs, rels, report, ver),
                                             hits_at(recs, rels, report, ver, at=5), base1)
    res["fusion"]["base_norm_deployed_verifier"] = fusion_summary(
        hits_at(recs, rels, report, dep, base=base_norm),
        hits_at(recs, rels, report, dep, base=base_norm, at=5), base1)
    res["fusion"]["base_norm_alone"] = fusion_summary(
        [hit_at(_order(bn), rels[i], 1) for bn, i in zip(base_norm, report)],
        [hit_at(_order(bn), rels[i], 5) for bn, i in zip(base_norm, report)], base1)

    csls1 = hits_at(recs, rels, report, variants["csls"])
    res["P1"] = dict(contrast(csls1, dep1, n_rep), mcnemar=mcnemar_exact(csls1, dep1),
                     metric="R@1 difference, proportion of queries (CSLS - deployed)")
    for name, ver in [("deployed", dep)] + list(variants.items()):
        vals = window_aucs(rels, report, ver)[1]
        res["auroc"][name] = float(np.mean(vals))
    res["exposure"] = exposure_diagnostic(recs, report, pen)
    return res


BANK_SUBSET_FRACTION = 0.8


def bank_subsets(bank_idx: Sequence[int]) -> Dict[int, List[int]]:
    """Seeded 80% subsets of the bank half, one per sensitivity seed. They are
    drawn only from bank_idx, so the evaluation half can never change."""
    m = int(BANK_SUBSET_FRACTION * len(bank_idx))
    return {seed: sorted(random.Random(seed).sample(list(bank_idx), m))
            for seed in EXTRA_BANK_SEEDS}


def _validate_fiq_set(loaded) -> None:
    cats = [m["category"] for _, _, _, m in loaded]
    if sorted(cats) != sorted(FIQ_EXPECTED):
        raise ProtocolError("FashionIQ needs dress, shirt and toptee exactly once; got %s" % cats)
    for recs, _, _, m in loaded:
        if m["dataset"] != "fashioniq" or len(recs) != FIQ_EXPECTED[m["category"]]:
            raise ProtocolError("%s: wrong dataset or query count" % m["category"])
    if len({m["checkpoint_sha256"] for _, _, _, m in loaded}) != 1:
        raise ProtocolError("FashionIQ categories come from different checkpoints")


def analyse_fiq(cirr_cache, cirr_npz, pairs, n_rep: int = N_REP) -> dict:
    c_recs, _, c_vec, c_man = load_pair(cirr_cache, cirr_npz)
    if c_man["dataset"] != "cirr":
        raise ProtocolError("the source bank must come from a CIRR manifest")
    loaded = [load_pair(c, z) for c, z in pairs]
    _validate_fiq_set(loaded)
    all_man = [c_man] + [m for _, _, _, m in loaded]
    for key in ("module_combo", "config_sha256", "dim"):      # validated in load_pair
        if len({m[key] for m in all_man}) != 1:
            raise ProtocolError("CIRR and FashionIQ dumps differ in %s" % key)
    if len({m["k"] for _, _, _, m in loaded}) != 1:
        raise ProtocolError("FashionIQ categories have different candidate windows")
    if loaded[0][3]["checkpoint_sha256"] != c_man["checkpoint_sha256"]:
        raise ProtocolError("FashionIQ and CIRR vectors come from different checkpoints")
    c_tune, _ = split_indices(len(c_recs), SPLIT_SEED)

    arms = ("deployed", "csls_target", "csls_source", "csls_transductive")
    auc = {a: [] for a in arms}
    hits1 = {a: [] for a in arms}
    hits5 = {a: [] for a in arms}
    base1_all, base5_all, per_cat = [], [], {}
    sens = {s: [] for s in EXTRA_BANK_SEEDS}
    for recs, rels, vec, man in loaded:
        bank_idx, eval_idx = split_indices(len(recs), SPLIT_SEED)
        tgt = {name: vec[name][bank_idx] for name in ("e_inv", "t_hat")}
        src_pick = sorted(random.Random(SPLIT_SEED).sample(c_tune, len(bank_idx)))
        src = {name: c_vec[name][src_pick] for name in ("e_inv", "t_hat")}
        loo = {name: vec[name] for name in ("e_inv", "t_hat")}
        dep, parity = deployed_scores(recs, rels, eval_idx, man)
        scores = {
            "deployed": dep,
            "csls_target": corrected_scores(recs, eval_idx, term_penalties(vec, eval_idx, tgt), man),
            "csls_source": corrected_scores(recs, eval_idx, term_penalties(vec, eval_idx, src), man),
            "csls_transductive": corrected_scores(
                recs, eval_idx, term_penalties(vec, eval_idx, loo, self_col=eval_idx), man),
        }
        base1_all += base_hits(recs, rels, eval_idx)
        base5_all += base_hits(recs, rels, eval_idx, 5)
        cat_qs, cat_auc = None, {}
        for a in arms:
            qs, vals = window_aucs(rels, eval_idx, scores[a])
            if cat_qs is None:
                cat_qs = qs
            elif qs != cat_qs:
                raise ProtocolError("arms cover different queries")
            auc[a] += vals
            cat_auc[a] = float(np.mean(vals))
            hits1[a] += hits_at(recs, rels, eval_idx, scores[a])
            hits5[a] += hits_at(recs, rels, eval_idx, scores[a], at=5)
        per_cat[man["category"]] = {"n_eval": len(eval_idx), "n_bank": len(bank_idx),
                                    "n_covered": len(cat_qs), "auroc": cat_auc,
                                    "deployed_parity": parity}
        # 80% bank-subset sensitivity: the evaluation half stays fixed and only
        # the bank changes. Smaller than the primary bank by design.
        dep_v = window_aucs(rels, eval_idx, dep)[1]
        for seed, pick in bank_subsets(bank_idx).items():
            alt = {name: vec[name][pick] for name in ("e_inv", "t_hat")}
            cor_v = window_aucs(rels, eval_idx, corrected_scores(
                recs, eval_idx, term_penalties(vec, eval_idx, alt), man))[1]
            sens[seed] += [c - d for c, d in zip(cor_v, dep_v)]

    res = {"provenance": provenance("fiq", [c_man] + [m for _, _, _, m in loaded], n_rep),
           "n_eval": len(base1_all), "n_covered": len(auc["deployed"]), "K": K_PRIMARY,
           "coef": CSLS_COEF, "per_category": per_cat,
           "auroc": {a: float(np.mean(auc[a])) for a in arms},
           "fusion": {a: fusion_summary(hits1[a], hits5[a], base1_all) for a in arms},
           "bank_subset_80pct_sensitivity_auroc_diff": {
               str(s): float(np.mean(v)) for s, v in sens.items()}}
    res["fusion"]["base"] = fusion_summary(base1_all, base5_all, base1_all)
    res["P2"] = dict(contrast(auc["csls_target"], auc["deployed"], n_rep),
                     metric="mean within-window AUROC difference (CSLS target bank - deployed)")
    res["source_minus_deployed_auroc"] = contrast(auc["csls_source"], auc["deployed"], n_rep)
    res["transductive_minus_deployed_auroc"] = contrast(
        auc["csls_transductive"], auc["deployed"], n_rep)
    return res


def decide(cirr_json: str, fiq_json: str) -> dict:
    """Holm over the two pre-registered primary tests (two-sided, alpha 0.05)."""
    with open(cirr_json, encoding="utf-8") as fh:
        cirr = json.load(fh)
    with open(fiq_json, encoding="utf-8") as fh:
        fiq = json.load(fh)
    pc, pf = cirr.get("provenance", {}), fiq.get("provenance", {})
    if pc.get("mode") != "cirr" or pf.get("mode") != "fiq":
        raise ProtocolError("decide needs a CIRR result and a FashionIQ result")
    for key in PROVENANCE_KEYS:
        if key not in pc or pc[key] != pf.get(key):
            raise ProtocolError("results disagree on %s; they are not one experiment" % key)
    if pc["result_schema"] != RESULT_SCHEMA:
        raise ProtocolError("unknown result schema %r" % pc["result_schema"])
    src_c, src_f = _source_input(pc, "CIRR"), _source_input(pf, "FashionIQ")
    if (src_c["cache_sha256"], src_c["npz_sha256"]) != (src_f["cache_sha256"], src_f["npz_sha256"]):
        raise ProtocolError("the two results use different CIRR cache/vector inputs")
    if len(pc["inputs"]) != 1:
        raise ProtocolError("the CIRR result must have exactly one input")
    cats = sorted(i.get("category") for i in pf["inputs"] if i.get("dataset") == "fashioniq")
    if cats != sorted(FIQ_EXPECTED) or len(pf["inputs"]) != 1 + len(FIQ_EXPECTED):
        raise ProtocolError("the FashionIQ result must cover dress, shirt and toptee once each")
    tests = {"P1": _validated_test(cirr.get("P1"), "P1"),
             "P2": _validated_test(fiq.get("P2"), "P2")}
    order = sorted(tests, key=lambda t: tests[t]["p_two_sided"])
    out, still, prev = {}, True, 0.0
    for r, name in enumerate(order):
        p = tests[name]["p_two_sided"]
        adj = max(prev, min(1.0, p * (len(order) - r)))
        prev = adj
        still = still and adj <= 0.05
        out[name] = {"diff": tests[name]["diff"], "ci_low": tests[name]["ci_low"],
                     "ci_high": tests[name]["ci_high"], "p": p, "p_holm": adj,
                     "rejected": still, "metric": tests[name]["metric"]}
    out["provenance"] = {key: pc[key] for key in PROVENANCE_KEYS}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("cirr", "fiq", "decide"))
    ap.add_argument("first", help="CIRR cache (cirr/fiq) or the CIRR result JSON (decide)")
    ap.add_argument("second", help="CIRR vectors .npz (cirr/fiq) or the FashionIQ result JSON (decide)")
    ap.add_argument("--fiq", nargs="*", default=[], help="cache npz pairs: dress, shirt, toptee")
    ap.add_argument("--bootstrap", type=int, default=N_REP)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)
    if args.bootstrap < 1:
        ap.error("--bootstrap must be positive")
    try:
        if args.mode == "cirr":
            res = analyse_cirr(args.first, args.second, args.bootstrap)
        elif args.mode == "fiq":
            if len(args.fiq) != 6:
                ap.error("--fiq needs three cache/npz pairs")
            res = analyse_fiq(args.first, args.second,
                              list(zip(args.fiq[0::2], args.fiq[1::2])), args.bootstrap)
        else:
            res = decide(args.first, args.second)
    except (CacheError, ProtocolError, OSError, json.JSONDecodeError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps(res, indent=1, allow_nan=False))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, allow_nan=False)
        print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
