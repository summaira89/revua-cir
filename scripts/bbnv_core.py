"""Bidirectional bank-normalised verification (BBNV): shared finite-bank scoring core.

Design: Bidirectional_Verification_Design_v3_2026-10-19.md as amended by
Bidirectional_Verification_Design_v3.1_Amendments_2026-10-19.md (binding).
Everything here is numpy float64 and deterministic. Learned heads (torch,
elsewhere) only supply hypotheses, mixture weights and concentrations.

Contract:
  * every distribution is categorical over a declared, de-duplicated, finite
    support; nothing here is a continuous density;
  * mixtures are component-normalised, q(x) = sum_m pi_m q_m(x), so pi are
    the mixture masses;
  * the query reference always has positive prior mass on R_q (v3.1 A);
  * all forward conditionals entering one marginalisation are normalised
    over the same, ordered candidate support C_q;
  * integration and gate features are within-window centred or standardised;
    absolute, support-dependent log-probabilities are never compared across
    queries;
  * every public function validates its inputs' types, ranks and shapes
    before any broadcasting, and never coerces a malformed value into a
    valid-looking one (source audit, 19 Oct 2026, findings 1-6).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

NORM_TOL = 1e-6          # unit-norm tolerance for embeddings and hypotheses
NORMALISED_TOL = 1e-9    # |logsumexp(log p)| tolerance for a normalised pmf
SD_FLOOR = 1e-6          # per-component window SD below which z-scores are zeroed
KAPPA_RANGE = (1.0, 100.0)
# Cyclic seed pairs for the two-head ensemble controls, fixed in advance
# (v3.1 §8). The deployed policy uses seed 0.
SEED_PAIRS: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 2), (2, 0))
DEPLOYED_SEED = 0


class ContractError(ValueError):
    """An input violates the BBNV scoring contract."""


# --------------------------------------------------------------- validators --
def _is_int(x) -> bool:
    """A genuine integer: Python int or numpy integer, never a bool."""
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def as_index(x, n: int, name: str) -> int:
    """A genuine integer index in [0, n)."""
    if not _is_int(x):
        raise ContractError("%s must be an integer, got %r" % (name, x))
    if not 0 <= int(x) < n:
        raise ContractError("%s %d outside [0, %d)" % (name, int(x), n))
    return int(x)


def as_count(x, name: str, minimum: int = 1) -> int:
    """A genuine integer count >= minimum."""
    if not _is_int(x) or int(x) < minimum:
        raise ContractError("%s must be an integer >= %d, got %r" % (name, minimum, x))
    return int(x)


def window_vector(v, name: str) -> np.ndarray:
    """One window's scores: a finite, non-empty, 1-D float64 array."""
    a = np.asarray(v, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise ContractError("%s must be a non-empty 1-D window, got shape %s" % (name, a.shape))
    if not np.all(np.isfinite(a)):
        raise ContractError("%s has non-finite values" % name)
    return a


def _check_prior(log_rho, n: int) -> np.ndarray:
    """A 1-D log-prior of length n: normalised, every element finite (positive mass)."""
    a = np.asarray(log_rho, dtype=np.float64)
    if a.ndim != 1 or a.shape[0] != n:
        raise ContractError("prior must have shape (%d,), got %s" % (n, a.shape))
    if not np.all(np.isfinite(a)):
        raise ContractError("every element of R_q needs positive prior mass")
    return check_log_pmf(a, "reference prior")


def _check_table(log_t, name: str) -> np.ndarray:
    """A non-empty 2-D table whose rows are normalised log-pmfs."""
    a = np.asarray(log_t, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
        raise ContractError("%s must be a non-empty 2-D table, got shape %s" % (name, a.shape))
    return check_log_pmf(a, name, axis=1)


def _require_finite_marginal(log_marg: np.ndarray) -> np.ndarray:
    """PMI is undefined where the bank marginal is zero (v3 review 2.1)."""
    if not np.all(np.isfinite(log_marg)):
        raise ContractError("a window candidate has zero marginal mass on this bank: "
                            "PMI is undefined there")
    return log_marg


# ------------------------------------------------------------------ basics --
def logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable log(sum(exp(a))) along `axis`; -inf rows give -inf."""
    a = np.asarray(a, dtype=np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):            # an all -inf row has log(0) = -inf by design
        out = np.log(np.sum(np.exp(a - m_safe), axis=axis, keepdims=True)) + m_safe
    return np.squeeze(out, axis=axis)


def log_softmax(a: np.ndarray, axis: int = -1) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a - np.expand_dims(logsumexp(a, axis=axis), axis)


def check_unit_rows(x: np.ndarray, name: str) -> np.ndarray:
    """Finite rows of unit L2 norm (embeddings, hypotheses)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 2 or x.size == 0:
        raise ContractError("%s must be a non-empty array of row vectors" % name)
    if not np.all(np.isfinite(x)):
        raise ContractError("%s has non-finite values" % name)
    norms = np.linalg.norm(x, axis=-1)
    if np.max(np.abs(norms - 1.0)) > NORM_TOL:
        raise ContractError("%s rows are not unit-norm (max deviation %.3g)"
                            % (name, float(np.max(np.abs(norms - 1.0)))))
    return x


def check_log_pmf(logp: np.ndarray, name: str, axis: int = -1) -> np.ndarray:
    """A log-pmf: no NaN/+inf, sums to one along `axis`."""
    logp = np.asarray(logp, dtype=np.float64)
    if logp.size == 0:
        raise ContractError("%s is empty" % name)
    if np.any(np.isnan(logp)) or np.any(logp == np.inf):
        raise ContractError("%s contains NaN or +inf" % name)
    dev = np.max(np.abs(logsumexp(logp, axis=axis)))
    if dev > NORMALISED_TOL:
        raise ContractError("%s is not normalised (|logsumexp| up to %.3g)" % (name, float(dev)))
    return logp


def check_kappa(kappa) -> np.ndarray:
    k = np.asarray(kappa, dtype=np.float64)
    if not np.all(np.isfinite(k)) or np.any(k < KAPPA_RANGE[0]) or np.any(k > KAPPA_RANGE[1]):
        raise ContractError("kappa must lie in [%g, %g]" % KAPPA_RANGE)
    return k


# --------------------------------------------------------------- universes --
@dataclass(frozen=True)
class Universe:
    """A de-duplicated finite support: image ids, their unit embeddings, and
    the position of each id."""
    ids: Tuple
    emb: np.ndarray

    def index(self, image_id) -> int:
        return self.ids.index(image_id)


def universe_digest(ids: Sequence[Hashable]) -> str:
    """SHA-256 of an ordered support's ids: identifies the support AND its order."""
    h = hashlib.sha256()
    for i in ids:
        h.update(repr(i).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def build_universe(bank_ids: Sequence, bank_emb: np.ndarray,
                   extra_ids: Sequence = (), extra_emb: Optional[np.ndarray] = None) -> Universe:
    """bank ∪ extra, de-duplicated by image id, bank order first.

    The bank must itself be duplicate-free. An extra id already in the bank
    is not added again (v3.1 A), but its embedding must equal the bank's:
    the same image id with a different vector is a provenance error."""
    bank_ids = list(bank_ids)
    bank_emb = check_unit_rows(bank_emb, "bank embeddings")
    if bank_emb.ndim != 2 or len(bank_ids) != bank_emb.shape[0]:
        raise ContractError("bank ids and embeddings differ in length")
    if len(set(bank_ids)) != len(bank_ids):
        raise ContractError("bank contains duplicate image ids")
    ids, rows = list(bank_ids), [bank_emb]
    pos = {i: n for n, i in enumerate(bank_ids)}
    extra_ids = list(extra_ids)
    if extra_ids:
        if extra_emb is None:
            raise ContractError("extra ids need embeddings")
        extra_emb = check_unit_rows(extra_emb, "extra embeddings")
        if extra_emb.ndim != 2 or len(extra_ids) != extra_emb.shape[0] \
                or extra_emb.shape[1] != bank_emb.shape[1]:
            raise ContractError("extra ids and embeddings differ in length or dimension")
        for i, e in zip(extra_ids, extra_emb):
            if i in pos:
                if np.max(np.abs(_rows_lookup(rows, pos[i]) - e)) > NORM_TOL:
                    raise ContractError("image id %r appears with two different embeddings" % (i,))
                continue
            pos[i] = len(ids)
            ids.append(i)
            rows.append(e[None, :])
    return Universe(ids=tuple(ids), emb=np.concatenate(rows, axis=0))


def _rows_lookup(row_blocks: List[np.ndarray], n: int) -> np.ndarray:
    """Row n of the vertical concatenation of `row_blocks`, without copying it."""
    for block in row_blocks:
        if n < block.shape[0]:
            return block[n]
        n -= block.shape[0]
    raise IndexError("row index out of range")


# ------------------------------------------------------------------ priors --
def log_prior_uniform(n: int) -> np.ndarray:
    """Uniform prior over a support of size n: every element, including the
    query reference, has mass 1/n > 0 (v3.1 A, primary)."""
    n = as_count(n, "prior support size")
    return np.full(n, -math.log(n))


def log_prior_instruction(t_emb: np.ndarray, support_emb: np.ndarray, beta: float) -> np.ndarray:
    """Instruction-weighted prior over the same support:
    rho(x | t) ∝ exp(beta * t·x), normalised over the support (v3.1 A, sensitivity)."""
    if isinstance(beta, bool) or not isinstance(beta, (int, float, np.floating, np.integer)) \
            or not math.isfinite(float(beta)):
        raise ContractError("beta must be a finite number")
    t = np.asarray(t_emb, dtype=np.float64)
    if t.ndim != 1:
        raise ContractError("the instruction embedding must be 1-D")
    t = check_unit_rows(t[None, :], "instruction embedding")[0]
    x = check_unit_rows(support_emb, "prior support embeddings")
    if x.ndim != 2 or x.shape[1] != t.shape[0]:
        raise ContractError("support and instruction dimensions differ")
    return log_softmax(float(beta) * (x @ t))


# ---------------------------------------------------------------- mixtures --
def mixture_log_prob(hyp: np.ndarray, log_pi: np.ndarray, kappa: float,
                     support_emb: np.ndarray) -> np.ndarray:
    """Component-normalised vMF-kernel mixture over a finite support.

    hyp (M, d) unit hypotheses, log_pi (M,) normalised log mixture masses,
    kappa in KAPPA_RANGE. Returns log q(x) for every x in the support (N,):
        q(x) = sum_m pi_m softmax_x(kappa * hyp_m · x)."""
    hyp = np.asarray(hyp, dtype=np.float64)
    if hyp.ndim != 2:
        raise ContractError("hypotheses must have shape (M, d)")
    out = mixture_log_prob_batch(hyp[None], np.asarray(log_pi)[None], np.asarray([kappa]),
                                 support_emb)
    return out[0]


def mixture_log_prob_batch(hyp: np.ndarray, log_pi: np.ndarray, kappa: np.ndarray,
                           support_emb: np.ndarray) -> np.ndarray:
    """Batched mixture_log_prob: hyp (B, M, d), log_pi (B, M), kappa (B,) ->
    (B, N) log-pmfs, each normalised over the same support."""
    hyp = np.asarray(hyp, dtype=np.float64)
    if hyp.ndim != 3 or 0 in hyp.shape:
        raise ContractError("hypotheses must have non-empty shape (B, M, d)")
    check_unit_rows(hyp.reshape(-1, hyp.shape[-1]), "hypotheses")
    log_pi = np.asarray(log_pi, dtype=np.float64)
    if log_pi.shape != hyp.shape[:2]:
        raise ContractError("log_pi shape %s does not match hypotheses %s"
                            % (log_pi.shape, hyp.shape[:2]))
    log_pi = check_log_pmf(log_pi, "log mixture masses")
    kappa = check_kappa(kappa)
    if kappa.ndim != 1 or kappa.shape[0] != hyp.shape[0]:
        raise ContractError("one kappa per conditioning item is required")
    x = check_unit_rows(support_emb, "support embeddings")
    if x.ndim != 2 or x.shape[1] != hyp.shape[2]:
        raise ContractError("support and hypothesis dimensions differ")
    logits = kappa[:, None, None] * np.einsum("bmd,nd->bmn", hyp, x)      # (B, M, N)
    comp = log_softmax(logits, axis=-1)                                    # each q_m normalised
    return logsumexp(log_pi[:, :, None] + comp, axis=1)                    # (B, N)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats for pmfs with supp(p) ⊆ supp(q); 0 log 0 = 0."""
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))


def mixture_masses_report(log_pi: np.ndarray, comp_log_prob: np.ndarray) -> Dict[str, object]:
    """Diagnostics for one conditioning item (v3.1 §1.2).

    log_pi (M,) normalised log masses; comp_log_prob (M, N), each row a
    normalised component log-pmf. Returns the actual masses and the mean
    pairwise Jensen-Shannon DIVERGENCE between components (nats, in
    [0, log 2]): 0 means two components are identical; larger values mean
    better-separated components. None when M = 1."""
    log_pi = np.asarray(log_pi, dtype=np.float64)
    comp = np.asarray(comp_log_prob, dtype=np.float64)
    if log_pi.ndim != 1 or comp.ndim != 2 or log_pi.shape[0] == 0 or comp.shape[1] == 0:
        raise ContractError("need log_pi (M,) and component log-pmfs (M, N)")
    if comp.shape[0] != log_pi.shape[0]:
        raise ContractError("%d masses but %d component distributions"
                            % (log_pi.shape[0], comp.shape[0]))
    log_pi = check_log_pmf(log_pi, "log mixture masses")
    comp = check_log_pmf(comp, "component log-pmfs", axis=-1)
    p = np.exp(comp)
    m = p.shape[0]
    js = []
    for i in range(m):
        for j in range(i + 1, m):
            mid = 0.5 * (p[i] + p[j])
            js.append(0.5 * _kl(p[i], mid) + 0.5 * _kl(p[j], mid))
    return {"masses": np.exp(log_pi).tolist(),
            "mean_pairwise_js_divergence": float(np.mean(js)) if js else None}


# ----------------------------------------------------------- query scoring --
def forward_query_scores(log_f: np.ndarray, log_rho: np.ndarray, ref_index: int,
                         window_index: Sequence[int]) -> Dict[str, np.ndarray]:
    """Forward-side quantities for one query.

    log_f (|R_q|, |C_q|): log q_F(c | x, t) for every x in R_q, each row
    normalised over the SAME C_q. log_rho (|R_q|,): the prior on R_q.
    ref_index: the query reference's row; window_index: the window
    candidates' columns in C_q. Returns, for the window candidates (K,):
      log_marginal  log p_C(c | t) = logsumexp_x(log rho(x) + log q_F(c | x))
      pmi_f         log q_F(c | r) - log p_C(c | t)
      ie_induced    log q_I^F(r | c) = log rho(r) + pmi_f   (v3.1 A, B)
      log_f_raw     log q_F(c | r), the uncorrected forward score."""
    log_f = _check_table(log_f, "forward conditionals")
    log_rho = _check_prior(log_rho, log_f.shape[0])
    ref_index = as_index(ref_index, log_f.shape[0], "reference index")
    win = _window(window_index, log_f.shape[1])
    log_marg = logsumexp(log_rho[:, None] + log_f[:, win], axis=0)
    _require_finite_marginal(log_marg)
    raw = log_f[ref_index, win]
    pmi = raw - log_marg
    # With a validated 1-D prior and 2-D table, every output has shape (K,)
    # and ie_induced <= 0 holds by construction (rho_r q(c|r) <= sum_x rho_x
    # q(c|x)); both are pinned by tests, not re-checked here.
    return {"log_marginal": log_marg, "pmi_f": pmi, "ie_induced": log_rho[ref_index] + pmi,
            "log_f_raw": raw}


def forward_lambda_score(log_f_raw: np.ndarray, log_marginal: np.ndarray, lam: float) -> np.ndarray:
    """Interpolated forward evidence (v3.1 §9; LEDOM, ACL 2026, Eqs. 7-9):
        s_lambda(c) = log q_F(c | r, t) - lambda * log p_C(c | t)
                    = (1 - lambda) * raw + lambda * PMI_F,   lambda in [0, 1].
    lambda = 0 is the raw forward score, lambda = 1 is full PMI. The
    forward-induced inverse remains tied to lambda = 1 only."""
    if isinstance(lam, bool) or not isinstance(lam, (int, float, np.floating, np.integer)) \
            or not 0.0 <= float(lam) <= 1.0:
        raise ContractError("lambda must be a number in [0, 1]")
    raw = window_vector(log_f_raw, "raw forward scores")
    marg = window_vector(log_marginal, "log marginal")
    if raw.shape != marg.shape:
        raise ContractError("raw scores and marginal cover different windows")
    return raw - float(lam) * marg


def chunked_log_marginal(row_fn: Callable[[int, int], Tuple[np.ndarray, Sequence[Hashable]]],
                         log_rho: np.ndarray, window_index: Sequence[int], chunk: int,
                         candidate_ids: Sequence[Hashable]) -> np.ndarray:
    """log p_C(c | t) for the window, accumulated over R_q in chunks with an
    online log-sum-exp, so the full |R_q| x |C_q| table never has to be held.

    row_fn(start, stop) returns (rows, ids): the normalised forward rows for
    x in [start, stop) and the ordered candidate ids of their columns. Every
    chunk must carry exactly `candidate_ids` (same universe, same order), and
    the prior follows the same contract as forward_query_scores."""
    candidate_ids = tuple(candidate_ids)
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ContractError("candidate ids must be non-empty and duplicate-free")
    chunk = as_count(chunk, "chunk")
    log_rho = np.asarray(log_rho, dtype=np.float64)
    if log_rho.ndim != 1:
        raise ContractError("prior must be 1-D")
    n_rows = log_rho.shape[0]
    log_rho = _check_prior(log_rho, n_rows)
    win = _window(window_index, len(candidate_ids))
    acc = None
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        rows, ids = row_fn(start, stop)
        if tuple(ids) != candidate_ids:
            raise ContractError("rows %d-%d use a different or reordered candidate universe"
                                % (start, stop))
        rows = _check_table(rows, "forward conditionals (chunk)")
        if rows.shape != (stop - start, len(candidate_ids)):
            raise ContractError("chunk %d-%d has shape %s, expected %s"
                                % (start, stop, rows.shape, (stop - start, len(candidate_ids))))
        part = logsumexp(log_rho[start:stop, None] + rows[:, win], axis=0)
        acc = part if acc is None else np.logaddexp(acc, part)
    return _require_finite_marginal(acc)


def inverse_query_scores(log_i: np.ndarray, ref_index: int) -> np.ndarray:
    """Learned inverse evidence for one query: log q_I(r | c, t) for each window
    candidate. log_i (K, |R_q|): one normalised log-pmf over R_q per candidate."""
    log_i = _check_table(log_i, "inverse conditionals")
    ref_index = as_index(ref_index, log_i.shape[1], "reference index")
    return log_i[:, ref_index].copy()


def _window(window_index: Sequence[int], n_support: int) -> np.ndarray:
    """Window positions in the candidate support: genuine integers, no bools,
    no floats, in range, without repeats. Nothing is coerced."""
    items = list(window_index) if not isinstance(window_index, np.ndarray) \
        else window_index.tolist() if window_index.ndim == 1 else None
    if items is None or not items:
        raise ContractError("window must be a non-empty 1-D index list")
    if isinstance(window_index, np.ndarray) and not np.issubdtype(window_index.dtype, np.integer):
        raise ContractError("window index array must have an integer dtype")
    for x in items:
        if not _is_int(x):
            raise ContractError("window index %r is not an integer" % (x,))
    win = np.asarray([int(x) for x in items], dtype=np.int64)
    if np.any(win < 0) or np.any(win >= n_support):
        raise ContractError("window index outside the candidate support")
    if np.unique(win).size != win.size:
        raise ContractError("window contains a candidate twice")
    return win


# ---------------------------------------------------- window transformations --
def center(v: np.ndarray) -> np.ndarray:
    """Centre ONE window (a finite, non-empty 1-D array)."""
    v = window_vector(v, "window scores")
    return v - v.mean()


def zscore(v: np.ndarray, floor: float = SD_FLOOR) -> np.ndarray:
    """Within-window z-score of ONE window; a component whose window SD is
    below `floor` contributes zeros for that window (v3.1 §4.1), never
    amplified noise. `floor` must be finite and strictly positive."""
    if isinstance(floor, bool) or not isinstance(floor, (int, float, np.floating)) \
            or not math.isfinite(float(floor)) or float(floor) <= 0.0:
        raise ContractError("floor must be a finite number > 0")
    c = center(v)
    sd = float(np.sqrt(np.mean(c * c)))
    return np.zeros_like(c) if sd < floor else c / sd


def integrate(head_scores: Sequence[np.ndarray], edit_cos: np.ndarray, gamma: float) -> np.ndarray:
    """V_H(c) = (1/|H|) sum_h z_W(s_h(c)) + gamma * z_W(edit_cos(c))  (v3.1 §8),
    for ONE window. Averaging over heads keeps the verification-to-edit
    balance identical for one-head and two-head arms; gamma is frozen and
    shared by every arm."""
    if not head_scores:
        raise ContractError("an arm needs at least one head")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float, np.floating)) \
            or not math.isfinite(float(gamma)) or float(gamma) < 0:
        raise ContractError("gamma must be a finite number >= 0")
    edit = window_vector(edit_cos, "edit scores")
    zs = []
    for s in head_scores:
        s = window_vector(s, "head scores")
        if s.shape != edit.shape:
            raise ContractError("head and edit scores cover different windows")
        zs.append(zscore(s))
    return np.mean(zs, axis=0) + gamma * zscore(edit)


def gate_window_features(ie_learned: np.ndarray, ie_induced: np.ndarray,
                         log_marginal: np.ndarray) -> Dict[str, np.ndarray]:
    """Centred window representations for the gates (v3.1 §4.2). `delta` is
    computed from exactly the arrays returned as c_ie_learned and
    c_ie_induced, so a linear stacker given those two can represent it."""
    c_learned = center(ie_learned)
    c_induced = center(ie_induced)
    c_marg = center(log_marginal)
    if not (c_learned.shape == c_induced.shape == c_marg.shape):
        raise ContractError("gate features cover different windows")
    return {"c_ie_learned": c_learned, "c_ie_induced": c_induced,
            "delta": c_learned - c_induced, "c_log_marginal": c_marg}


# ----------------------------------------------------------------- metrics --
def _binary_labels(positive, n: int) -> np.ndarray:
    """Labels as booleans: a bool array, or finite values that are exactly 0 or 1."""
    lab = np.asarray(positive)
    if lab.ndim != 1 or lab.shape[0] != n:
        raise ContractError("labels must be 1-D and match the scores")
    if lab.dtype == bool:
        return lab
    if not np.issubdtype(lab.dtype, np.number):
        raise ContractError("labels must be boolean or numeric 0/1")
    labf = lab.astype(np.float64)
    if not np.all(np.isfinite(labf)) or not np.all((labf == 0.0) | (labf == 1.0)):
        raise ContractError("labels must be exactly 0 or 1")
    return labf == 1.0


def pairwise_auroc(scores: np.ndarray, positive: np.ndarray) -> Optional[float]:
    """Per-query AUROC: the fraction of (positive, negative) pairs ordered
    correctly, ties counting one half. None if the window lacks a positive or
    a negative (v3.1 §4.5)."""
    s = window_vector(scores, "AUROC scores")
    pos = _binary_labels(positive, s.shape[0])
    sp, sn = s[pos], s[~pos]
    if sp.size == 0 or sn.size == 0:
        return None
    diff = sp[:, None] - sn[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)


def _check_auroc_value(v, where: str) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float, np.floating)) \
            or not math.isfinite(float(v)) or not 0.0 <= float(v) <= 1.0:
        raise ContractError("%s: %r is not None or a finite AUROC in [0, 1]" % (where, v))
    return float(v)


def macro_auroc(per_query: Sequence[Optional[float]]) -> Tuple[Optional[float], int]:
    """Mean over the queries that have a defined AUROC, and how many those are.
    Every entry must be None (undefined) or a finite value in [0, 1]."""
    vals = [v for v in (_check_auroc_value(x, "macro_auroc") for x in per_query) if v is not None]
    return (float(np.mean(vals)) if vals else None), len(vals)


def paired_macro_auroc(method: Mapping[Hashable, Optional[float]],
                       control: Mapping[Hashable, Optional[float]]) -> Dict[str, object]:
    """Method and control compared on the SAME queries: per-query AUROCs keyed
    by query id; only queries where both are defined enter (shared mask)."""
    if set(method) != set(control):
        raise ContractError("method and control cover different query ids")
    # validate EVERY value first, so an invalid entry is never skipped because
    # its partner happens to be undefined
    m_vals = {k: _check_auroc_value(method[k], "method %r" % (k,)) for k in method}
    c_vals = {k: _check_auroc_value(control[k], "control %r" % (k,)) for k in control}
    keys = [k for k in method if m_vals[k] is not None and c_vals[k] is not None]
    if not keys:
        return {"n_shared": 0, "method": None, "control": None, "difference": None}
    m = np.array([m_vals[k] for k in keys], dtype=np.float64)
    c = np.array([c_vals[k] for k in keys], dtype=np.float64)
    return {"n_shared": len(keys), "method": float(m.mean()), "control": float(c.mean()),
            "difference": float((m - c).mean())}


def validate_seed_pairs(pairs: Sequence[Tuple[int, int]] = SEED_PAIRS) -> None:
    """The ensemble seed pairs are the fixed cyclic pairs (v3.1 §8)."""
    if tuple(tuple(p) for p in pairs) != SEED_PAIRS:
        raise ContractError("seed pairs must be the pre-declared cyclic pairs %s" % (SEED_PAIRS,))
