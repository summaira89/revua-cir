"""BBNV Phase A: enumerated CPU oracle (v3 §3 as amended by v3.1 §6 and §8).

A fully specified discrete joint with known forward, inverse, marginal and
PMI tables, synthetic unit embeddings, and small heads trained on samples
from the joint. Its two jobs, kept separate:
  1. software correctness: the Bayes identity and every scoring invariant
     hold on enumerated tables (the unit tests in tests/test_bbnv.py);
  2. a falsifiable first look at the hypotheses: the outcomes are RECORDED
     in the output, never asserted, and none is required to come out a
     particular way (v3.1 §6, invariant 8).

Generative model (stated in full, v3 review §3.2):
  items       regular (u, a), u < U identities, a < A attributes; plus H hubs
  embeddings  regular (e_u + e_a)/sqrt(2); hub e_hub; orthonormal bases
  instruction t_v = "set the attribute to v", v < A; one-hot embedding
  applicability  t_v applies to references with a != v; rho(r | t_v) is
                 uniform over those (U * (A - 1) references)
  forward     p(c | r=(u,a), t_v) = (1 - eps - eta) [c = (u, v)]
                                    + eta / H [c is a hub] + eps / N_c
              so every candidate has positive mass (eps > 0) and hubs
              receive mass from every reference through the joint itself
  candidates  all regular items and all hubs (N_c = U*A + H)

Outputs, in a FRESH directory (refused if it already holds files):
predictions.jsonl (every query's window, labels and per-arm scores) is written
first, results.json (results + run manifest + each companion's SHA-256) last,
as the completion marker; both atomically. bbnv_provenance.verify_run checks
a directory.

Usage
-----
  python scripts/bbnv_oracle.py --out-dir ../bbnv_runs/phaseA_<date>
"""
from __future__ import annotations

import os
import sys

# Windows/Anaconda: numpy's MKL and pip's torch each ship an Intel OpenMP
# runtime; once both start threads the process aborts (OMP Error #15), and the
# runtime's own "KMP_DUPLICATE_LIB_OK" override can corrupt results, so it is
# not used. MKL's documented sequential mode keeps numpy off OpenMP, leaving
# torch's runtime as the only one. It is only effective if set before numpy is
# first imported, which is recorded (runtime.mkl_guard_effective); an explicit
# user setting wins; it is harmless on Linux.
_NUMPY_PRELOADED = "numpy" in sys.modules
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import argparse  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bbnv_core as core  # noqa: E402
import bbnv_provenance as prov  # noqa: E402

DIRECTIONS = ("forward", "inverse")
DISTRACTOR_KINDS = ("wrong_identity", "reference_like", "hub")
# Phase A columns (v3.1 §2). F, I, F+I and F+F' are the Q1 arms; the others are
# reference columns: the oracle's own PMI and raw forward probability, the
# learned uncorrected forward score, and the forward-induced inverse (which
# must rank exactly like F; invariant 4).
ARM_NAMES = ("F", "I", "F+I", "F+F'", "true_pmi", "true_raw", "F_raw_uncorrected",
             "induced_inverse")
SEEDS = (0, 1, 2)          # the three seeds the cyclic pairs (0,1), (1,2), (2,0) need


def _check_direction(direction: str) -> str:
    if direction not in DIRECTIONS:
        raise ValueError("direction must be one of %s, got %r" % (DIRECTIONS, direction))
    return direction


# ------------------------------------------------------------ the joint --
@dataclass(frozen=True)
class OracleSpec:
    n_identity: int = 4
    n_attribute: int = 8
    n_hub: int = 4
    eps: float = 0.02
    eta: float = 0.10

    def validate(self) -> None:
        for name in ("n_identity", "n_attribute", "n_hub"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError("%s must be an integer" % name)
        if self.n_identity < 2 or self.n_attribute < 2 or self.n_hub < 0:
            raise ValueError("need >= 2 identities, >= 2 attributes and >= 0 hubs")
        if not (0 < self.eps < 1) or not (0 <= self.eta < 1) or self.eps + self.eta >= 1:
            raise ValueError("need 0 < eps, 0 <= eta, eps + eta < 1")
        if self.n_hub == 0 and self.eta != 0:
            raise ValueError("eta must be 0 without hubs")


class Oracle:
    """Items, embeddings and exact tables for every instruction."""

    def __init__(self, spec: OracleSpec):
        spec.validate()
        self.spec = spec
        U, A, H = spec.n_identity, spec.n_attribute, spec.n_hub
        self.dim = U + A + H
        eye = np.eye(self.dim)
        self.items: List[Tuple[str, int, int]] = (
            [("reg", u, a) for u in range(U) for a in range(A)] + [("hub", h, -1) for h in range(H)])
        self.item_index = {it: n for n, it in enumerate(self.items)}
        emb = []
        for kind, x, y in self.items:
            v = eye[x] + eye[U + y] if kind == "reg" else eye[U + A + x]
            emb.append(v / np.linalg.norm(v))
        self.emb = core.check_unit_rows(np.array(emb), "oracle embeddings")
        self.t_emb = np.eye(A)                                   # instruction embeddings
        self.n_c = len(self.items)
        self._tables = {v: self._build_tables(v) for v in range(A)}

    def references(self, v: int) -> List[int]:
        """Item indices of the references instruction v applies to (a != v)."""
        return [self.item_index[("reg", u, a)] for u in range(self.spec.n_identity)
                for a in range(self.spec.n_attribute) if a != v]

    def target_of(self, ref_item: int, v: int) -> int:
        _, u, _ = self.items[ref_item]
        return self.item_index[("reg", u, v)]

    def _build_tables(self, v: int) -> Dict[str, np.ndarray]:
        s = self.spec
        refs = self.references(v)
        p_f = np.full((len(refs), self.n_c), s.eps / self.n_c)
        hubs = [n for n, it in enumerate(self.items) if it[0] == "hub"]
        for row, r in enumerate(refs):
            p_f[row, self.target_of(r, v)] += 1.0 - s.eps - s.eta
            for h in hubs:
                p_f[row, h] += s.eta / len(hubs)
        log_f = np.log(p_f)
        log_rho = core.log_prior_uniform(len(refs))
        log_marg = core.logsumexp(log_rho[:, None] + log_f, axis=0)          # (N_c,)
        pmi = log_f - log_marg[None, :]                                       # (R, C)
        log_inv = (log_rho[:, None] + log_f - log_marg[None, :]).T           # (C, R) by Bayes
        return {"refs": np.array(refs), "log_f": log_f, "log_rho": log_rho,
                "log_marg": log_marg, "pmi": pmi, "log_inv": log_inv}

    def tables(self, v: int) -> Dict[str, np.ndarray]:
        """Exact tables for instruction v over R_v (rows) x C (columns)."""
        return self._tables[v]

    def row_of(self, ref_item: int, v: int) -> int:
        return self.references(v).index(ref_item)

    def window(self, ref_item: int, v: int) -> Tuple[List[int], List[str]]:
        """Target, every wrong identity with the same new attribute (supported,
        query-incompatible), the unedited reference (reference-like, eps-only
        support) and every hub."""
        _, u, _ = self.items[ref_item]
        cand, kind = [self.target_of(ref_item, v)], ["target"]
        for u2 in range(self.spec.n_identity):
            if u2 != u:
                cand.append(self.item_index[("reg", u2, v)])
                kind.append("wrong_identity")
        cand.append(ref_item)
        kind.append("reference_like")
        for n, it in enumerate(self.items):
            if it[0] == "hub":
                cand.append(n)
                kind.append("hub")
        return cand, kind


def bayes_check(oracle: Oracle) -> Dict[str, float]:
    """Before any training: every table normalised, and the inverse equals
    log rho + PMI exactly (v3 review §3.2 step 2)."""
    worst_norm = worst_bayes = 0.0
    for v in range(oracle.spec.n_attribute):
        t = oracle.tables(v)
        worst_norm = max(worst_norm, float(np.max(np.abs(core.logsumexp(t["log_f"], 1)))),
                         float(np.max(np.abs(core.logsumexp(t["log_inv"], 1)))))
        recon = t["log_rho"][None, :] + t["pmi"].T
        worst_bayes = max(worst_bayes, float(np.max(np.abs(t["log_inv"] - recon))))
    return {"max_normalisation_error": worst_norm, "max_bayes_error": worst_bayes}


# --------------------------------------------------------------- heads --
def _torch():
    import torch  # local import: the scoring core and its tests do not need torch
    return torch


def build_head(in_dim: int, out_dim: int, hidden: int, n_comp: int, seed: int):
    """Two-layer MLP -> n_comp unit hypotheses, component masses, bounded kappa."""
    for name, v in (("in_dim", in_dim), ("out_dim", out_dim), ("hidden", hidden), ("n_comp", n_comp)):
        core.as_count(v, name)
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(seed)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_comp, self.out_dim = n_comp, out_dim
            self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                      nn.Linear(hidden, hidden), nn.GELU())
            self.hyp = nn.Linear(hidden, n_comp * out_dim)
            self.mix = nn.Linear(hidden, n_comp) if n_comp > 1 else None
            self.kappa_raw = nn.Parameter(torch.zeros(()))     # kappa = 1 + 99 * sigmoid(raw)

        def forward(self, x):
            h = self.body(x)
            hyp = torch.nn.functional.normalize(
                self.hyp(h).view(-1, self.n_comp, self.out_dim), dim=-1)
            if self.mix is None:
                log_pi = torch.zeros(x.shape[0], 1, dtype=x.dtype)
            else:
                log_pi = torch.log_softmax(self.mix(h), dim=-1)
            lo, hi = core.KAPPA_RANGE
            kappa = lo + (hi - lo) * torch.sigmoid(self.kappa_raw)
            return hyp, log_pi, kappa.expand(x.shape[0])

    return Head().double()


def param_count(head) -> int:
    return int(sum(p.numel() for p in head.parameters()))


def torch_mixture_log_prob(hyp, log_pi, kappa, support):
    """Torch twin of core.mixture_log_prob_batch (same formula, autograd)."""
    torch = _torch()
    logits = kappa[:, None, None] * torch.einsum("bmd,nd->bmn", hyp, support)
    comp = torch.log_softmax(logits, dim=-1)
    return torch.logsumexp(log_pi[:, :, None] + comp, dim=1)


def width_matched_hidden(in_dim: int, out_dim: int, hidden_m8: int, m: int = 8) -> int:
    """Hidden width at which an M=1 head has (nearly) as many parameters as an
    M=m head of width hidden_m8 (v3 review: width-matched M=1 control). The
    closed form is checked against the real modules in main and in the tests."""
    def count(hid, comps):
        mix = (hid * comps + comps) if comps > 1 else 0
        return (in_dim * hid + hid) + (hid * hid + hid) + (hid * comps * out_dim + comps * out_dim) + mix + 1
    target = count(hidden_m8, m)
    return min(range(1, 8 * hidden_m8 + 1), key=lambda h: abs(count(h, 1) - target))


@dataclass(frozen=True)
class TrainConfig:
    hidden: int = 64
    n_comp: int = 1
    steps: int = 400
    batch: int = 256
    lr: float = 1e-2

    def validate(self) -> None:
        for name in ("hidden", "n_comp", "steps", "batch"):
            core.as_count(getattr(self, name), name)
        if not isinstance(self.lr, float) or not 0 < self.lr < 1:
            raise ValueError("lr must be a float in (0, 1)")


def sample_triplets(oracle: Oracle, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 3) array of (reference item, instruction v, candidate item) drawn from the joint."""
    n = core.as_count(n, "number of samples")
    out = np.empty((n, 3), dtype=np.int64)
    A = oracle.spec.n_attribute
    for k in range(n):
        v = int(rng.integers(A))
        t = oracle.tables(v)
        row = int(rng.integers(len(t["refs"])))
        c = int(rng.choice(oracle.n_c, p=np.exp(t["log_f"][row])))
        out[k] = (t["refs"][row], v, c)
    return out


def _check_samples(oracle: Oracle, data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] != 3 or data.shape[0] == 0 \
            or not np.issubdtype(data.dtype, np.integer):
        raise ValueError("samples must be a non-empty integer array of shape (n, 3)")
    if np.any(data[:, 1] < 0) or np.any(data[:, 1] >= oracle.spec.n_attribute) \
            or np.any(data[:, [0, 2]] < 0) or np.any(data[:, [0, 2]] >= oracle.n_c):
        raise ValueError("sample indices outside the oracle")
    return data


def train_head(oracle: Oracle, data: np.ndarray, direction: str, cfg: TrainConfig, seed: int):
    """Fit one head with full-support categorical cross-entropy (the oracle
    contract: exact enumeration of the support, v3 review 2.4 option 1).
    forward: predict the candidate over all items from (reference, t);
    inverse: predict the reference over R_v from (candidate, t)."""
    _check_direction(direction)
    cfg.validate()
    data = _check_samples(oracle, data)
    torch = _torch()
    head = build_head(oracle.dim + oracle.spec.n_attribute, oracle.dim, cfg.hidden, cfg.n_comp, seed)
    opt = torch.optim.Adam(head.parameters(), lr=cfg.lr)
    emb = torch.tensor(oracle.emb)
    temb = torch.tensor(oracle.t_emb)
    ref_sets = {v: torch.tensor(oracle.references(v)) for v in range(oracle.spec.n_attribute)}
    gen = np.random.default_rng(seed)
    for _ in range(cfg.steps):
        idx = gen.integers(len(data), size=cfg.batch)
        batch = data[idx]
        r, v, c = (torch.tensor(batch[:, j]) for j in range(3))
        cond = emb[r] if direction == "forward" else emb[c]
        hyp, log_pi, kappa = head(torch.cat([cond, temb[v]], dim=1))
        if direction == "forward":
            lp = torch_mixture_log_prob(hyp, log_pi, kappa, emb)              # over all items
            loss = -lp.gather(1, c[:, None]).mean()
        else:
            loss = 0.0
            for vv in torch.unique(v).tolist():                               # support depends on v
                sel = v == vv
                refs = ref_sets[vv]
                lp = torch_mixture_log_prob(hyp[sel], log_pi[sel], kappa[sel], emb[refs])
                match = refs[None, :] == r[sel][:, None]
                if not bool(match.any(1).all()):
                    raise RuntimeError("a sampled reference is outside its instruction's support")
                pos = match.double().argmax(1)
                loss = loss - lp.gather(1, pos[:, None]).sum()
            loss = loss / len(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head


def head_outputs(head, cond: np.ndarray, t_vec: np.ndarray):
    """Numpy hypotheses, log masses and kappa for conditioning rows `cond`."""
    torch = _torch()
    with torch.no_grad():
        x = torch.cat([torch.tensor(cond), torch.tensor(t_vec)], dim=1)
        hyp, log_pi, kappa = head(x)
    return hyp.numpy(), log_pi.numpy(), kappa.numpy()


def learned_query_scores(oracle: Oracle, f_head, i_head, ref_item: int, v: int,
                         window: Sequence[int]) -> Dict[str, np.ndarray]:
    """Score one query's window through bbnv_core, exactly as real data will be."""
    refs = oracle.references(v)
    r_pos = refs.index(ref_item)
    t_rows = np.repeat(oracle.t_emb[v][None, :], len(refs), axis=0)
    out: Dict[str, np.ndarray] = {}
    if f_head is not None:
        hyp, lpi, kap = head_outputs(f_head, oracle.emb[refs], t_rows)
        log_f = core.mixture_log_prob_batch(hyp, lpi, kap, oracle.emb)             # (R_v, C)
        out.update(core.forward_query_scores(log_f, core.log_prior_uniform(len(refs)), r_pos, window))
    if i_head is not None:
        cand = oracle.emb[list(window)]
        t_c = np.repeat(oracle.t_emb[v][None, :], len(window), axis=0)
        hyp, lpi, kap = head_outputs(i_head, cand, t_c)
        log_i = core.mixture_log_prob_batch(hyp, lpi, kap, oracle.emb[refs])        # (K, R_v)
        out["ie_learned"] = core.inverse_query_scores(log_i, r_pos)
    return out


def heldout_nll(oracle: Oracle, head, data: np.ndarray, direction: str) -> float:
    """Mean held-out negative log-likelihood of a head on joint samples."""
    _check_direction(direction)
    data = _check_samples(oracle, data)
    total = 0.0
    for r, v, c in data:
        refs = oracle.references(int(v))
        if direction == "forward":
            hyp, lpi, kap = head_outputs(head, oracle.emb[[r]], oracle.t_emb[[v]])
            total -= core.mixture_log_prob_batch(hyp, lpi, kap, oracle.emb)[0, c]
        else:
            hyp, lpi, kap = head_outputs(head, oracle.emb[[c]], oracle.t_emb[[v]])
            total -= core.mixture_log_prob_batch(hyp, lpi, kap, oracle.emb[refs])[0, refs.index(r)]
    return float(total / len(data))


def true_nll(oracle: Oracle, data: np.ndarray, direction: str) -> float:
    _check_direction(direction)
    data = _check_samples(oracle, data)
    total = 0.0
    for r, v, c in data:
        t = oracle.tables(int(v))
        row = oracle.row_of(int(r), int(v))
        total -= t["log_f"][row, c] if direction == "forward" else t["log_inv"][c, row]
    return float(total / len(data))


# ------------------------------------------------------------ experiments --
def eval_queries(oracle: Oracle) -> List[Tuple[int, int]]:
    """Every (reference, instruction) pair the instruction applies to."""
    return [(r, v) for v in range(oracle.spec.n_attribute) for r in oracle.references(v)]


def truth_window(oracle: Oracle, ref_item: int, v: int, window: Sequence[int]) -> Dict[str, np.ndarray]:
    """Oracle PMI, raw forward and marginal log-probabilities for a window."""
    t = oracle.tables(v)
    row = oracle.row_of(ref_item, v)
    w = list(window)
    return {"pmi": t["pmi"][row, w], "raw": t["log_f"][row, w], "log_marg": t["log_marg"][w]}


def arm_scores(scores_by_head: Dict[str, Dict[str, np.ndarray]], arm: Sequence[Tuple[str, str]],
               k: int) -> np.ndarray:
    """An arm is a list of (head name, score key); integrated with gamma = 0
    because the oracle has no edit score (the formula is the v3.1 §8 one)."""
    return core.integrate([scores_by_head[h][key] for h, key in arm], np.zeros(k), gamma=0.0)


def _query_key(ref_item: int, v: int) -> str:
    return "r%d_v%d" % (ref_item, v)


def _kind_aurocs(vals: np.ndarray, kinds: np.ndarray) -> Dict[str, Optional[float]]:
    """AUROC of the target against all distractors and against each kind."""
    out = {"all": core.pairwise_auroc(vals, kinds == "target")}
    for dk in DISTRACTOR_KINDS:
        keep = (kinds == "target") | (kinds == dk)
        out[dk] = core.pairwise_auroc(vals[keep], (kinds == "target")[keep])
    return out


def run_q1_and_representation(oracle: Oracle, train: np.ndarray, test: np.ndarray,
                              forward_cfg: TrainConfig, inverse_cfg: TrainConfig,
                              seeds: Sequence[int]) -> Dict[str, object]:
    """Train forward and inverse heads per seed; record held-out NLL, the
    oracle-versus-learned diagnostic (conditional and marginal log errors by
    distractor kind), every arm's per-query AUROC, and query-paired Q1
    contrasts. Outcomes are recorded, not asserted (invariant 8)."""
    if tuple(seeds) != SEEDS:
        raise ValueError("Phase A needs exactly seeds %s for the cyclic pairs %s"
                         % (SEEDS, core.SEED_PAIRS))
    core.validate_seed_pairs()
    f_heads = {s: train_head(oracle, train, "forward", forward_cfg, s) for s in seeds}
    i_heads = {s: train_head(oracle, train, "inverse", inverse_cfg, s) for s in seeds}
    rep = {"forward_nll": {str(s): heldout_nll(oracle, f_heads[s], test, "forward") for s in seeds},
           "inverse_nll": {str(s): heldout_nll(oracle, i_heads[s], test, "inverse") for s in seeds},
           "true_forward_nll": true_nll(oracle, test, "forward"),
           "true_inverse_nll": true_nll(oracle, test, "inverse"),
           "params_forward": param_count(f_heads[seeds[0]]),
           "params_inverse": param_count(i_heads[seeds[0]]),
           "train_config_forward": asdict(forward_cfg), "train_config_inverse": asdict(inverse_cfg)}

    per_query: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]] = {}
    errors = {dk: {"conditional": [], "marginal": []} for dk in ("target",) + DISTRACTOR_KINDS}
    predictions: List[Dict[str, object]] = []
    rank_identity = True
    for s0, s1 in core.SEED_PAIRS:
        pair = "pair%d%d" % (s0, s1)
        per_query[pair] = {a: {} for a in ARM_NAMES}
        for ref_item, v in eval_queries(oracle):
            win, kinds = oracle.window(ref_item, v)
            kinds_arr = np.array(kinds)
            k = len(win)
            sc = {"F0": learned_query_scores(oracle, f_heads[s0], None, ref_item, v, win),
                  "F1": learned_query_scores(oracle, f_heads[s1], None, ref_item, v, win),
                  "I0": learned_query_scores(oracle, None, i_heads[s0], ref_item, v, win)}
            truth = truth_window(oracle, ref_item, v, win)
            rank_identity &= bool(np.array_equal(np.argsort(-sc["F0"]["pmi_f"], kind="stable"),
                                                 np.argsort(-sc["F0"]["ie_induced"], kind="stable")))
            cond_err = np.abs(sc["F0"]["log_f_raw"] - truth["raw"])
            marg_err = np.abs(sc["F0"]["log_marginal"] - truth["log_marg"])
            for dk in errors:
                sel = kinds_arr == dk
                errors[dk]["conditional"].extend(cond_err[sel].tolist())
                errors[dk]["marginal"].extend(marg_err[sel].tolist())
            arm_vals = {
                "F": arm_scores(sc, [("F0", "pmi_f")], k),
                "I": arm_scores(sc, [("I0", "ie_learned")], k),
                "F+I": arm_scores(sc, [("F0", "pmi_f"), ("I0", "ie_learned")], k),
                "F+F'": arm_scores(sc, [("F0", "pmi_f"), ("F1", "pmi_f")], k),
                "true_pmi": truth["pmi"], "true_raw": truth["raw"],
                "F_raw_uncorrected": sc["F0"]["log_f_raw"],
                "induced_inverse": sc["F0"]["ie_induced"]}
            if tuple(arm_vals) != ARM_NAMES:
                raise RuntimeError("arm set drifted from ARM_NAMES")
            qk = _query_key(ref_item, v)
            for name, vals in arm_vals.items():
                per_query[pair][name][qk] = _kind_aurocs(np.asarray(vals), kinds_arr)
            predictions.append({"pair": [s0, s1], "query": qk, "reference_item": ref_item,
                                "instruction": v, "candidates": list(win), "kinds": kinds,
                                "scores": {n: np.asarray(x).tolist() for n, x in arm_vals.items()}})

    q1: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}
    for pair, arms in per_query.items():
        for name, by_query in arms.items():
            for dk in ("all",) + DISTRACTOR_KINDS:
                mean, n = core.macro_auroc([by_query[q][dk] for q in by_query])
                q1.setdefault(name, {}).setdefault(pair, {})[dk] = {"macro": mean, "n_defined": n}
    contrasts = {}
    for pair, arms in per_query.items():
        for label, (m, c) in (("F+I_vs_F", ("F+I", "F")), ("F+I_vs_F+F'", ("F+I", "F+F'")),
                              ("I_vs_F", ("I", "F"))):
            contrasts.setdefault(pair, {})[label] = core.paired_macro_auroc(
                {q: arms[m][q]["all"] for q in arms[m]}, {q: arms[c][q]["all"] for q in arms[c]})
    diag = {dk: {"mean_abs_conditional_log_error": float(np.mean(e["conditional"])),
                 "mean_abs_marginal_log_error": float(np.mean(e["marginal"]))}
            for dk, e in errors.items()}
    return {"representation": rep, "q1": q1, "paired_contrasts": contrasts,
            "forward_log_error_by_kind": diag,
            "induced_ranks_like_corrected_forward": rank_identity,
            "_predictions": predictions}


def sample_size_sensitivity(oracle: Oracle, sizes: Sequence[int], cfg: TrainConfig,
                            seed: int) -> Dict[str, object]:
    """Development-only diagnostic (source audit, finding 7): the same forward
    head (seed 0) trained on joint-sample pools of different sizes, with the
    optimisation budget fixed (cfg.steps x cfg.batch draws). Records the
    learned raw and PMI discrimination of the reference-like distractor and the
    conditional/marginal log errors there, beside the oracle's own values."""
    out = {}
    for n in sizes:
        rng = np.random.default_rng(seed + n)
        data = sample_triplets(oracle, n, rng)
        head = train_head(oracle, data, "forward", cfg, 0)
        raw_au, pmi_au, cond_e, marg_e = [], [], [], []
        for ref_item, v in eval_queries(oracle):
            win, kinds = oracle.window(ref_item, v)
            kinds_arr = np.array(kinds)
            keep = (kinds_arr == "target") | (kinds_arr == "reference_like")
            sc = learned_query_scores(oracle, head, None, ref_item, v, win)
            truth = truth_window(oracle, ref_item, v, win)
            lab = (kinds_arr == "target")[keep]
            raw_au.append(core.pairwise_auroc(sc["log_f_raw"][keep], lab))
            pmi_au.append(core.pairwise_auroc(sc["pmi_f"][keep], lab))
            refl = kinds_arr == "reference_like"
            cond_e.extend(np.abs(sc["log_f_raw"] - truth["raw"])[refl].tolist())
            marg_e.extend(np.abs(sc["log_marginal"] - truth["log_marg"])[refl].tolist())
        expected_obs = n / oracle.spec.n_attribute * oracle.spec.eps / oracle.n_c
        out[str(n)] = {"learned_raw_auroc_reference_like": core.macro_auroc(raw_au)[0],
                       "learned_pmi_auroc_reference_like": core.macro_auroc(pmi_au)[0],
                       "mean_abs_conditional_log_error_reference_like": float(np.mean(cond_e)),
                       "mean_abs_marginal_log_error_reference_like": float(np.mean(marg_e)),
                       "expected_training_observations_per_eps_candidate_per_instruction":
                           round(expected_obs, 3),
                       "pool_digest": prov.sha256_array(data)}
    return out


def hub_bonus_perturbation(oracle: Oracle, bonuses: Sequence[float]) -> Dict[str, Dict[str, float]]:
    """Score-bonus hubs (a perturbation, not an oracle hub): multiply every
    forward row's hub probabilities by e^b and renormalise; record the macro
    AUROC of the raw and the prior-corrected forward score."""
    out = {}
    bump_kind = np.array([1.0 if it[0] == "hub" else 0.0 for it in oracle.items])
    for b in bonuses:
        raw_auc, cor_auc = [], []
        for ref_item, v in eval_queries(oracle):
            t = oracle.tables(v)
            log_f = core.log_softmax(t["log_f"] + b * bump_kind[None, :], axis=1)
            win, kinds = oracle.window(ref_item, v)
            fq = core.forward_query_scores(log_f, t["log_rho"], oracle.row_of(ref_item, v), win)
            lab = np.array(kinds) == "target"
            raw_auc.append(core.pairwise_auroc(fq["log_f_raw"], lab))
            cor_auc.append(core.pairwise_auroc(fq["pmi_f"], lab))
        out[str(b)] = {"raw": core.macro_auroc(raw_auc)[0], "corrected": core.macro_auroc(cor_auc)[0]}
    return out


def error_injection(oracle: Oracle, sigma: float, seed: int) -> Dict[str, object]:
    """Inject controlled errors into the oracle's centred PMI and record how
    the disagreement Delta = (g + e_I) - (g + e_F) responds (v3 review §3.2
    step 7), including the cases where it cannot identify the faulty side."""
    rng = np.random.default_rng(seed)
    corr = {"inverse_only": [], "forward_only": [], "marginal_only": [], "shared": [], "scale": []}
    shared_max_delta, shared_max_err = 0.0, 0.0
    for ref_item, v in eval_queries(oracle):
        win, _ = oracle.window(ref_item, v)
        g = core.center(truth_window(oracle, ref_item, v, win)["pmi"])
        k = len(win)
        e = rng.normal(0, sigma, k)
        scen = {"inverse_only": (e, np.zeros(k)), "forward_only": (np.zeros(k), e),
                "marginal_only": (np.zeros(k), -rng.normal(0, sigma, k)),   # error in log p_C
                "shared": (e, e), "scale": (np.zeros(k), 0.5 * g)}          # forward 1.5x scale
        for name, (e_i, e_f) in scen.items():
            delta = core.center(g + e_i) - core.center(g + e_f)
            target = (e_i - e_f) - np.mean(e_i - e_f)
            if np.std(target) > 0:
                corr[name].append(float(np.corrcoef(delta, target)[0, 1]))
            if name == "shared":
                shared_max_delta = max(shared_max_delta, float(np.max(np.abs(delta))))
                shared_max_err = max(shared_max_err, float(np.max(np.abs(e_i))))
    g = np.array([0.0, 1.0, -1.0])
    bump = np.array([0.5, 0.0, 0.0])
    over_i = core.center(g + bump) - core.center(g)
    under_f = core.center(g) - core.center(g - bump)
    return {"corr_delta_with_eI_minus_eF": {k: (float(np.mean(v)) if v else None) for k, v in corr.items()},
            "shared_error": {"max_abs_delta": shared_max_delta, "max_abs_error": shared_max_err},
            "sign_ambiguity_identical_delta": bool(np.allclose(over_i, under_f))}


CONFIG_NOTES = {
    "M1_forward_misspecified": "misspecification diagnostic: forward hidden width 2 vs inverse 64; "
                               "not capacity-matched, not evidence of an intrinsic inverse advantage",
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True, help="fresh (new or empty) output directory")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--sensitivity-sizes", type=int, nargs="+", default=[5000, 20000, 80000])
    args = ap.parse_args(argv)
    if args.n_train < 1 or args.n_test < 1 or args.torch_threads < 1 \
            or any(n < 1 for n in args.sensitivity_sizes):
        ap.error("sizes and thread counts must be positive")
    try:
        prov.prepare_fresh_dir(args.out_dir)          # fail before any training time is spent
    except IOError as exc:
        ap.error(str(exc))
    torch = _torch()
    torch.set_num_threads(args.torch_threads)
    t0 = time.time()
    spec = OracleSpec()
    oracle = Oracle(spec)
    out: Dict[str, object] = {"bayes_check": bayes_check(oracle)}
    if out["bayes_check"]["max_bayes_error"] > 1e-12 or out["bayes_check"]["max_normalisation_error"] > 1e-12:
        print("FATAL: oracle tables fail the Bayes/normalisation check", file=sys.stderr)
        return 1
    rng = np.random.default_rng(args.seed)
    train = sample_triplets(oracle, args.n_train, rng)
    test = sample_triplets(oracle, args.n_test, rng)
    in_dim = oracle.dim + spec.n_attribute
    wide = width_matched_hidden(in_dim, oracle.dim, 64, 8)
    m8 = param_count(build_head(in_dim, oracle.dim, 64, 8, 0))
    m1w = param_count(build_head(in_dim, oracle.dim, wide, 1, 0))
    if abs(m1w - m8) > 0.05 * m8:
        print("FATAL: width-matched M=1 head (%d params) is not within 5%% of M=8 (%d)" % (m1w, m8),
              file=sys.stderr)
        return 1
    configs = {
        "M1": (TrainConfig(hidden=64, n_comp=1), TrainConfig(hidden=64, n_comp=1)),
        "M8": (TrainConfig(hidden=64, n_comp=8), TrainConfig(hidden=64, n_comp=8)),
        "M1_width_matched": (TrainConfig(hidden=wide, n_comp=1), TrainConfig(hidden=wide, n_comp=1)),
        "M1_forward_misspecified": (TrainConfig(hidden=2, n_comp=1), TrainConfig(hidden=64, n_comp=1)),
    }
    out["runs"], predictions = {}, []
    for name, (fc, ic) in configs.items():
        print("[phaseA] %s ..." % name, flush=True)
        res = run_q1_and_representation(oracle, train, test, fc, ic, SEEDS)
        for row in res.pop("_predictions"):
            predictions.append(dict(row, config=name))
        if name in CONFIG_NOTES:
            res["note"] = CONFIG_NOTES[name]
        out["runs"][name] = res
    print("[phaseA] sample-size sensitivity ...", flush=True)
    out["sample_size_sensitivity"] = sample_size_sensitivity(
        oracle, args.sensitivity_sizes, TrainConfig(hidden=64, n_comp=1), args.seed)
    out["hub_bonus_perturbation"] = hub_bonus_perturbation(oracle, (0.0, 1.0, 2.0, 4.0))
    out["error_injection"] = error_injection(oracle, sigma=0.5, seed=args.seed)
    out["manifest"] = {
        "schema": prov.MANIFEST_SCHEMA, "script": "bbnv_oracle.py", "args": vars(args),
        "spec": asdict(spec), "seeds": list(SEEDS), "seed_pairs": [list(p) for p in core.SEED_PAIRS],
        "train_configs": {n: {"forward": asdict(f), "inverse": asdict(i)} for n, (f, i) in configs.items()},
        "width_matched_hidden": wide,
        "params": {"M8_hidden64": m8, "M1_width_matched": m1w,
                   "M1_hidden64": param_count(build_head(in_dim, oracle.dim, 64, 1, 0))},
        "sample_digests": {"train": prov.sha256_array(train), "test": prov.sha256_array(test)},
        "source_sha256": prov.source_hashes(),
        "runtime": prov.runtime_info(_NUMPY_PRELOADED, args.torch_threads),
        "wall_seconds": round(time.time() - t0, 1)}
    prov.write_run(args.out_dir, out, {"predictions.jsonl": predictions})
    prov.verify_run(args.out_dir)
    print("[phaseA] bayes %s | wrote and verified %s in %.0fs"
          % (out["bayes_check"], args.out_dir, out["manifest"]["wall_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
