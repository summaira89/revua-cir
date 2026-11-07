"""BBNV Phase D real-data scoring (v3.1 §1, §9, §12). CPU, float64.

Shared pieces for every real-data scoring step:
  * load_heads     the verified Phase C heads (bbnv_heads), bound to the
                   features they were trained on;
  * make_banks     the two disjoint 1,024-image banks B_R, B_C from the
                   negbank partition (BANK_SEED, §12);
  * forward_window per query: R_q = B_R ∪ {r} (uniform prior), C_q = B_C ∪ W(q),
                   q_F(. | x, t) for every x in R_q over C_q, then the core's
                   forward_query_scores (raw, marginal, PMI, induced inverse).

Command `lambda` (§12): on the 2,258 dev-calibrate triplets, each with its
frozen-feature hard-negative window (target + the 9 dev-calibrate images with
the highest (r_hat + t_hat) . x_hat, excluding the reference, the target and
every known positive of either), the macro within-window AUROC of
s_lambda = raw - lambda * log p_C for every forward head and lambda in
LAMBDAS. lambda* = highest mean over the three seeds (ties -> smaller lambda).

Usage
-----
  python scripts/bbnv_score.py lambda --features-dir ../bbnv_feats/v1/cirr_train \\
      --manifest-dir ../bbnv_runs/manifest_groups_cluster --heads-dir ../bbnv_heads/v1 --out-dir ../bbnv_runs/lambda_v1

Command `windows` (§13): every validation window of a deployed score cache
scored by every head (F0-F2: raw forward, marginal S, PMI, induced inverse;
I0-I2: learned inverse), with R_q = B_R ∪ {r} and C_q = B_C ∪ W(q); the
cache's base_score, edit_cos and verification_score are carried unchanged.
The cache must be the one its verifier-vector dump recorded, and every
window's target must agree with the dataset. No arm, metric or gate is
computed here.

  python scripts/bbnv_score.py windows --train-features ../bbnv_feats/v1/cirr_train       --manifest-dir ../bbnv_runs/manifest_groups_cluster --heads-dir ../bbnv_heads/v1       --val-features ../bbnv_feats/v1/cirr_val --cache ../bbnv_caches/rv_score_cache_fc_aep_rv.jsonl       --vectors-manifest ../bbnv_caches/vec_cirr_val.manifest.json --out-dir ../bbnv_runs/windows_cirr_v1

Output (fresh directory): per_query.jsonl (lambda) or windows.jsonl
(windows), then results.json last.
"""
from __future__ import annotations

import os
import sys

# see bbnv_oracle.py: keep numpy's MKL off OpenMP so torch's runtime is the only one
_NUMPY_PRELOADED = "numpy" in sys.modules
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bbnv_core as core  # noqa: E402
import bbnv_extract as ext  # noqa: E402
import bbnv_manifest as man  # noqa: E402
import bbnv_provenance as prov  # noqa: E402
import bbnv_train as trn  # noqa: E402
import fusion_controls as fc  # noqa: E402

BANK_SEED = 20261022
BANK_SIZE = 1024
WINDOW = 10
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


class ScoreError(ValueError):
    pass


# ------------------------------------------------------------------ heads --
def load_heads(heads_dir: str, features_dir: str) -> Dict[str, Dict[str, object]]:
    """{'forward_s0': {'head': module (eval), 'kappa': float}, ...} from a
    verified, non-smoke Phase C run trained on exactly these features."""
    import torch
    res = prov.verify_run(heads_dir)
    if res.get("schema") != trn.SCHEMA or res.get("smoke") is not False:
        raise ScoreError("%s is not a full Phase C run (schema %s, smoke must be false)" % (heads_dir, trn.SCHEMA))
    if res["inputs"]["features_results_sha256"] != prov.sha256_file(os.path.join(features_dir, prov.RESULTS_NAME)):
        raise ScoreError("the heads were trained on other features than %s" % features_dir)
    with np.load(os.path.join(heads_dir, trn.HEADS_NAME), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    heads = {}
    for key, sel in res["selections"].items():
        head = trn.build_head(0)
        state = {k.split("/", 1)[1]: torch.from_numpy(v) for k, v in arrays.items()
                 if k.startswith(key + "/") and not k.endswith("/kappa")}
        head.load_state_dict(state, strict=True)
        head.eval()
        kappa = float(arrays[key + "/kappa"])
        if kappa != sel["selected_kappa"] or not core.KAPPA_RANGE[0] <= kappa <= core.KAPPA_RANGE[1]:
            raise ScoreError("%s: stored kappa %g disagrees with its selection" % (key, kappa))
        heads[key] = {"head": head, "kappa": kappa}
    expected = {"%s_s%d" % (d, s) for d in trn.DIRECTIONS for s in trn.SEEDS}
    if set(heads) != expected:
        raise ScoreError("heads present %s, expected %s" % (sorted(heads), sorted(expected)))
    return heads


def hypotheses(head, cond: np.ndarray, text: np.ndarray) -> np.ndarray:
    """(n, 1, d) float64 unit hypotheses for conditioning rows `cond` under one text."""
    import torch
    x = np.concatenate([cond, np.broadcast_to(text, cond.shape)], axis=1).astype(np.float32)
    with torch.no_grad():
        h = head(torch.from_numpy(x)).numpy().astype(np.float64)
    return (h / np.linalg.norm(h, axis=1, keepdims=True))[:, None, :]


# ------------------------------------------------------------------ banks --
def make_banks(image_ids: np.ndarray, partition: np.ndarray, seed: int = BANK_SEED,
               size: int = BANK_SIZE) -> Dict[str, np.ndarray]:
    """Rows of two disjoint banks from the negbank partition (§12): the pool
    sorted by image id, permuted with `seed`; B_R first, B_C next."""
    pool = np.flatnonzero(partition == "negbank")
    pool = pool[np.argsort(image_ids[pool], kind="stable")]
    if len(pool) < 2 * size:
        raise ScoreError("negbank has %d images; two disjoint banks of %d need %d" % (len(pool), size, 2 * size))
    perm = np.random.default_rng(seed).permutation(len(pool))
    return {"B_R": pool[perm[:size]], "B_C": pool[perm[size:2 * size]]}


# ---------------------------------------------------------------- windows --
def hard_negative_window(query_vec: np.ndarray, cand_rows: np.ndarray, cand_emb: np.ndarray,
                         target_row: int, excluded: Set[int], k: int = WINDOW) -> np.ndarray:
    """Target first, then the k - 1 candidates with the highest query_vec . x,
    skipping every excluded row (the reference, the target, known positives).
    Ties are broken by row (stable)."""
    scores = cand_emb @ query_vec
    order = np.argsort(-scores, kind="stable")
    picked = [int(cand_rows[i]) for i in order if int(cand_rows[i]) not in excluded][:k - 1]
    if len(picked) < k - 1:
        raise ScoreError("only %d eligible window negatives" % len(picked))
    return np.asarray([target_row] + picked, dtype=np.int64)


def reference_universe(bank_r: np.ndarray, ref_row: int) -> List[int]:
    """R_q = B_R ∪ {r} as rows, bank order first; r is not added twice (v3.1 §1)."""
    r_ids = [int(i) for i in bank_r] + ([ref_row] if ref_row not in set(bank_r.tolist()) else [])
    return r_ids


def _unit64(rows: np.ndarray) -> np.ndarray:
    out = rows.astype(np.float64)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def forward_window(head, kappa: float, image_emb: np.ndarray, text: np.ndarray, ref_row: int,
                   bank_r: np.ndarray, bank_c: np.ndarray, window_rows: np.ndarray) -> Dict[str, np.ndarray]:
    """Forward quantities for one query's window (core.forward_query_scores),
    with R_q = B_R ∪ {r}, uniform prior, C_q = B_C ∪ W(q), every q_F row
    normalised over the same C_q (v3.1 §1)."""
    r_ids = reference_universe(bank_r, ref_row)
    c_ids = [int(i) for i in bank_c]
    c_pos = {i: n for n, i in enumerate(c_ids)}
    win_cols = []
    for w in window_rows.tolist():
        if w not in c_pos:
            c_pos[w] = len(c_ids)
            c_ids.append(w)
        win_cols.append(c_pos[w])
    hyp = hypotheses(head, image_emb[r_ids], text[None, :])
    log_f = core.mixture_log_prob_batch(hyp, np.zeros((len(r_ids), 1)), np.full(len(r_ids), kappa),
                                        _unit64(image_emb[c_ids]))
    return core.forward_query_scores(log_f, core.log_prior_uniform(len(r_ids)), r_ids.index(ref_row), win_cols)


def inverse_window(head, kappa: float, image_emb: np.ndarray, text: np.ndarray, ref_row: int,
                   bank_r: np.ndarray, window_rows: np.ndarray) -> np.ndarray:
    """Learned inverse evidence log q_I(r | c, t) for each window candidate c,
    each a normalised pmf over the SAME R_q as the forward side."""
    r_ids = reference_universe(bank_r, ref_row)
    hyp = hypotheses(head, image_emb[window_rows], text[None, :])
    log_i = core.mixture_log_prob_batch(hyp, np.zeros((len(window_rows), 1)), np.full(len(window_rows), kappa),
                                        _unit64(image_emb[r_ids]))
    return core.inverse_query_scores(log_i, r_ids.index(ref_row))


def select_lambda(macro: Mapping[float, Sequence[float]]) -> float:
    """lambda with the highest mean macro AUROC over seeds; ties -> smaller lambda."""
    means = {lam: float(np.mean(v)) for lam, v in macro.items()}
    best = max(means.values())
    return min(lam for lam, m in means.items() if m == best)


# ------------------------------------------------------------- commands --
def cmd_lambda(args) -> int:
    import torch
    torch.set_num_threads(args.threads)
    t0 = time.time()
    inp = trn.load_inputs(args.features_dir, args.manifest_dir)
    feat, positives = inp["feat"], inp["positives"]
    heads = load_heads(args.heads_dir, args.features_dir)
    ids, part = feat["image_ids"], feat["partition"]
    row_of = {img: i for i, img in enumerate(ids.tolist())}
    banks = make_banks(ids, part)
    img = feat["image_emb"].astype(np.float64)
    txt = feat["text_emb"].astype(np.float64)
    q_idx = np.flatnonzero(part[feat["ref_row"]] == "dev_calibrate")
    cand_rows = np.flatnonzero(part == "dev_calibrate")
    cand_emb = img[cand_rows]
    print("[lambda] %d dev-calibrate queries | %d candidate images | banks %d + %d"
          % (len(q_idx), len(cand_rows), len(banks["B_R"]), len(banks["B_C"])), flush=True)
    keys = ["forward_s%d" % s for s in trn.SEEDS]
    per_query: List[Dict[str, object]] = []
    auroc = {k: {lam: [] for lam in LAMBDAS} for k in keys}
    for n, q in enumerate(q_idx.tolist()):
        r, g = int(feat["ref_row"][q]), int(feat["tgt_row"][q])
        excluded = {r, g} | {row_of[p] for p in positives.get(ids[r], set()) | positives.get(ids[g], set())}
        win = hard_negative_window(img[r] + txt[q], cand_rows, cand_emb, g, excluded)
        labels = np.zeros(WINDOW, dtype=bool)
        labels[0] = True
        row = {"triplet": int(q), "pairid": int(feat["pairid"][q]), "window": ids[win].tolist(), "auroc": {}}
        for k in keys:
            fq = forward_window(heads[k]["head"], heads[k]["kappa"], img, txt[q], r, banks["B_R"], banks["B_C"], win)
            row["auroc"][k] = {}
            for lam in LAMBDAS:
                a = core.pairwise_auroc(core.forward_lambda_score(fq["log_f_raw"], fq["log_marginal"], lam), labels)
                auroc[k][lam].append(a)
                row["auroc"][k][str(lam)] = a
        per_query.append(row)
        if (n + 1) % 250 == 0:
            print("[lambda] %d/%d queries, %.0fs" % (n + 1, len(q_idx), time.time() - t0), flush=True)
    macro = {k: {lam: core.macro_auroc(auroc[k][lam]) for lam in LAMBDAS} for k in keys}
    by_lambda = {lam: [macro[k][lam][0] for k in keys] for lam in LAMBDAS}
    lam_star = select_lambda(by_lambda)
    results = {
        "schema": "bbnv-lambda/1", "amendment": "v3.1 §12", "lambda_star": lam_star,
        "lambdas": list(LAMBDAS), "mean_macro_auroc": {str(lam): float(np.mean(v)) for lam, v in by_lambda.items()},
        "macro_auroc": {k: {str(lam): {"auroc": m[0], "n_defined": m[1]} for lam, m in v.items()}
                        for k, v in macro.items()},
        "n_queries": len(q_idx), "window": WINDOW,
        "banks": {"seed": BANK_SEED, "size": BANK_SIZE,
                  "B_R_digest": core.universe_digest(ids[banks["B_R"]].tolist()),
                  "B_C_digest": core.universe_digest(ids[banks["B_C"]].tolist()),
                  "B_R": ids[banks["B_R"]].tolist(), "B_C": ids[banks["B_C"]].tolist()},
        "kappas": {k: heads[k]["kappa"] for k in keys},
        "inputs": {"heads_results_sha256": prov.sha256_file(os.path.join(args.heads_dir, prov.RESULTS_NAME)),
                   "features_results_sha256": prov.sha256_file(os.path.join(args.features_dir, prov.RESULTS_NAME))},
        "manifest": {"schema": prov.MANIFEST_SCHEMA, "script": "bbnv_score.py lambda", "args": vars(args),
                     "source_sha256": prov.source_hashes(),
                     "runtime": prov.runtime_info(_NUMPY_PRELOADED, args.threads)},
        "wall_seconds": round(time.time() - t0, 1)}
    prov.write_run(args.out_dir, results, {"per_query.jsonl": per_query})
    prov.verify_run(args.out_dir)
    print("[lambda] lambda* = %g | mean macro AUROC by lambda %s | wrote and verified %s"
          % (lam_star, results["mean_macro_auroc"], args.out_dir), flush=True)
    return 0


def load_validation(val_features: str, cache: str, vectors_manifest: str,
                    checkpoint_sha256: str) -> Dict[str, object]:
    """Phase B validation features and the deployed score cache, bound
    together: the cache bytes equal the cache_sha256 its verifier-vector dump
    recorded, the dump and the features share the checkpoint, the dump's
    vectors are the ones the features reproduced, and every window's target
    agrees with the dataset."""
    fres = prov.verify_run(val_features)
    if fres.get("schema") != ext.SCHEMA or fres.get("target") not in ("cirr_val", "fiq_val", "edir_test"):
        raise ScoreError("%s is not a validation feature run of schema %s" % (val_features, ext.SCHEMA))
    with open(vectors_manifest, encoding="utf-8") as fh:
        vman = json.load(fh)
    if fres["manifest"]["checkpoint_sha256"] != checkpoint_sha256 or vman.get("checkpoint_sha256") != checkpoint_sha256:
        raise ScoreError("validation features, vector dump and heads are not of one checkpoint")
    if vman.get("npz_sha256") != fres.get("vectors_sha256"):
        raise ScoreError("the features were checked against a different vector dump")
    if vman.get("cache_sha256") != prov.sha256_file(cache):
        raise ScoreError("%s is not the cache this vector dump was made from" % cache)
    recs, multipositive = fc.load_records(cache)
    with np.load(os.path.join(val_features, ext.FEATURES_NAME), allow_pickle=False) as z:
        feat = {k: z[k] for k in z.files}
    if multipositive or len(recs) != len(feat["text_emb"]):
        raise ScoreError("cache and features disagree in query count or target type")
    for q, rec in enumerate(recs):
        cand = rec["candidate_gallery_ids"]
        if rec["query_id"] != q or len(cand) != WINDOW or rec["target_gallery_index"] != int(feat["tgt_row"][q]):
            raise ScoreError("cache record %d disagrees with the dataset (order, window size or target)" % q)
        ec = rec.get("edit_cos")
        if not isinstance(ec, list) or len(ec) != WINDOW or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) for v in ec):
            raise ScoreError("cache record %d: edit_cos missing, mis-sized or non-finite" % q)
        tpos = cand.index(rec["target_gallery_index"]) if rec["target_gallery_index"] in cand else -1
        if rec["target_pos"] != tpos or max(cand) >= len(feat["image_emb"]):
            raise ScoreError("cache record %d: inconsistent target position or candidate id" % q)
    return {"feat": feat, "recs": recs, "features_results": fres, "vectors_manifest": vman}


def edir_scoring_table(tf: Mapping[str, np.ndarray], banks: Mapping[str, np.ndarray],
                       vf: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, int]]:
    """EDIR's head-scoring table (v3.1 §19.1, conditions 1-2): B_R rows, B_C
    rows, the corpus, then the separate reference block, with an explicit,
    unique id for every row. The banks occupy rows [0, n) and [n, 2n), as the
    kernels' bank_r/bank_c ranges assume; references only supply r for
    R_q = B_R ∪ {r}, and never enter C_q or any bank."""
    nb = BANK_SIZE
    ids = (["cirr_train:%s" % i for i in tf["image_ids"][banks["B_R"]]]
           + ["cirr_train:%s" % i for i in tf["image_ids"][banks["B_C"]]]
           + [str(i) for i in vf["image_ids"]] + [str(i) for i in vf["ref_ids"]])
    emb = np.concatenate([tf["image_emb"][banks["B_R"]], tf["image_emb"][banks["B_C"]],
                          vf["image_emb"], vf["ref_emb"]])
    if len(ids) != len(emb) or len(set(ids)) != len(ids) or len(ids) != 2 * nb + len(vf["image_emb"]) + len(vf["ref_emb"]):
        raise ScoreError("EDIR scoring table: ids and rows do not correspond one to one")
    if not (all(i.startswith("corpus:") for i in vf["image_ids"]) and all(i.startswith("ref:") for i in vf["ref_ids"])):
        raise ScoreError("EDIR ids must be namespaced corpus:/ref:")
    return emb, {i: r for r, i in enumerate(ids)}


def edir_rows(row_of: Mapping[str, int], vf: Mapping[str, np.ndarray], q: int,
              candidates: Sequence[int]) -> Tuple[np.ndarray, int]:
    """(window rows, reference row) of EDIR query q, looked up by identity and checked."""
    nb = BANK_SIZE
    ref_id = str(vf["ref_ids"][int(vf["ref_index"][q])])
    ref = row_of[ref_id]
    win = np.asarray([row_of[str(vf["image_ids"][c])] for c in candidates], dtype=np.int64)
    if not ref_id.startswith("ref:") or ref < 2 * nb or win.min() < 2 * nb or ref in set(win.tolist()):
        raise ScoreError("EDIR query %d: reference or window rows fall in a bank or overlap" % q)
    return win, ref


def score_query(heads: Mapping[str, Mapping[str, object]], emb: np.ndarray, text: np.ndarray, ref_row: int,
                bank_r: np.ndarray, bank_c: np.ndarray, window_rows: np.ndarray) -> Dict[str, Dict[str, list]]:
    """Every head's window scores for one query (§13)."""
    out: Dict[str, Dict[str, list]] = {}
    for s in trn.SEEDS:
        f, i = heads["forward_s%d" % s], heads["inverse_s%d" % s]
        fq = forward_window(f["head"], f["kappa"], emb, text, ref_row, bank_r, bank_c, window_rows)
        out["F%d" % s] = {"raw": fq["log_f_raw"].tolist(), "marginal": fq["log_marginal"].tolist(),
                          "pmi": fq["pmi_f"].tolist(), "induced": fq["ie_induced"].tolist()}
        out["I%d" % s] = {"ie": inverse_window(i["head"], i["kappa"], emb, text, ref_row, bank_r,
                                                window_rows).tolist()}
    return out


def cmd_windows(args) -> int:
    import torch
    torch.set_num_threads(args.threads)
    t0 = time.time()
    tr = trn.load_inputs(args.train_features, args.manifest_dir)
    heads = load_heads(args.heads_dir, args.train_features)
    ckpt = tr["features_results"]["manifest"]["checkpoint_sha256"]
    val = load_validation(args.val_features, args.cache, args.vectors_manifest, ckpt)
    tf, vf, recs = tr["feat"], val["feat"], val["recs"]
    banks = make_banks(tf["image_ids"], tf["partition"])
    edir = val["features_results"]["target"] == "edir_test"
    nb = BANK_SIZE
    if edir:
        emb, row_of = edir_scoring_table(tf, banks, vf)
    else:
        # One embedding table: B_R rows, then B_C rows, then the validation gallery.
        emb = np.concatenate([tf["image_emb"][banks["B_R"]], tf["image_emb"][banks["B_C"]], vf["image_emb"]])
    bank_r, bank_c = np.arange(nb), np.arange(nb, 2 * nb)
    n_q = len(recs) if args.max_queries is None else min(args.max_queries, len(recs))
    split = {}
    if val["features_results"]["target"] == "cirr_val":
        tune, report = fc.split_indices(len(recs), fc.SPLIT_SEED)
        split = {**{q: "tune" for q in tune}, **{q: "report" for q in report}}
    print("[windows] %s | %d queries | gallery %d | banks %d + %d"
          % (val["features_results"]["target"], n_q, len(vf["image_emb"]), nb, nb), flush=True)
    rows = []
    for q in range(n_q):
        rec = recs[q]
        if edir:
            win, ref = edir_rows(row_of, vf, q, rec["candidate_gallery_ids"])
        else:
            win = 2 * nb + np.asarray(rec["candidate_gallery_ids"], dtype=np.int64)
            ref = 2 * nb + int(vf["ref_row"][q])
        rows.append({"query": q, "split": split.get(q), "target_pos": rec["target_pos"],
                     "window": rec["candidate_gallery_ids"], "base_score": rec["base_score"],
                     "edit_cos": rec["edit_cos"], "verification_score": rec["verification_score"],
                     "heads": score_query(heads, emb, vf["text_emb"][q].astype(np.float64),
                                          ref, bank_r, bank_c, win)})
        if (q + 1) % 250 == 0:
            print("[windows] %d/%d queries, %.0fs" % (q + 1, n_q, time.time() - t0), flush=True)
    results = {
        "schema": "bbnv-windows/1", "amendment": "v3.1 §13", "smoke": args.max_queries is not None,
        **({"protocol": "edir_retrieval"} if edir else {}),
        "target": val["features_results"]["target"], "category": val["features_results"]["category"],
        "n_queries": n_q, "window": WINDOW, "lambda_star_frozen": 0.0,
        "banks": {"seed": BANK_SEED, "size": BANK_SIZE,
                  "B_R_digest": core.universe_digest(tf["image_ids"][banks["B_R"]].tolist()),
                  "B_C_digest": core.universe_digest(tf["image_ids"][banks["B_C"]].tolist())},
        "kappas": {k: h["kappa"] for k, h in heads.items()},
        "split_seed": fc.SPLIT_SEED if split else None,
        "inputs": {"cache_sha256": prov.sha256_file(args.cache),
                   "vectors_manifest_sha256": prov.sha256_file(args.vectors_manifest),
                   "val_features_results_sha256": prov.sha256_file(os.path.join(args.val_features, prov.RESULTS_NAME)),
                   "heads_results_sha256": prov.sha256_file(os.path.join(args.heads_dir, prov.RESULTS_NAME)),
                   "checkpoint_sha256": ckpt},
        "manifest": {"schema": prov.MANIFEST_SCHEMA, "script": "bbnv_score.py windows", "args": vars(args),
                     "source_sha256": prov.source_hashes(),
                     "runtime": prov.runtime_info(_NUMPY_PRELOADED, args.threads)},
        "wall_seconds": round(time.time() - t0, 1)}
    prov.write_run(args.out_dir, results, {"windows.jsonl": rows})
    prov.verify_run(args.out_dir)
    print("[windows] wrote and verified %s in %.0fs" % (args.out_dir, time.time() - t0), flush=True)
    return 0


def load_annotated(val_features: str, annotated_cache: str, vectors_manifest: str,
                   checkpoint_sha256: str) -> Dict[str, object]:
    """EDIR features and the `edir_annotated` scores, bound by hash through the
    chain: annotated manifest -> retrieval cache -> vector dump -> features, one
    checkpoint throughout. Every record is checked against the features."""
    fres = prov.verify_run(val_features)
    if fres.get("schema") != ext.SCHEMA or fres.get("target") != "edir_test":
        raise ScoreError("%s is not an edir_test feature run" % val_features)
    with open(vectors_manifest, encoding="utf-8") as fh:
        vman = json.load(fh)
    with open(annotated_cache[:-6] + ".manifest.json", encoding="utf-8") as fh:
        aman = json.load(fh)
    if aman.get("protocol") != "edir_annotated" or aman.get("cache_sha256") != prov.sha256_file(annotated_cache):
        raise ScoreError("%s is not the verified edir_annotated output" % annotated_cache)
    if not aman["cross_check"]["shared_pairs"] or aman["cross_check"]["violations"]:
        raise ScoreError("the annotated scores did not pass their cross-check")
    if {fres["manifest"]["checkpoint_sha256"], vman.get("checkpoint_sha256"), aman.get("checkpoint_sha256")} \
            != {checkpoint_sha256}:
        raise ScoreError("features, vector dump, annotated scores and heads are not of one checkpoint")
    if vman.get("npz_sha256") != fres.get("vectors_sha256") or vman.get("cache_sha256") != aman.get("retrieval_cache_sha256"):
        raise ScoreError("the annotated scores, retrieval cache, vector dump and features are not one chain")
    if aman.get("gallery_sha256") != fres.get("gallery_sha256"):
        raise ScoreError("the annotated scores were made on another gallery")
    with np.load(os.path.join(val_features, ext.FEATURES_NAME), allow_pickle=False) as z:
        feat = {k: z[k] for k in z.files}
    recs = man.read_jsonl(annotated_cache)
    if len(recs) != len(feat["text_emb"]):
        raise ScoreError("annotated scores and features disagree in query count")
    for q, rec in enumerate(recs):
        cand = rec["candidate_gallery_ids"]
        ok = (rec.get("protocol") == "edir_annotated" and rec["query_id"] == q and len(cand) in (5, 6)
              and cand == sorted(set(cand)) and rec["target_gallery_index"] == int(feat["tgt_row"][q])
              and 0 <= rec["target_pos"] < len(cand) and cand[rec["target_pos"]] == rec["target_gallery_index"]
              and max(cand) < len(feat["image_emb"])
              and all(len(rec[f]) == len(cand) and all(np.isfinite(rec[f]))
                      for f in ("base_score", "edit_cos", "verification_score")))
        if not ok:
            raise ScoreError("annotated record %d is malformed or disagrees with the features" % q)
    return {"feat": feat, "recs": recs, "features_results": fres, "annotated_manifest": aman}


def cmd_annotated(args) -> int:
    """Track A Q1 windows (v3.1 §15.2): every EDIR query's annotated candidates
    scored by every head, with R_q = B_R ∪ {r} and C_q = B_C ∪ W(q), W(q) the
    annotated set; same row format as `windows`, protocol `edir_annotated`."""
    import torch
    torch.set_num_threads(args.threads)
    t0 = time.time()
    tr = trn.load_inputs(args.train_features, args.manifest_dir)
    heads = load_heads(args.heads_dir, args.train_features)
    ckpt = tr["features_results"]["manifest"]["checkpoint_sha256"]
    ann = load_annotated(args.val_features, args.annotated_cache, args.vectors_manifest, ckpt)
    tf, vf, recs = tr["feat"], ann["feat"], ann["recs"]
    banks = make_banks(tf["image_ids"], tf["partition"])
    emb, row_of = edir_scoring_table(tf, banks, vf)
    bank_r, bank_c = np.arange(BANK_SIZE), np.arange(BANK_SIZE, 2 * BANK_SIZE)
    n_q = len(recs) if args.max_queries is None else min(args.max_queries, len(recs))
    print("[annotated] %d queries | banks %d + %d" % (n_q, BANK_SIZE, BANK_SIZE), flush=True)
    rows = []
    for q in range(n_q):
        rec = recs[q]
        win, ref = edir_rows(row_of, vf, q, rec["candidate_gallery_ids"])
        rows.append({"query": q, "split": None, "target_pos": rec["target_pos"],
                     "window": rec["candidate_gallery_ids"], "base_score": rec["base_score"],
                     "edit_cos": rec["edit_cos"], "verification_score": rec["verification_score"],
                     "heads": score_query(heads, emb, vf["text_emb"][q].astype(np.float64),
                                          ref, bank_r, bank_c, win)})
        if (q + 1) % 250 == 0:
            print("[annotated] %d/%d queries, %.0fs" % (q + 1, n_q, time.time() - t0), flush=True)
    sizes = [len(r["window"]) for r in rows]
    results = {
        "schema": "bbnv-windows/1", "amendment": "v3.1 §15.2, §19", "smoke": args.max_queries is not None,
        "protocol": "edir_annotated", "target": "edir_test", "category": None,
        "n_queries": n_q, "window": None, "window_sizes": {str(k): sizes.count(k) for k in sorted(set(sizes))},
        "lambda_star_frozen": 0.0,
        "banks": {"seed": BANK_SEED, "size": BANK_SIZE,
                  "B_R_digest": core.universe_digest(tf["image_ids"][banks["B_R"]].tolist()),
                  "B_C_digest": core.universe_digest(tf["image_ids"][banks["B_C"]].tolist())},
        "kappas": {k: h["kappa"] for k, h in heads.items()}, "split_seed": None,
        "inputs": {"annotated_sha256": prov.sha256_file(args.annotated_cache),
                   "annotated_manifest_sha256": prov.sha256_file(args.annotated_cache[:-6] + ".manifest.json"),
                   "vectors_manifest_sha256": prov.sha256_file(args.vectors_manifest),
                   "val_features_results_sha256": prov.sha256_file(os.path.join(args.val_features, prov.RESULTS_NAME)),
                   "heads_results_sha256": prov.sha256_file(os.path.join(args.heads_dir, prov.RESULTS_NAME)),
                   "checkpoint_sha256": ckpt},
        "manifest": {"schema": prov.MANIFEST_SCHEMA, "script": "bbnv_score.py annotated", "args": vars(args),
                     "source_sha256": prov.source_hashes(),
                     "runtime": prov.runtime_info(_NUMPY_PRELOADED, args.threads)},
        "wall_seconds": round(time.time() - t0, 1)}
    prov.write_run(args.out_dir, results, {"windows.jsonl": rows})
    prov.verify_run(args.out_dir)
    print("[annotated] wrote and verified %s in %.0fs" % (args.out_dir, time.time() - t0), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    lam = sub.add_parser("lambda", help="choose lambda on dev-calibrate (§12)")
    lam.add_argument("--features-dir", required=True)
    lam.add_argument("--manifest-dir", required=True)
    lam.add_argument("--heads-dir", required=True)
    lam.add_argument("--out-dir", required=True)
    lam.add_argument("--threads", type=int, default=4)
    win = sub.add_parser("windows", help="score every validation window with every head (§13)")
    win.add_argument("--train-features", required=True)
    win.add_argument("--manifest-dir", required=True)
    win.add_argument("--heads-dir", required=True)
    win.add_argument("--val-features", required=True)
    win.add_argument("--cache", required=True)
    win.add_argument("--vectors-manifest", required=True)
    win.add_argument("--out-dir", required=True)
    win.add_argument("--threads", type=int, default=4)
    win.add_argument("--max-queries", type=int, default=None, help="smoke check on the first N queries")
    ann = sub.add_parser("annotated", help="score EDIR's annotated sets with every head (§15.2, §19)")
    ann.add_argument("--train-features", required=True)
    ann.add_argument("--manifest-dir", required=True)
    ann.add_argument("--heads-dir", required=True)
    ann.add_argument("--val-features", required=True, help="the edir_test feature run")
    ann.add_argument("--annotated-cache", required=True, help="edir_annotated_cache.py output (.jsonl)")
    ann.add_argument("--vectors-manifest", required=True, help="the EDIR vector dump's manifest")
    ann.add_argument("--out-dir", required=True)
    ann.add_argument("--threads", type=int, default=4)
    ann.add_argument("--max-queries", type=int, default=None, help="smoke check on the first N queries")
    args = ap.parse_args(argv)
    if args.threads < 1:
        ap.error("--threads must be positive")
    if getattr(args, "max_queries", None) is not None and args.max_queries < 1:
        ap.error("--max-queries must be positive")
    try:
        prov.prepare_fresh_dir(args.out_dir)
    except IOError as exc:
        ap.error(str(exc))
    return {"lambda": cmd_lambda, "windows": cmd_windows, "annotated": cmd_annotated}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
