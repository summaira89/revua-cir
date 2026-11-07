"""Text-cosine baseline follow-ups, amendment v3.1 §21 (retrospective). CPU only.

The §20.2 result showed the text cosine T = cos(text, candidate), computed from the
trained backbone with no additional training, outperforming the learned verifier as
a second signal. §21 fixed these analyses before this file existed:

  §21.1 A  does V (or the reconstruction term) add beyond base + T?   three-term fusion
  §21.2 B  T under every fusion family, recall- and utility-optimal
  §21.3    T fusion tuned for utility, cross-fitted as the gate study; vs published gates
  §21.4 D  confidence: verification vs a text-based partner
  §21.5 C  FashionIQ with frozen CIRR settings

T is within-window reranking of the base top 10, never full-gallery text retrieval.
Helpers that define published numbers are imported: fusion_controls (split, z-score,
hit@1, LAMBDA_GRID), fusion_suite (rules, CONVEX_GRID, urisk_trisk), bbnv_policy
(clustered bootstrap), selective_retrieval_analysis (confidence, AUROC), and
revision_followups (§20 statistics, input hashing).

  (from revua_cir/) python scripts/revision_followups2.py --root .. --out ../revision_work/followups21_2026-11-03
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import List, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import bbnv_policy as pol  # noqa: E402
import fusion_controls as fc  # noqa: E402
import fusion_suite as fs  # noqa: E402
import revision_followups as rf  # noqa: E402
import selective_retrieval_analysis as sra  # noqa: E402

AMENDMENT = "v3.1 §21 (2026-11-03)"
FIQ = ("dress", "shirt", "toptee")
INPUTS = {   # project-root-relative path, sha256 recorded before the run
    "cache": ("bbnv_caches/rv_score_cache_fc_aep_rv.jsonl",
              "26d7fc552b4ae902d9ecfd0bd4ec2bad329e627536887ae4b241e5d9a85c0308"),
    "features": ("bbnv_feats/v1/cirr_val/features.npz",
                 "3a91e82caa69479ba6fe18ce93adbcca9345a454cc53812c3105d4927495033f"),
    "s20": ("revision_work/followups_2026-11-03/results.json",
            "a8ef5f658dad278ef66843c79b59365db20cabcb3e9041b025c57a53a357dd30"),
    "policy": ("bbnv_runs/policy_cirr_v1/results.json",
               "4078a199e1128025aa372c87ffbc873b5e7aaa263ce1c14b1c39e607da8fd77e"),
    "transfer": ("bbnv_runs/transfer_fiq_v1/results.json",
                 "03559e239733e94707e007bdbf525c2c78a1c7d31cdc404b102029bd0a180886"),
    "fiq_dress_cache": ("bbnv_caches/fiq_dress_val.jsonl",
                        "02d3b622bdf6e5c909e4c2c4a053fa6edb7540801c0cb9152f659fd5cfcf5083"),
    "fiq_shirt_cache": ("bbnv_caches/fiq_shirt_val.jsonl",
                        "d11850041dc96fed2deedda3d3d4fe393c380618ccf38deb19ab810b8c1869c4"),
    "fiq_toptee_cache": ("bbnv_caches/fiq_toptee_val.jsonl",
                         "cd1f114c8e39bc34e233c6025e3f3433cdc80b3fa0087609093d378915039331"),
    "fiq_dress_feat": ("bbnv_feats/v1/fiq_dress/features.npz",
                       "db71e603a8fab694fce812a244505fac42c5b4337ad8de64a2f00dccdd4aa283"),
    "fiq_shirt_feat": ("bbnv_feats/v1/fiq_shirt/features.npz",
                       "614b0278f55c6695739bc59f1fbb3c013a29254df709df3a8838eaa2edbd8150"),
    "fiq_toptee_feat": ("bbnv_feats/v1/fiq_toptee/features.npz",
                        "b3ebdbb4af08a3c73c0229f30f5e3a120a795789a872d8d5ddb5ceb1f1aed02b"),
}
LAMBDA_T_S20 = 4.0                      # §20.2, locked; re-read from the §20 results and asserted
PUBLISHED_AUROC_ENT_RV = 0.7078         # §20.5 (4 d.p.)
PUBLISHED_FIQ = {"base": 9.2753, "deployed": 8.9761}   # transfer_fiq_v1 pooled, 4 d.p.
N_REP = 10000
SCORE_RULES = ("bounded", "linear", "convex")
RANK_RULES = ("rank_wsum", "rrf60", "wrrf")


# ---------------------------------------------------------------------------
# signals and rules
# ---------------------------------------------------------------------------
def text_cosines(recs, feats, gallery_size: int) -> List[List[float]]:
    """T per window candidate, with the §20.2 alignment checks (fatal)."""
    img = feats["image_emb"].astype(np.float64)
    txt = feats["text_emb"].astype(np.float64)
    if len(img) != gallery_size or len(txt) != len(recs):
        raise SystemExit("ABORT: features %s/%s do not match %d queries, gallery %d"
                         % (txt.shape, img.shape, len(recs), gallery_size))
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    out = []
    for q, r in enumerate(recs):
        if r["target_gallery_index"] != int(feats["tgt_row"][q]):
            raise SystemExit("ABORT: query %d target does not match the feature file" % q)
        if int(feats["ref_row"][q]) in r["candidate_gallery_ids"]:
            raise SystemExit("ABORT: query %d has its reference in the window" % q)
        out.append((img[r["candidate_gallery_ids"]] @ txt[q]).tolist())
    return out


def three_term(b: Sequence[float], t: Sequence[float], x: Sequence[float], lt: float, lx: float) -> List[float]:
    """z(B) + lt*tanh(z(T)) + lx*tanh(z(X)); with lx = 0 it equals fc.rule_bounded(b, t, lt)."""
    zb, zt, zx = fc._zscore(b), fc._zscore(t), fc._zscore(x)
    return [zb[i] + lt * math.tanh(zt[i]) + lx * math.tanh(zx[i]) for i in range(len(zb))]


def hits_of(recs, rel, score_fn, idx) -> List[int]:
    return [fc.hit_at_1(score_fn(i), rel[i]) for i in idx]


def utility(hits: Sequence[int], base: Sequence[int]) -> float:
    return fs.urisk_trisk(hits, base, 1.0)["urisk"]


# ---------------------------------------------------------------------------
# §21.1 A
# ---------------------------------------------------------------------------
def select_three(recs, rel, T, X, idx, base, objective: str):
    """argmax over (lt, lx) in LAMBDA_GRID^2; ties to the smaller lx, then the smaller lt."""
    best = None
    for lx in fc.LAMBDA_GRID:
        for lt in fc.LAMBDA_GRID:
            h = hits_of(recs, rel, lambda i: three_term(recs[i]["base_score"], T[i], X[i], lt, lx), idx)
            val = sum(h) if objective == "r1" else utility(h, [base[i] for i in idx])
            key = (val, -lx, -lt)
            if best is None or key > best[0]:
                best = (key, lt, lx)
    return best[1], best[2]


def part_a(recs, rel, T, tune, report, base) -> dict:
    X = {"V": [r["verification_score"] for r in recs], "recon": [r["recon_cos"] for r in recs]}
    bt_rep = hits_of(recs, rel, lambda i: fc.rule_bounded(recs[i]["base_score"], T[i], LAMBDA_T_S20), report)
    bh = [base[i] for i in report]
    out = {"reference": {"lambda_T": LAMBDA_T_S20, **rf.outcome(bt_rep, bh)}, "contrasts": {}, "utility_selected": {}}
    for name, key in (("Q1", "V"), ("Q2", "recon")):
        lt, lx = select_three(recs, rel, T, X[key], tune, base, "r1")
        h = hits_of(recs, rel, lambda i: three_term(recs[i]["base_score"], T[i], X[key][i], lt, lx), report)
        c = {"term": key, "lambda_T": lt, "lambda_X": lx, "excluded_by_tuning": lx == 0.0,
             **rf.outcome(h, bh), "vs_BT": rf.paired(h, bt_rep)}
        full = list(range(len(recs)))
        hf = hits_of(recs, rel, lambda i: three_term(recs[i]["base_score"], T[i], X[key][i], lt, lx), full)
        btf = hits_of(recs, rel, lambda i: fc.rule_bounded(recs[i]["base_score"], T[i], LAMBDA_T_S20), full)
        c["fullset_vs_BT"] = rf.paired(hf, btf)
        out["contrasts"][name] = c
        ult, ulx = select_three(recs, rel, T, X[key], tune, base, "u")
        hu = hits_of(recs, rel, lambda i: three_term(recs[i]["base_score"], T[i], X[key][i], ult, ulx), report)
        out["utility_selected"][key] = {"lambda_T": ult, "lambda_X": ulx, **rf.outcome(hu, bh)}
    adj = rf.holm_adjusted([out["contrasts"][q]["vs_BT"]["p_mcnemar"] for q in ("Q1", "Q2")])
    for q, p in zip(("Q1", "Q2"), adj):
        out["contrasts"][q]["vs_BT"]["p_holm"] = p
        out["contrasts"][q]["vs_BT"]["holm_significant"] = p <= 0.05
    return out


# ---------------------------------------------------------------------------
# §21.2 B
# ---------------------------------------------------------------------------
def select_rule(recs, rel, T, fn, grid, idx, base, objective: str):
    """Best parameter on idx; ties to the smaller parameter. grid None = parameter-free."""
    if grid is None:
        return 0.0
    best = None
    for p in grid:
        h = hits_of(recs, rel, lambda i: fn(recs[i]["base_score"], T[i], p), idx)
        val = sum(h) if objective == "r1" else utility(h, [base[i] for i in idx])
        if best is None or val > best[0]:
            best = (val, p)
    return best[1]


def part_b(recs, rel, T, tune, report, base) -> dict:
    bh = [base[i] for i in report]
    out = {}
    for objective in ("r1", "u"):
        rows, hits = {}, {}
        for name in SCORE_RULES + RANK_RULES:
            fn, grid, fixed, _ = fs.RULES[name]
            p = select_rule(recs, rel, T, fn, grid, tune, base, objective) if grid is not None else fixed
            h = hits_of(recs, rel, lambda i: fn(recs[i]["base_score"], T[i], p), report)
            tune_r1 = sum(hits_of(recs, rel, lambda i: fn(recs[i]["base_score"], T[i], p), tune))
            rows[name] = {"param": p, "tune_hits": tune_r1, **rf.outcome(h, bh), "vs_base": rf.paired(h, bh),
                          "trisk_a1": fs.urisk_trisk(h, bh, 1.0)["trisk"], "trisk_a5": fs.urisk_trisk(h, bh, 5.0)["trisk"]}
            hits[name] = h
        names = list(rows)
        for n, p in zip(names, rf.holm_adjusted([rows[n]["vs_base"]["p_mcnemar"] for n in names])):
            rows[n]["vs_base"]["p_holm"] = p
        # reference: highest tune-half hits among score rules; ties to the fixed order of SCORE_RULES
        ref = max(SCORE_RULES, key=lambda n: (rows[n]["tune_hits"], -SCORE_RULES.index(n)))
        direct = {n: rf.paired(hits[n], hits[ref]) for n in RANK_RULES}
        for n, p in zip(RANK_RULES, rf.holm_adjusted([direct[n]["p_mcnemar"] for n in RANK_RULES])):
            direct[n]["p_holm"] = p
        out[objective] = {"rules": rows, "reference_rule": ref, "rank_vs_reference": direct,
                          "status": "registered (§21.2)" if objective == "r1" else "descriptive"}
    return out


# ---------------------------------------------------------------------------
# §21.3 cross-fitted utility, vs the published gates
# ---------------------------------------------------------------------------
def part_risk(recs, rel, T, tune, report, base, cluster, policy) -> dict:
    hits = [None] * len(recs)
    lams = {}
    for fold, (fit, ev) in (("fit_tune", (tune, report)), ("fit_report", (report, tune))):
        lam = select_rule(recs, rel, T, fc.rule_bounded, fc.LAMBDA_GRID, fit, base, "u")
        lams[fold] = lam
        for i, h in zip(ev, hits_of(recs, rel, lambda i: fc.rule_bounded(recs[i]["base_score"], T[i], lam), ev)):
            hits[i] = h
    if any(h is None for h in hits):
        raise SystemExit("ABORT: cross-fitting left a query unscored")
    x = [(h - b) if h >= b else 2.0 * (h - b) for h, b in zip(hits, base)]       # URisk alpha = 1 per query
    if len(set(cluster)) != policy["n_clusters"]:
        raise SystemExit("ABORT: %d reference clusters, gate study used %d" % (len(set(cluster)), policy["n_clusters"]))
    pooled = policy["pooled"]
    return {"lambda_by_fold": lams, **rf.outcome(hits, base),
            "utility_ci": pol.clustered_bootstrap(x, cluster, N_REP),
            "published_gates": {g: {"u": pooled[g]["u"], "r1": pooled[g]["r1"], "rescues": pooled[g]["rescues"],
                                    "breaks": pooled[g]["breaks"]} for g in ("FW", "control", "A")},
            "comparison": "descriptive only: per-query gate decisions were not saved"}


# ---------------------------------------------------------------------------
# §21.4 D
# ---------------------------------------------------------------------------
def mean_midranks(*proxies: Sequence[float]) -> List[float]:
    ranks = [sra._midranks(p) for p in proxies]
    return [sum(col) / len(col) for col in zip(*ranks)]


def part_d(cache_path, T) -> dict:
    recs, mp, _ = sra.load_records(cache_path)
    if len(recs) != len(T):
        raise SystemExit("ABORT: confidence cache and T differ in length")
    correct = [sra.is_correct(r, 1, mp) for r in recs]
    ent = [sra.confidence(r, "base_entropy") for r in recs]
    rvm = [sra.confidence(r, "rv_margin") for r in recs]
    tm = [t[0] - t[1] for t in T]                      # same positions as rv_margin
    ent_rv = sra.combine_by_rank(ent, rvm)
    if mean_midranks(ent, rvm) != ent_rv:
        raise SystemExit("ABORT: mean_midranks disagrees with combine_by_rank")
    if round(sra.auroc(ent_rv, correct), 4) != PUBLISHED_AUROC_ENT_RV:
        raise SystemExit("ABORT: entropy+rv_margin AUROC does not reproduce %.4f" % PUBLISHED_AUROC_ENT_RV)
    ent_t = sra.combine_by_rank(ent, tm)
    ent_t_rv = mean_midranks(ent, tm, rvm)
    au = {n: sra.auroc(c, correct) for n, c in (("t_margin", tm), ("entropy+t_margin", ent_t),
                                                 ("entropy+rv_margin", ent_rv), ("entropy+t_margin+rv_margin", ent_t_rv))}
    r1 = sra.paired_bootstrap_auroc_delta(ent_rv, ent_t, correct)
    r2 = sra.paired_bootstrap_auroc_delta(ent_t_rv, ent_t, correct)
    for r, p in zip((r1, r2), rf.holm_adjusted([r1["p_two_sided"], r2["p_two_sided"]])):
        r["p_holm"] = p
    return {"auroc": au, "R1_entV_minus_entT": r1, "R2_entTV_minus_entT": r2}


# ---------------------------------------------------------------------------
# §21.5 C
# ---------------------------------------------------------------------------
def part_c(paths, sel_v) -> dict:
    pooled = {"base": [], "deployed": [], "BT": [], "BTV": []}
    per_cat = {}
    for c in FIQ:
        recs, mp = fc.load_records(paths["fiq_%s_cache" % c])
        if mp:
            raise SystemExit("ABORT: FashionIQ cache %s is multi-positive" % c)
        rel = [fc.relevant_set(r, mp) for r in recs]
        with np.load(paths["fiq_%s_feat" % c], allow_pickle=False) as z:
            feats = {k: z[k] for k in ("image_emb", "text_emb", "ref_row", "tgt_row")}
        T = text_cosines(recs, feats, len(feats["image_emb"]))
        idx = list(range(len(recs)))
        sys_hits = {
            "base": hits_of(recs, rel, lambda i: recs[i]["base_score"], idx),
            "deployed": hits_of(recs, rel, lambda i: fc.rule_bounded(recs[i]["base_score"], recs[i]["verification_score"], 1.0), idx),
            "BT": hits_of(recs, rel, lambda i: fc.rule_bounded(recs[i]["base_score"], T[i], LAMBDA_T_S20), idx),
            "BTV": hits_of(recs, rel, lambda i: three_term(recs[i]["base_score"], T[i], recs[i]["verification_score"],
                                                            sel_v[0], sel_v[1]), idx)}
        per_cat[c] = {s: rf.outcome(h, sys_hits["base"]) for s, h in sys_hits.items()}
        for s in pooled:
            pooled[s] += sys_hits[s]
    pre = {"base": round(100.0 * sum(pooled["base"]) / len(pooled["base"]), 4),
           "deployed": round(100.0 * sum(pooled["deployed"]) / len(pooled["deployed"]), 4)}
    if pre != PUBLISHED_FIQ or len(pooled["base"]) != 6016:
        raise SystemExit("ABORT: FashionIQ pre-flight %s != published %s" % (pre, PUBLISHED_FIQ))
    out = {"preflight": pre, "per_category": per_cat,
           "pooled": {s: rf.outcome(h, pooled["base"]) for s, h in pooled.items()},
           "BT_vs_base": rf.paired(pooled["BT"], pooled["base"]),
           "BT_vs_deployed": rf.paired(pooled["BT"], pooled["deployed"]),
           "BTV_vs_BT_descriptive": rf.paired(pooled["BTV"], pooled["BT"])}
    for k, p in zip(("BT_vs_base", "BT_vs_deployed"),
                    rf.holm_adjusted([out["BT_vs_base"]["p_mcnemar"], out["BT_vs_deployed"]["p_mcnemar"]])):
        out[k]["p_holm"] = p
    return out


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        ap.error("%s exists; refusing to overwrite" % args.out)
    t0 = time.time()
    paths = {k: os.path.join(args.root, p) for k, (p, _) in INPUTS.items()}
    hashes = {k: rf.sha256(p) for k, p in paths.items()}
    for k, (_, want) in INPUTS.items():
        if hashes[k] != want:
            raise SystemExit("ABORT: %s sha256 %s != recorded %s" % (k, hashes[k], want))
    with open(paths["s20"], encoding="utf-8") as fh:
        s20 = json.load(fh)
    if s20["m2a"]["signals"]["T"]["lambda"] != LAMBDA_T_S20:
        raise SystemExit("ABORT: §20 T lambda is not %s" % LAMBDA_T_S20)
    with open(paths["policy"], encoding="utf-8") as fh:
        policy = json.load(fh)

    recs, mp = fc.load_records(paths["cache"])
    if mp or len(recs) != 4181:
        raise SystemExit("ABORT: expected the 4181-query CIRR cache")
    rel = [fc.relevant_set(r, mp) for r in recs]
    tune, report = fc.split_indices(len(recs))
    base = [fc.hit_at_1(r["base_score"], rel[i]) for i, r in enumerate(recs)]
    with np.load(paths["features"], allow_pickle=False) as z:
        feats = {k: z[k] for k in ("image_emb", "text_emb", "ref_row", "tgt_row")}
    T = text_cosines(recs, feats, 2297)
    bt = hits_of(recs, rel, lambda i: fc.rule_bounded(recs[i]["base_score"], T[i], LAMBDA_T_S20), report)
    pre = round(100.0 * sum(bt) / len(bt), 3)
    if pre != round(s20["m2a"]["signals"]["T"]["r1"], 3):
        raise SystemExit("ABORT: base+T does not reproduce §20 (%s)" % pre)
    print("[pre-flight] base+T reproduces §20: %.3f on the report half" % pre, flush=True)

    res = {"amendment": AMENDMENT, "inputs_sha256": hashes,
           "label": "no additional training, scored with the trained backbone; within-window reranking"}
    res["A"] = part_a(recs, rel, T, tune, report, base)
    print("[A] done, %.0fs" % (time.time() - t0), flush=True)
    res["B"] = part_b(recs, rel, T, tune, report, base)
    print("[B] done, %.0fs" % (time.time() - t0), flush=True)
    res["risk"] = part_risk(recs, rel, T, tune, report, base, feats["ref_row"].astype(np.int64).tolist(), policy)
    print("[risk] done, %.0fs" % (time.time() - t0), flush=True)
    res["D"] = part_d(paths["cache"], T)
    print("[D] done, %.0fs" % (time.time() - t0), flush=True)
    q1 = res["A"]["contrasts"]["Q1"]
    res["C"] = part_c(paths, (q1["lambda_T"], q1["lambda_X"]))
    print("[C] done, %.0fs" % (time.time() - t0), flush=True)
    res["source_sha256"] = {f: rf.sha256(os.path.join(_HERE, f)) for f in
                            ("revision_followups2.py", "revision_followups.py", "fusion_controls.py",
                             "fusion_suite.py", "bbnv_policy.py", "selective_retrieval_analysis.py")}
    res["wall_seconds"] = round(time.time() - t0, 1)
    os.makedirs(args.out)
    out = os.path.join(args.out, "results.json")
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("[done] wrote %s (sha256 %s) in %.0fs" % (out, rf.sha256(out), time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
