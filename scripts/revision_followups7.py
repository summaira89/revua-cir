"""CIRCO val transfer of the text-cosine and verifier findings: amendment v3.1 §27. CPU.

Every arm reorders the SAME cached CIRCO top-10 windows of the deployed retriever; positions
11-50 keep their base order (circo_bounded_map's splice), and AP@K is the official CIRCO
formula (circo_bounded_map.average_precision_at_k). Arms:
  base | +T_pretrained | +T_fc_only | +T_ours | +V | +T_ours+V
T_E = cos(text_E, image_E) with matched towers of encoder E (circo_towers.py); V = the cache's
verification_score.

Fusion weights are never tuned on CIRCO. Primary source: tuned on the CIRR tune half, on our
own CIRR windows (the same first-stage retriever as CIRCO), by R@1 over LAMBDA_GRID -- so the
primary result tests transfer across DATASET only. Secondary sources, labelled: frozen
(lambda_T 4 of §20.2, lambda_V 1.0 deployed) and the §26c CLIP4Cir-window weights (transfer
across dataset AND first-stage retriever).

Primary endpoint mAP@5 (paired bootstrap over queries, seed 54321); Holm over the two primary
contrasts P1 (+T_fc_only vs base) and P2 (+T_ours+V vs +T_ours). mAP@10/25/50 supporting.

  (from revua_cir/) python scripts/revision_followups7.py --root .. --circo-towers deployed=../revision_work/circo_towers_deployed_2026-11-04 --circo-towers pretrained=../revision_work/circo_towers_pretrained_2026-11-04 --circo-towers fc_only=../revision_work/circo_towers_fc_2026-11-04 --out ../revision_work/followups27_2026-11-04
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import circo_bounded_map as cbm  # noqa: E402
import fusion_controls as fc  # noqa: E402
import revision_followups as rf  # noqa: E402
import revision_followups2 as r2  # noqa: E402
import revision_followups3 as r3  # noqa: E402
import revision_followups5 as r5  # noqa: E402
import revision_followups6 as r6  # noqa: E402

AMENDMENT = "v3.1 §27 (2026-11-04)"
CIRCO_CACHE = ("rv_score_cache_circo_fc_aep_rv.jsonl",
               "5c1c77d4457d7ec7b72f914b6763d53e69493110358e68bd797bdba63195ba6b")
CIRR_TOWERS = {   # dir, locked results.json sha256 (edir_work/LOCK_followups26c*_results_2026-11-04)
    "pretrained": ("revision_work/towers_pretrained_2026-11-04",
                   "bce5c145799483d9e24693364d76f8b580fd066df39845044ac51e145d99d0a8"),
    "fc_only": ("revision_work/towers_fc_2026-11-04",
                "6e20ba845f05991555da01bf96c6618af5c890cac5470c3a2dc5ac921fc12456"),
    "deployed": ("revision_work/towers_deployed_2026-11-04",
                 "7413dd3a76758196038c20b56c35899d65c91f51262cb0e7135fe57b5db66fc9")}
CHECKPOINT_SHA = {"deployed": "677eddfb2238a4d729678346d3b5af49ddf6edb83b0ee565b089a03bf596c597",
                  "fc_only": "8d5c6932ec5183d464d180ac27096c029f8f749afebe34da627f107109edf9dc",
                  "pretrained": None}
S26C_FC = ("revision_work/followups26c_fc_2026-11-04/results.json",
           "7012dd9aef2608e34a58bf1a3265156721dc3c92e67d3aef3b2e46c25295e436")
CIRCO_SCHEMA = "circo-towers/1"
PUBLISHED_BASE_MAP5 = 24.49
RANKS = cbm.RANKS
ENCODERS = ("pretrained", "fc_only", "ours")


# ---------------------------------------------------------------------------
# CIRR side: the primary fusion weights
# ---------------------------------------------------------------------------
def cirr_weights(root: str) -> Tuple[dict, dict]:
    """lambda per arm, tuned on the CIRR tune half on our own windows (R@1, ties smaller)."""
    hashes, _, recs, ours, _ = r5.load_all(root)
    with open(os.path.join(root, r6.GPU_RUN[0], "results.json"), encoding="utf-8") as fh:
        gallery_sha = json.load(fh)["gallery_sha256"]
    rel = [fc.relevant_set(r, False) for r in recs]
    pos = [next(iter(s)) if s else None for s in rel]
    B = [r["base_score"] for r in recs]
    V = [r["verification_score"] for r in recs]
    wb = np.asarray([r["candidate_gallery_ids"] for r in recs], dtype=np.int64)
    T = {"ours": r2.text_cosines(recs, ours, 2297)}
    for name in ("deployed", "pretrained", "fc_only"):
        d, want = CIRR_TOWERS[name]
        if rf.sha256(os.path.join(root, d, "results.json")) != want:
            raise SystemExit("ABORT: CIRR towers %s differ from the lock" % name)
        n, arr, prov_ = r6.load_towers("%s=%s" % (name, os.path.join(root, d)), len(recs), 2297, gallery_sha)
        if prov_["checkpoint_sha256"] != CHECKPOINT_SHA[name]:
            raise SystemExit("ABORT: CIRR towers %s came from checkpoint %r" % (name, prov_["checkpoint_sha256"]))
        if n != "deployed":
            T[n] = r5.window_cosines(arr["text_emb"], arr["image_emb"], wb).tolist()
    tune, _ = fc.split_indices(len(recs))
    lam = {"T_" + e: r5.tune_one(B, T[e], pos, tune) for e in ENCODERS}
    lam["V"] = r5.tune_one(B, V, pos, tune)
    lt, lx = r5.tune_three(B, T["ours"], V, pos, tune)
    lam["T_ours+V"] = [lt, lx]
    if lam["T_ours"] != r2.LAMBDA_T_S20:
        raise SystemExit("ABORT: CIRR-tuned lambda for our T is %r, not §20.2's %r" % (lam["T_ours"], r2.LAMBDA_T_S20))
    return lam, hashes


def clip4cir_weights(root: str) -> dict:
    path = os.path.join(root, S26C_FC[0])
    if rf.sha256(path) != S26C_FC[1]:
        raise SystemExit("ABORT: §26c (with FC-only) results differ from the lock")
    with open(path, encoding="utf-8") as fh:
        lt = json.load(fh)["E"]["lambda_tuned"]
    with open(os.path.join(root, r6.S25C[0]), encoding="utf-8") as fh:
        v = json.load(fh)["V"]["lambda_tuned"]
    return {"T_pretrained": lt["pretrained"], "T_fc_only": lt["fc_only"], "T_ours": lt["ours"],
            "V": v["A4"], "T_ours+V": [v["A5"]["lambda_T"], v["A5"]["lambda_V"]]}


# ---------------------------------------------------------------------------
# CIRCO side
# ---------------------------------------------------------------------------
def load_circo_towers(spec: str, cache_sha: str, rows: Sequence[int], n_q: int) -> Tuple[str, Dict[str, np.ndarray], dict]:
    name, _, d = spec.partition("=")
    if name not in ("deployed",) + ENCODERS[:2]:
        raise SystemExit("ABORT: unexpected CIRCO towers name %r" % name)
    with open(os.path.join(d, "results.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    npz = os.path.join(d, res["output_name"])
    if res["schema"] != CIRCO_SCHEMA or rf.sha256(npz) != res["output_files"][res["output_name"]]:
        raise SystemExit("ABORT: %s is not an intact circo_towers.py run" % d)
    if res["manifest"]["cache_sha256"] != cache_sha:
        raise SystemExit("ABORT: %s was built from a different CIRCO cache" % d)
    want_src = r6.TOWER_SOURCE[name]
    if res["source"] != want_src:
        raise SystemExit("ABORT: towers %r have source %r" % (name, res["source"]))
    if res["manifest"]["checkpoint_sha256"] != CHECKPOINT_SHA[name]:
        raise SystemExit("ABORT: CIRCO towers %s came from checkpoint %r" % (name, res["manifest"]["checkpoint_sha256"]))
    if name == "deployed" and not (res["parity_base"] and res["parity_base"]["base_score_worst"] <= 1e-3):
        raise SystemExit("ABORT: the deployed CIRCO towers run did not reproduce the base scores")
    with np.load(npz, allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    if arr["rows"].tolist() != list(rows) or arr["text_emb"].shape[0] != n_q:
        raise SystemExit("ABORT: %s covers different window rows or queries" % d)
    return name, arr, {"dir": os.path.basename(os.path.normpath(d)), "results_sha256": rf.sha256(os.path.join(d, "results.json")),
                       "checkpoint_sha256": res["manifest"]["checkpoint_sha256"], "parity_base": res["parity_base"]}


def circo_cosines(arr: Dict[str, np.ndarray], recs: Sequence[dict]) -> List[List[float]]:
    at = {int(g): k for k, g in enumerate(arr["rows"])}
    img, txt = r3.unit_rows(arr["image_emb"]), r3.unit_rows(arr["text_emb"])
    return [(img[[at[int(g)] for g in r["candidate_gallery_ids"]]] @ txt[i]).tolist() for i, r in enumerate(recs)]


def rerank_top50(rec: dict, scores: Sequence[float]) -> List[int]:
    """Window reordered by `scores` (ties to the lower window position), ranks 11-50 kept."""
    cand = list(rec["candidate_gallery_ids"])
    top50 = list(rec["base_top50_gallery_ids"])
    if top50[:len(cand)] != cand or len(scores) != len(cand):
        raise SystemExit("ABORT: window is not the head of the top-50, or scores misaligned")
    order = sorted(range(len(cand)), key=lambda j: (-scores[j], j))
    return [cand[j] for j in order] + top50[len(cand):]


def arm_scores(B, T: Dict[str, list], V, lam: dict) -> Dict[str, Callable[[int], Sequence[float]]]:
    lt, lx = lam["T_ours+V"]
    arms = {"base": lambda i: B[i], "V": lambda i: fc.rule_bounded(B[i], V[i], lam["V"]),
            "T_ours+V": lambda i: r2.three_term(B[i], T["ours"][i], V[i], lt, lx)}
    for e in ENCODERS:
        arms["T_" + e] = (lambda e_: lambda i: fc.rule_bounded(B[i], T[e_][i], lam["T_" + e_]))(e)
    return arms


def evaluate(recs, arms, n_rep: int) -> dict:
    gts = [r["relevant_gallery_indices"] for r in recs]
    ap = {a: {k: [cbm.average_precision_at_k(rerank_top50(r, fn(i)), gts[i], k) for i, r in enumerate(recs)]
              for k in RANKS} for a, fn in arms.items()}
    maps = {a: {k: 100.0 * float(np.mean(v[k])) for k in RANKS} for a, v in ap.items()}

    def con(a, b):
        return {k: cbm.paired_bootstrap(ap[a][k], ap[b][k], n_rep) for k in RANKS}
    primary = {"P1_T_fc_only_vs_base": con("T_fc_only", "base"), "P2_T_ours+V_vs_T_ours": con("T_ours+V", "T_ours")}
    for (name, c), p in zip(primary.items(), rf.holm_adjusted([c[5]["p_two_sided"] for c in primary.values()])):
        c["mAP@5_p_holm"] = p
    secondary = {"T_ours_vs_base": con("T_ours", "base"), "T_pretrained_vs_base": con("T_pretrained", "base"),
                 "V_vs_base": con("V", "base"), "T_ours_vs_T_fc_only": con("T_ours", "T_fc_only"),
                 "T_ours_vs_T_pretrained": con("T_ours", "T_pretrained")}
    changed = {a: sum(1 for i, r in enumerate(recs) if rerank_top50(r, fn(i))[:10] != list(r["candidate_gallery_ids"]))
               for a, fn in arms.items()}
    return {"maps": maps, "primary": primary, "secondary": secondary, "queries_reordered": changed}


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--circo-towers", action="append", required=True, help="NAME=DIR: deployed, pretrained, fc_only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        ap.error("%s exists; refusing to overwrite" % args.out)
    t0 = time.time()
    cache = os.path.join(args.root, CIRCO_CACHE[0])
    if rf.sha256(cache) != CIRCO_CACHE[1]:
        raise SystemExit("ABORT: CIRCO cache hash differs")
    recs = cbm.load_circo_records(cache)
    rows = sorted({int(g) for r in recs for g in r["candidate_gallery_ids"]})
    base_map5 = 100.0 * float(np.mean([cbm.average_precision_at_k(r["base_top50_gallery_ids"], r["relevant_gallery_indices"], 5)
                                       for r in recs]))
    if round(base_map5, 2) != PUBLISHED_BASE_MAP5:
        raise SystemExit("ABORT: base mAP@5 %.4f does not reproduce %.2f" % (base_map5, PUBLISHED_BASE_MAP5))
    towers, T = {}, {}
    for spec in args.circo_towers:
        name, arr, prov_ = load_circo_towers(spec, CIRCO_CACHE[1], rows, len(recs))
        if name in towers:
            raise SystemExit("ABORT: repeated towers %r" % name)
        towers[name] = prov_
        if name != "deployed":
            T[name] = circo_cosines(arr, recs)
        else:
            T["ours"] = circo_cosines(arr, recs)
    if set(towers) != {"deployed", "pretrained", "fc_only"}:
        raise SystemExit("ABORT: need CIRCO towers deployed, pretrained and fc_only; got %s" % sorted(towers))
    lam_cirr, hashes = cirr_weights(args.root)
    lam_c4c = clip4cir_weights(args.root)
    lam_frozen = {"T_pretrained": 4.0, "T_fc_only": 4.0, "T_ours": 4.0, "V": rf.DEPLOYED_LAMBDA,
                  "T_ours+V": [4.0, rf.DEPLOYED_LAMBDA]}
    print("[27] CIRCO cache, towers and CIRR weights checked; base mAP@5 %.2f, %.0fs" % (base_map5, time.time() - t0),
          flush=True)
    B = [r["base_score"] for r in recs]
    V = [r["verification_score"] for r in recs]
    res = {"amendment": AMENDMENT, "status": "specified before execution; CIRCO val (220 queries)",
           "inputs_sha256": {**hashes, "circo_cache": CIRCO_CACHE[1]}, "towers": towers,
           "weights": {"primary_cirr_own_windows": lam_cirr, "secondary_frozen": lam_frozen,
                       "secondary_clip4cir_windows": lam_c4c},
           "base_map5": base_map5, "n_window_rows": len(rows)}
    for label, lam in (("primary", lam_cirr), ("secondary_frozen", lam_frozen), ("secondary_clip4cir", lam_c4c)):
        res[label] = evaluate(recs, arm_scores(B, T, V, lam), cbm.N_BOOTSTRAP)
    res["source_sha256"] = {f: rf.sha256(os.path.join(_HERE, f)) for f in
                            ("revision_followups7.py", "revision_followups6.py", "revision_followups5.py",
                             "circo_bounded_map.py", "fusion_controls.py", "revision_followups2.py")}
    res["wall_seconds"] = round(time.time() - t0, 1)
    os.makedirs(args.out)
    out = os.path.join(args.out, "results.json")
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("[done] wrote %s (sha256 %s) in %.0fs" % (out, rf.sha256(out), time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
