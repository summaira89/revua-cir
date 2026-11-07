"""Candidate conditioning, paired gate comparison and the CIRR subset: amendment v3.1 §22. CPU.

Exploratory: specified before execution, after inspecting the §20/§21 CIRR findings.

  §22.1  reconstruct the cross-fitted forward gate's per-query decisions with
         bbnv_policy.run_fold, require exact parity with every saved fold quantity and a
         deterministic second reconstruction, then compare cross-fitted base+T fusion with
         FW and with the control on paired per-query utility (clustered bootstrap)
  §22.2  Stage 1 of candidate conditioning: the deployed base window W_B versus a
         reference-image-similarity window W_I, scored by B, T and B+T (lambda frozen at 4);
         coverage, R@1, and two diagnostic selected-population views; full-gallery text
         retrieval reported separately as its own baseline
  §22.3  official CIRR subset (img_set members minus the reference), lambda frozen at 4,
         tuned lambda secondary

The verifier is not scored on any new candidate set here; §22 makes no claim about its
incremental value under other windows (that needs a later matched-window GPU run).

  (from revua_cir/) python scripts/revision_followups3.py --root .. --out ../revision_work/followups22_2026-11-03
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import bbnv_analyze as ana  # noqa: E402
import bbnv_policy as pol  # noqa: E402
import fusion_controls as fc  # noqa: E402
import revision_followups as rf  # noqa: E402
import revision_followups2 as r2  # noqa: E402

AMENDMENT = "v3.1 §22 (2026-11-03)"
INPUTS = {
    "cache": ("bbnv_caches/rv_score_cache_fc_aep_rv.jsonl",
              "26d7fc552b4ae902d9ecfd0bd4ec2bad329e627536887ae4b241e5d9a85c0308"),
    "features": ("bbnv_feats/v1/cirr_val/features.npz",
                 "3a91e82caa69479ba6fe18ce93adbcca9345a454cc53812c3105d4927495033f"),
    "vectors": ("edir_work/parity/vec_cirr_val.npz",
                "b5fa0ac2e6b95b60a8c984bdabbfc6ea2e8f3b2a9620b79591052531dfdf9be7"),
    "windows_results": ("bbnv_runs/windows_cirr_v1/results.json",
                        "cb5766eed3f18eafcc531aaeee4bebe034080c603bac18cafd0f4f4fc7c573ea"),
    "windows_rows": ("bbnv_runs/windows_cirr_v1/windows.jsonl",
                     "5060062ed23c9a521fb612f1db079c0bf04f1fd7ac3691e032658affb7efbb5e"),
    "policy": ("bbnv_runs/policy_cirr_v1/results.json",
               "4078a199e1128025aa372c87ffbc873b5e7aaa263ce1c14b1c39e607da8fd77e"),
    "s20": ("revision_work/followups_2026-11-03/results.json",
            "a8ef5f658dad278ef66843c79b59365db20cabcb3e9041b025c57a53a357dd30"),
    "captions": ("cirr_val_unpacked/cirr_val/captions/cap.rc2.val.json",
                 "a85c3a1aa464f1af7229918e8018d08b8b20ce5dab479ffdf39d61113140f919"),
}
LAMBDA_T = 4.0                       # §20.2, locked
K = 10
PUBLISHED = {"base_report": 36.633, "bt_report": 46.294, "base_full": 37.072,
             "fw_pooled_u": 0.028462090408993064, "control_pooled_u": 0.0}
FOLDS = (("tune", "report"), ("report", "tune"))


# ---------------------------------------------------------------------------
# §22.1 gate reconstruction
# ---------------------------------------------------------------------------
def reconstruct_gate(windows_dir: str):
    """Exactly the fold loop of bbnv_policy.main for seed 0; returns per-query vectors."""
    res, rows = ana.load_windows(windows_dir)
    lam = float(res["lambda_star_frozen"])
    halves = {"tune": [i for i, r in enumerate(rows) if r["split"] == "tune"],
              "report": [i for i, r in enumerate(rows) if r["split"] == "report"]}
    views = ana.gate_views(rows, lam, 0)
    n = len(rows)
    out = {"fw_hits": [None] * n, "fw_accept": [None] * n, "control_hits": [None] * n,
           "base": [None] * n, "folds": {}}
    for fit_name, ev_name in FOLDS:
        fi, ei = halves[fit_name], halves[ev_name]
        f = pol.run_fold([views[i] for i in fi], [views[i] for i in ei], [rows[i] for i in fi],
                         [rows[i] for i in ei], [views[i] for i in ei], [views[i] for i in fi])
        out["folds"]["fit_%s" % fit_name] = {
            "FW": pol.summary(f["hits"]["FW"], f["base"], f["acc"]["FW"]),
            "control": f["control"], "sw_lambda": f["sw_lambda"]}
        for j, i in enumerate(ei):
            out["fw_hits"][i] = int(f["hits"]["FW"][j])
            out["fw_accept"][i] = bool(f["acc"]["FW"][j])
            out["control_hits"][i] = int(f["hits"][f["control"]][j])
            out["base"][i] = int(f["base"][j])
    if any(v is None for k in ("fw_hits", "fw_accept", "control_hits", "base") for v in out[k]):
        raise SystemExit("ABORT: gate reconstruction left a query undecided")
    return out, rows


def check_gate_parity(rec: dict, again: dict, saved: dict) -> Dict[str, object]:
    """Every saved quantity must match exactly; a second reconstruction must be identical."""
    for fold, got in rec["folds"].items():
        want = saved["per_fold"][fold]
        if got["FW"] != want["policies"]["FW"]:
            raise SystemExit("ABORT: %s FW %s != saved %s" % (fold, got["FW"], want["policies"]["FW"]))
        if got["control"] != want["control"] or got["sw_lambda"] != want["sw_lambda"]:
            raise SystemExit("ABORT: %s control/SW lambda differ from the saved run" % fold)
    fw_u = pol.utility(rec["fw_hits"], rec["base"])
    ctl_u = pol.utility(rec["control_hits"], rec["base"])
    if fw_u != PUBLISHED["fw_pooled_u"] or ctl_u != PUBLISHED["control_pooled_u"]:
        raise SystemExit("ABORT: pooled FW/control utility %r/%r != saved" % (fw_u, ctl_u))
    for k in ("fw_hits", "fw_accept", "control_hits", "base"):
        if rec[k] != again[k]:
            raise SystemExit("ABORT: gate reconstruction is not deterministic (%s)" % k)
    return {"fold_parity": "exact", "pooled_fw_u": fw_u, "pooled_control_u": ctl_u, "deterministic": True,
            "label": "reconstructed decisions (original per-query predictions were not saved)"}


def text_crossfit(recs, rel, T, tune, report, base) -> Dict[str, object]:
    hits = [None] * len(recs)
    lams = {}
    for fit_name, (fit, ev) in (("fit_tune", (tune, report)), ("fit_report", (report, tune))):
        lam = r2.select_rule(recs, rel, T, fc.rule_bounded, fc.LAMBDA_GRID, fit, base, "u")
        lams[fit_name] = lam
        for i, h in zip(ev, r2.hits_of(recs, rel, lambda i: fc.rule_bounded(recs[i]["base_score"], T[i], lam), ev)):
            hits[i] = h
    if any(h is None for h in hits):
        raise SystemExit("ABORT: text cross-fit left a query unscored")
    return {"hits": hits, "lambda_by_fold": lams}


def per_query_utility(hits: Sequence[int], base: Sequence[int]) -> List[float]:
    return [(1.0 if h > b else -2.0 if h < b else 0.0) for h, b in zip(hits, base)]


def part_g(paths, recs, rel, T, tune, report, base, cluster, saved) -> dict:
    wdir = os.path.dirname(paths["windows_results"])
    rec, rows = reconstruct_gate(wdir)
    again, _ = reconstruct_gate(wdir)
    if [int(r["query"]) for r in rows] != [r["query_id"] for r in recs]:
        raise SystemExit("ABORT: gate windows and cache disagree in query order")
    if rec["base"] != base:
        raise SystemExit("ABORT: gate base hits differ from the cache's")
    parity = check_gate_parity(rec, again, saved)
    tx = text_crossfit(recs, rel, T, tune, report, base)
    ut, ufw, uct = (per_query_utility(h, base) for h in (tx["hits"], rec["fw_hits"], rec["control_hits"]))
    g1 = pol.clustered_bootstrap([a - b for a, b in zip(ut, ufw)], cluster, 10000)
    g2 = pol.clustered_bootstrap([a - b for a, b in zip(ut, uct)], cluster, 10000)
    for g, p in zip((g1, g2), rf.holm_adjusted([g1["p_two_sided"], g2["p_two_sided"]])):
        g["p_holm"] = p
    return {"parity": parity, "text_lambda_by_fold": tx["lambda_by_fold"],
            "systems": {"text_fusion": pol.summary(tx["hits"], base),
                        "FW": pol.summary(rec["fw_hits"], base, rec["fw_accept"]),
                        "control": pol.summary(rec["control_hits"], base)},
            "G1_text_minus_FW": g1, "G2_text_minus_control": g2,
            "decisions": {"fw_accept": rec["fw_accept"], "fw_hits": rec["fw_hits"], "text_hits": tx["hits"]}}


# ---------------------------------------------------------------------------
# §22.2 candidate conditioning
# ---------------------------------------------------------------------------
def unit_rows(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def top_k_excluding(scores: np.ndarray, exclude: int, k: int) -> List[int]:
    """Top-k gallery rows by score, `exclude` removed, ties to the lower row."""
    s = scores.astype(np.float64)
    order = np.lexsort((np.arange(len(s)), -s))
    return [int(x) for x in order if x != exclude][:k]


def window_hits(win: Sequence[int], target: int, b: Sequence[float], t: Sequence[float]) -> Dict[str, int]:
    rel = {win.index(target)} if target in win else set()
    return {"B": fc.hit_at_1(b, rel), "T": fc.hit_at_1(t, rel),
            "BT": fc.hit_at_1(fc.rule_bounded(b, t, LAMBDA_T), rel), "covered": int(bool(rel))}


def rank_of(scores: np.ndarray, target: int, exclude: int) -> int:
    """0-based rank of `target` with `exclude` removed; ties to the lower row (as top_k_excluding)."""
    s = scores.astype(np.float64)
    better = (s > s[target]) | ((s == s[target]) & (np.arange(len(s)) < target))
    better[exclude] = False
    return int(better.sum())


def part_w(recs, q, img, txt, ref_row, tgt_row, tune, report) -> dict:
    n = len(recs)
    sim_ref = img @ img.T
    rows = {"W_B": [], "W_I": []}
    for i, r in enumerate(recs):
        t_row = int(tgt_row[i])
        wb = r["candidate_gallery_ids"]
        wi = top_k_excluding(sim_ref[int(ref_row[i])], int(ref_row[i]), K)
        for name, win in (("W_B", wb), ("W_I", wi)):
            C = img[win]
            h = window_hits(win, t_row, (C @ q[i]).tolist(), (C @ txt[i]).tolist())
            if name == "W_I":
                h["I"] = fc.hit_at_1(sim_ref[int(ref_row[i])][win].tolist(), {win.index(t_row)} if t_row in win else set())
            rows[name].append(h)
    pre = round(100.0 * np.mean([rows["W_B"][i]["B"] for i in report]), 3), \
        round(100.0 * np.mean([rows["W_B"][i]["BT"] for i in report]), 3)
    if pre != (PUBLISHED["base_report"], PUBLISHED["bt_report"]):
        raise SystemExit("ABORT: W_B pre-flight %s != published (36.633, 46.294)" % (pre,))

    def summarise(idx):
        out = {}
        for name, hs in rows.items():
            cov = [hs[i]["covered"] for i in idx]
            sc = {}
            for s in [k for k in hs[0] if k != "covered"]:
                all_q = [hs[i][s] for i in idx]
                cond = [hs[i][s] for i in idx if hs[i]["covered"]]
                sc[s] = {"r1_all": 100.0 * np.mean(all_q), "r1_given_covered": 100.0 * np.mean(cond) if cond else None}
            out[name] = {"coverage": float(np.mean(cov)), "n_covered": int(sum(cov)), "scorings": sc}
        both = [i for i in idx if rows["W_B"][i]["covered"] and rows["W_I"][i]["covered"]]
        out["both_covered"] = {"n": len(both),
                               "BT_WB_minus_WI": rf.paired([rows["W_B"][i]["BT"] for i in both], [rows["W_I"][i]["BT"] for i in both]),
                               "T_WB_minus_WI": rf.paired([rows["W_B"][i]["T"] for i in both], [rows["W_I"][i]["T"] for i in both]),
                               "status": "diagnostic: selected population"}
        out["all_queries"] = {"BT_WB_minus_WI": rf.paired([rows["W_B"][i]["BT"] for i in idx], [rows["W_I"][i]["BT"] for i in idx]),
                              "T_WB_minus_WI": rf.paired([rows["W_B"][i]["T"] for i in idx], [rows["W_I"][i]["T"] for i in idx])}
        return out

    full = list(range(n))
    # full-gallery retrieval baselines (separate from the windows)
    ranks_t = [rank_of(img @ txt[i], int(tgt_row[i]), int(ref_row[i])) for i in range(n)]
    ranks_b = [rank_of(img @ q[i], int(tgt_row[i]), int(ref_row[i])) for i in range(n)]
    if round(100.0 * np.mean([r == 0 for r in ranks_b]), 3) != PUBLISHED["base_full"]:
        raise SystemExit("ABORT: full-gallery B R@1 does not reproduce 37.072")

    def recall(ranks, idx):
        return {"R@%d" % k: 100.0 * np.mean([ranks[i] < k for i in idx]) for k in (1, 10, 50)}
    gallery = {h: {"text_only": recall(ranks_t, idx), "composed_B": recall(ranks_b, idx)}
               for h, idx in (("report", report), ("full", full))}
    return {"report": summarise(report), "full": summarise(full), "full_gallery_baseline": gallery,
            "note": "full-gallery retrieval is a separate baseline, not a third window"}


# ---------------------------------------------------------------------------
# §22.3 official subset
# ---------------------------------------------------------------------------
def subset_members(captions, recs, name_to_row, ref_row, tgt_row) -> List[List[int]]:
    if len(captions) != len(recs):
        raise SystemExit("ABORT: captions and cache differ in length")
    out = []
    for i, c in enumerate(captions):
        rows = [name_to_row[m] for m in c["img_set"]["members"]]
        if name_to_row[c["reference"]] != int(ref_row[i]) or name_to_row[c["target_hard"]] != int(tgt_row[i]):
            raise SystemExit("ABORT: caption %d does not match the feature rows" % i)
        mem = [m for m in rows if m != int(ref_row[i])]
        if len(mem) != 5 or int(tgt_row[i]) not in mem:
            raise SystemExit("ABORT: query %d subset is not 5 members containing the target" % i)
        out.append(mem)
    return out


def subset_rank(scores: Sequence[float], pos: int) -> int:
    order = sorted(range(len(scores)), key=lambda j: (-scores[j], j))
    return order.index(pos)


def part_s(members, q, img, txt, tgt_row, tune, report) -> dict:
    B = [(img[m] @ q[i]).tolist() for i, m in enumerate(members)]
    T = [(img[m] @ txt[i]).tolist() for i, m in enumerate(members)]
    pos = [m.index(int(tgt_row[i])) for i, m in enumerate(members)]

    def ranks(fn, idx):
        return [subset_rank(fn(i), pos[i]) for i in idx]

    lam_tuned, best = fc.LAMBDA_GRID[0], -1
    for lam in fc.LAMBDA_GRID:
        h = sum(r == 0 for r in ranks(lambda i: fc.rule_bounded(B[i], T[i], lam), tune))
        if h > best:
            lam_tuned, best = lam, h
    systems = {"B": lambda i: B[i], "T": lambda i: T[i],
               "BT_frozen": lambda i: fc.rule_bounded(B[i], T[i], LAMBDA_T),
               "BT_tuned": lambda i: fc.rule_bounded(B[i], T[i], lam_tuned)}
    out = {"lambda_frozen": LAMBDA_T, "lambda_tuned_secondary": lam_tuned}
    for half, idx in (("report", report), ("full", list(range(len(members))))):
        rk = {s: ranks(fn, idx) for s, fn in systems.items()}
        hit = {s: [int(r == 0) for r in v] for s, v in rk.items()}
        out[half] = {"recall_subset": {s: {"R@%d" % k: 100.0 * np.mean([r < k for r in v]) for k in (1, 2, 3)}
                                       for s, v in rk.items()},
                     "BT_frozen_vs_B": rf.paired(hit["BT_frozen"], hit["B"]),
                     "T_vs_B": rf.paired(hit["T"], hit["B"])}
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
    with open(paths["policy"], encoding="utf-8") as fh:
        saved = json.load(fh)
    with open(paths["s20"], encoding="utf-8") as fh:
        if json.load(fh)["m2a"]["signals"]["T"]["lambda"] != LAMBDA_T:
            raise SystemExit("ABORT: §20 T lambda is not %s" % LAMBDA_T)
    with open(paths["captions"], encoding="utf-8") as fh:
        captions = json.load(fh)

    recs, mp = fc.load_records(paths["cache"])
    if mp or len(recs) != 4181:
        raise SystemExit("ABORT: expected the 4181-query CIRR cache")
    rel = [fc.relevant_set(r, mp) for r in recs]
    tune, report = fc.split_indices(len(recs))
    base = [fc.hit_at_1(r["base_score"], rel[i]) for i, r in enumerate(recs)]
    with np.load(paths["features"], allow_pickle=False) as z:
        feats = {k: z[k] for k in ("image_emb", "text_emb", "ref_row", "tgt_row", "image_paths")}
    with np.load(paths["vectors"], allow_pickle=False) as z:
        if [int(x) for x in z["query_id"]] != [r["query_id"] for r in recs]:
            raise SystemExit("ABORT: vector dump and cache disagree in query order")
        q = unit_rows(z["q_comp"])
    img, txt = unit_rows(feats["image_emb"]), unit_rows(feats["text_emb"])
    T = r2.text_cosines(recs, feats, 2297)
    cluster = feats["ref_row"].astype(np.int64).tolist()
    name_to_row = {os.path.splitext(os.path.basename(str(p)))[0]: i for i, p in enumerate(feats["image_paths"])}

    res = {"amendment": AMENDMENT, "inputs_sha256": hashes, "status": "exploratory; specified before execution"}
    res["G"] = part_g(paths, recs, rel, T, tune, report, base, cluster, saved)
    print("[22.1] gate parity exact and deterministic; paired comparison done, %.0fs" % (time.time() - t0), flush=True)
    res["W"] = part_w(recs, q, img, txt, feats["ref_row"], feats["tgt_row"], tune, report)
    print("[22.2] candidate conditioning done, %.0fs" % (time.time() - t0), flush=True)
    members = subset_members(captions, recs, name_to_row, feats["ref_row"], feats["tgt_row"])
    res["S"] = part_s(members, q, img, txt, feats["tgt_row"], tune, report)
    print("[22.3] official subset done, %.0fs" % (time.time() - t0), flush=True)
    res["source_sha256"] = {f: rf.sha256(os.path.join(_HERE, f)) for f in
                            ("revision_followups3.py", "revision_followups2.py", "revision_followups.py",
                             "bbnv_policy.py", "bbnv_analyze.py", "fusion_controls.py")}
    res["wall_seconds"] = round(time.time() - t0, 1)
    os.makedirs(args.out)
    out = os.path.join(args.out, "results.json")
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("[done] wrote %s (sha256 %s) in %.0fs" % (out, rf.sha256(out), time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
