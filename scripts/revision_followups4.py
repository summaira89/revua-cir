"""Distractor structure on shared-target queries: amendment v3.1 §23. CPU, descriptive.

On the queries whose target lies in both §22.2 windows (W_B, the deployed composed-query
window; W_I, the reference-image-similarity window), describe the text-distractor
configuration: candidate overlap, hardest-distractor text cosine, target margins, and the
candidates composition swaps in (W_B only) versus out (W_I only). No causal claim.

  (from revua_cir/) python scripts/revision_followups4.py --root .. --out ../revision_work/followups23_2026-11-03
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

import fusion_controls as fc  # noqa: E402
import revision_followups as rf  # noqa: E402
import revision_followups3 as r3  # noqa: E402

AMENDMENT = "v3.1 §23 (2026-11-03)"
INPUTS = {k: r3.INPUTS[k] for k in ("cache", "features", "vectors")}
INPUTS["s22"] = ("revision_work/followups22_2026-11-03/results.json",
                 "516c1aea7b504abfce4ed27208f3f08482028c58a2cfda66f49dc19e295b6a7c")
N_REP, SEED = 10000, 0


def query_profile(win_b: Sequence[int], win_i: Sequence[int], target: int, tv: np.ndarray) -> Dict[str, object]:
    """Text-distractor structure of one query whose target is in both windows.

    tv holds the text cosine of every gallery row. The hardest distractor is the first
    maximum in window order, i.e. ties go to the lowest window position (fc.hit_at_1's rule).
    """
    win_b, win_i = [int(r) for r in win_b], [int(r) for r in win_i]
    if target not in win_b or target not in win_i:
        raise ValueError("target must lie in both windows")
    db = [r for r in win_b if r != target]
    di = [r for r in win_i if r != target]
    hb = db[int(np.argmax(tv[db]))]
    hi = di[int(np.argmax(tv[di]))]
    sb, si = set(win_b), set(win_i)
    only_b = [r for r in db if r not in si]
    only_i = [r for r in di if r not in sb]
    p = {"shared": len(set(db) & set(di)), "di_subset_wb": int(set(di) <= sb), "hi_in_wb": int(hi in sb),
         "d_b": float(tv[hb]), "d_i": float(tv[hi]),
         "m_b": float(tv[target] - tv[hb]), "m_i": float(tv[target] - tv[hi]),
         "t_b": fc.hit_at_1(tv[win_b].tolist(), {win_b.index(target)}),
         "t_i": fc.hit_at_1(tv[win_i].tolist(), {win_i.index(target)}),
         "hb": hb, "hb_swapped_in": int(hb not in si), "only_b": only_b, "only_i": only_i}
    for m, t in (("m_b", "t_b"), ("m_i", "t_i")):
        if (p[m] > 0 and not p[t]) or (p[m] < 0 and p[t]):
            raise SystemExit("ABORT: margin %s=%r disagrees with T hit %s=%r" % (m, p[m], t, p[t]))
    return p


def gallery_ranks(tv: np.ndarray, exclude: int) -> np.ndarray:
    """0-based full-gallery rank of every row by tv, `exclude` removed (-1), ties to the lower row."""
    order = np.lexsort((np.arange(len(tv)), -tv.astype(np.float64)))
    order = order[order != exclude]
    rank = np.full(len(tv), -1, dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank


def boot_mean(x: Sequence[float], n_rep: int = N_REP, seed: int = SEED) -> Dict[str, float]:
    """Mean with a percentile bootstrap CI over queries."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_rep, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)), "n": int(len(x))}


def rank_summary(r: Sequence[int]) -> Dict[str, float]:
    r = np.asarray(r)
    return {"n": int(len(r)), "median": float(np.median(r)),
            "share_top10": float(np.mean(r < 10)), "share_top50": float(np.mean(r < 50))}


def summarise(P: List[dict]) -> Dict[str, object]:
    def a(k):
        return np.array([p[k] for p in P], dtype=np.float64)
    d_b, d_i, m_b, m_i = a("d_b"), a("d_i"), a("m_b"), a("m_i")
    t_b, t_i, hin = a("t_b"), a("t_i"), a("hi_in_wb")
    shared = a("shared").astype(int)
    broken = [p for p in P if p["t_i"] and not p["t_b"]]
    sw = [p for p in P if p["only_b"]]
    return {
        "n": len(P),
        "overlap": {"shared_mean": float(shared.mean()), "shared_median": float(np.median(shared)),
                    "shared_hist": [int((shared == k).sum()) for k in range(10)],
                    "share_di_subset_wb": float(a("di_subset_wb").mean()), "share_hi_in_wb": float(hin.mean())},
        "hardest_distractor_t": {"W_B": {"mean": float(d_b.mean()), "median": float(np.median(d_b))},
                                 "W_I": {"mean": float(d_i.mean()), "median": float(np.median(d_i))},
                                 "d_b_minus_d_i": boot_mean(d_b - d_i),
                                 "share_b_gt_i": float(np.mean(d_b > d_i)), "share_equal": float(np.mean(d_b == d_i)),
                                 "share_b_lt_i": float(np.mean(d_b < d_i)),
                                 "status": "largely mechanical when h_I is in W_B"},
        "margin": {w: {"mean": float(m.mean()), "median": float(np.median(m)), "share_pos": float(np.mean(m > 0)),
                       "n_exact_tie": int((m == 0).sum()), "t_success": float(t.mean())}
                   for w, m, t in (("W_B", m_b, t_b), ("W_I", m_i, t_i))},
        "t_discordance": {"fixed_hi_in_wb": int(((t_b == 1) & (t_i == 0) & (hin == 1)).sum()),
                          "fixed_hi_not_in_wb": int(((t_b == 1) & (t_i == 0) & (hin == 0)).sum()),
                          "broken": len(broken)},
        "h_b_source": {"share_swapped_in_all": float(a("hb_swapped_in").mean()),
                       "share_swapped_in_broken": float(np.mean([p["hb_swapped_in"] for p in broken])) if broken else None,
                       "status_broken": "mechanical: 1.0 by construction",
                       "gallery_text_rank": rank_summary([p["hb_rank"] for p in P])},
        "swapped": {"n_queries": len(sw),
                    "mean_t_in_minus_out": boot_mean([np.mean(p["only_b_t"]) - np.mean(p["only_i_t"]) for p in sw]),
                    "mean_t_in": float(np.mean([np.mean(p["only_b_t"]) for p in sw])),
                    "mean_t_out": float(np.mean([np.mean(p["only_i_t"]) for p in sw])),
                    "gallery_text_rank_in": rank_summary([r for p in sw for r in p["only_b_rank"]]),
                    "gallery_text_rank_out": rank_summary([r for p in sw for r in p["only_i_rank"]])},
    }


def profiles(recs, img, txt, ref_row, tgt_row, idx_all) -> Dict[int, dict]:
    sim_ref = img @ img.T
    out = {}
    for i in idx_all:
        t_row, ref = int(tgt_row[i]), int(ref_row[i])
        wb = [int(r) for r in recs[i]["candidate_gallery_ids"]]
        wi = r3.top_k_excluding(sim_ref[ref], ref, r3.K)
        if t_row not in wb or t_row not in wi:
            continue
        tv = img @ txt[i]
        p = query_profile(wb, wi, t_row, tv)
        rank = gallery_ranks(tv, ref)
        p["hb_rank"] = int(rank[p["hb"]])
        p["only_b_t"] = [float(tv[r]) for r in p["only_b"]]
        p["only_i_t"] = [float(tv[r]) for r in p["only_i"]]
        p["only_b_rank"] = [int(rank[r]) for r in p["only_b"]]
        p["only_i_rank"] = [int(rank[r]) for r in p["only_i"]]
        out[i] = p
    return out


def preflight(P: Dict[int, dict], idx: Sequence[int], want: dict) -> None:
    sel = [P[i] for i in idx if i in P]
    fixed = sum(1 for p in sel if p["t_b"] and not p["t_i"])
    broken = sum(1 for p in sel if p["t_i"] and not p["t_b"])
    got = (len(sel), fixed, broken)
    exp = (want["both_covered"]["n"], want["both_covered"]["T_WB_minus_WI"]["fixed"],
           want["both_covered"]["T_WB_minus_WI"]["broken"])
    if got != exp:
        raise SystemExit("ABORT: shared-target (n, fixed, broken) %s != §22 %s" % (got, exp))


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
    with open(paths["s22"], encoding="utf-8") as fh:
        s22 = json.load(fh)["W"]

    recs, mp = fc.load_records(paths["cache"])
    if mp or len(recs) != 4181:
        raise SystemExit("ABORT: expected the 4181-query CIRR cache")
    tune, report = fc.split_indices(len(recs))
    with np.load(paths["features"], allow_pickle=False) as z:
        feats = {k: z[k] for k in ("image_emb", "text_emb", "ref_row", "tgt_row")}
    with np.load(paths["vectors"], allow_pickle=False) as z:
        if [int(x) for x in z["query_id"]] != [r["query_id"] for r in recs]:
            raise SystemExit("ABORT: vector dump and cache disagree in query order")
    img, txt = r3.unit_rows(feats["image_emb"]), r3.unit_rows(feats["text_emb"])
    full = list(range(len(recs)))
    P = profiles(recs, img, txt, feats["ref_row"], feats["tgt_row"], full)
    preflight(P, report, s22["report"])
    preflight(P, full, s22["full"])
    print("[23] pre-flight matches §22 (n, fixed, broken), %.0fs" % (time.time() - t0), flush=True)

    res = {"amendment": AMENDMENT, "inputs_sha256": hashes, "status": "descriptive; exploratory; no causal claim",
           "report": summarise([P[i] for i in report if i in P]),
           "full": summarise([P[i] for i in full if i in P])}
    res["source_sha256"] = {f: rf.sha256(os.path.join(_HERE, f)) for f in
                            ("revision_followups4.py", "revision_followups3.py", "revision_followups.py",
                             "fusion_controls.py")}
    res["wall_seconds"] = round(time.time() - t0, 1)
    os.makedirs(args.out)
    out = os.path.join(args.out, "results.json")
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("[done] wrote %s (sha256 %s) in %.0fs" % (out, rf.sha256(out), time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
