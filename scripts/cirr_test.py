"""CIRR test-server evaluation of frozen systems (v3.1 §18). CPU.

  freeze   fit, on ALL of CIRR validation, the strongest simple policy (T3:
           R, BM, SW or A by validation utility; ties R, BM, SW, A) and the
           forward-only gate (T4: the §14 procedure, seed 0), and write them
           to a fresh, verified, hashed run. Must run before any test output.
  build    from a split's window file (cirr_split_windows.py) and the frozen
           policies, write the recall submission of every system to submit
           (T1, T2, T3 unless it equals T1 or T2, T4), each validated against
           the server's format. On validation (--split val) it also scores
           the files locally and requires base R@1 37.072 and ungated 39.369.

Usage
-----
  python scripts/cirr_test.py freeze --windows ../bbnv_runs/windows_cirr_v1 --out-dir ../bbnv_runs/cirr_test_freeze
  python scripts/cirr_test.py build --freeze ../bbnv_runs/cirr_test_freeze --split-windows ../bbnv_runs/split_windows_test1 \\
      --scored ../bbnv_runs/test1_scored --out-dir ../cirr_test_submission
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bbnv_analyze as ana  # noqa: E402
import bbnv_policy as pol  # noqa: E402
import bbnv_provenance as prov  # noqa: E402
import bbnv_score as scr  # noqa: E402
import bbnv_train as trn  # noqa: E402
import bbnv_transfer as tfr  # noqa: E402
from fusion_suite import r_bounded  # noqa: E402

FREEZE_SCHEMA = "cirr-test-freeze/1"
SPLIT_SCHEMA = "cirr-split-windows/1"
SCORED_SCHEMA = "cirr-test-scored/1"
SYSTEMS = ("T1", "T2", "T3", "T4")
GATE_SEED = 0
MAX_IDS = 50
MAX_BYTES = 5 * 1024 * 1024
VAL_ANCHORS = {"T1": 37.072, "T2": 39.369}


class TestError(ValueError):
    pass


# ----------------------------------------------------------------- freeze --
def choose_simple(rows: Sequence[Mapping], views: Sequence[Mapping]) -> Dict[str, object]:
    """T3 on all validation queries: R, BM, SW, A by utility (ties in that order)."""
    base = [int(v["hit_base"]) for v in views]
    bm, t_bm = pol.fit_bm(views)
    lam = pol.sw_param(rows)
    hits = {"R": base,
            "BM": [pol.outcome(v, a) for v, a in zip(views, bm(views))],
            "SW": pol.sw_hits(rows, lam),
            "A": [pol.outcome(v, bool(v["changed"])) for v in views]}
    u = {k: pol.utility(h, base) for k, h in hits.items()}
    best = max(u.values())
    choice = next(k for k in pol.SIMPLE if u[k] == best)
    return {"choice": choice, "utility": u, "bm_threshold": pol.threshold_record(t_bm), "sw_lambda": lam}


def cmd_freeze(args) -> int:
    res, rows = ana.load_windows(args.windows)
    if res["target"] != "cirr_val":
        raise TestError("policies are frozen on CIRR validation")
    views = ana.gate_views(rows, float(res["lambda_star_frozen"]), GATE_SEED)
    t3 = choose_simple(rows, views)
    names = ana.feature_names("forward_only")
    C, cv = ana.select_C(views, names)
    _, params = ana._fit_gate(views, names, C)
    out = {"schema": FREEZE_SCHEMA, "amendment": "v3.1 §18", "fitted_on": "all of CIRR validation (n=4181)",
           "lambda_star": float(res["lambda_star_frozen"]), "gate_seed": GATE_SEED,
           "T3": t3, "T4": {"gate": "forward_only", "features": names, "params": params, "cv_by_C": cv},
           "inputs": {"windows_results_sha256": prov.sha256_file(os.path.join(args.windows, prov.RESULTS_NAME))},
           "manifest": {"schema": prov.MANIFEST_SCHEMA, "script": "cirr_test.py freeze", "args": vars(args),
                        "source_sha256": prov.source_hashes(prov.BBNV_SOURCES + ("bbnv_policy.py", "bbnv_transfer.py",
                                                                                 "cirr_test.py"))}}
    prov.write_run(args.out_dir, out, {})
    prov.verify_run(args.out_dir)
    print("[freeze] T3 = %s (validation utility %s) | T4 C = %g | wrote %s, results.json sha256 %s"
          % (t3["choice"], {k: round(v, 4) for k, v in t3["utility"].items()}, C, args.out_dir,
             prov.sha256_file(os.path.join(args.out_dir, prov.RESULTS_NAME))))
    return 0


# ------------------------------------------------------------------ build --
def fused_order(base: Sequence[float], ver: Sequence[float], lam: float = 1.0) -> List[int]:
    """Window positions in bounded-fusion order (ties by position)."""
    s = r_bounded(list(base), list(ver), lam)
    return sorted(range(len(s)), key=lambda i: (-s[i], i))


def system_ranking(top50: Sequence[int], window_order: Sequence[int]) -> List[int]:
    """The window (top 10) in the given order, then base ranks 11-50."""
    k = len(window_order)
    return [top50[i] for i in window_order] + list(top50[k:])


def validate_submission(sub: Mapping[str, object], pairids: Sequence[int], image_ids: set) -> None:
    """The server's format: version/metric plus every pair ID exactly once,
    each with at most 50 distinct known image IDs as strings, at most 5 MiB."""
    if sub.get("version") != "rc2" or sub.get("metric") != "recall":
        raise TestError("version must be rc2 and metric recall")
    keys = set(sub) - {"version", "metric"}
    if keys != {str(p) for p in pairids}:
        raise TestError("the submission must hold every pair ID exactly once and nothing else")
    for k in keys:
        v = sub[k]
        if (not isinstance(v, list) or not 1 <= len(v) <= MAX_IDS or len(set(v)) != len(v)
                or not all(isinstance(x, str) and x in image_ids for x in v)):
            raise TestError("pair %s: predictions must be 1-50 distinct known image IDs as strings" % k)
    if len(compact(sub).encode("utf-8")) > MAX_BYTES:
        raise TestError("submission exceeds 5 MiB")


def compact(obj: Mapping) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False)


def load_split(split_dir: str) -> Dict[str, object]:
    res = prov.verify_run(split_dir)
    if res.get("schema") != SPLIT_SCHEMA:
        raise TestError("%s is not a %s run" % (split_dir, SPLIT_SCHEMA))
    with np.load(os.path.join(split_dir, "windows.npz"), allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    if np.any(arr["ref_row"] < 0):
        raise TestError("a query reference is not in its split gallery")
    return {"results": res, "arr": arr}


def val_targets(val_features: str, n: int) -> np.ndarray:
    """Dataset target gallery rows of CIRR validation (Phase B features)."""
    prov.verify_run(val_features)
    with np.load(os.path.join(val_features, "features.npz"), allow_pickle=False) as z:
        tgt = z["tgt_row"].astype(np.int64)
    if len(tgt) != n:
        raise TestError("validation targets and windows disagree in query count")
    return tgt


# ------------------------------------------------------------------ score --
def cmd_score(args) -> int:
    """Frozen BBNV head scores for every window of a split (same rows as §13)."""
    import torch
    torch.set_num_threads(args.threads)
    t0 = time.time()
    sp = load_split(args.split_windows)
    arr, k = sp["arr"], int(sp["results"]["k"])
    tr = trn.load_inputs(args.train_features, args.manifest_dir)
    ckpt = tr["features_results"]["manifest"]["checkpoint_sha256"]
    if sp["results"]["manifest"]["checkpoint_sha256"] != ckpt:
        raise TestError("split windows and heads come from different checkpoints")
    heads = scr.load_heads(args.heads_dir, args.train_features)
    tf = tr["feat"]
    banks = scr.make_banks(tf["image_ids"], tf["partition"])
    nb = scr.BANK_SIZE
    emb = np.concatenate([tf["image_emb"][banks["B_R"]], tf["image_emb"][banks["B_C"]], arr["image_emb"]])
    bank_r, bank_c = np.arange(nb), np.arange(nb, 2 * nb)
    n = len(arr["pairid"])
    tgt = val_targets(args.val_features, n) if args.val_features else None
    rows = []
    for q in range(n):
        cand = arr["top50"][q, :k].astype(np.int64)
        tp = int(list(cand).index(tgt[q])) if tgt is not None and tgt[q] in cand else -1
        rows.append({"query": q, "pairid": int(arr["pairid"][q]), "split": None, "target_pos": tp,
                     "window": cand.tolist(), "base_score": arr["top50_scores"][q, :k].astype(float).tolist(),
                     "edit_cos": arr["edit_cos"][q].astype(float).tolist(),
                     "verification_score": arr["verification_score"][q].astype(float).tolist(),
                     "heads": scr.score_query(heads, emb, arr["text_emb"][q].astype(np.float64),
                                              2 * nb + int(arr["ref_row"][q]), bank_r, bank_c, 2 * nb + cand)})
        if (q + 1) % 500 == 0:
            print("[score] %d/%d queries, %.0fs" % (q + 1, n, time.time() - t0), flush=True)
    info = {}
    if args.compare_windows:
        _, ref_rows = ana.load_windows(args.compare_windows)
        diff = max(max(abs(a - b) for a, b in zip(r["heads"][h][f], o["heads"][h][f]))
                   for r, o in zip(rows, ref_rows) for h in r["heads"] for f in r["heads"][h])
        if len(ref_rows) != n or any(r["window"] != o["window"] or r["target_pos"] != o["target_pos"]
                                     for r, o in zip(rows, ref_rows)) or diff > 1e-6:
            raise TestError("head scores do not reproduce %s (max diff %.3g)" % (args.compare_windows, diff))
        info["reproduces_windows_max_abs_diff"] = diff
    res = {"schema": SCORED_SCHEMA, "split": sp["results"]["split"], "n_queries": n, "k": k,
           "lambda_star_frozen": 0.0, **info,
           "inputs": {"split_windows_results_sha256": prov.sha256_file(os.path.join(args.split_windows,
                                                                                    prov.RESULTS_NAME)),
                      "heads_results_sha256": prov.sha256_file(os.path.join(args.heads_dir, prov.RESULTS_NAME)),
                      "checkpoint_sha256": ckpt},
           "manifest": {"schema": prov.MANIFEST_SCHEMA, "script": "cirr_test.py score", "args": vars(args)},
           "wall_seconds": round(time.time() - t0, 1)}
    prov.write_run(args.out_dir, res, {"windows.jsonl": rows})
    prov.verify_run(args.out_dir)
    print("[score] wrote and verified %s in %.0fs %s" % (args.out_dir, time.time() - t0, info), flush=True)
    return 0


# ------------------------------------------------------------------ build --
def rankings(freeze: Mapping, arr: Mapping[str, np.ndarray], rows: Sequence[Mapping], k: int) -> Dict[str, List[List[int]]]:
    """Gallery-row rankings (top 50) of T1-T4 for every query (§18)."""
    views = ana.gate_views(rows, freeze["lambda_star"], freeze["gate_seed"])
    t4 = freeze["T4"]
    accept4 = tfr.frozen_gate(t4["params"], t4["features"])(views)
    t3 = freeze["T3"]
    if t3["choice"] == "BM" and t3["bm_threshold"] in ("+inf", "-inf"):
        raise TestError("a BM threshold of +/-inf would make T3 equal to T1 or T2")
    out = {s: [] for s in SYSTEMS}
    for q, (row, v) in enumerate(zip(rows, views)):
        top = arr["top50"][q].tolist()
        fused = system_ranking(top, fused_order(row["base_score"], row["verification_score"], 1.0))
        out["T1"].append(top)
        out["T2"].append(fused)
        if t3["choice"] == "SW":
            out["T3"].append(system_ranking(top, fused_order(row["base_score"], row["verification_score"],
                                                             float(t3["sw_lambda"]))))
        elif t3["choice"] == "BM":
            take = v["changed"] and v["feat"]["base_margin"] <= float(t3["bm_threshold"])
            out["T3"].append(fused if take else top)
        out["T4"].append(fused if (v["changed"] and accept4[q]) else top)
    if not out["T3"]:
        del out["T3"]                      # T3 is R (= T1) or A (= T2): not submitted separately
    return out


def recall_at(rank: Sequence[Sequence[int]], tgt: np.ndarray, ks=(1, 5, 10, 50)) -> Dict[str, float]:
    return {"R@%d" % kk: 100.0 * float(np.mean([t in r[:kk] for r, t in zip(rank, tgt)])) for kk in ks}


def cmd_build(args) -> int:
    fres = prov.verify_run(args.freeze)
    if fres.get("schema") != FREEZE_SCHEMA:
        raise TestError("%s is not the frozen policy run" % args.freeze)
    sp = load_split(args.split_windows)
    sres = prov.verify_run(args.scored)
    if sres.get("schema") != SCORED_SCHEMA or sres["inputs"]["split_windows_results_sha256"] != prov.sha256_file(
            os.path.join(args.split_windows, prov.RESULTS_NAME)):
        raise TestError("the scored run is not of these split windows")
    arr, k = sp["arr"], int(sp["results"]["k"])
    rows = ana.man.read_jsonl(os.path.join(args.scored, "windows.jsonl"))
    if [r["pairid"] for r in rows] != arr["pairid"].tolist():
        raise TestError("scored rows and split windows disagree in pair order")
    ranks = rankings(fres, arr, rows, k)
    ids = arr["image_ids"]
    known = set(ids.tolist())
    pairids = arr["pairid"].tolist()
    subs = {}
    for s, rk in ranks.items():
        sub = {"version": "rc2", "metric": "recall"}
        sub.update({str(p): [str(ids[i]) for i in r] for p, r in zip(pairids, rk)})
        validate_submission(sub, pairids, known)
        subs[s] = sub
    info: Dict[str, object] = {"systems": sorted(subs), "T3_choice": fres["T3"]["choice"]}
    if sp["results"]["split"] == "val":
        tgt = val_targets(args.val_features, len(pairids))
        info["val_recall"] = {s: recall_at(rk, tgt) for s, rk in ranks.items()}
        for s, anchor in VAL_ANCHORS.items():
            if abs(info["val_recall"][s]["R@1"] - anchor) > 0.001:
                raise TestError("%s validation R@1 %.3f does not reproduce %.3f" % (s, info["val_recall"][s]["R@1"],
                                                                                    anchor))
    staging = ana.man.staging_dir(args.out_dir)
    prov.prepare_fresh_dir(staging)
    files = {}
    for s, sub in subs.items():
        name = "%s_pred_ranks_recall_%s.json" % (sp["results"]["split"], s)
        files[name] = prov._atomic_write_text(os.path.join(staging, name), compact(sub))
        with open(os.path.join(staging, name), encoding="utf-8") as fh:
            validate_submission(json.load(fh), pairids, known)
    res = {"schema": "cirr-test-submission/1", "amendment": "v3.1 §18", "split": sp["results"]["split"], **info,
           "inputs": {"freeze_results_sha256": prov.sha256_file(os.path.join(args.freeze, prov.RESULTS_NAME)),
                      "scored_results_sha256": prov.sha256_file(os.path.join(args.scored, prov.RESULTS_NAME))},
           "output_files": files}
    prov.atomic_write_json(os.path.join(staging, prov.RESULTS_NAME), res)
    prov.verify_run(staging)
    os.rmdir(args.out_dir)
    os.replace(staging, args.out_dir)
    print("[build] %s: wrote %s | %s" % (sp["results"]["split"], sorted(files), info.get("val_recall", "")), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    fr = sub.add_parser("freeze")
    fr.add_argument("--windows", required=True)
    fr.add_argument("--out-dir", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--split-windows", required=True)
    sc.add_argument("--train-features", required=True)
    sc.add_argument("--manifest-dir", required=True)
    sc.add_argument("--heads-dir", required=True)
    sc.add_argument("--val-features", help="val only: targets")
    sc.add_argument("--compare-windows", help="val only: the §13 window run to reproduce")
    sc.add_argument("--out-dir", required=True)
    sc.add_argument("--threads", type=int, default=6)
    bd = sub.add_parser("build")
    bd.add_argument("--freeze", required=True)
    bd.add_argument("--split-windows", required=True)
    bd.add_argument("--scored", required=True)
    bd.add_argument("--val-features", help="val only: targets for the local dry run")
    bd.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    try:
        prov.prepare_fresh_dir(args.out_dir)
    except IOError as exc:
        ap.error(str(exc))
    t0 = time.time()
    rc = {"freeze": cmd_freeze, "score": cmd_score, "build": cmd_build}[args.command](args)
    print("[cirr_test] done in %.0fs" % (time.time() - t0))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
