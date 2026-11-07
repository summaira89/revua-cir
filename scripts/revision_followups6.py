"""Encoder comparison and post-hoc checks on the fixed CLIP4Cir windows: amendment v3.1 §26. CPU.

  --part posthoc   §26a (exploratory, proposed after §25): (i) the verifier's isolated effect,
                   lambda_T held at A3's tuned value and only lambda_V tuned; (ii) paired utility
                   intervals of the §25 accuracy-tuned policies, clustered by reference image;
                   (iii) direct text-only contrast of the two encoders available so far
  --part encoders  §26c: every encoder's text cosine on the same CLIP4Cir windows -- text-only
                   ranking and tuned bounded fusion -- with paired contrasts between encoders.
                   Encoders: CLIP4Cir (§24 features), ours (deployed checkpoint, §21 T) and each
                   --towers NAME=DIR from encode_towers.py (pretrained, later FC-only)

Windows, first-stage scores, fusion rule, tune/report split and lambda grid are those of §25
throughout (revision_followups5), re-derived and re-checked here.

  (from revua_cir/) python scripts/revision_followups6.py --part posthoc --root .. --out ../revision_work/followups26a_2026-11-04
  (from revua_cir/) python scripts/revision_followups6.py --part encoders --root .. --towers deployed=../revision_work/towers_deployed_2026-11-04 --towers pretrained=../revision_work/towers_pretrained_2026-11-04 --out ../revision_work/followups26c_2026-11-04
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import bbnv_extract as ext  # noqa: E402
import bbnv_policy as pol  # noqa: E402
import fusion_controls as fc  # noqa: E402
import revision_followups as rf  # noqa: E402
import revision_followups2 as r2  # noqa: E402
import revision_followups3 as r3  # noqa: E402
import revision_followups5 as r5  # noqa: E402

AMENDMENT = "v3.1 §26 (2026-11-04)"
GPU_RUN = ("revision_work/external_windows_c_2026-11-04",
           "3320c78cc9115b0ca39a9703bb6f3d8ab71ca8f87ba16c4478ceaa03595d23c7")
S25C = ("revision_work/followups25c_2026-11-04/results.json",
        "df703d87f098ebec4684b339b61fc9dff769cb3ee697f3808981431a1e930719")
TOWERS_SCHEMA = "cirr-towers/1"
N_REP = 10000


def per_query_utility(hits: Sequence[int], base: Sequence[int]) -> List[float]:
    return r3.per_query_utility(hits, base)


def utility_interval(hits: Sequence[int], ref: Sequence[int], cluster: Sequence[int]) -> Dict[str, float]:
    """Mean per-query utility difference vs ref (+1 rescue, -2 break), clustered bootstrap CI."""
    return pol.clustered_bootstrap(per_query_utility(hits, ref), cluster, N_REP)


def tune_v_only(B, T, X, lt: float, pos, idx) -> float:
    """lambda_T fixed; highest R@1 on idx over lambda_V in LAMBDA_GRID, ties to the smaller."""
    best, best_h = None, -1
    for lx in fc.LAMBDA_GRID:
        h = sum(r5.hits(lambda i: r2.three_term(B[i], T[i], X[i], lt, lx), pos, idx))
        if h > best_h:
            best, best_h = lx, h
    return best


def load_s25(root: str):
    hashes, probe, recs, ours, c4c = r5.load_all(root)
    sig = r5.build_signals(recs, ours, c4c, probe)
    s25c_path = os.path.join(root, S25C[0])
    if rf.sha256(s25c_path) != S25C[1]:
        raise SystemExit("ABORT: §25c results hash differs from the lock")
    with open(s25c_path, encoding="utf-8") as fh:
        s25c = json.load(fh)["V"]
    gpu_dir = os.path.join(root, GPU_RUN[0])
    if rf.sha256(os.path.join(gpu_dir, "results.json")) != GPU_RUN[1]:
        raise SystemExit("ABORT: §25b results hash differs from the lock")
    arrays, gres = r5.load_gpu_run(gpu_dir, sig["cand"])
    V = arrays["verification_score"].astype(np.float64).tolist()
    hashes.update({"s25c": S25C[1], "s25b": GPU_RUN[1]})
    return hashes, recs, ours, sig, V, s25c, gres["gallery_sha256"]


# ---------------------------------------------------------------------------
# §26a
# ---------------------------------------------------------------------------
def part_posthoc(sig, V, s25c, cluster) -> dict:
    B, TO, TC, pos, tune, report = sig["B"], sig["T_O"], sig["T_C"], sig["pos"], sig["tune"], sig["report"]
    lam3, lam4 = r5.tune_one(B, TO, pos, tune), r5.tune_one(B, V, pos, tune)
    lt5, lx5 = r5.tune_three(B, TO, V, pos, tune)
    want = s25c["lambda_tuned"]
    if (lam3, lam4, lt5, lx5) != (want["A3"], want["A4"], want["A5"]["lambda_T"], want["A5"]["lambda_V"]):
        raise SystemExit("ABORT: re-derived §25 lambdas differ from the locked §25c")
    lx_fixed = tune_v_only(B, TO, V, lam3, pos, tune)
    arms = {"A1": lambda i: B[i], "A3": lambda i: fc.rule_bounded(B[i], TO[i], lam3),
            "A4": lambda i: fc.rule_bounded(B[i], V[i], lam4),
            "A5": lambda i: r2.three_term(B[i], TO[i], V[i], lt5, lx5),
            "A5_fixedT": lambda i: r2.three_term(B[i], TO[i], V[i], lam3, lx_fixed),
            "T_C_alone": lambda i: TC[i], "T_O_alone": lambda i: TO[i]}
    out = {"status": "exploratory: proposed after seeing §25",
           "lambda": {"A3": lam3, "A4": lam4, "A5": [lt5, lx5], "A5_fixedT": [lam3, lx_fixed]},
           "A5_fixedT_verifier_excluded_by_tuning": lx_fixed == 0.0}
    for half, idx in (("report", report), ("full", list(range(len(pos))))):
        h = {a: r5.hits(fn, pos, idx) for a, fn in arms.items()}
        cl = [cluster[i] for i in idx]
        out[half] = {
            "r1": {a: 100.0 * np.mean(v) for a, v in h.items()},
            "verifier_isolated_A5fixedT_minus_A3": rf.paired(h["A5_fixedT"], h["A3"]),
            "utility_vs_A1": {a: {**rf.outcome(h[a], h["A1"]), "clustered": utility_interval(h[a], h["A1"], cl)}
                              for a in ("A3", "A4", "A5", "A5_fixedT")},
            "utility_A5_minus_A3": pol.clustered_bootstrap(
                [x - y for x, y in zip(per_query_utility(h["A5"], h["A1"]), per_query_utility(h["A3"], h["A1"]))],
                cl, N_REP),
            "encoder_text_only_T_O_minus_T_C": rf.paired(h["T_O_alone"], h["T_C_alone"])}
    out["utility_note"] = "utility of the §25 policies, which were tuned for R@1, not for utility"
    return out


# ---------------------------------------------------------------------------
# §26c
# ---------------------------------------------------------------------------
TOWER_SOURCE = {"deployed": "checkpoint", "pretrained": "pretrained", "fc_only": "checkpoint"}


def load_towers(spec: str, n_q: int, gallery: int, gallery_sha: str) -> Tuple[str, Dict[str, np.ndarray], dict]:
    name, _, d = spec.partition("=")
    if not name or not d:
        raise SystemExit("ABORT: --towers expects NAME=DIR, got %r" % spec)
    with open(os.path.join(d, "results.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    npz = os.path.join(d, res["output_name"])
    if res["schema"] != TOWERS_SCHEMA or rf.sha256(npz) != res["output_files"][res["output_name"]]:
        raise SystemExit("ABORT: %s is not an intact encode_towers.py run" % d)
    if name not in TOWER_SOURCE or res["source"] != TOWER_SOURCE[name]:
        raise SystemExit("ABORT: towers %r have source %r" % (name, res["source"]))
    if res["gallery_sha256"] != gallery_sha:
        raise SystemExit("ABORT: %s used a different gallery from the §25b run" % d)
    if name == "deployed":
        par = res["parity_vs_phase_b"]
        if par is None or max(par.values()) > ext.COS_TOL:
            raise SystemExit("ABORT: the deployed towers run does not reproduce Phase B: %r" % par)
    with np.load(npz, allow_pickle=False) as z:
        arr = {k: z[k] for k in ("image_emb", "text_emb")}
    if arr["text_emb"].shape[0] != n_q or arr["image_emb"].shape[0] != gallery:
        raise SystemExit("ABORT: %s has %s/%s rows" % (d, arr["text_emb"].shape, arr["image_emb"].shape))
    prov_ = {"dir": os.path.basename(os.path.normpath(d)), "results_sha256": rf.sha256(os.path.join(d, "results.json")),
             "source": res["source"], "checkpoint_sha256": res["manifest"]["checkpoint_sha256"],
             "parity_vs_phase_b": res["parity_vs_phase_b"]}
    return name, arr, prov_


def part_encoders(sig, enc: Dict[str, List[List[float]]], pairs: Sequence[Tuple[str, str]]) -> dict:
    B, pos, tune, report = sig["B"], sig["pos"], sig["tune"], sig["report"]
    lam = {e: r5.tune_one(B, T, pos, tune) for e, T in enc.items()}
    arms = {"A1": lambda i: B[i]}
    for e, T in enc.items():
        arms["text_" + e] = (lambda T_: lambda i: T_[i])(T)
        arms["fused_" + e] = (lambda T_, l_: lambda i: fc.rule_bounded(B[i], T_[i], l_))(T, lam[e])
    out = {"lambda_tuned": lam, "pairs": [list(p) for p in pairs]}
    for half, idx in (("report", report), ("full", list(range(len(pos))))):
        summ, h = r5.summarise_arms(arms, pos, idx, "A1")
        out[half] = {"arms": summ,
                     "fused_contrasts": r5.contrasts(h, [("%s_vs_%s" % p, "fused_" + p[0], "fused_" + p[1]) for p in pairs]),
                     "text_only_contrasts": r5.contrasts(h, [("%s_vs_%s" % p, "text_" + p[0], "text_" + p[1])
                                                            for p in pairs])}
    out["contrast_status"] = "report half primary; Holm within each family (fused, text-only)"
    return out


def encoder_pairs(names: Sequence[str]) -> List[Tuple[str, str]]:
    """Fixed contrasts: ours vs pretrained, pretrained vs CLIP4Cir, and (if present) ours vs FC-only
    and FC-only vs pretrained."""
    pairs = [("ours", "pretrained"), ("pretrained", "clip4cir")]
    if "fc_only" in names:
        pairs += [("ours", "fc_only"), ("fc_only", "pretrained")]
    missing = {n for p in pairs for n in p} - set(names)
    if missing:
        raise SystemExit("ABORT: encoders %s not supplied" % sorted(missing))
    return pairs


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", required=True, choices=("posthoc", "encoders"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--towers", action="append", default=[],
                    help="NAME=DIR (encoders part): deployed (required parity run), pretrained, fc_only")
    args = ap.parse_args(argv)
    if (args.part == "encoders") != bool(args.towers):
        ap.error("--towers is required for --part encoders and only for it")
    if os.path.exists(args.out):
        ap.error("%s exists; refusing to overwrite" % args.out)
    t0 = time.time()
    hashes, recs, ours, sig, V, s25c, gallery_sha = load_s25(args.root)
    cluster = ours["ref_row"].astype(np.int64).tolist()
    print("[26] §25 windows, T and verifier scores re-derived and checked, %.0fs" % (time.time() - t0), flush=True)
    res = {"amendment": AMENDMENT, "part": args.part, "inputs_sha256": hashes}
    if args.part == "posthoc":
        res["P"] = part_posthoc(sig, V, s25c, cluster)
    else:
        enc = {"clip4cir": sig["T_C"], "ours": sig["T_O"]}
        res["towers"] = {}
        for spec in args.towers:
            name, arr, prov_ = load_towers(spec, len(recs), len(ours["image_emb"]), gallery_sha)
            if name in res["towers"]:
                raise SystemExit("ABORT: repeated encoder name %r" % name)
            res["towers"][name] = prov_
            if name != "deployed":      # the deployed run is a parity check; ours is the §21 T
                enc[name] = r5.window_cosines(arr["text_emb"], arr["image_emb"], sig["cand"]).tolist()
        if "deployed" not in res["towers"]:
            raise SystemExit("ABORT: --towers deployed=DIR (the encode_towers.py parity run) is required")
        res["E"] = part_encoders(sig, enc, encoder_pairs(list(enc)))
    res["source_sha256"] = {f: rf.sha256(os.path.join(_HERE, f)) for f in
                            ("revision_followups6.py", "revision_followups5.py", "revision_followups3.py",
                             "revision_followups2.py", "revision_followups.py", "fusion_controls.py", "bbnv_policy.py")}
    res["wall_seconds"] = round(time.time() - t0, 1)
    os.makedirs(args.out)
    out = os.path.join(args.out, "results.json")
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("[done] wrote %s (sha256 %s) in %.0fs" % (out, rf.sha256(out), time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
